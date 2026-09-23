"""bench_abd_gpu_compile_included.py -- Addendum 143, Part 2.

Extends Addendum 141/142's A/B/D comparison (same construction, same
correctness fixes) to n=20..29 (or up to whatever VRAM ceiling
find_vram_ceiling.py reports), and -- unlike Addendum 141/142, which
deliberately excluded it -- now TIMES each route's own compile step, on
lightning.gpu only, reporting compile time, median per-call execution time,
and their sum, separately and combined.

Requires psf_compile.py and psf_smart_layout.py in the same directory.

Usage (inside psf_zero_wsl_env_312, after running find_vram_ceiling.py to
set N_VALUES below to the actual ceiling):
    python bench_abd_gpu_compile_included.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import statistics
import time

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler import CouplingMap

import psf_compile
from psf_smart_layout import smart_vf2_layout

N_VALUES = (20, 24, 25, 26, 27, 28)
# NOTE: an earlier version of this list went up to 29, following the
# pre-registration's P1 (28 or 29). Both n=29 and n=30 were found to
# "succeed" without an exception, but n=29 took 4,567 ms for a trivial
# single-Hadamard-plus-CNOT-chain circuit against n=28's 300 ms -- a 15x
# jump inconsistent with every smaller step, and n=30 failed outright on a
# more complex circuit (though it had "succeeded" on the trivial one one
# run earlier). The most likely explanation: WSL2's shared-GPU-memory
# mechanism lets CUDA allocations exceed the RTX 4070's 12 GB VRAM by
# spilling into system RAM, silently, without raising an error -- so a
# circuit "running" at n=29 is not evidence it fits in VRAM. Capped here at
# n=28 (4.29 GB, comfortably inside 12 GB) so this addendum's timed
# comparison measures genuine VRAM-resident execution, not disk/RAM
# swapping mislabelled as GPU speed.
LAYERS = 6
COMPILE_SEED = 1234
ANGLE_SEED = 0
REPEATS = 10
BASIS = ["rz", "sx", "x", "cx"]


def grid_edges(rows, cols):
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return edges


def fixed_angles(n):
    rng = np.random.default_rng(ANGLE_SEED)
    return rng.uniform(-np.pi, np.pi, size=2 * n * LAYERS)


def build_parameterized(n):
    """Symbolic Parameter objects -- compiling this does NOT see specific
    numeric values, so it cannot achieve the CX-count reduction (route A)."""
    qc = QuantumCircuit(n)
    params = []
    for layer in range(LAYERS):
        for q in range(n):
            p1, p2 = Parameter(f"ry_{layer}_{q}"), Parameter(f"rz_{layer}_{q}")
            params += [p1, p2]
            qc.ry(p1, q)
            qc.rz(p2, q)
        for a in range(0, n - 1, 2):
            qc.cx(a, a + 1)
    return qc, params


def build_bound(n, angles):
    """The same circuit, numeric angles bound in from the start -- routes B
    and D compile THIS, so synthesis can see the actual values."""
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(LAYERS):
        for q in range(n):
            qc.ry(float(angles[k]), q); k += 1
            qc.rz(float(angles[k]), q); k += 1
        for a in range(0, n - 1, 2):
            qc.cx(a, a + 1)
    return qc


def qiskit_circuit_to_pennylane_ops(qc):
    ops = []
    for inst in qc.data:
        name = inst.operation.name
        wires = [qc.find_bit(q).index for q in inst.qubits]
        params = list(inst.operation.params)
        if name == "rz":
            ops.append(qml.RZ(float(params[0]), wires=wires[0]))
        elif name == "sx":
            ops.append(qml.SX(wires=wires[0]))
        elif name == "x":
            ops.append(qml.PauliX(wires=wires[0]))
        elif name == "cx":
            ops.append(qml.CNOT(wires=wires))
        elif name == "id":
            pass
        else:
            raise ValueError(f"unexpected gate '{name}' outside the basis {BASIS}")
    return ops


def route_a(n, angles, cm):
    """Compile once (parameterized), THEN bind."""
    qc, params = build_parameterized(n)
    compiled_param = transpile(qc, coupling_map=cm, basis_gates=BASIS, optimization_level=3,
                               seed_transpiler=COMPILE_SEED)
    perm = list(compiled_param.layout.final_index_layout(filter_ancillas=True))
    # Bind the fixed angles (in the same order Parameters were created) into
    # the COMPILED circuit -- the whole point of route A.
    binding = dict(zip(params, [float(a) for a in angles]))
    bound = compiled_param.assign_parameters(binding)
    return bound, perm[0]


def route_b(n, angles, cm):
    """Bind first, then Qiskit re-compile."""
    qc = build_bound(n, angles)
    out = transpile(qc, coupling_map=cm, basis_gates=BASIS, optimization_level=3,
                    seed_transpiler=COMPILE_SEED)
    perm = list(out.layout.final_index_layout(filter_ancillas=True))
    return out, perm[0]


def route_d(n, angles, cm, pairs, device_size):
    """Bind first, then PSF-Zero layout + synthesis."""
    qc = build_bound(n, angles)
    layout_map, info = smart_vf2_layout(cm, pairs, n)
    if layout_map is None or any(i not in layout_map for i in range(n)):
        raise RuntimeError(
            f"no perfect layout found for n={n} on this device (info={info}); "
            "this device may be too saturated for this instance -- see "
            "Addenda 88-136 for that separate, deliberate experiment.")
    perm = [layout_map[i] for i in range(n)]
    with contextlib.redirect_stdout(io.StringIO()):
        synth = psf_compile.compile(qc, verify=False, entangling_basis="cx")
    placed = QuantumCircuit(device_size)
    placed.compose(synth, qubits=perm, inplace=True)
    out = transpile(placed, basis_gates=BASIS, optimization_level=1)
    return out, perm[0]


def run_and_time(ops, device_size, device_name, measure_wire):
    dev = qml.device(device_name, wires=device_size)

    def circuit():
        for op in ops:
            qml.apply(op)
        return qml.expval(qml.PauliZ(measure_wire))

    qnode = qml.QNode(circuit, dev)
    val = qnode()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        qnode()
        times.append(time.perf_counter() - t0)
    return float(val), times


def main():
    rows = []
    for n in N_VALUES:
        # Two earlier fix attempts both failed, caught before running the
        # expensive comparison: (1) the next larger perfect-square grid
        # (e.g. 6x6=36 for n=25) jumps by 11 qubits, needing 2**36
        # amplitudes -- far beyond the 12 GB budget this addendum exists to
        # respect; (2) keeping a SQUARE grid but requiring only n+2 qubits
        # still forces the same jump, since square grids only exist at
        # side**2 qubit counts. The actual fix: this circuit's own `pairs`
        # only need ADJACENT connectivity (2i, 2i+1), so a 1D CHAIN
        # coupling map -- not a 2D grid -- suffices, and a chain's qubit
        # count can be set to exactly n+2, in increments of 1, with no
        # wasted jump.
        device_size = n + 2
        cm = CouplingMap.from_line(device_size)  # 1D chain, exactly device_size qubits
        pairs = [(a, a + 1) for a in range(0, n - 1, 2)]
        angles = fixed_angles(n)

        print(f"{'='*94}\nn={n} (chain, device_size={device_size}), {LAYERS} layers\n{'='*94}")

        compiled = {}
        for label, fn in (("A", route_a), ("B", route_b), ("D", route_d)):
            t0 = time.perf_counter()
            if label == "D":
                out, wire0 = fn(n, angles, cm, pairs, device_size)
            else:
                out, wire0 = fn(n, angles, cm)
            compile_s = time.perf_counter() - t0
            cx = sum(1 for inst in out.data if len(inst.qubits) == 2)
            compiled[label] = (out, wire0, cx, compile_s)
            print(f"  route {label}: 2q gates={cx}, compile {compile_s*1000:9.3f} ms, "
                  f"logical qubit 0 at physical wire {wire0}")

        results = {}
        for label, (out, wire0, cx, compile_s) in compiled.items():
            ops = qiskit_circuit_to_pennylane_ops(out)
            val, times = run_and_time(ops, device_size, "lightning.gpu", wire0)
            med_exec = statistics.median(times) * 1000
            combined = compile_s * 1000 + med_exec
            results[label] = (val, compile_s * 1000, med_exec, combined)
            print(f"  {label}-lightning.gpu  compile {compile_s*1000:9.3f} ms  "
                  f"+ exec {med_exec:9.3f} ms  = {combined:9.3f} ms  value={val:.10f}")
            rows.append(dict(n=n, route=label, cx=cx, compile_ms=compile_s * 1000,
                             exec_ms=med_exec, combined_ms=combined, value=val))

        vals = [v for v, _, _, _ in results.values()]
        print(f"  max value spread across all three: {max(vals) - min(vals):.2e}")
        combined_b = results["B"][3]
        combined_d = results["D"][3]
        print(f"  combined (compile+exec): B={combined_b:.3f} ms  D={combined_d:.3f} ms  "
              f"D/B={combined_d/combined_b:.4f}x")
        compile_b, compile_d = results["B"][1], results["D"][1]
        print(f"  compile only: B={compile_b:.3f} ms  D={compile_d:.3f} ms  D/B={compile_d/compile_b:.4f}x")
        print()

    out = "compile_included_2026-09-23.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
