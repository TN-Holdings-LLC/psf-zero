"""bench_abd_gpu.py -- Addendum 141.

Redo of Addendum 139/140 with the CORRECT three-route comparison
(A/B/D, matching Addenda 110-126's own established framework):

  A (compile-once)  : a PARAMETERIZED circuit compiled once with Qiskit,
                      numeric values bound only after compilation --
                      cannot achieve CX reduction, by construction.
  B (Qiskit re-compile): bound values, then Qiskit transpile each time.
  D (PSF-Zero)       : bound values, then PSF-Zero layout + synthesis.

Both bugs found in Addendum 140 (device sizing across a coupling map larger
than the circuit; measuring the wrong physical wire after layout) are fixed
from the start here.

Requires psf_compile.py and psf_smart_layout.py in the same directory.

Usage (inside psf_zero_wsl_env_312):
    python bench_abd_gpu.py
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

N_VALUES = (20, 24)
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
    layout_map, _ = smart_vf2_layout(cm, pairs, n)
    if layout_map is None:
        raise RuntimeError("no perfect layout found for this instance")
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
        side = 1
        while side * side < n:
            side += 1
        device_size = side * side
        cm = CouplingMap.from_grid(side, side)
        pairs = [(a, a + 1) for a in range(0, n - 1, 2)]
        angles = fixed_angles(n)

        print(f"{'='*94}\nn={n} (grid {side}x{side}, device_size={device_size}), {LAYERS} layers\n{'='*94}")

        compiled = {}
        for label, fn in (("A", route_a), ("B", route_b), ("D", route_d)):
            if label == "D":
                out, wire0 = fn(n, angles, cm, pairs, device_size)
            else:
                out, wire0 = fn(n, angles, cm)
            cx = sum(1 for inst in out.data if len(inst.qubits) == 2)
            compiled[label] = (out, wire0, cx)
            print(f"  route {label}: 2q gates={cx}, logical qubit 0 at physical wire {wire0}")

        results = {}
        for label, (out, wire0, cx) in compiled.items():
            ops = qiskit_circuit_to_pennylane_ops(out)
            for device in ("lightning.qubit", "lightning.gpu"):
                val, times = run_and_time(ops, device_size, device, wire0)
                med = statistics.median(times) * 1000
                results[(label, device)] = (val, med)
                print(f"  {label}-{device:16s} median {med:9.3f} ms  value={val:.10f}")
                rows.append(dict(n=n, route=label, device=device, cx=cx, median_ms=med, value=val))

        vals = [v for v, _ in results.values()]
        print(f"  max value spread across all six: {max(vals) - min(vals):.2e}")
        for label in ("A", "B", "D"):
            r = results[(label, "lightning.gpu")][1] / results[(label, "lightning.qubit")][1]
            print(f"  {label}: GPU/CPU = {r:.4f}x")
        gpu_a = results[("A", "lightning.gpu")][1]
        for label in ("B", "D"):
            r = results[(label, "lightning.gpu")][1] / gpu_a
            print(f"  GPU execution, {label}/A = {r:.4f}x")
        print()

    out = "abd_gpu_2026-09-23.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
