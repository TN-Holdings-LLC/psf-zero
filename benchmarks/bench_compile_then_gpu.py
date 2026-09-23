"""bench_compile_then_gpu.py -- Addendum 139.

Chains PSF-Zero's compilation with lightning.gpu execution for the first
time: compile a circuit (Qiskit compile-once vs PSF-Zero), then actually
execute the compiled result on lightning.qubit (CPU) and lightning.gpu
(GPU), at n=20 and n=24 -- the sizes Addendum 138 found GPU-favoured.

Requires psf_compile.py and psf_smart_layout.py in the same directory
(the verified, post-Addendum-116 versions).

Usage (inside psf_zero_wsl_env_312):
    python bench_compile_then_gpu.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import statistics
import time

import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

import psf_compile
from psf_smart_layout import smart_vf2_layout

N_VALUES = (20, 24)
LAYERS = 6
COMPILE_SEED = 1234
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


def build_qiskit_ansatz(n):
    """Same-pair redundant block: LAYERS repetitions of ry/rz/cx on fixed
    disjoint pairs, so PSF-Zero's own re-synthesis has redundant CXs to
    remove (Addenda 109-118's own construction)."""
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(0)
    for _ in range(LAYERS):
        for q in range(n):
            qc.ry(float(rng.uniform(-np.pi, np.pi)), q)
            qc.rz(float(rng.uniform(-np.pi, np.pi)), q)
        for a in range(0, n - 1, 2):
            qc.cx(a, a + 1)
    return qc


def qiskit_circuit_to_pennylane_ops(qc):
    """Converts a Qiskit circuit's basis-gate operations into PennyLane ops,
    without using pennylane_qiskit's own converter (kept independent so this
    script has no pennylane-qiskit dependency)."""
    ops = []
    for inst in qc.data:
        name = inst.operation.name
        wires = [qc.find_bit(q).index for q in inst.qubits]
        params = list(inst.operation.params)
        if name == "rz":
            ops.append(qml.RZ(params[0], wires=wires[0]))
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


def compile_route_a(qc, cm):
    """Returns (circuit, physical_wire_of_logical_qubit_0). Qiskit's own
    layout pass may place logical qubit 0 on any physical qubit, and pads
    unused physical qubits as idle ancillas when the coupling map is larger
    than the circuit -- so PauliZ(0) on the OUTPUT circuit is only correct
    if physical wire 0 happens to be where logical qubit 0 landed, which is
    not guaranteed. Tracked via TranspileLayout.final_index_layout(), the
    same method verified and used throughout Addenda 103/116-118/121-126."""
    out = transpile(qc, coupling_map=cm, basis_gates=BASIS, optimization_level=3,
                    seed_transpiler=COMPILE_SEED)
    perm = list(out.layout.final_index_layout(filter_ancillas=True))
    return out, perm[0]


def compile_route_d(qc, cm, n, pairs, device_size):
    """Returns (circuit, physical_wire_of_logical_qubit_0). Same reasoning
    as compile_route_a: smart_vf2_layout's own `perm` already gives the
    physical placement of every logical qubit directly, so perm[0] is the
    physical wire logical qubit 0 landed on -- no separate lookup needed."""
    layout_map, _ = smart_vf2_layout(cm, pairs, n)
    if layout_map is None:
        raise RuntimeError("no perfect layout found for this instance")
    perm = [layout_map[i] for i in range(n)]
    with contextlib.redirect_stdout(io.StringIO()):
        synth = psf_compile.compile(qc, verify=False, entangling_basis="cx")
    placed = QuantumCircuit(device_size)
    placed.compose(synth, qubits=perm, inplace=True)
    out = transpile(placed, basis_gates=BASIS, optimization_level=1)  # 1q cleanup only; layout/routing already done
    return out, perm[0]


def run_and_time(ops, n, device_name, measure_wire):
    dev = qml.device(device_name, wires=n)

    def circuit():
        for op in ops:
            qml.apply(op)
        return qml.expval(qml.PauliZ(measure_wire))

    qnode = qml.QNode(circuit, dev)
    val = qnode()  # warm-up, also captures the value
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        qnode()
        times.append(time.perf_counter() - t0)
    return float(val), times


def main():
    rows = []
    for n in N_VALUES:
        # Square-ish grid with at least n qubits, then use the first n.
        side = 1
        while side * side < n:
            side += 1
        cm = CouplingMap.from_grid(side, side)
        edges = grid_edges(side, side)
        pairs = [(a, a + 1) for a in range(0, n - 1, 2)]

        qc = build_qiskit_ansatz(n)
        print(f"{'='*90}\nn={n} (grid {side}x{side}, using first {n} qubits), {LAYERS} layers\n{'='*90}")

        device_size = side * side
        out_a, wire0_a = compile_route_a(qc, cm)
        out_d, wire0_d = compile_route_d(qc, cm, n, pairs, device_size)
        print(f"  logical qubit 0 physically at: A wire {wire0_a}, D wire {wire0_d}")
        cx_a = sum(1 for inst in out_a.data if len(inst.qubits) == 2)
        cx_d = sum(1 for inst in out_d.data if len(inst.qubits) == 2)
        print(f"  two-qubit gates: A(Qiskit)={cx_a}  D(PSF-Zero)={cx_d}")

        ops_a = qiskit_circuit_to_pennylane_ops(out_a)
        ops_d = qiskit_circuit_to_pennylane_ops(out_d)

        results = {}
        for label, ops, wire0 in (("A", ops_a, wire0_a), ("D", ops_d, wire0_d)):
            for device in ("lightning.qubit", "lightning.gpu"):
                # Both A and D's compiled circuits are sized to the full device
                # (device_size), NOT the original logical n -- confirmed the
                # cause of the WireError this script hit on its first run.
                # measure_wire=wire0 (not a hardcoded 0) fixes a second bug
                # found on the second run: PauliZ(0) on the raw output wire
                # measured an idle ancilla for route A, giving a suspicious
                # exact 1.0 every time -- logical qubit 0 must be tracked
                # through each route's own layout, the same fix pattern
                # already established in Addendum 103/122.
                val, times = run_and_time(ops, device_size, device, wire0)
                med = statistics.median(times) * 1000
                results[(label, device)] = (val, med)
                print(f"  {label}-{device:16s} median {med:9.3f} ms  value={val:.10f}")
                rows.append(dict(n=n, route=label, device=device, cx=(cx_a if label == "A" else cx_d),
                                 median_ms=med, value=val))

        vals = [v for v, _ in results.values()]
        spread = max(vals) - min(vals)
        print(f"  max value spread across all four: {spread:.2e}")

        gpu_a = results[("A", "lightning.gpu")][1]
        cpu_a = results[("A", "lightning.qubit")][1]
        gpu_d = results[("D", "lightning.gpu")][1]
        cpu_d = results[("D", "lightning.qubit")][1]
        print(f"  A: GPU/CPU = {gpu_a/cpu_a:.3f}x   D: GPU/CPU = {gpu_d/cpu_d:.3f}x")
        print(f"  GPU execution, D/A = {gpu_d/gpu_a:.3f}x  (does PSF-Zero's own compiled circuit run faster on GPU too?)")
        print()

    out = "compile_then_gpu_2026-09-23.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
