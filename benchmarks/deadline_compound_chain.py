"""deadline_compound_chain.py -- Addendum 183.

The timed exam (Addenda 178-180) repeated with compounding (Addenda
181-182): on FakeNighthawk, a PennyLane tape of the cliff circuit family is
compiled, mapped back to logical qubits, converted back to a PennyLane tape,
and fed into the next lap. Every lap's compile time is scored against
deadlines, and every lap's meaning is checked per qubit pair against
PennyLane's own matrices.

Runs spare in {0, 8} x arms {Q3, P}, one after another (never in parallel,
because this is a timing experiment). Rows are written after every lap.

Usage (WSL, repository root):
    python -u deadline_compound_chain.py --laps 10 2>&1 | tee deadline_chain_result.txt
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import platform
import sys
import time

import numpy as np
import pennylane as qml
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit_ibm_runtime.fake_provider import FakeNighthawk

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from psf_compile import compile_for_hardware  # noqa: E402
from psf_pennylane_gpu_prototype import qiskit_to_tape, tape_to_qiskit  # noqa: E402

GATES_PER_PAIR = 20
SPARES = (0, 8)
ARMS = ("Q3", "P")
DEADLINES = (0.01, 0.1, 1.0, 10.0)
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
SEED = 0
OUT_CSV = "deadline_compound_chain_2026-09-26.csv"


class SwapError(RuntimeError):
    pass


def initial_tape(n):
    rng = np.random.default_rng(SEED)
    ops = []
    for a in range(0, n - 1, 2):
        for _ in range(GATES_PER_PAIR):
            ops.append(qml.QubitUnitary(random_unitary(4, seed=int(rng.integers(0, 2**31))).data, wires=[a, a + 1]))
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


def pair_matrices(tape, n):
    """PennyLane's own 4x4 matrix of each pair's operations, in tape order.
    Every operation must stay inside one pair (the circuit family keeps
    pairs independent); anything else raises."""
    per_pair = {p: [] for p in range(n // 2)}
    for op in tape.operations:
        pairs = {int(w) // 2 for w in op.wires}
        if len(pairs) != 1:
            raise SwapError(f"operation {op.name} on wires {list(op.wires)} spans more than one pair")
        per_pair[pairs.pop()].append(op)
    mats = {}
    for p, ops in per_pair.items():
        wires = [2 * p, 2 * p + 1]
        if ops:
            mats[p] = qml.matrix(qml.tape.QuantumTape(ops, measurements=[], shots=None), wire_order=wires)
        else:
            mats[p] = np.eye(4, dtype=complex)
    return mats


def phase_aligned_distance(a, b):
    t = np.trace(a.conj().T @ b)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(b - ph * a))


def back_to_logical(routed, n):
    init = list(routed.layout.initial_index_layout(filter_ancillas=True))
    final = list(routed.layout.final_index_layout(filter_ancillas=True))
    if init != final:
        raise SwapError("routing permuted qubits")
    to_logical = {p: v for v, p in enumerate(init)}
    out = QuantumCircuit(n)
    for inst in routed.data:
        if inst.operation.name in ("barrier", "delay", "measure"):
            continue
        phys = [routed.find_bit(q).index for q in inst.qubits]
        if any(p not in to_logical for p in phys):
            if len(phys) == 1:
                continue  # a single-qubit operation on an unused (ancilla) qubit
            raise SwapError(f"two-qubit operation on a qubit outside the layout: {phys}")
        out.append(inst.operation, [to_logical[p] for p in phys])
    return out, tuple(init)


def to_tape(logical):
    wrapped = QuantumCircuit(logical.num_qubits)
    for inst in logical.data:
        qs = [logical.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2:
            wrapped.append(UnitaryGate(Operator(inst.operation).data), qs)
        else:
            wrapped.append(inst.operation, qs)
    return qiskit_to_tape(wrapped, list(range(logical.num_qubits)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=10)
    args = ap.parse_args()

    backend = FakeNighthawk()
    native = [g for g in backend.operation_names if g in NATIVE]
    print(f"laps={args.laps} | platform {platform.platform()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__} | pennylane {qml.__version__}", flush=True)

    rows = []
    for spare in SPARES:
        n = backend.coupling_map.size() - spare
        for arm in ARMS:
            tape = initial_tape(n)
            u0 = pair_matrices(tape, n)
            prev_layout = None
            for lap in range(1, args.laps + 1):
                t_lap = time.perf_counter()
                qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
                t0 = time.perf_counter()
                if arm == "Q3":
                    routed = transpile(qc, backend, optimization_level=3, seed_transpiler=0)
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        routed = compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                                      entangling_basis="cx", layout_search=True,
                                                      on_unsupported="raise", seed_transpiler=0)
                compile_s = time.perf_counter() - t0
                status = "OK"
                try:
                    logical, layout = back_to_logical(routed, n)
                    tape = to_tape(logical)
                    mats = pair_matrices(tape, n)
                    dist = max(phase_aligned_distance(u0[p], mats[p]) for p in u0)
                except SwapError as e:
                    status, dist, layout = f"STOP: {e}", None, None
                lap_s = time.perf_counter() - t_lap
                row = dict(spare=spare, arm=arm, lap=lap, status=status, compile_s=compile_s, lap_s=lap_s,
                           max_pair_distance=dist, ops=len(tape.operations),
                           routed_twoq=sum(1 for i in routed.data if len(i.qubits) == 2 and i.operation.name != "barrier"),
                           layout_changed=(prev_layout is not None and layout != prev_layout))
                for d in DEADLINES:
                    row[f"within_{d}s"] = compile_s <= d
                rows.append(row)
                prev_layout = layout
                with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                d_txt = "-" if dist is None else f"{dist:.2e}"
                print(f"spare={spare} {arm:2s} lap {lap:3d}  compile {compile_s:8.3f}s  lap {lap_s:7.2f}s  "
                      f"within1s={compile_s <= 1.0}  max_pair_dist={d_txt}  2q={row['routed_twoq']}  "
                      f"layout_changed={row['layout_changed']}  {status}", flush=True)
                if status != "OK":
                    break
            sel = [r for r in rows if r["spare"] == spare and r["arm"] == arm]
            print(f"  -> spare={spare} {arm}: laps within 1 s {sum(r['within_1.0s'] for r in sel)}/{len(sel)}, "
                  f"total compile {sum(r['compile_s'] for r in sel):.1f}s", flush=True)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
