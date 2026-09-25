"""compound_pipeline_chain.py -- Addendum 181.

Compound test of the whole PennyLane -> synthesis -> IBM-topology pipeline:
the output tape of each lap is the input of the next, for 20,000 laps, and
every lap is checked against PennyLane's own matrix of the tape.

One lap: tape -> tape_to_qiskit -> compile (arm) -> route onto FakeNighthawk
at a fixed 4-qubit path -> map back to logical qubits -> wrap 2-qubit gates
as unitaries -> qiskit_to_tape -> qml.matrix.

Arms:
  A   Qiskit TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX") per block
  P   PSF-Zero SU4GeodesicPSFSynthesizer(entangling_basis="cx") per block
  Q3  Qiskit transpile(optimization_level=3) on the unsynthesized circuit
  C   control: as A, but the tape -> Qiskit step uses the PRE-FIX
      conversion (no qubit-order reversal), reintroducing Addendum 164's bug

Each arm is meant to run as its own process:
    python -u compound_pipeline_chain.py --arm A  2>&1 | tee chain_A.txt &
    python -u compound_pipeline_chain.py --arm P  2>&1 | tee chain_P.txt &
    python -u compound_pipeline_chain.py --arm Q3 2>&1 | tee chain_Q3.txt &
    python -u compound_pipeline_chain.py --arm C  2>&1 | tee chain_C.txt &
    wait
Results are written at every checkpoint, so a stopped run keeps its data.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import platform
import statistics as st
import sys
import time

import numpy as np
import pennylane as qml
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate, UnitaryGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit_ibm_runtime.fake_provider import FakeNighthawk

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from psf_compile import GeodesicPSFHyper, SU4GeodesicPSFSynthesizer  # noqa: E402
from psf_pennylane_gpu_prototype import collect_and_consolidate, qiskit_to_tape, tape_to_qiskit  # noqa: E402

WIRES = [0, 1, 2, 3]
CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
TAPE_SEED = 181


def find_path(backend, length=4):
    """First simple path of `length` qubits by deterministic DFS from qubit 0."""
    adj = {}
    for a, b in backend.coupling_map.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def dfs(path):
        if len(path) == length:
            return path
        for nxt in sorted(adj.get(path[-1], ())):
            if nxt not in path:
                found = dfs(path + [nxt])
                if found:
                    return found
        return None

    for start in sorted(adj):
        p = dfs([start])
        if p:
            return p
    raise RuntimeError("no path found")


def initial_tape():
    rng = np.random.default_rng(TAPE_SEED)
    ops = [qml.QubitUnitary(random_unitary(4, seed=int(rng.integers(0, 2**31))).data, wires=w)
           for w in ([0, 1], [2, 3], [1, 2])]
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


def tape_to_qiskit_prefix(tape):
    """Arm C only: the PRE-FIX conversion (Addendum 164's bug) for 2-qubit
    QubitUnitary -- the matrix is handed over without reversing the qubit
    list. Single-qubit ops carry no ordering and are converted exactly."""
    qc = QuantumCircuit(len(WIRES))
    for op in tape.operations:
        idx = [WIRES.index(w) for w in op.wires]
        mat = np.asarray(qml.matrix(op), dtype=complex)
        qc.unitary(mat, idx)  # no [::-1]: deliberately wrong for 2 qubits
    return qc


def phase_aligned_distance(a, b):
    t = np.trace(a.conj().T @ b)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(b - ph * a))


def make_compile(arm, backend, layout):
    if arm == "Q3":
        return lambda qc: transpile(qc, backend, optimization_level=3, initial_layout=layout, seed_transpiler=0), None
    if arm == "P":
        psf = SU4GeodesicPSFSynthesizer(GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"), verify=True)
        synth = psf.synthesize
    else:
        psf = None
        synth = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")

    def compile_fn(qc):
        blocked = collect_and_consolidate(qc, block_gate_floor=0)
        out = QuantumCircuit(qc.num_qubits)
        for inst in blocked.data:
            qs = [blocked.find_bit(q).index for q in inst.qubits]
            if len(qs) == 2 and inst.operation.name == "unitary":
                with contextlib.redirect_stdout(io.StringIO()):
                    sub = synth(inst.operation.to_matrix())
                out.compose(sub, qubits=qs, inplace=True)
            else:
                out.append(inst.operation, qs)
        return transpile(out, backend, optimization_level=1, initial_layout=layout, seed_transpiler=0)

    return compile_fn, psf


def back_to_logical(routed):
    """Routed circuit -> 4-qubit logical circuit, using the routed circuit's
    own layout. Raises if routing permuted qubits (a SWAP) or touched a
    qubit outside the layout."""
    init = list(routed.layout.initial_index_layout(filter_ancillas=True))
    final = list(routed.layout.final_index_layout(filter_ancillas=True))
    if init != final:
        raise RuntimeError(f"routing permuted qubits: initial {init} final {final}")
    to_logical = {p: v for v, p in enumerate(init)}
    out = QuantumCircuit(len(init))
    out.global_phase = routed.global_phase
    for inst in routed.data:
        if inst.operation.name in ("barrier", "delay", "measure"):
            continue
        phys = [routed.find_bit(q).index for q in inst.qubits]
        if any(p not in to_logical for p in phys):
            raise RuntimeError(f"operation {inst.operation.name} on qubit outside the layout: {phys}")
        out.append(inst.operation, [to_logical[p] for p in phys])
    return out


def to_tape(logical):
    wrapped = QuantumCircuit(logical.num_qubits)
    for inst in logical.data:
        qs = [logical.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2:
            wrapped.append(UnitaryGate(Operator(inst.operation).data), qs)
        else:
            wrapped.append(inst.operation, qs)
    return qiskit_to_tape(wrapped, WIRES)


def growth_exponent(ks, ds):
    pts = [(math.log(k), math.log(d)) for k, d in zip(ks, ds) if k >= 10 and d > 0]
    if len(pts) < 2:
        return float("nan")
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "P", "Q3", "C"])
    ap.add_argument("--laps", type=int, default=20000)
    args = ap.parse_args()

    backend = FakeNighthawk()
    layout = find_path(backend)
    print(f"arm={args.arm} laps={args.laps} | platform {platform.platform()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__} | pennylane {qml.__version__} | layout {layout}", flush=True)

    compile_fn, psf = make_compile(args.arm, backend, layout)
    to_qiskit = tape_to_qiskit_prefix if args.arm == "C" else (lambda t: tape_to_qiskit(t, wire_order=WIRES)[0])

    tape = initial_tape()
    u0 = qml.matrix(tape, wire_order=WIRES)
    u_prev = u0
    step_err, fixed = [], 0
    ck_rows = []
    out_ck = f"compound_chain_{args.arm}_checkpoints_2026-09-25.csv"
    t_start = time.perf_counter()
    for k in range(1, args.laps + 1):
        qc = to_qiskit(tape)
        routed = compile_fn(qc)
        tape = to_tape(back_to_logical(routed))
        u = qml.matrix(tape, wire_order=WIRES)
        e = phase_aligned_distance(u_prev, u)
        step_err.append(e)
        if np.array_equal(u, u_prev):
            fixed += 1
        u_prev = u
        if k in CHECKPOINTS or k == args.laps:
            row = dict(arm=args.arm, lap=k, delta=phase_aligned_distance(u0, u), ops=len(tape.operations),
                       step_err_median_so_far=float(np.median(step_err)), step_err_max_so_far=float(np.max(step_err)),
                       fixed_point_laps_so_far=fixed, elapsed_s=time.perf_counter() - t_start,
                       psf_fallbacks=psf.fallback_count if psf is not None else "")
            ck_rows.append(row)
            with open(out_ck, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerows(ck_rows)
            print(f"lap {k:6d}  delta={row['delta']:.3e}  ops={row['ops']}  step_med={row['step_err_median_so_far']:.2e} "
                  f"step_max={row['step_err_max_so_far']:.2e}  fixed={fixed}  {row['elapsed_s']:.0f}s", flush=True)

    ks = [r["lap"] for r in ck_rows]
    alpha = growth_exponent(ks, [r["delta"] for r in ck_rows])
    print(f"\nDONE arm={args.arm}  alpha={alpha:+.3f}  delta_final={ck_rows[-1]['delta']:.3e}  "
          f"ops lap1={ck_rows[0]['ops']} final={ck_rows[-1]['ops']}  step_median={st.median(step_err):.2e}  "
          f"step_max={max(step_err):.2e}  fixed_laps={fixed}  wall={time.perf_counter() - t_start:.0f}s  "
          f"fallbacks={psf.fallback_count if psf is not None else '-'}", flush=True)
    print(f"Wrote {out_ck}")


if __name__ == "__main__":
    main()
