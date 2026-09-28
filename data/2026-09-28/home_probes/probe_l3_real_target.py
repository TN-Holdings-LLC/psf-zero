"""probe_l3_real_target.py -- exploratory (home, 2026-09-28): the two Qiskit L3 outputs
of real_target_cliff.py (Addendum 245) whose worst pair was off by 1.1e-4 and 7.1e-5
(spare 0 input 0, spare 2 input 1). Is it the Weyl specialization seen at the workplace
(Addendum 242: a block near a = b snapped within Qiskit's 1e-9 fidelity tolerance)?

Uses the pickled Target of that run (real_target_2026-09-28.pkl, no IBM connection),
recompiles the two circuits exactly as the run did, and for every pair above 1e-12
prints the pair's Weyl coordinates, the specialization Qiskit's default Weyl
decomposition chooses for it, and the operator distance of Qiskit's CZ-basis
synthesis of that pair alone. Also recompiles with approximation_degree=1.0 given
explicitly. Two other circuits (spare 0 inputs 1, 2) are included as controls.

    python -u benchmarks/probe_l3_real_target.py 2>&1 | tee probe_l3_real_target.txt
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit.library import CZGate  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402
from qiskit.synthesis import TwoQubitBasisDecomposer, TwoQubitWeylDecomposition  # noqa: E402

import loop_endurance as le  # noqa: E402
import real_target_cliff as rtc  # noqa: E402

CASES = [(0, 0, "reported 1.08e-4"), (2, 1, "reported 7.1e-5"), (0, 1, "control"), (0, 2, "control")]


def per_pair(qc_logical, qc_out, n):
    """Phase-aligned distance of every pair (dict k -> (distance, logical 4x4))."""
    lay = qc_out.layout
    phys = list(lay.final_index_layout(filter_ancillas=True))
    pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
    owner = {}
    for k, (a, b) in enumerate(pairs):
        owner[phys[a]] = k
        owner[phys[b]] = k
    per = {k: QuantumCircuit(2) for k in range(len(pairs))}
    for inst in qc_out.data:
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        qs = [qc_out.find_bit(q).index for q in inst.qubits]
        ks = {owner.get(q) for q in qs}
        if None in ks:
            continue
        k = ks.pop()
        a, _ = pairs[k]
        per[k].append(inst.operation, [0 if q == phys[a] else 1 for q in qs])
    out = {}
    for k, (a, b) in enumerate(pairs):
        ref = QuantumCircuit(2)
        for inst in qc_logical.data:
            qs = [qc_logical.find_bit(q).index for q in inst.qubits]
            if set(qs) <= {a, b}:
                ref.append(inst.operation, [0 if q == a else 1 for q in qs])
        u = Operator(ref).data
        out[k] = (rtc.aligned(u, Operator(per[k]).data), u)
    return out


def main():
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    with open(rtc.TARGET_PKL, "rb") as f:
        target = pickle.load(f)
    import rustworkx as rx
    cmap = target.build_coupling_map()
    g = rx.PyGraph()
    g.add_nodes_from(range(cmap.size()))
    g.add_edges_from_no_data(sorted({tuple(sorted(e)) for e in cmap.get_edges()}))
    m = len(rx.max_weight_matching(g, max_cardinality=True))
    dec = TwoQubitBasisDecomposer(CZGate())
    for spare, k_in, note in CASES:
        n = 2 * m - spare
        qc = rtc.build(n, 1000 * spare + k_in)
        for label, kw in (("default", {}), ("approximation_degree=1.0", {"approximation_degree": 1.0})):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = transpile(qc, target=target, optimization_level=3, seed_transpiler=0, **kw)
            pp = per_pair(qc, out, n)
            bad = sorted(((d, k) for k, (d, _) in pp.items() if d > 1e-12), reverse=True)
            worst = max(d for d, _ in pp.values())
            print(f"\nspare {spare} input {k_in} ({note}), {label}: worst pair {worst:.3e}; pairs above 1e-12: {len(bad)}")
            if label != "default":
                continue
            for d, k in bad:
                u = pp[k][1]
                w0 = TwoQubitWeylDecomposition(u, fidelity=None)
                w9 = TwoQubitWeylDecomposition(u)
                syn = Operator(dec(u)).data
                print(f"  pair {k}: distance {d:.3e} | Weyl ({w0.a:.6f}, {w0.b:.6f}, {w0.c:.6e}) "
                      f"a-b {abs(w0.a - w0.b):.2e} b-|c| {abs(w0.b - abs(w0.c)):.2e} |c| {abs(w0.c):.2e} | "
                      f"default specialization {getattr(w9, 'specialization', None)} | "
                      f"CZ decomposer alone {rtc.aligned(u, syn):.3e}", flush=True)
    print("\nDONE")


if __name__ == "__main__":
    main()
