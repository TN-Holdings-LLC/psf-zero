"""probe_l3_cancel.py -- exploratory (home, 2026-09-28): which gates does Qiskit's
CommutativeCancellation remove from the two real-target pairs of Addendum 245 whose
error appears in that pass (probe_l3_passes_r2.py: pair 7 of spare 0 input 0, pair 47
of spare 2 input 1)?

Runs the level-3 preset pass manager on the same pickled Target and circuits, keeps the
pair's gate list just before and just after the first CommutativeCancellation, prints
both lists, and then runs CommutativeCancellation alone on the "before" list (as a
2-qubit circuit) to see whether the pass by itself reproduces the error.

    python -u benchmarks/probe_l3_cancel.py 2>&1 | tee probe_l3_cancel.txt
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qiskit.converters import dag_to_circuit  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402
from qiskit.transpiler import PassManager, generate_preset_pass_manager  # noqa: E402
from qiskit.transpiler.passes import CommutativeCancellation  # noqa: E402

import loop_endurance as le  # noqa: E402
import probe_l3_passes_r2 as plp  # noqa: E402
import real_target_cliff as rtc  # noqa: E402

CASES = [(0, 0, 7), (2, 1, 47)]


def listing(sub):
    out = []
    for i, inst in enumerate(sub.data):
        qs = [sub.find_bit(q).index for q in inst.qubits]
        ps = ",".join(f"{float(p):+.9f}" for p in inst.operation.params)
        out.append(f"{i:2d} {inst.operation.name:3s} q{qs} {ps}")
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
    for spare, k_in, pair in CASES:
        n = 2 * m - spare
        qc = rtc.build(n, 1000 * spare + k_in)
        va, vb = qc.qubits[2 * pair], qc.qubits[2 * pair + 1]
        u_ref = Operator(plp.pair_sub(qc, 2 * pair, 2 * pair + 1)).data
        st = {"applied": False, "prev": None, "before": None, "after": None}

        def cb(**kw):
            name = type(kw["pass_"]).__name__
            lay = kw["property_set"]["layout"]
            if name == "ApplyLayout":
                st["applied"] = True
            if not st["applied"] or lay is None:
                return
            sub = plp.pair_sub(dag_to_circuit(kw["dag"]), lay[va], lay[vb])
            if name == "CommutativeCancellation" and st["before"] is None:
                st["before"], st["after"] = st["prev"], sub
            st["prev"] = sub

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=0).run(qc, callback=cb)
        b, a = st["before"], st["after"]
        print(f"\n=== spare {spare} input {k_in} pair {pair} ===")
        print(f"before CommutativeCancellation: {len(b.data)} gates, distance {rtc.aligned(u_ref, Operator(b).data):.3e}")
        print("\n".join("  " + s for s in listing(b)))
        print(f"after: {len(a.data)} gates, distance {rtc.aligned(u_ref, Operator(a).data):.3e}")
        print("\n".join("  " + s for s in listing(a)))
        alone = PassManager([CommutativeCancellation(target=target)]).run(b)
        alone_nt = PassManager([CommutativeCancellation()]).run(b)
        print(f"CommutativeCancellation alone on the 'before' list: {len(alone.data)} gates, distance "
              f"{rtc.aligned(u_ref, Operator(alone).data):.3e} (with target); {len(alone_nt.data)} gates, "
              f"{rtc.aligned(u_ref, Operator(alone_nt).data):.3e} (without)")
    print("\nDONE")


if __name__ == "__main__":
    main()
