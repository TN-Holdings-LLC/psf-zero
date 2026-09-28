"""probe_l3_passes.py -- exploratory (home, 2026-09-28): in which pass of Qiskit's
optimization level 3 does the error of the two real-target outputs of Addendum 245
(pair 7 of spare 0 input 0: 1.08e-4; pair 47 of spare 2 input 1: 7.1e-5) enter?

Same pickled Target (real_target_2026-09-28.pkl, no IBM connection), same circuits and
the same preset pass manager transpile() uses (level 3, seed 0). A callback records,
after every pass, the phase-aligned Frobenius distance between the logical pair's
4x4 and the gates the current circuit applies to that pair's qubits (logical indices
until ApplyLayout has run, physical ones after; revision 2: the first version switched
to physical indices as soon as ancillas were added, which misread EnlargeWithAncilla). When the distance first exceeds 1e-12 it
prints the pass and the two-qubit runs on that pair just before and just after it,
with the Weyl coordinates of each run (fidelity=None) and of the whole pair.

    python -u benchmarks/probe_l3_passes.py 2>&1 | tee probe_l3_passes.txt
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

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.converters import dag_to_circuit  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402
from qiskit.synthesis import TwoQubitWeylDecomposition  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402

import loop_endurance as le  # noqa: E402
import real_target_cliff as rtc  # noqa: E402

CASES = [(0, 0, 7), (2, 1, 47)]  # (spare, input, pair)


def pair_sub(circ, q0, q1):
    """Gates of `circ` on qubits q0, q1 as a 2-qubit circuit; None if a 2-qubit gate
    joins one of them with another qubit."""
    sub = QuantumCircuit(2)
    for inst in circ.data:
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        qs = [circ.find_bit(q).index for q in inst.qubits]
        if not set(qs) & {q0, q1}:
            continue
        if not set(qs) <= {q0, q1}:
            return None
        sub.append(inst.operation, [0 if q == q0 else 1 for q in qs])
    return sub


def runs(sub):
    """Gate count, two-qubit gate count and gate names of a pair's gate list."""
    twoq = [i for i, inst in enumerate(sub.data) if len(inst.qubits) == 2]
    return len(sub.data), len(twoq), sorted({inst.operation.name for inst in sub.data})


def weyl(u):
    w = TwoQubitWeylDecomposition(u, fidelity=None)
    a, b, c = float(w.a), float(w.b), float(w.c)
    return (a, b, c), dict(a_b=abs(a - b), b_c=abs(b - abs(c)), c=abs(c), a_bc=abs(a - b - abs(c)),
                           pi4_a=abs(np.pi / 4 - a))


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
        ref = pair_sub(qc, 2 * pair, 2 * pair + 1)
        u_ref = Operator(ref).data
        (a, b, c), sd = weyl(u_ref)
        print(f"\n=== spare {spare} input {k_in} pair {pair}: logical pair Weyl ({a:.6f}, {b:.6f}, {c:.6f}) "
              f"distances {', '.join(f'{k} {v:.2e}' for k, v in sd.items())}")
        pm = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=0)
        state = {"prev": None, "prev_name": None, "found": False, "i": 0}

        def cb(**kw):
            state["i"] += 1
            name = type(kw["pass_"]).__name__
            ps = kw["property_set"]
            circ = dag_to_circuit(kw["dag"])
            lay = ps["layout"]
            if name == "ApplyLayout":
                state["applied"] = True
            if state.get("applied") and lay is not None:
                q0, q1 = lay[va], lay[vb]
            else:
                q0, q1 = 2 * pair, 2 * pair + 1
            sub = pair_sub(circ, q0, q1)
            d = None if sub is None else rtc.aligned(u_ref, Operator(sub).data)
            if d is not None and d > 1e-12 and not state["found"]:
                state["found"] = True
                print(f"  first above 1e-12 after pass #{state['i']} {name}: distance {d:.3e} "
                      f"(qubits {q0},{q1}; previous pass {state['prev_name']})")
                for label, s in (("before", state["prev"]), ("after", sub)):
                    if s is None:
                        print(f"    {label}: not applicable")
                        continue
                    ng, n2, names = runs(s)
                    (a2, b2, c2), sd2 = weyl(Operator(s).data)
                    print(f"    {label}: {ng} gates, {n2} two-qubit, {names}; Weyl of the pair's gates "
                          f"({a2:.6f}, {b2:.6f}, {c2:.6f})")
                # each two-qubit-gate-delimited segment before the pass: Weyl of every 2-gate window
                s = state["prev"]
                if s is not None:
                    idx = [i for i, inst in enumerate(s.data) if len(inst.qubits) == 2]
                    for j in range(len(idx) - 1):
                        seg = QuantumCircuit(2)
                        for inst in s.data[idx[j]:idx[j + 1] + 1]:
                            seg.append(inst.operation, [s.find_bit(q).index for q in inst.qubits])
                        (a3, b3, c3), sd3 = weyl(Operator(seg).data)
                        near = min(sd3, key=sd3.get)
                        print(f"      window {j}: 2q gates {j}..{j + 1}, Weyl ({a3:.6f}, {b3:.6f}, {c3:.6f}), "
                              f"nearest special {near} {sd3[near]:.2e}")
            state["prev"], state["prev_name"] = sub, name

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pm.run(qc, callback=cb)
        final = pair_sub(out, out.layout.initial_layout[va], out.layout.initial_layout[vb]) if out.layout else None
        fd = None if final is None else rtc.aligned(u_ref, Operator(final).data)
        print(f"  final output: pair distance {fd} ({state['i']} passes); first-above found: {state['found']}")
    print("\nDONE")


if __name__ == "__main__":
    main()
