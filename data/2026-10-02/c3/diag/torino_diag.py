"""torino_diag.py -- exploratory diagnosis for Addendum 301 (not a test): which physical qubits and couplers do C2
and L3T use for the GAP F1 circuits on a device, and how bad are they in the Target?

For the first circuit of every (n, L) cell of GAP F1 (same seeds as the scored run), it prints per arm: the qubits
touched, the two-qubit edges used with their reported error, the worst sx error, the shortest T1/T2, and the readout
errors of the final-layout qubits; then the same Target statistics for the whole device for comparison.

    python torino_diag.py --repo <repo> [--device FakeTorino]
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--device", default="FakeTorino")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    import gap_eval as G
    st = G.Stack(args.repo)
    st.device(args.device)
    tgt = st.d["tgt"]
    g2 = [g for g in ("cz", "cx", "ecr") if g in tgt.operation_names][0]
    e2 = {tuple(q): (p.error if p is not None and p.error is not None else float("nan")) for q, p in tgt[g2].items()}
    esx = {q[0]: (p.error if p is not None and p.error is not None else float("nan")) for q, p in tgt["sx"].items()}
    qp = tgt.qubit_properties
    t1 = {i: qp[i].t1 for i in range(tgt.num_qubits)}
    t2 = {i: qp[i].t2 for i in range(tgt.num_qubits)}
    allv = np.array([v for v in e2.values() if v == v])
    print(f"{args.device}: {g2} edges {len(e2)}, error median {np.median(allv):.4f}, p90 {np.percentile(allv, 90):.4f}, "
          f"max {allv.max():.4f}, edges with error >= 0.05: {int((allv >= 0.05).sum())}")
    print(f"  T2 median {np.median([v for v in t2.values() if v]) * 1e6:.0f} us; qubits with T2 < 30 us: "
          f"{sorted(i for i, v in t2.items() if v and v < 30e-6)}")
    seen = set()
    for params, qc in G.family("F1", False):
        key = (params["n"], params["L"])
        if key in seen:
            continue
        seen.add(key)
        print(f"\nF1 n={params['n']} L={params['L']} seed={params['seed']}")
        for arm in ("C2", "L3T"):
            out = st.compile(arm, qc)
            fin = list(out.layout.final_index_layout(filter_ancillas=True)[:qc.num_qubits])
            edges = {}
            qs = set()
            for ins in out.data:
                idx = tuple(out.find_bit(q).index for q in ins.qubits)
                qs.update(idx)
                if len(idx) == 2:
                    k = idx if idx in e2 else idx[::-1]
                    edges[k] = edges.get(k, 0) + 1
            worst = sorted(((e2.get(k, float("nan")), k, c) for k, c in edges.items()), reverse=True)[:3]
            print(f"  {arm:3s} qubits {sorted(qs)}  2q {sum(edges.values())}  worst edges "
                  + ", ".join(f"{k}x{c}:{e:.4f}" for e, k, c in worst)
                  + f"  max sx {max(esx.get(q, 0) for q in qs):.2e}  min T1 {min(t1[q] for q in qs) * 1e6:.0f} us"
                  + f"  min T2 {min(t2[q] for q in qs) * 1e6:.0f} us  readout(final) "
                  + ", ".join(f"{st.readout(q):.3f}" for q in fin))


if __name__ == "__main__":
    main()
