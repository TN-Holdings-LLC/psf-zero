#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(5/6) Does the cliff appear on topologies other than a grid -- heavy-hex,
linear, ring, real backend maps?

## Why this is needed

The cliff has only ever been measured on `CouplingMap.from_grid`'s rectangular
grid. Real IBM hardware is **heavy-hex**, with a maximum degree of 3 and sparse
edges. The document has honestly left this open: "Untested on heavy-hex, linear,
or real backend maps."

**There is an important precondition to check first.** On a sparse topology, a
saturated circuit's required perfect matching **may simply not exist**. If it
doesn't, `NO_SOLUTION_FOUND` is the correct answer, not a bug and not a cliff.

So this script checks whether the matching exists (`networkx.max_weight_matching`)
**before** timing anything, and only treats an instance as a cliff candidate if a
matching does exist.

Failing to separate this and writing "heavy-hex is slow too" would be reporting
correct behaviour as a defect -- the same failure mode as the upstream episode.
Better to rule it out first.

## Pre-registered predictions

(P1) **The cliff appears only on topologies where a perfect matching exists.**
     Where none exists, the pass either returns `NO_SOLUTION_FOUND` quickly
     (small search space) or, even if slow, that is "confirming no solution
     exists", not a cliff.
(P2) Heavy-hex has low degree and sparse edges, so at saturation a perfect
     matching is often absent. So the cliff should appear **less often** on
     heavy-hex.
(P3) A line (path) and a ring always have a perfect matching (even node count),
     and the adjacent-pair circuit maps onto them directly, so the ordering
     should find it straightforwardly and **quickly**. No cliff.
(P4) Adding spare qubits makes every topology faster (the existing general rule).

**If this fails**: if heavy-hex shows the cliff despite a matching existing, the
phenomenon is not grid-specific and can occur on real hardware maps too. That
would substantially change its practical weight.

## Usage

    python verify_vf2_topologies.py
    python verify_vf2_topologies.py --levels 2 3 --reps 2
"""
from __future__ import annotations

import argparse

from vf2_probe_common import (BASIS_GATES, banner, build_dense_pair_blocks_circuit,
                              default_out, environment, has_perfect_matching, timed,
                              write_csv)


def topologies():
    """(name, CouplingMap) list. All aligned to an even node count."""
    from qiskit.transpiler import CouplingMap
    out = [
        ("grid_6x7", CouplingMap.from_grid(6, 7)),          # 42, the known cliff
        ("grid_7x8", CouplingMap.from_grid(7, 8)),          # 56, the known cliff
        ("line_42", CouplingMap.from_line(42)),
        ("ring_42", CouplingMap.from_ring(42)),
        ("full_20", CouplingMap.from_full(20)),
    ]
    for d in (3, 5):
        try:
            out.append((f"heavy_hex_d{d}", CouplingMap.from_heavy_hex(d)))
        except Exception:
            pass
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        out.append(("fake_sherbrooke", FakeSherbrooke().coupling_map))
    except Exception:
        pass
    return out


def run(qc, cmap, level, reps):
    from qiskit import transpile
    mn, md, _ = timed(
        lambda: transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
                          optimization_level=level, seed_transpiler=0), reps)
    return mn, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--spare", type=int, nargs="+", default=[0, 4],
                    help="number of spare qubits; 0 = saturated")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(5/6) Topologies beyond the grid -- how far does the cliff extend?", [
        "P1 the cliff appears only on topologies where a perfect matching exists",
        "P2 heavy-hex is sparse, so a matching is often absent at saturation, making "
        "the cliff less likely",
        "P3 line and ring always have a matching and are found straightforwardly -- no cliff",
        "P4 adding spare qubits speeds up every topology",
    ])
    print("**The existence of a perfect matching is checked before timing anything.** "
          "If none exists, NO_SOLUTION_FOUND is correct behaviour, not a cliff.\n")

    rows = []
    hdr = f"{'topology':<18} {'phys':>5} {'spare':>6} {'qubits':>7} {'PM?':>6}"
    for lv in args.levels:
        hdr += f" {'L' + str(lv) + ' ms':>10}"
    print(hdr)

    for name, cmap in topologies():
        phys = cmap.size()
        for spare in args.spare:
            n = phys - spare
            if n < 4 or n % 2:
                continue
            pm = has_perfect_matching(cmap, n // 2)
            qc = build_dense_pair_blocks_circuit(n, seed=0)
            line = f"{name:<18} {phys:>5} {spare:>6} {n:>7} {str(pm):>6}"
            rec = dict(Topology=name, Physical=phys, Spare=spare, Qubits=n,
                       PerfectMatchingExists=pm)
            for lv in args.levels:
                mn, md = run(qc, cmap, lv, args.reps)
                line += f" {mn*1000:10.1f}"
                rec[f"L{lv}_min_s"] = mn
                rec[f"L{lv}_median_s"] = md
            print(line, flush=True)
            rows.append(rec)

    df = write_csv(rows, args.out or default_out("vf2_topologies"), environment())

    print("\n" + "=" * 78)
    print("Verdict -- ratio of saturated (spare=0) to spare. Known grids give 40x-275x")
    print("=" * 78)
    for lv in args.levels:
        col = f"L{lv}_min_s"
        print(f"\noptimization_level = {lv}")
        for name in df.Topology.unique():
            sub = df[df.Topology == name].set_index("Spare")
            if 0 in sub.index and len(sub) > 1:
                other = [s for s in sub.index if s != 0][0]
                ratio = sub.loc[0, col] / sub.loc[other, col]
                pm = sub.loc[0, "PerfectMatchingExists"]
                tag = "cliff" if ratio > 10 else "no cliff"
                note = "" if pm else "  (no matching -> not a cliff)"
                print(f"  {name:<18} {ratio:8.1f}x  {tag}{note}")
    print("\n  -> If any non-grid topology shows the cliff despite a matching existing,")
    print("     this is not grid-specific. Its practical weight changes.")


if __name__ == "__main__":
    main()
