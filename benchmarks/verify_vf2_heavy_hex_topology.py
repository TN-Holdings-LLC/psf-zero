#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 5) Heavy-hex and real backend maps were never actually measured at all

## Why this is needed

`verify_vf2_topologies.py` (2026-09-14) included heavy-hex and `FakeSherbrooke` as
candidates, but not a single row for either appears in the results CSV. The cause
is in the code's precondition:

    if n < 4 or n % 2:
        continue

`CouplingMap.from_heavy_hex(3)` has 19 nodes, `from_heavy_hex(5)` has 57,
`FakeSherbrooke` has 127 -- **all odd.** The dense-pair-blocks circuit generator
only produces even qubit counts, so drawing `spare in (0, 4)` left `n = phys -
spare` odd as well, and `n % 2` was true every time, so every row was silently
dropped by `continue`. These were missing due to a bug, not because "heavy-hex
shows no cliff". Addendum-5's P2 was **written without ever having been tested**.

Here `spare` is chosen from the odd side instead (e.g. 1, 3, 5 for heavy-hex; 1, 3
for the real backend), so an even-qubit comparison pair can actually be built.
This also tests addendum-6 section 5's "vertex transitivity" hypothesis on a
**real-world** topology that is sparse and irregular in degree (heavy-hex mixes
degree 2 and 3, and is not vertex-transitive -- putting it on the same side of
the prediction as grid/line).

## Pre-registered predictions

(P1) Both heavy-hex and `FakeSherbrooke` have some configuration, up to near
     saturation, for which a perfect matching exists (low degree makes this less
     likely per addendum-5's P2, but the existence check itself is redone first).
(P2) Wherever a perfect matching exists, both heavy-hex and the real backend map
     **show the cliff** (consistent with addendum-6's "not vertex-transitive"
     hypothesis).
(P3) The size of the cliff may be less extreme than on a grid (lower degree means
     a smaller search space to begin with), but is predicted not to be absent.

**If this fails**: if (P1) fails and no perfect matching is ever found, the cliff
cannot even be discussed on this topology family (which would mean addendum-5's
original P2 concern was right after all). If (P2) fails and no cliff appears, the
"vertex transitivity" hypothesis may not extend beyond the grid family.

## Usage

    python verify_vf2_heavy_hex_topology.py
    python verify_vf2_heavy_hex_topology.py --levels 2 3 --reps 2
"""
from __future__ import annotations

import argparse

import pandas as pd

from vf2_probe_common import (BASIS_GATES, banner, build_dense_pair_blocks_circuit,
                              default_out, environment, has_perfect_matching, timed,
                              write_csv)


def topologies():
    """(name, CouplingMap, list of spare values to try). Heavy-hex and real backend
    maps have an odd node count, so spare is also chosen from the odd side. Lower
    degree can require a larger spare for a matching to exist, so the range is
    kept generous and searched automatically rather than betting on a handful of
    fixed points."""
    from qiskit.transpiler import CouplingMap
    out = [
        ("grid_6x7", CouplingMap.from_grid(6, 7), [0, 4]),   # control (known cliff)
    ]
    for d in (3, 5):
        try:
            cm = CouplingMap.from_heavy_hex(d)
            spares = list(range(1, min(cm.size() - 4, 41), 2))
            out.append((f"heavy_hex_d{d}", cm, spares))
        except Exception as e:
            print(f"failed to build heavy_hex_d{d}: {e}")
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        cm = FakeSherbrooke().coupling_map
        spares = list(range(1, min(cm.size() - 4, 61), 2))
        out.append(("fake_sherbrooke", cm, spares))
    except Exception as e:
        print(f"failed to obtain FakeSherbrooke: {e}")
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 5) Heavy-hex and real backend maps -- previously missing due to a bug", [
        "P1 both heavy-hex and the real backend have a configuration near saturation "
        "where a perfect matching exists",
        "P2 wherever a matching exists, the cliff appears (not vertex-transitive)",
        "P3 the cliff's size may be less extreme than on a grid, but should not be absent",
    ])
    print("**The existence of a perfect matching is checked before timing anything.**\n")

    rows = []
    hdr = f"{'topology':<16} {'phys':>5} {'spare':>6} {'qubits':>7} {'PM?':>6}"
    for lv in args.levels:
        hdr += f" {'L' + str(lv) + ' ms':>10}"
    print(hdr)

    for name, cmap, spares in topologies():
        phys = cmap.size()
        # Stage 1: cheaply check matching existence across all spare values first
        # (transpile is not called yet).
        pm_by_spare = {}
        for spare in spares:
            n = phys - spare
            if n < 4 or n % 2:
                continue
            pm_by_spare[spare] = has_perfect_matching(cmap, n // 2)
        has_pm = [s for s, ok in pm_by_spare.items() if ok]
        print(f"{name}: spare values with a matching = {has_pm if has_pm else '(none)'}")

        # Stage 2: among those with a matching, time only the two extremes --
        # smallest (most saturated) and largest (most spare).
        if not has_pm:
            for spare, pm in pm_by_spare.items():
                n = phys - spare
                rows.append(dict(Topology=name, Physical=phys, Spare=spare, Qubits=n,
                                 PerfectMatchingExists=pm))
            continue
        targets = sorted({min(has_pm), max(has_pm)})
        for spare in targets:
            n = phys - spare
            pm = True
            line = f"{name:<16} {phys:>5} {spare:>6} {n:>7} {str(pm):>6}"
            rec = dict(Topology=name, Physical=phys, Spare=spare, Qubits=n,
                       PerfectMatchingExists=pm)
            qc = build_dense_pair_blocks_circuit(n, seed=0)
            for lv in args.levels:
                mn, md = run(qc, cmap, lv, args.reps)
                line += f" {mn*1000:10.1f}"
                rec[f"L{lv}_min_s"] = mn
                rec[f"L{lv}_median_s"] = md
            print(line, flush=True)
            rows.append(rec)

    df = write_csv(rows, args.out or default_out("vf2_heavy_hex_topology"), environment())

    print("\n" + "=" * 78)
    print("Verdict -- per topology, ratio of the smallest spare with a matching to the "
          "largest (most saturated vs most spare)")
    print("=" * 78)
    for lv in args.levels:
        col = f"L{lv}_min_s"
        if col not in df.columns:
            continue
        print(f"\noptimization_level = {lv}")
        for name in df.Topology.unique():
            sub = df[(df.Topology == name) & df.PerfectMatchingExists].sort_values("Spare")
            if len(sub) < 2 or col not in sub.columns:
                continue
            tight, loose = sub.iloc[0], sub.iloc[-1]
            if pd.isna(tight.get(col)) or pd.isna(loose.get(col)):
                continue
            ratio = tight[col] / loose[col]
            tag = "cliff" if ratio > 10 else "no cliff"
            print(f"  {name:<16} spare {int(tight.Spare)} / spare {int(loose.Spare)}: "
                  f"{ratio:8.1f}x  {tag}")
    print("\n  -> A cliff on either heavy-hex or the real backend is direct evidence that")
    print("     this phenomenon extends to real-world topologies beyond the grid.")


if __name__ == "__main__":
    main()
