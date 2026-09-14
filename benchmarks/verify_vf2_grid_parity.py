#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 12) Does grid row/column parity really govern the success rate of
   a corner-start depth-first search?

## Why this is needed

Followup 9's (`verify_vf2_neighbor_order_full.py`) results showed the failure
counts among the 24 patterns from a corner start (corner_TL / corner_BR) lining
up cleanly across the four grids tested:

    8x8 (even x even)  24/24 failed (all failed)
    6x7 (even x odd)   11/24 failed
    8x9 (even x odd)   15/24 failed
    9x9 (odd x odd)     0/24 failed (all succeeded)

The hypothesis is "both even is worst, both odd is best, one of each is in
between" -- but with only 4 grids (and only 2 examples of even x odd), it is
not yet safe to say "parity is the operative factor." Here, each of the 2x2
parity combinations is filled out with two examples:

    even x even: 6x6, 8x6
    odd x odd:   7x7, 9x7
    even x odd:  6x7 (already covered in followup 9, repeated here as a control), 6x9
    odd x even:  7x6, 9x6

Even x odd and odd x even are kept separate to also check transpose symmetry
at the same time (whether 6x7 and its transpose 7x6 behave the same way). To
save time, only corner starts (corner_TL, corner_BR) are covered here
(edge_mid and center already have adequate results from followup 9).

## Pre-registered predictions

(P1) Even x even grids (6x6, 8x6) have a high failure count from a corner start
     (close to 8x8's 24/24, or at least a majority failing).
(P2) Odd x odd grids (7x7, 9x7) have a low failure count from a corner start
     (close to 9x9's 0/24).
(P3) Even x odd / odd x even grids (6x7, 6x9, 7x6, 9x6) have an intermediate
     failure count (not pinned at 0 or 24).
(P4) Behaviour is unchanged under transpose -- 6x7 and 7x6, and 6x9 and 9x6,
     each land on similar failure counts.

**If this fails**: if P1 or P2 fails (e.g. 8x6 turns out all-succeed like
9x9), "parity" is the wrong explanation, and some other factor (e.g. one side
having length 8, overall area) should be suspected instead. If P4 fails (6x7
and 7x6 diverge substantially), that is a new clue that an asymmetric factor
-- which of rows or columns is "closer to the corner" -- is at work.

## Usage

    python verify_vf2_grid_parity.py
    python verify_vf2_grid_parity.py --call-limit 1000000
"""
from __future__ import annotations

import argparse
import itertools

from vf2_probe_common import banner, default_out, environment, write_csv
from verify_vf2_neighbor_order_full import (DIRS, NAMES, START_POINTS, edge_locality,
                                            probe, traverse)

GRIDS = {
    "even_even": ["6x6", "8x6"],
    "odd_odd": ["7x7", "9x7"],
    "even_odd": ["6x7", "6x9"],
    "odd_even": ["7x6", "9x6"],
}
CORNER_STARTS = ("corner_TL", "corner_BR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call-limit", type=int, default=3_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 12) Filling out the grid-parity hypothesis with a 2x2", [
        "P1 even x even (6x6, 8x6) has a high corner-start failure count",
        "P2 odd x odd (7x7, 9x7) has a low corner-start failure count",
        "P3 even x odd / odd x even (6x7, 6x9, 7x6, 9x6) has an intermediate failure count",
        "P4 behaviour is unchanged under transpose (6x7 vs 7x6, 6x9 vs 9x6 give similar results)",
    ])

    perms = list(itertools.permutations(DIRS))
    rows_out = []
    print(f"{'parity':<10} {'grid':>6} {'start':<10} {'order':<6} {'locality':>9} "
          f"{'found':>6} {'ms':>10}")

    for parity, grids in GRIDS.items():
        for g in grids:
            r, c = (int(x) for x in g.lower().split("x"))
            n = r * c
            n_pairs = n // 2
            for start_name in CORNER_STARTS:
                start = START_POINTS[start_name](r, c)
                for perm in perms:
                    order = traverse(r, c, start, depth_first=True, neighbor_order=perm)
                    loc = edge_locality(r, c, order)
                    order_name = "".join(NAMES[d] for d in perm)
                    ok, el = probe(r, c, order, n_pairs, args.call_limit, id_order=True)
                    print(f"{parity:<10} {g:>6} {start_name:<10} {order_name:<6} "
                          f"{loc:9.2f} {str(ok):>6} {el*1000:10.2f}", flush=True)
                    rows_out.append(dict(Parity=parity, Grid=g, Nodes=n, Pairs=n_pairs,
                                         Start=start_name, NeighborOrder=order_name,
                                         EdgeLocality=loc, Found=ok, Time_s=el,
                                         CallLimit=args.call_limit))

    df = write_csv(rows_out, args.out or default_out("vf2_grid_parity"), environment())

    print("\n" + "=" * 78)
    print("Verdict 1 -- failure count by parity (how many of the 24 patterns failed, per grid x start)")
    print("=" * 78)
    fail_counts = (df.assign(Failed=~df.Found)
                     .groupby(["Parity", "Grid", "Start"]).Failed.sum())
    print(fail_counts.to_string())

    print("\n" + "=" * 78)
    print("Verdict 2 -- mean failure count by parity class (direct test of P1/P2/P3)")
    print("=" * 78)
    parity_summary = (df.assign(Failed=~df.Found)
                        .groupby("Parity").Failed.agg(["mean", "sum", "count"]))
    print(parity_summary.to_string())
    print("\n  -> High for even_even, low for odd_odd, intermediate for even_odd/odd_even")
    print("     supports P1/P2/P3.")

    print("\n" + "=" * 78)
    print("Verdict 3 -- transpose symmetry (P4): 6x7 vs 7x6, 6x9 vs 9x6")
    print("=" * 78)
    for a, b in (("6x7", "7x6"), ("6x9", "9x6")):
        fa = (~df[df.Grid == a].Found).sum() if a in df.Grid.values else None
        fb = (~df[df.Grid == b].Found).sum() if b in df.Grid.values else None
        print(f"  {a}: {fa} of 48 failed   {b}: {fb} of 48 failed")
    print("\n  -> The two being close supports P4. A large gap suggests an asymmetric role "
          "for rows versus columns.")


if __name__ == "__main__":
    main()
