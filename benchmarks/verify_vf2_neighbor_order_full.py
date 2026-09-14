#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 9) What about the neighbor-visit order is doing the work -- all 24
   permutations instead of 4, and can a locality number predict failure?

## Why this is needed

Followup 6 (`verify_vf2_dfs_mechanism.py`) tried only 4 depth-first
neighbor-visit orders (RDLU/DRUL/LURD/ULDR -- cyclic shifts of the 4 directions
only), and found that from the 6x7 top-left corner, RDLU alone failed while the
other three succeeded. But those four are only the 4 cyclic shifts out of 24
possible orderings of "4 directions", and the other 20 (e.g. non-cyclic orderings
that put the same two directions first) have not been tried. This fills that
gap first.

Beyond that, this tries to pin down **why** RDLU alone fails with a number
instead of a guess. In depth-first search (a stack pop), the search moves first
into whichever direction was pushed **last** among the neighbor-visit order, so
the order determines "how close together adjacent grid pairs end up in the
resulting visit order (locality)". For each ordering, this computes

    edge_locality = mean over all grid edges (u,v) of |position(u) in the visit order - position(v)|

and tests the hypothesis that the smaller this is (i.e. the more adjacent nodes
stay close together in the visit order), the more likely VF2's `id_order` search
is to succeed.

## Pre-registered predictions

(P1) Even trying all 24 patterns, only a minority fail -- neither every pattern
     nor none of them (confirming followup 6's "not a coincidence of one
     particular ordering" across 24 patterns too -- note followup 6 did not
     claim "all four fail consistently", so here the claim under test is that
     "failure is skewed toward specific patterns").
(P2) Orderings with higher edge_locality (worse locality) are more likely to fail
     and take longer (a monotonic relationship between edge_locality and
     success rate / time).
(P3) This trend holds consistently across grid shapes (6x7, 8x8, 8x9, 9x9).

**If this fails**: if (P2) fails and there is no relationship between
edge_locality and success rate, the "visit-order locality" hypothesis itself is
wrong, and some other structural factor (e.g. the parity of distance from the
corner, the ordering of the degree sequence) needs to be sought instead.

## Usage

    python verify_vf2_neighbor_order_full.py
    python verify_vf2_neighbor_order_full.py --grids 6x7 8x8 --call-limit 1000000
"""
from __future__ import annotations

import argparse
import itertools
import time

from vf2_probe_common import banner, default_out, environment, write_csv

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
NAMES = {(0, 1): "R", (1, 0): "D", (0, -1): "L", (-1, 0): "U"}

START_POINTS = {
    "corner_TL": lambda rows, cols: (0, 0),
    "corner_BR": lambda rows, cols: (rows - 1, cols - 1),
    "edge_mid":  lambda rows, cols: (0, cols // 2),
    "center":    lambda rows, cols: (rows // 2, cols // 2),
}


def grid_edges(rows, cols):
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append(((r, c), (r, c + 1)))
            if r + 1 < rows:
                e.append(((r, c), (r + 1, c)))
    return e


def traverse(rows, cols, start, depth_first, neighbor_order):
    seen = {start}
    frontier = [start]
    out = []
    while frontier:
        n = frontier.pop() if depth_first else frontier.pop(0)
        out.append(n)
        r, c = n
        for dr, dc in neighbor_order:
            m = (r + dr, c + dc)
            if 0 <= m[0] < rows and 0 <= m[1] < cols and m not in seen:
                seen.add(m)
                frontier.append(m)
    return out


def edge_locality(rows, cols, order):
    pos = {node: i for i, node in enumerate(order)}
    total, count = 0, 0
    for a, b in grid_edges(rows, cols):
        total += abs(pos[a] - pos[b])
        count += 1
    return total / count


def probe(rows, cols, order, n_pairs, call_limit, id_order=True):
    import rustworkx as rx
    cm = rx.PyGraph()
    idx = {n: cm.add_node(n) for n in order}
    for a, b in grid_edges(rows, cols):
        cm.add_edge(idx[a], idx[b], None)
    im = rx.PyGraph()
    for _ in range(n_pairs):
        a, b = im.add_node(None), im.add_node(None)
        im.add_edge(a, b, None)
    t0 = time.perf_counter()
    it = rx.vf2_mapping(cm, im, subgraph=True, id_order=id_order, induced=False,
                        call_limit=call_limit)
    m = next(iter(it), None)
    return m is not None, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=["6x7", "8x8", "8x9", "9x9"])
    ap.add_argument("--call-limit", type=int, default=3_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 9) All 24 neighbor-visit orders, plus a locality metric to pin down the mechanism", [
        "P1 of the 24 patterns, failure is skewed toward a minority (neither all-fail nor all-succeed)",
        "P2 orderings with higher edge_locality (worse locality) are more likely to fail and take longer",
        "P3 this trend holds consistently regardless of grid shape",
    ])

    perms = list(itertools.permutations(DIRS))
    print(f"All permutations of the neighbor-visit order: {len(perms)}\n")

    rows_out = []
    print(f"{'grid':>6} {'start':<10} {'order':<6} {'locality':>9} "
          f"{'found':>6} {'ms':>10}")
    for g in args.grids:
        r, c = (int(x) for x in g.lower().split("x"))
        n = r * c
        n_pairs = n // 2
        for start_name, start_fn in START_POINTS.items():
            start = start_fn(r, c)
            for perm in perms:
                order = traverse(r, c, start, depth_first=True, neighbor_order=perm)
                loc = edge_locality(r, c, order)
                order_name = "".join(NAMES[d] for d in perm)
                ok, el = probe(r, c, order, n_pairs, args.call_limit, id_order=True)
                print(f"{g:>6} {start_name:<10} {order_name:<6} {loc:9.2f} "
                      f"{str(ok):>6} {el*1000:10.2f}")
                rows_out.append(dict(Grid=g, Nodes=n, Pairs=n_pairs, Start=start_name,
                                     NeighborOrder=order_name, EdgeLocality=loc,
                                     Found=ok, Time_s=el, CallLimit=args.call_limit))

    df = write_csv(rows_out, args.out or default_out("vf2_neighbor_order_full"), environment())

    print("\n" + "=" * 78)
    print("Verdict 1 -- distribution of failing patterns (how many of the 24 fail, per grid x start)")
    print("=" * 78)
    fail_counts = (df.assign(Failed=~df.Found)
                     .groupby(["Grid", "Start"]).Failed.sum())
    print(fail_counts.to_string())
    print("\n  -> All 0s or all 24s means P1 is rejected (all-succeed or all-fail).")
    print("     A scattering of a minority (1 to a few) of failures supports P1.")

    print("\n" + "=" * 78)
    print("Verdict 2 -- relationship between edge_locality and success rate / time")
    print("=" * 78)
    try:
        corr = df["EdgeLocality"].corr(df["Time_s"])
        print(f"Correlation coefficient between EdgeLocality and Time_s: {corr:.3f}")
    except Exception as e:
        print(f"Failed to compute the correlation coefficient: {e}")
    q = df["EdgeLocality"].quantile([0.0, 0.5, 1.0])
    median_loc = df["EdgeLocality"].median()
    lo = df[df.EdgeLocality <= median_loc]
    hi = df[df.EdgeLocality > median_loc]
    print(f"\nSplit at the EdgeLocality median={median_loc:.2f}:")
    print(f"  low-locality group:  success rate={lo.Found.mean():.3f}  "
          f"mean ms={lo.Time_s.mean()*1000:.2f}  (n={len(lo)})")
    print(f"  high-locality group: success rate={hi.Found.mean():.3f}  "
          f"mean ms={hi.Time_s.mean()*1000:.2f}  (n={len(hi)})")
    print("\n  -> Lower success rate / longer time in the high-locality group supports P2.")
    print("     A small correlation and no difference between the two groups means P2 "
          "is rejected -- another factor needs to be sought.")

    print("\n" + "=" * 78)
    print("Verdict 3 -- consistency by grid shape (for each grid, is a failing order's locality high?)")
    print("=" * 78)
    for g in args.grids:
        sub = df[df.Grid == g]
        if sub.Found.all() or (~sub.Found).all():
            print(f"  {g}: all-succeed or all-fail (undecidable)")
            continue
        fail_loc = sub[~sub.Found].EdgeLocality.mean()
        ok_loc = sub[sub.Found].EdgeLocality.mean()
        print(f"  {g}: mean locality of failing orders={fail_loc:.2f}  "
              f"mean locality of successful orders={ok_loc:.2f}  "
              f"{'failing is higher (supports P2/P3)' if fail_loc > ok_loc else 'reversed (needs reconsideration)'}")


if __name__ == "__main__":
    main()
