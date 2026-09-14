#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 6) Why does dfs_corner fail -- is it "depth-first" or "starting from
a corner"?

## Why this is needed

Followup 3 (`verify_vf2_ordering_id_order_true.py`) found that under
`id_order=True`, `dfs_corner` alone failed on 3 of 4 grids, while every other
structured ordering (including row_major) succeeded on all of them. That is a new
finding, but "why" admits two compatible explanations:

  (a) **The depth-first traversal itself** is bad (it builds long, thin candidate
      chains).
  (b) **Starting from the top-left corner** is bad (it anchors the search at a
      special low-degree point).

`dfs_corner` traverses depth-first from a corner, so both factors are mixed
together. Separating them needs (1) varying **only the starting point** while
keeping depth-first traversal (corner, opposite corner, edge midpoint, center),
and (2) trying several **neighbor-visit orderings** within depth-first, to confirm
the result is not a coincidence of one particular ordering.

## Pre-registered predictions

(P1) Depth-first fails **regardless of starting point** (also fails starting from
     the opposite corner, an edge midpoint, or the center). -> Supports "the
     depth-first traversal itself is the cause."
(P2) Varying the depth-first neighbor-visit order (right->down->left->up,
     down->right->up->left, and others) does not change the outcome -- the
     failure is consistent. -> Not a coincidence of one particular ordering.
(P3) Breadth-first succeeds regardless of starting point (followup 3 found
     `bfs_corner` succeeds; this confirms it does not break with a different
     starting point).

**If this fails**: if depth-first succeeds once the starting point is changed,
explanation (b) (the corner as a special point) is the correct one, and
addendum-6's "depth-first itself" account needs correcting.

## Usage

    python verify_vf2_dfs_mechanism.py
    python verify_vf2_dfs_mechanism.py --grids 6x7 8x8 8x9 --call-limit 3000000
"""
from __future__ import annotations

import argparse
import time

from vf2_probe_common import banner, default_out, environment, write_csv


def grid_nodes(rows, cols):
    return [(r, c) for r in range(rows) for c in range(cols)]


def grid_edges(rows, cols):
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append(((r, c), (r, c + 1)))
            if r + 1 < rows:
                e.append(((r, c), (r + 1, c)))
    return e


# Four neighbor-visit-order patterns. dfs_corner matches the original
# implementation's (right, down, left, up).
NEIGHBOR_ORDERS = {
    "RDLU": [(0, 1), (1, 0), (0, -1), (-1, 0)],   # same as the original verify_vf2_ordering_structure.py
    "DRUL": [(1, 0), (0, 1), (-1, 0), (0, -1)],
    "LURD": [(0, -1), (-1, 0), (0, 1), (1, 0)],
    "ULDR": [(-1, 0), (0, -1), (1, 0), (0, 1)],
}

START_POINTS = {
    "corner_TL": lambda rows, cols: (0, 0),
    "corner_BR": lambda rows, cols: (rows - 1, cols - 1),
    "edge_mid":  lambda rows, cols: (0, cols // 2),
    "center":    lambda rows, cols: (rows // 2, cols // 2),
}


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


def probe(rows, cols, order, n_pairs, call_limit, id_order):
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
    ap.add_argument("--grids", nargs="+", default=["6x7", "8x8", "8x9"])
    ap.add_argument("--call-limit", type=int, default=3_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 6) Why does dfs_corner fail -- depth-first itself, or starting from a corner?", [
        "P1 depth-first fails regardless of starting point (opposite corner, edge midpoint, center)",
        "P2 depth-first's failure is consistent across different neighbor-visit orders",
        "P3 breadth-first succeeds regardless of starting point",
    ])

    rows_out = []
    print(f"{'grid':>6} {'mode':<6} {'start':<10} {'nbr_order':<6} {'found':>6} {'ms':>10}")
    for g in args.grids:
        r, c = (int(x) for x in g.lower().split("x"))
        n = r * c
        n_pairs = n // 2
        for depth_first, mode in ((True, "dfs"), (False, "bfs")):
            for start_name, start_fn in START_POINTS.items():
                for nbr_name, nbr_order in NEIGHBOR_ORDERS.items():
                    order = traverse(r, c, start_fn(r, c), depth_first, nbr_order)
                    ok, el = probe(r, c, order, n_pairs, args.call_limit, id_order=True)
                    print(f"{g:>6} {mode:<6} {start_name:<10} {nbr_name:<6} "
                          f"{str(ok):>6} {el*1000:10.2f}")
                    rows_out.append(dict(Grid=g, Nodes=n, Pairs=n_pairs, Mode=mode,
                                         Start=start_name, NeighborOrder=nbr_name,
                                         Found=ok, Time_s=el, CallLimit=args.call_limit))

    df = write_csv(rows_out, args.out or default_out("vf2_dfs_mechanism"), environment())

    print("\n" + "=" * 78)
    print("Verdict -- success rate by mode (dfs/bfs) x start (pooling the 4 neighbor-visit orders)")
    print("=" * 78)
    summ = df.groupby(["Mode", "Start"]).Found.agg(["sum", "count"])
    summ["rate"] = (summ["sum"] / summ["count"]).round(2)
    print(summ.to_string())
    print("\n  -> If the dfs rows are near 0.00 regardless of start, that supports P1 "
          "(depth-first itself is the cause).")
    print("     If the dfs rows' success rate varies a lot by start, the corner as a "
          "special point (b) is the issue.")
    print("     If the bfs rows are near 1.00 regardless of start, that supports P3.")


if __name__ == "__main__":
    main()
