#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(3/6) **What about** the VF2++ ordering matters -- probed structurally on the
rustworkx side

## Why this is needed

"The failure is ordering-dependent" is established. **What about** the ordering
is not.

On the Qiskit side the only lever on ordering is `shuffle_seed` (i.e. random), and
random draws cannot answer "what". **The rustworkx side lets the node insertion
order be set structurally.** VF2++ decides order from node degree and index, so
changing how indices are assigned changes the order.

This is a separate implementation, not evidence about Qiskit -- that stance from
the existing document is unchanged. What this is meant to give is a **structural
clue about which orderings pass**, not a claim about Qiskit's internals.

## Orderings tried (same graph throughout; only the index assignment differs)

  row_major     row-major (the grid's natural order)
  col_major     column-major
  snake         boustrophedon (each row reverses direction)
  bfs_corner    breadth-first from the top-left corner
  dfs_corner    depth-first from the top-left corner
  bipartite     checkerboard (all of one colour first, then the other)
  reverse       row-major, reversed
  random_k      random (several seeds)

## Pre-registered predictions

(P1) Locality-preserving orders (row_major, snake, bfs_corner) **succeed**.
(P2) Random orders mostly fail (consistent with the existing near-total failure
     from 24 nodes up).
(P3) **`bipartite` is the dividing line.** A perfect matching connects one side of
     a bipartite graph to the other, so an order grouped by colour should let
     VF2++ traverse it straightforwardly. If it succeeds, that narrows the
     mechanism toward "what matters is following the matching's structure, not
     geometric locality".
(P4) `col_major` and `reverse` give the same result as `row_major` (symmetry).

**If this fails**: if everything succeeds or everything fails, the ordering's
structure is not what matters, and whatever `shuffle_seed` was changing was
something else (e.g. only the search's starting point).

## Usage

    python verify_vf2_ordering_structure.py
    python verify_vf2_ordering_structure.py --grids 6x7 7x8 8x8 8x9 --random-seeds 5
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


def order_row_major(rows, cols):
    return grid_nodes(rows, cols)


def order_col_major(rows, cols):
    return [(r, c) for c in range(cols) for r in range(rows)]


def order_snake(rows, cols):
    out = []
    for r in range(rows):
        rng = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        out += [(r, c) for c in rng]
    return out


def _traverse(rows, cols, depth_first):
    start = (0, 0)
    seen = {start}
    frontier = [start]
    out = []
    while frontier:
        n = frontier.pop() if depth_first else frontier.pop(0)
        out.append(n)
        r, c = n
        for m in ((r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)):
            if 0 <= m[0] < rows and 0 <= m[1] < cols and m not in seen:
                seen.add(m)
                frontier.append(m)
    return out


def order_bfs(rows, cols):
    return _traverse(rows, cols, False)


def order_dfs(rows, cols):
    return _traverse(rows, cols, True)


def order_bipartite(rows, cols):
    ns = grid_nodes(rows, cols)
    return [n for n in ns if (n[0] + n[1]) % 2 == 0] + \
           [n for n in ns if (n[0] + n[1]) % 2 == 1]


def order_reverse(rows, cols):
    return list(reversed(grid_nodes(rows, cols)))


def order_random(rows, cols, seed):
    import random
    ns = grid_nodes(rows, cols)
    random.Random(seed).shuffle(ns)
    return ns


def probe(rows, cols, order, n_pairs, call_limit):
    """Build the grid with node indices assigned in `order`, and search for a
    perfect-matching-shaped interaction graph as a subgraph isomorphism.
    Returns (found, elapsed)."""
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
    it = rx.vf2_mapping(cm, im, subgraph=True, id_order=False, induced=False,
                        call_limit=call_limit)
    m = next(iter(it), None)
    return m is not None, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=["6x7", "7x8", "8x8", "8x9"])
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--call-limit", type=int, default=3_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(3/6) VF2++ ordering structure -- controlling insertion order on the rustworkx side", [
        "P1 locality-preserving orders (row_major, snake, bfs_corner) succeed",
        "P2 random orders mostly fail",
        "P3 bipartite (checkerboard, grouped by colour) is the dividing line. If it "
        "succeeds, what matters is \"following the matching's structure\", not "
        "\"geometric locality\"",
        "P4 col_major and reverse give the same result as row_major (symmetry)",
    ])
    print("Note: this is rustworkx's VF2++, not Qiskit's implementation. What comes "
          "out is a structural clue, not a claim about Qiskit.\n")

    builders = [("row_major", order_row_major), ("col_major", order_col_major),
                ("snake", order_snake), ("bfs_corner", order_bfs),
                ("dfs_corner", order_dfs), ("bipartite", order_bipartite),
                ("reverse", order_reverse)]

    rows_out = []
    print(f"{'grid':>6} {'order':<14} {'found':>6} {'ms':>10}")
    for g in args.grids:
        r, c = (int(x) for x in g.lower().split("x"))
        n = r * c
        n_pairs = n // 2
        for name, fn in builders:
            ok, el = probe(r, c, fn(r, c), n_pairs, args.call_limit)
            print(f"{g:>6} {name:<14} {str(ok):>6} {el*1000:10.1f}")
            rows_out.append(dict(Grid=g, Nodes=n, Pairs=n_pairs, Order=name,
                                 OrderSeed=None, Found=ok, Time_s=el,
                                 CallLimit=args.call_limit))
        for s in range(args.random_seeds):
            ok, el = probe(r, c, order_random(r, c, s), n_pairs, args.call_limit)
            print(f"{g:>6} {'random_' + str(s):<14} {str(ok):>6} {el*1000:10.1f}")
            rows_out.append(dict(Grid=g, Nodes=n, Pairs=n_pairs, Order="random",
                                 OrderSeed=s, Found=ok, Time_s=el,
                                 CallLimit=args.call_limit))

    df = write_csv(rows_out, args.out or default_out("vf2_ordering_structure"),
                   environment())

    print("\n" + "=" * 78)
    print("Verdict -- success rate by order")
    print("=" * 78)
    summ = df.groupby("Order").Found.agg(["sum", "count"])
    summ["rate"] = (summ["sum"] / summ["count"]).round(2)
    print(summ.to_string())
    print("\n  -> bipartite at 1.00 and random at 0.00 supports P3.")
    print("     Everything at 1.00 or everything at 0.00 means the order's structure "
          "does not matter.")


if __name__ == "__main__":
    main()
