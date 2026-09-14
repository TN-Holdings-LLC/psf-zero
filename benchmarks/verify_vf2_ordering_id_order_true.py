#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 3/4) Does a structural ordering differentiate results under
id_order=True -- testing the tie-break hypothesis

## Why this is needed

The 2026-09-14 `verify_vf2_ordering_structure.py` run (`id_order=False`) found
that seven hand-built orderings -- row_major, col_major, snake, bipartite,
reverse, and others -- **all** fail the same way, with only 1 of 20 random
draws succeeding. If none of the hand-built relabelings change the outcome,
that suggests `id_order=False` largely ignores the supplied node-ID order and
constructs its own order from structural criteria such as degree (addendum-5
section 4). The single random success might then be coincidental, arising from
however ties are broken among equal-degree nodes.

This is directly testable. **Run the same seven orderings plus random draws
under `id_order=True`** (plain VF2, already known from the root-cause
investigation to "find it in under a millisecond" on this pattern) and see
whether the results differentiate. If every ordering behaves the same way
under `id_order=True` too (all fast, or all slow), the supplied order is
likely genuinely ignored. If they differentiate, that is evidence the supplied
order *is* used, and the "ignored" account of `id_order=False` should be
doubted.

`id_order=False` is re-measured in the same script alongside, to also confirm
whether the 2026-09-14 result reproduces.

## Pre-registered predictions

(P1) Under `id_order=True`, at least some structural orderings are found
     quickly (consistent with the root-cause finding that `id_order=True`
     takes under a millisecond on every saturated instance).
(P2) Under `id_order=True`, results differentiate by ordering (not the uniform
     behaviour across all orderings seen under `id_order=False`).
(P3) The bipartite (checkerboard) ordering is found especially quickly under
     `id_order=True` (a guess that this ordering meshes with a perfect
     matching's structure of connecting the two sides of a bipartite graph;
     stated explicitly as a weak prediction whose failure does not affect the
     other predictions).
(P4) The `id_order=False` side matches 2026-09-14: little difference between
     orderings (a reproducibility check).

**If this fails**: if (P2) fails and every ordering behaves the same way even
under `id_order=True`, the hypothesis that "`id_order=False` ignores the
supplied order" is itself wrong, and something else (perhaps something in the
graph construction itself) is the real factor.

## Usage

    python verify_vf2_ordering_id_order_true.py
    python verify_vf2_ordering_id_order_true.py --grids 6x7 7x8 8x8 8x9 --random-seeds 5
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
    ap.add_argument("--grids", nargs="+", default=["6x7", "7x8", "8x8", "8x9"])
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--call-limit", type=int, default=3_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 3/4) Does structural order differentiate under id_order=True -- tie-break hypothesis", [
        "P1 under id_order=True, at least some structural orders are found quickly",
        "P2 under id_order=True, results differentiate by ordering (not uniform)",
        "P3 (weak) bipartite is especially fast under id_order=True",
        "P4 the id_order=False side matches 2026-09-14: little difference between orderings (reproducibility)",
    ])

    builders = [("row_major", order_row_major), ("col_major", order_col_major),
                ("snake", order_snake), ("bfs_corner", order_bfs),
                ("dfs_corner", order_dfs), ("bipartite", order_bipartite),
                ("reverse", order_reverse)]

    rows_out = []
    print(f"{'grid':>6} {'order':<14} {'id_order':>9} {'found':>6} {'ms':>10}")
    for g in args.grids:
        r, c = (int(x) for x in g.lower().split("x"))
        n = r * c
        n_pairs = n // 2
        for id_order in (False, True):
            for name, fn in builders:
                ok, el = probe(r, c, fn(r, c), n_pairs, args.call_limit, id_order)
                print(f"{g:>6} {name:<14} {str(id_order):>9} {str(ok):>6} {el*1000:10.2f}")
                rows_out.append(dict(Grid=g, Nodes=n, Pairs=n_pairs, Order=name,
                                     OrderSeed=None, IdOrder=id_order, Found=ok,
                                     Time_s=el, CallLimit=args.call_limit))
            for s in range(args.random_seeds):
                ok, el = probe(r, c, order_random(r, c, s), n_pairs, args.call_limit,
                               id_order)
                print(f"{g:>6} {'random_' + str(s):<14} {str(id_order):>9} "
                      f"{str(ok):>6} {el*1000:10.2f}")
                rows_out.append(dict(Grid=g, Nodes=n, Pairs=n_pairs, Order="random",
                                     OrderSeed=s, IdOrder=id_order, Found=ok,
                                     Time_s=el, CallLimit=args.call_limit))

    df = write_csv(rows_out, args.out or default_out("vf2_ordering_id_order_true"),
                   environment())

    print("\n" + "=" * 78)
    print("Verdict -- success rate by id_order and by ordering")
    print("=" * 78)
    summ = df.groupby(["IdOrder", "Order"]).Found.agg(["sum", "count"])
    summ["rate"] = (summ["sum"] / summ["count"]).round(2)
    print(summ.to_string())
    print("\n  -> Varying success rates in the id_order=True rows support P1/P2.")
    print("     Uniform rows (all 0 or all 1) under id_order=False support P4 (reproduction).")
    print("     If both are uniform with no difference, the 'ignores the supplied order' "
          "hypothesis itself becomes doubtful.")


if __name__ == "__main__":
    main()
