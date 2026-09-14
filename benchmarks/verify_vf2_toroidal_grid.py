#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 4/4) Does the cliff appear on a toroidal grid (periodic boundary) --
testing the "degree uniformity" hypothesis

## Why this is needed

The 2026-09-14 `verify_vf2_topologies.py` found that a line shows the cliff
about as much as a grid does, while a ring and a full graph show no cliff at
all. All five always have a perfect matching, so "whether a matching exists"
cannot explain it.

The common thread noticed: the line and grid, where the cliff appears, are
**not vertex-transitive** (they have boundaries and endpoints, so node degree
varies across the graph), while the ring and full graph, where it does not
appear, **are vertex-transitive** (every node has the same degree, and the
graph looks the same from every node's point of view) -- the new hypothesis
in addendum-5 section 5.

If this is really the dividing line, **a "toroidal grid" -- one with the same
local structure as the ordinary grid (a degree-4 lattice) but made
vertex-transitive via periodic boundary conditions -- should show no cliff.**
This is the first experiment that directly separates "the cliff appears
because it's a grid" from "the cliff appears because it isn't
vertex-transitive".

`CouplingMap.from_grid` has no periodic-boundary variant, so this builds one
via `networkx.grid_2d_graph(..., periodic=True)` and constructs a
`CouplingMap` from its edge list.

## Pre-registered predictions

(P1) The **toroidal** 6x7 / 7x8 grids (degree-4, uniform, vertex-transitive)
     show no cliff even when saturated (the ratio against the unsaturated case
     falls well below 10x -- as a rough target, near the 0.8-1.2x level seen
     for the ring/full graph).
(P2) In the same run, the **open** (non-periodic) grid is measured side by side,
     confirming that the 2026-09-14 cliff (57x-414x) reproduces within the
     same session (a control ruling out the torus "no cliff" result being a
     false negative from a difference in measurement conditions).
(P3) A perfect matching exists on both the torus and the open grid (with
     higher degree than the open grid, it should if anything be at least as
     likely to exist; this being contradicted is nearly inconceivable, but is
     checked before timing anything, per this project's standing discipline).

**If this fails**: if the torus also shows the cliff, the "vertex transitivity"
hypothesis is rejected, and the dividing line more likely lies in some other
property (e.g. simply whether the graph is bipartite-like, or the grid-shaped
local structure itself).

## Usage

    python verify_vf2_toroidal_grid.py
    python verify_vf2_toroidal_grid.py --shapes 6x7 7x8 --levels 2 3 --reps 2
"""
from __future__ import annotations

import argparse

from vf2_probe_common import (BASIS_GATES, banner, build_dense_pair_blocks_circuit,
                              default_out, environment, has_perfect_matching, timed,
                              write_csv)


def torus_coupling_map(rows, cols):
    """A 2D grid with periodic boundaries. Every node has degree 4, uniformly
    (vertex-transitive)."""
    import networkx as nx
    from qiskit.transpiler import CouplingMap
    g = nx.grid_2d_graph(rows, cols, periodic=True)
    idx = {n: i for i, n in enumerate(g.nodes())}
    edges = []
    for a, b in g.edges():
        edges.append((idx[a], idx[b]))
        edges.append((idx[b], idx[a]))  # CouplingMap is a directed edge list; add both directions
    return CouplingMap(edges)


def open_grid_coupling_map(rows, cols):
    from qiskit.transpiler import CouplingMap
    return CouplingMap.from_grid(rows, cols)


def run(qc, cmap, level, reps):
    from qiskit import transpile
    mn, md, _ = timed(
        lambda: transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
                          optimization_level=level, seed_transpiler=0), reps)
    return mn, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["6x7", "7x8"])
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--spare", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 4/4) Toroidal grid -- is vertex transitivity the dividing line for the cliff?", [
        "P1 the toroidal grid (uniform degree 4) shows no cliff even saturated (~1x, like ring/full)",
        "P2 the open grid's cliff (57x-414x) reproduces within the same session (control)",
        "P3 a perfect matching exists on both the torus and the open grid",
    ])
    print("**The existence of a perfect matching is checked before timing anything.**\n")

    topologies = []
    for shape in args.shapes:
        r, c = (int(x) for x in shape.lower().split("x"))
        topologies.append((f"torus_{shape}", torus_coupling_map(r, c)))
        topologies.append((f"open_{shape}", open_grid_coupling_map(r, c)))

    rows = []
    hdr = f"{'topology':<14} {'phys':>5} {'spare':>6} {'qubits':>7} {'PM?':>6}"
    for lv in args.levels:
        hdr += f" {'L' + str(lv) + ' ms':>10}"
    print(hdr)

    for name, cmap in topologies:
        phys = cmap.size()
        for spare in args.spare:
            n = phys - spare
            if n < 4 or n % 2:
                continue
            pm = has_perfect_matching(cmap, n // 2)
            qc = build_dense_pair_blocks_circuit(n, seed=0)
            line = f"{name:<14} {phys:>5} {spare:>6} {n:>7} {str(pm):>6}"
            rec = dict(Topology=name, Physical=phys, Spare=spare, Qubits=n,
                       PerfectMatchingExists=pm)
            for lv in args.levels:
                mn, md = run(qc, cmap, lv, args.reps)
                line += f" {mn*1000:10.1f}"
                rec[f"L{lv}_min_s"] = mn
                rec[f"L{lv}_median_s"] = md
            print(line, flush=True)
            rows.append(rec)

    df = write_csv(rows, args.out or default_out("vf2_toroidal_grid"), environment())

    print("\n" + "=" * 78)
    print("Verdict -- ratio of saturated (spare=0) to spare")
    print("=" * 78)
    for lv in args.levels:
        col = f"L{lv}_min_s"
        print(f"\noptimization_level = {lv}")
        for name in df.Topology.unique():
            sub = df[df.Topology == name].set_index("Spare")
            if 0 in sub.index and len(sub) > 1:
                other = [s for s in sub.index if s != 0][0]
                ratio = sub.loc[0, col] / sub.loc[other, col]
                tag = "cliff" if ratio > 10 else "no cliff"
                print(f"  {name:<14} {ratio:8.1f}x  {tag}")
    print("\n  -> torus_* with no cliff and open_* with a cliff supports P1/P2 -- the")
    print("     'vertex transitivity' hypothesis. A cliff on torus_* too rejects it.")


if __name__ == "__main__":
    main()
