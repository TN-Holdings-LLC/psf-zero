#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 8) Between grid and heavy-hex -- is sparsity itself a third factor
in the cliff?

## Why this is needed

Addendum-7's two-factor hypothesis was: the cliff appears when **both**
(a) not vertex-transitive **and** (b) a perfect matching can exist all the way
down to zero spare. Evidence gathered so far:

  - grid, line          : (a) yes, (b) yes -> cliff present
  - ring, full graph     : (a) no,  (b) yes -> no cliff
  - torus                : (a) no,  (b) yes -> no cliff
  - heavy-hex, real chip : (a) yes, (b) NO (cannot reach zero spare) -> no cliff

Heavy-hex fails to satisfy (b) precisely because its degree is low (a mix of
degree 2 and 3, average degree < 3), so "sparsity" and "cannot satisfy (b)"
happen to coincide on heavy-hex. That means the (a)+(b) theory has not
distinguished "the cliff vanishes because it's sparse" from "the cliff vanishes
because (b) cannot be satisfied" using heavy-hex alone.

A line (fixed degree 2, the lowest possible average degree) is already known to
show the cliff, which might suggest "sparsity alone doesn't make the cliff
vanish" -- but a line is one-dimensional and too simple a structure (no
branching, and it differs from a grid in more ways than just sparsity). Here,
several topologies **between** grid and line are built -- keeping the
two-dimensional structure while lowering only the average degree -- to see
whether the cliff keeps appearing as long as (b) can be satisfied, or whether it
weakens or vanishes in some gray-zone degree range as degree is lowered.

Two candidate families:
  - brick: each row is a fully connected chain (as in a line) + vertical edges
    kept on only half the checkerboard pattern (a "brick-laid" pattern, similar
    to the coupling topology of some real superconducting devices). Average
    degree sits roughly between a line and a grid.
  - diluted_p: horizontal edges are all kept (to guarantee connectivity) +
    vertical edges are kept independently with probability p. Sweeping
    p = 0.25 / 0.5 / 0.75 moves the average degree continuously.

Both have different degree at the boundary (corners, edges, interior) versus the
interior, so they are non-vertex-transitive (the same side as grid and line).
Whether a perfect matching exists at spare=0 is checked directly after
construction (if none exists, that candidate itself becomes an example of a
sparse topology failing to satisfy (b)).

## Pre-registered predictions

(P1) Among brick and diluted_p, whichever has a perfect matching near spare=0
     shows the cliff regardless of average degree (supports the two-factor
     hypothesis -- sparsity is not a third factor).
(P2) Regardless of whether a perfect matching exists, the cliff's magnitude
     itself (the tight/loose ratio) may shrink as average degree drops (the
     search space shrinks, so while this would not amount to "the cliff
     vanishes," a tendency toward "the cliff gets shallower" is plausible).

**If this fails**: if (P1) fails and a sparse topology is found where a perfect
matching exists but no cliff appears, the two-factor hypothesis is insufficient,
and average degree (the breadth of the search space) itself was an independent
third factor.

## Usage

    python verify_vf2_sparse_topology.py
    python verify_vf2_sparse_topology.py --grid 8x8 --levels 2 3
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

from vf2_probe_common import (BASIS_GATES, banner, build_dense_pair_blocks_circuit,
                              default_out, environment, has_perfect_matching, timed,
                              write_csv)


def _cmap_from_edges(n_nodes, edges):
    from qiskit.transpiler import CouplingMap
    cm = CouplingMap()
    for i in range(n_nodes):
        cm.add_physical_qubit(i)
    for a, b in edges:
        cm.add_edge(a, b)
        cm.add_edge(b, a)
    return cm


def grid_edges(rows, cols):
    idx = lambda r, c: r * cols + c
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                e.append((idx(r, c), idx(r + 1, c)))
    return e


def brick_edges(rows, cols):
    """Each row is fully connected as a chain; vertical edges are kept on only
    half the checkerboard pattern."""
    idx = lambda r, c: r * cols + c
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows and (r + c) % 2 == 0:
                e.append((idx(r, c), idx(r + 1, c)))
    return e


def diluted_grid_edges(rows, cols, p, seed):
    """Horizontal edges are all kept (guarantees connectivity). Vertical edges
    are kept independently with probability p."""
    rng = random.Random(seed)
    idx = lambda r, c: r * cols + c
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows and rng.random() < p:
                e.append((idx(r, c), idx(r + 1, c)))
    return e


def avg_degree(n_nodes, edges):
    return 2 * len(edges) / n_nodes


def is_connected(n_nodes, edges):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(edges)
    return nx.is_connected(g)


def topologies(rows, cols):
    """(name, CouplingMap, average degree, list of spare values to try)."""
    n = rows * cols
    out = []

    ge = grid_edges(rows, cols)
    out.append(("grid", _cmap_from_edges(n, ge), avg_degree(n, ge)))

    be = brick_edges(rows, cols)
    assert is_connected(n, be), "brick ended up disconnected"
    out.append(("brick", _cmap_from_edges(n, be), avg_degree(n, be)))

    for p in (0.25, 0.5, 0.75):
        # Retry with a different seed until connected (horizontal edges alone
        # already guarantee connectivity, so this should normally succeed on
        # the first try).
        for seed in range(20):
            de = diluted_grid_edges(rows, cols, p, seed)
            if is_connected(n, de):
                break
        else:
            print(f"diluted_p{p}: did not become connected after 20 seeds "
                  f"(unexpected, since horizontal edges alone should connect it)")
            continue
        out.append((f"diluted_p{p}", _cmap_from_edges(n, de), avg_degree(n, de)))

    # Include a line with the same node count as a control (a single-row
    # chain -- an extreme case with no two-dimensional structure).
    from qiskit.transpiler import CouplingMap
    line_cm = CouplingMap.from_line(n, bidirectional=True)
    out.append(("line", line_cm, 2 * (n - 1) / n))

    spares = list(range(0, min(n - 4, 41), 2))
    return [(name, cm, deg, spares) for name, cm, deg in out]


def run(qc, cmap, level, reps):
    from qiskit import transpile
    mn, md, _ = timed(
        lambda: transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
                          optimization_level=level, seed_transpiler=0), reps)
    return mn, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="8x8",
                    help="base physical-qubit rows x cols (kept as the common "
                         "physical count across all candidates)")
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 8) Between grid and heavy-hex -- is sparsity a third factor?", [
        "P1 a candidate with a perfect matching shows the cliff regardless of average degree",
        "P2 the cliff's magnitude itself may shrink as average degree drops",
    ])
    print("**The existence of a perfect matching is checked before timing anything.**\n")

    rows, cols = (int(x) for x in args.grid.lower().split("x"))
    rows_out = []
    hdr = f"{'topology':<14} {'avg_deg':>8} {'phys':>5} {'spare':>6} {'qubits':>7} {'PM?':>6}"
    for lv in args.levels:
        hdr += f" {'L' + str(lv) + ' ms':>10}"
    print(hdr)

    for name, cmap, deg, spares in topologies(rows, cols):
        phys = cmap.size()
        pm_by_spare = {}
        for spare in spares:
            n = phys - spare
            if n < 4 or n % 2:
                continue
            pm_by_spare[spare] = has_perfect_matching(cmap, n // 2)
        has_pm = [s for s, ok in pm_by_spare.items() if ok]
        print(f"{name} (avg_deg={deg:.2f}): spare values with a matching = "
              f"{has_pm if has_pm else '(none)'}")

        if not has_pm:
            for spare, pm in pm_by_spare.items():
                n = phys - spare
                rows_out.append(dict(Topology=name, AvgDegree=deg, Physical=phys,
                                     Spare=spare, Qubits=n, PerfectMatchingExists=pm))
            continue

        targets = sorted({min(has_pm), max(has_pm)})
        for spare in targets:
            n = phys - spare
            line = f"{name:<14} {deg:>8.2f} {phys:>5} {spare:>6} {n:>7} {'True':>6}"
            rec = dict(Topology=name, AvgDegree=deg, Physical=phys, Spare=spare,
                       Qubits=n, PerfectMatchingExists=True)
            qc = build_dense_pair_blocks_circuit(n, seed=0)
            for lv in args.levels:
                mn, md = run(qc, cmap, lv, args.reps)
                line += f" {mn*1000:10.1f}"
                rec[f"L{lv}_min_s"] = mn
                rec[f"L{lv}_median_s"] = md
            print(line, flush=True)
            rows_out.append(rec)

    df = write_csv(rows_out, args.out or default_out("vf2_sparse_topology"), environment())

    print("\n" + "=" * 78)
    print("Verdict -- per topology, ratio of the smallest spare with a matching to the largest")
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
            deg = sub.iloc[0].AvgDegree
            print(f"  {name:<14} avg_deg={deg:5.2f}  spare {int(tight.Spare)} / "
                  f"spare {int(loose.Spare)}: {ratio:8.1f}x  {tag}")
    print("\n  -> If plotting the cliff's magnitude against average degree shows a degree")
    print("     range where the cliff vanishes or weakens partway through, P1 is")
    print("     insufficient (sparsity is an independent factor).")
    print("     If the cliff keeps appearing at every degree as long as a perfect")
    print("     matching exists, P1 is supported.")


if __name__ == "__main__":
    main()
