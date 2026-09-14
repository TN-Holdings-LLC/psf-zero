#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(4/6) Do Qiskit's VF2 and rustworkx's VF2 fail on the same instances?

## Why this is needed

`spare-qubit-cliff.md` states: "both use VF2++ and both fall into ordering
dependence on this pattern. A family resemblance and a shared symptom, not a
shared cause." **The same set of grids has never been run through both
implementations for comparison.**

The rustworkx side is known to fail past 24 nodes (3x8 and 4x6 are the smallest
failures, and the boundary agrees across all 14 grids tested). The Qiskit side has
only been checked at one point, 6x7.

Building the same table for both settles whether "fails on the same instance" can
be said, or whether the boundary shifts.

## Pre-registered predictions

(P1) The Qiskit side also fails from 24 nodes up. found/not-found agrees between
     the two implementations on every grid.
(P2) If they agree, 24 is a property of the VF2++ heuristic itself, not of an
     implementation detail.
(P3) Grids with no perfect matching (odd node counts, etc.) give not-found on
     both, which is correct behaviour and excluded from the tally.

**If this fails**: if the boundary shifts (e.g. Qiskit passes at 4x6 but fails at
6x7), the two are different implementations of the same heuristic, not the same
failure. In that case using the rustworkx measurement in an argument about Qiskit
should be treated even more cautiously than it already is.

## Usage

    python verify_vf2_cross_implementation.py
    python verify_vf2_cross_implementation.py --max-nodes 72
"""
from __future__ import annotations

import argparse
import time

from vf2_probe_common import (banner, build_dense_pair_blocks_circuit, default_out,
                              environment, has_perfect_matching, write_csv)


def grids(max_nodes):
    out = []
    for r in range(2, 13):
        for c in range(r, 13):
            n = r * c
            if n % 2 == 0 and 4 <= n <= max_nodes:
                out.append((r, c))
    return sorted(out, key=lambda rc: (rc[0] * rc[1], rc))


def qiskit_probe(rows, cols, call_limit):
    """Qiskit's VF2Layout. seed=-1 disables shuffling -- the same identity order
    the preset uses."""
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler import PropertySet
    from qiskit.transpiler.passes import VF2Layout
    n = rows * cols
    cmap = CouplingMap.from_grid(rows, cols)
    qc = build_dense_pair_blocks_circuit(n, seed=0)
    p = VF2Layout(coupling_map=cmap, seed=-1, call_limit=call_limit, max_trials=1)
    ps = PropertySet()   # a plain dict raises KeyError on an unset key
    p.property_set = ps
    t0 = time.perf_counter()
    p.run(circuit_to_dag(qc))
    el = time.perf_counter() - t0
    return str(ps.get("VF2Layout_stop_reason")).endswith(".SOLUTION_FOUND"), el, cmap


def rustworkx_probe(rows, cols, call_limit):
    """rustworkx's vf2_mapping, with the same arguments the older Qiskit used."""
    import rustworkx as rx
    n = rows * cols
    cm = rx.PyGraph()
    idx = {}
    for r in range(rows):
        for c in range(cols):
            idx[(r, c)] = cm.add_node((r, c))
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                cm.add_edge(idx[(r, c)], idx[(r, c + 1)], None)
            if r + 1 < rows:
                cm.add_edge(idx[(r, c)], idx[(r + 1, c)], None)
    im = rx.PyGraph()
    for _ in range(n // 2):
        a, b = im.add_node(None), im.add_node(None)
        im.add_edge(a, b, None)
    t0 = time.perf_counter()
    it = rx.vf2_mapping(cm, im, subgraph=True, id_order=False, induced=False,
                        call_limit=call_limit)
    m = next(iter(it), None)
    return m is not None, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=72)
    ap.add_argument("--call-limit", type=int, default=3_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(4/6) Do Qiskit VF2 and rustworkx VF2 fail on the same instance?", [
        "P1 the Qiskit side also fails from 24 nodes up; found/not-found agrees on every grid",
        "P2 if they agree, 24 is a property of the VF2++ heuristic itself",
        "P3 grids with no perfect matching give not-found on both -- correct behaviour, excluded",
    ])

    rows_out = []
    print(f"{'grid':>7} {'nodes':>6} {'PM?':>5} {'qiskit':>7} {'rustwx':>7} "
          f"{'agree':>6} {'q ms':>9} {'r ms':>9}")
    for r, c in grids(args.max_nodes):
        n = r * c
        q_ok, q_t, cmap = qiskit_probe(r, c, args.call_limit)
        r_ok, r_t = rustworkx_probe(r, c, args.call_limit)
        pm = has_perfect_matching(cmap, n // 2)
        agree = (q_ok == r_ok)
        print(f"{f'{r}x{c}':>7} {n:>6} {str(pm):>5} {str(q_ok):>7} {str(r_ok):>7} "
              f"{str(agree):>6} {q_t*1000:9.1f} {r_t*1000:9.1f}")
        rows_out.append(dict(Grid=f"{r}x{c}", Rows=r, Cols=c, Nodes=n,
                             PerfectMatchingExists=pm,
                             QiskitFound=q_ok, RustworkxFound=r_ok, Agree=agree,
                             Qiskit_s=q_t, Rustworkx_s=r_t,
                             CallLimit=args.call_limit))

    df = write_csv(rows_out, args.out or default_out("vf2_cross_implementation"),
                   environment())

    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    valid = df[df.PerfectMatchingExists != False]  # noqa: E712  (keep None too)
    print(f"Grids with a perfect matching: {len(valid)} / {len(df)}")
    print(f"Agreement between the two implementations: {int(valid.Agree.sum())} / {len(valid)}")
    for name, col in (("Qiskit", "QiskitFound"), ("rustworkx", "RustworkxFound")):
        f = valid[~valid[col]]
        if len(f):
            print(f"{name}'s first failing node count: {int(f.Nodes.min())} "
                  f"({', '.join(f[f.Nodes == f.Nodes.min()].Grid)})")
        else:
            print(f"{name}: succeeds on every grid")
    print("\n  -> 100% agreement at the same node count supports P1/P2.")
    print("     A shift means the two are separate failures of the same heuristic.")


if __name__ == "__main__":
    main()
