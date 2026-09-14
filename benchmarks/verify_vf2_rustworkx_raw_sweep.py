#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 7) Is "burning the whole budget" a quirk of Qiskit's VF2Layout pass,
or of rustworkx itself?

## Why this is needed

Followup 4 (the call_limit sweep) found that seed 8, given a large `call_limit`
through Qiskit's `VF2Layout` pass (`max_trials=1`), consumes **essentially the same
amount of time as the failing seed (seed 0)** even though it must have found a
solution early.

That measured `VF2Layout.run(dag)` -- a call that simply returns a result. The
ordering-family scripts in this batch, by contrast, all call
`rustworkx.vf2_mapping()` directly and pull out only the first item through
Python's **lazy iterator** protocol (`next(iter(it))`). An ordinary Python
generator should stop computing the moment the first value is produced.

If the raw rustworkx call really does "stop once found," then "burning the whole
budget" is specific to Qiskit's `VF2Layout` pass implementation (plausibly, the
Rust side does the `call_limit` bookkeeping to completion while constructing
`max_trials` internally), not a property of rustworkx's search algorithm itself.

`VF2Layout` shuffles internally by seed, but `rustworkx.vf2_mapping()` has no
equivalent seed argument, so "seed 8" cannot be carried over directly. Instead
this uses a two-stage design: **scan several random relabelings to find one
medium-difficulty instance** (found, but not instantly), then sweep `call_limit`
against that one instance.

## Pre-registered predictions

(P1) If a medium-difficulty instance is found, sweeping call_limit against it
     shows time **plateauing past some threshold** (i.e. not the "burns the whole
     budget" behaviour seen in Qiskit's VF2Layout pass).
(P2) An instance with no solution, swept over call_limit, scales roughly
     proportionally with the budget, the same as on the Qiskit side (exhausting
     the search space is a natural behaviour that should hold in both
     implementations).

**If this fails**: if (P1) fails and the raw rustworkx call also burns the whole
budget, "burning the whole budget" is a property of rustworkx's search algorithm
itself, not an implementation quirk specific to the `VF2Layout` pass.

**Another possible outcome**: the scan might not find a single medium-difficulty
instance at all (success is instant, failure is hopeless, with nothing in
between). That is not undecidable -- it is itself worth recording as a finding:
this configuration may simply have no "found only after a struggle" middle
ground analogous to Qiskit's seed 8.

## Usage

    python verify_vf2_rustworkx_raw_sweep.py
    python verify_vf2_rustworkx_raw_sweep.py --grid 8x8 --scan-seeds 60 --scan-limit 200000
"""
from __future__ import annotations

import argparse
import time

from vf2_probe_common import banner, default_out, environment, write_csv


def grid_edges(rows, cols):
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append(((r, c), (r, c + 1)))
            if r + 1 < rows:
                e.append(((r, c), (r + 1, c)))
    return e


def order_random(rows, cols, seed):
    import random
    ns = [(r, c) for r in range(rows) for c in range(cols)]
    random.Random(seed).shuffle(ns)
    return ns


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


def find_medium_instance(rows, cols, n_pairs, scan_seeds, scan_limit):
    """Search for a random relabeling of medium difficulty (found, but not
    instantly). If no candidate qualifies, return the slowest one found."""
    candidates = []
    for s in range(scan_seeds):
        order = order_random(rows, cols, s)
        ok, el = probe(rows, cols, order, n_pairs, scan_limit)
        if ok:
            candidates.append((s, el))
        print(f"  scan seed={s:<4} found={str(ok):<5} {el*1000:8.2f} ms")
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="8x8")
    ap.add_argument("--scan-seeds", type=int, default=200)
    ap.add_argument("--scan-limit", type=int, default=1_000_000)
    ap.add_argument("--limits", type=int, nargs="+",
                    default=[43, 1_000, 10_000, 100_000, 300_000, 1_000_000, 3_000_000])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 7) Does the raw rustworkx call also burn the whole budget?", [
        "P1 a medium-difficulty instance plateaus in time past some threshold",
        "P2 a failing instance scales roughly proportionally with the budget (should hold "
        "in both implementations)",
    ])

    r, c = (int(x) for x in args.grid.lower().split("x"))
    n = r * c
    n_pairs = n // 2
    print(f"grid = {n} nodes, id_order=True. First scanning {args.scan_seeds} random "
          f"relabelings at call_limit={args.scan_limit:,} to pick one "
          f"medium-difficulty instance:\n")

    medium_seed = find_medium_instance(r, c, n_pairs, args.scan_seeds, args.scan_limit)
    fail_seed = args.scan_seeds + 1   # a number outside the scan, used as the "failing instance" label

    rows_out = []
    if medium_seed is None:
        print("\nNo medium-difficulty instance found (everything succeeded instantly, "
              "or nothing succeeded at all). P1 is undecidable. Measuring only the "
              "failing instance.")
    else:
        print(f"\nUsing random seed={medium_seed} as the medium-difficulty instance. "
              f"Sweeping call_limit:\n")
        print(f"{'call_limit':>12} {'found':>6} {'ms':>10}")
        order = order_random(r, c, medium_seed)
        for cl in args.limits:
            ok, el = probe(r, c, order, n_pairs, cl)
            print(f"{cl:>12,} {str(ok):>6} {el*1000:10.3f}")
            rows_out.append(dict(Kind="medium", Seed=medium_seed, CallLimit=cl,
                                 Found=ok, Time_s=el, Grid=args.grid, Nodes=n))

    # Control: an instance known to fail (identity order, id_order=False is the known failure case)
    print(f"\nAs a control, sweeping the same call_limit values against a known-failing "
          f"instance (row_major, id_order=False):\n")
    print(f"{'call_limit':>12} {'found':>6} {'ms':>10}")
    identity_order = [(rr, cc) for rr in range(r) for cc in range(c)]
    for cl in args.limits:
        ok, el = probe(r, c, identity_order, n_pairs, cl, id_order=False)
        print(f"{cl:>12,} {str(ok):>6} {el*1000:10.3f}")
        rows_out.append(dict(Kind="fail_control", Seed=fail_seed, CallLimit=cl,
                             Found=ok, Time_s=el, Grid=args.grid, Nodes=n))

    df = write_csv(rows_out, args.out or default_out("vf2_rustworkx_raw_sweep"), environment())

    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    if medium_seed is not None:
        med = df[df.Kind == "medium"].sort_values("CallLimit")
        print("Time progression for the medium instance (ms):")
        print((med.set_index("CallLimit").Time_s * 1000).round(3).to_string())
        print("  -> Plateauing partway through supports P1 (different behaviour from "
              "Qiskit's VF2Layout pass).")
        print("     Growing monotonically all the way to 3,000,000 means rustworkx "
              "itself burns the whole budget.")
    fail = df[df.Kind == "fail_control"].sort_values("CallLimit")
    print("\nTime progression for the failing instance (ms):")
    print((fail.set_index("CallLimit").Time_s * 1000).round(3).to_string())
    print("  -> Scaling roughly proportionally with the budget supports P2 (as expected).")


if __name__ == "__main__":
    main()
