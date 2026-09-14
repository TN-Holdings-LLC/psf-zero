#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(2/6) Why is seed 8 alone slow to reach its first match -- measured in
**steps**, not time.

## Why this is needed

The existing measurement reports "seed 1 matches in 3.5 ms, seed 8 in 336 ms" in
wall-clock time. Time is machine-dependent, so someone else reproducing this will
not get the same numbers. And it answers nothing about *why* the two differ by
~100x.

`call_limit` is an upper bound on **how many times the search tries to extend the
mapping**, which is machine-independent. `max_trials=1` stops at the first match.
Combining the two:

  "the smallest call_limit L for which max_trials=1 still succeeds for this seed"
  = the number of steps that seed needed to reach its first match (to the precision
    of a binary search)

This is a **machine-independent quantity**, and it can translate the 100x time gap
into a step-count ratio.

## Pre-registered predictions

(P1) Step counts among the four successful seeds vary widely.
     Seed 1 is smallest, seed 8 largest, and the ratio is the same order of
     magnitude as the time ratio (~96x).
(P2) The 26 failing seeds never succeed no matter how high the limit is raised
     (at least up to 3,000,000). Consistent with the existing finding that
     raising call_limit only makes the failure linearly slower.
(P3) Seed 8's threshold sits comfortably below the budget the failing seeds burn
     through (3,000,000) -- i.e. seed 8 is "just barely making it", not a
     different kind of case.

**If this fails**: if (P1) fails and all successful seeds have similar step
counts, the time gap is not about search depth but about a per-step cost
difference, and the cause lies elsewhere.

## Usage

    python verify_vf2_steps_to_first_match.py
    python verify_vf2_steps_to_first_match.py --seeds 1 8 25 29 --max-limit 3000000
"""
from __future__ import annotations

import argparse

from vf2_probe_common import (banner, build_dense_pair_blocks_circuit, default_out,
                              environment, get_grid_cmap, write_csv)

N_QUBITS = 42


def found(qc, cmap, seed, call_limit):
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import PropertySet
    from qiskit.transpiler.passes import VF2Layout
    p = VF2Layout(coupling_map=cmap, seed=seed, call_limit=call_limit, max_trials=1)
    ps = PropertySet()   # a plain dict raises KeyError on an unset key
    p.property_set = ps
    p.run(circuit_to_dag(qc))
    # Note: "NO_SOLUTION_FOUND" contains "SOLUTION_FOUND" as a substring.
    # Checking with `in` would read a failure as a success. Use endswith.
    return str(ps.get("VF2Layout_stop_reason")).endswith(".SOLUTION_FOUND")


def threshold(qc, cmap, seed, lo, hi):
    """Binary search for the smallest call_limit that succeeds. None if not found."""
    if not found(qc, cmap, seed, hi):
        return None, 0
    calls = 1
    while lo < hi:
        mid = (lo + hi) // 2
        calls += 1
        if found(qc, cmap, seed, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[1, 8, 25, 29, 0, 2, 3, 4])
    ap.add_argument("--max-limit", type=int, default=3_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(2/6) Steps to the first match (machine-independent)", [
        "P1 successful seeds' step counts differ widely; seed1 smallest, seed8 largest, "
        "ratio same order as the time ratio",
        "P2 the 26 failing seeds never succeed even with a higher limit",
        "P3 seed8's threshold is comfortably below 3,000,000 (barely making it, not a "
        "different case)",
    ])

    cmap = get_grid_cmap(N_QUBITS)
    qc = build_dense_pair_blocks_circuit(N_QUBITS, seed=0)
    print(f"grid = {cmap.size()} physical, circuit = {N_QUBITS} qubits, "
          f"binary search over call_limit in [1, {args.max_limit}]\n")

    rows = []
    print(f"{'seed':>5} {'steps to first match':>22} {'probes':>7}")
    for seed in args.seeds:
        th, probes = threshold(qc, cmap, seed, 1, args.max_limit)
        label = f"{th:,}" if th is not None else f"none (> {args.max_limit:,})"
        print(f"{seed:>5} {label:>22} {probes:>7}")
        rows.append(dict(Seed=seed, StepsToFirstMatch=th, Probes=probes,
                         MaxLimit=args.max_limit, Qubits=N_QUBITS,
                         Physical=cmap.size(),
                         Outcome="found" if th is not None else "not_found"))

    df = write_csv(rows, args.out or default_out("vf2_steps_to_first_match"),
                   environment())

    ok = df[df.Outcome == "found"]
    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    if len(ok) >= 2:
        lo, hi = ok.StepsToFirstMatch.min(), ok.StepsToFirstMatch.max()
        print(f"Step counts among successful seeds: min {lo:,} / max {hi:,} / ratio {hi/lo:.1f}x")
        print("  -> The existing time ratio is seed1 3.5ms vs seed8 336ms = ~96x.")
        print("     A step-count ratio of the same order supports P1 (a search-depth")
        print("     difference). A small step-count ratio means the difference is in")
        print("     the per-step cost instead.")
    print(f"Seeds never found: {len(df) - len(ok)} / {len(df)}"
          f"  -> nonzero supports P2")


if __name__ == "__main__":
    main()
