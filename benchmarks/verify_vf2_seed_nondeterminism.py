#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 11) How much does VF2Layout's seed=-1 actually shake the result?

## Why this is needed

An important correction that came out of reading the source in addendum-9:
`transpile(..., seed_transpiler=0)` fixes `SabreLayout`'s seed, but
**`VF2Layout` is hardcoded to `seed=-1` and is not controlled by
`seed_transpiler`.** That means every topology experiment so far
(addendum-5 through 10) implicitly assumed "same conditions give the same
result," but `VF2Layout`'s internal shuffling can in principle vary from call
to call.

Addendum-10 recorded, as a "reassuring point," that `diluted_p0.75` reached
the same conclusion (VF2Layout succeeds) in all 4 runs (sandbox L2/L3, real
hardware L2/L3) -- but that is only one measurement per topology. Here,
`transpile()` is called repeatedly against the **same topology and same
circuit**, and how often `VF2Layout_stop_reason` actually flips is counted
directly.

## Pre-registered predictions

(P1) `grid` / `brick` / `diluted_p0.25` / `diluted_p0.5` / `line` (the
     candidates that were tight/cliff-showing in addendum-8/10) fail almost
     every time on repetition (a success rate near 0%) -- the difficulty is so
     extreme that `seed=-1`'s jitter does not change the outcome.
(P2) `diluted_p0.75` succeeds almost every time on repetition (a success rate
     near 100%) -- same reasoning as above.
(P3) If either P1 or P2 fails, and some candidate's success rate lands
     "mixed" (neither 0% nor 100%), that is a warning that the other topology
     experiments in addendum-5 through 10 -- which judged cliff-present or
     cliff-absent from a single measurement each (heavy-hex, torus, ring,
     full, etc.) -- may carry similar jitter too.

**If this fails**: if neither P1 nor P2 fails (i.e. everything splits cleanly
into 0% or 100%), then `seed=-1`'s non-determinism is not a practical problem
except at the knife-edge of difficulty.

## Usage

    python verify_vf2_seed_nondeterminism.py
    python verify_vf2_seed_nondeterminism.py --reps 30 --levels 2 3
"""
from __future__ import annotations

import argparse

from vf2_probe_common import (BASIS_GATES, banner, build_dense_pair_blocks_circuit,
                              default_out, environment, write_csv)
from verify_vf2_sparse_topology import topologies as sparse_topologies


def vf2_stop_reason_once(qc, cmap, level):
    from qiskit import transpile
    holder = {}

    def cb(**kwargs):
        if type(kwargs["pass_"]).__name__ == "VF2Layout":
            holder["stop"] = str(kwargs["property_set"].get("VF2Layout_stop_reason"))

    transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
              optimization_level=level, seed_transpiler=0, callback=cb)
    return holder.get("stop", "(VF2Layout did not run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="8x8")
    ap.add_argument("--levels", type=int, nargs="+", default=[2],
                    help="defaults to L2 only (an L3 failing case takes roughly 20 "
                         "seconds per call, so pass --levels 2 3 explicitly to include it)")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--reps-l3", type=int, default=5,
                    help="repetitions when L3 is included (kept lower than L2 to bound time)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 11) Directly counting the jitter from VF2Layout's seed=-1", [
        "P1 grid/brick/diluted_p0.25/diluted_p0.5/line fail almost every time (success rate ~=0%)",
        "P2 diluted_p0.75 succeeds almost every time (success rate ~=100%)",
        "P3 any candidate landing at an intermediate success rate calls the robustness of "
        "the other topology experiments into question too",
    ])

    rows, cols = (int(x) for x in args.grid.lower().split("x"))
    rows_out = []
    print(f"{'topology':<14} {'level':>5} {'reps':>5} {'succ':>5} {'fail':>5} {'rate':>7}")

    for name, cmap, deg, _ in sparse_topologies(rows, cols):
        phys = cmap.size()
        n = phys  # spare=0 (tight) -- the same condition used to judge cliff
                  # presence/absence in addendum-8/10
        qc = build_dense_pair_blocks_circuit(n, seed=0)
        for lv in args.levels:
            reps = args.reps if lv != 3 else args.reps_l3
            outcomes = [vf2_stop_reason_once(qc, cmap, lv) for _ in range(reps)]
            ok = sum(1 for o in outcomes if o.endswith(".SOLUTION_FOUND"))
            fail = reps - ok
            rate = ok / reps
            print(f"{name:<14} {lv:>5} {reps:>5} {ok:>5} {fail:>5} {rate:>7.2%}", flush=True)
            for i, o in enumerate(outcomes):
                rows_out.append(dict(Topology=name, AvgDegree=deg, Physical=phys,
                                     Level=lv, Rep=i, StopReason=o,
                                     Success=o.endswith(".SOLUTION_FOUND")))

    df = write_csv(rows_out, args.out or default_out("vf2_seed_nondeterminism"), environment())

    print("\n" + "=" * 78)
    print("Verdict -- are there any candidates whose success rate is pinned at neither 0% nor 100%?")
    print("=" * 78)
    summ = df.groupby(["Topology", "Level"]).Success.agg(["mean", "count"])
    print(summ.to_string())
    mixed = summ[(summ["mean"] > 0.05) & (summ["mean"] < 0.95)]
    if len(mixed):
        print("\n  -> The following have an intermediate, fluctuating success rate "
              "(this is what P3 concerns -- a single measurement cannot be trusted here):")
        print(mixed.to_string())
    else:
        print("\n  -> Every candidate is pinned cleanly at 0% or 100%. Supports P1/P2 -- ")
        print("     at least at this scale of extreme difficulty gap, seed=-1's jitter")
        print("     does not sway the outcome.")


if __name__ == "__main__":
    main()
