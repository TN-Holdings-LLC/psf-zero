#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 1/4) Does the **size** of call_limit itself change the search path --
swept on a single seed

## Why this is needed

The 2026-09-14 measurement found that seed 8's minimum call_limit for a first
match is 43 (the same as seed 25), yet giving `call_limit=3,000,000,
max_trials=1` took 332 ms (seed 25 took 34.9 ms). Giving the 2-tuple
`call_limit=(3,000,000, 10,000)` instead dropped it to 1.7 ms.

This suggests the implicit assumption "call_limit only decides where to cut off,
it does not affect which path is taken" is broken. If so, **for the same seed and
graph, varying only the value of call_limit should not make time increase smoothly
in proportion to the call count -- it should jump discontinuously somewhere.**

This sweeps call_limit directly for a single seed (seed 8, the largest gap), with
seed 1 (a successful seed expected to be smooth) and seed 0 (a failing seed with
no solution, expected to slow down linearly with the budget) as comparisons.

## Pre-registered predictions

(P1) As call_limit is raised logarithmically from 43 to 3,000,000 for seed 8,
     time jumps discontinuously past some threshold (not a smooth proportional
     relationship).
(P2) Seed 1 behaves relatively smoothly between its minimum sufficient value
     (186) and 3,000,000 (staying on the order of a few ms regardless of which
     value is used).
(P3) The failing seed (seed 0) slows down roughly linearly with the budget
     (consistent with the 2026-09-11 finding of "30M -> 6.68s, 100M -> 21.96s").

**If this fails**: if seed 8 also scales smoothly, the "the path changes"
hypothesis is wrong, and the 332 ms figure is more likely due to something else
(measurement noise, system load). In that case the external-load hypothesis in
followup 2 (the seed 3/4 reproducibility check) should be weighted more heavily.

## Usage

    python verify_vf2_call_limit_sweep.py
    python verify_vf2_call_limit_sweep.py --seeds 8 1 0 --limits 43 100 300 1000 3000 10000 30000 100000 300000 1000000 3000000
"""
from __future__ import annotations

import argparse

from vf2_probe_common import (banner, build_dense_pair_blocks_circuit, default_out,
                              environment, get_grid_cmap, timed, write_csv)

N_QUBITS = 42

DEFAULT_LIMITS = [43, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000,
                  300_000, 1_000_000, 3_000_000]


def run_once(qc, cmap, seed, call_limit):
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import PropertySet
    from qiskit.transpiler.passes import VF2Layout
    p = VF2Layout(coupling_map=cmap, seed=seed, call_limit=call_limit, max_trials=1)
    ps = PropertySet()   # a plain dict raises KeyError on an unset key
    p.property_set = ps
    p.run(circuit_to_dag(qc))
    # "NO_SOLUTION_FOUND" contains "SOLUTION_FOUND" as a substring, so check with endswith
    return str(ps.get("VF2Layout_stop_reason")).endswith(".SOLUTION_FOUND")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[8, 1, 0])
    ap.add_argument("--limits", type=int, nargs="+", default=DEFAULT_LIMITS)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 1/4) call_limit sweep -- does the path change with the budget?", [
        "P1 seed 8 jumps discontinuously near some threshold (not smoothly proportional)",
        "P2 seed 1 stays relatively smooth, on the order of a few ms, from its minimum "
        "sufficient value up to 3,000,000",
        "P3 the failing seed (seed 0) slows down roughly linearly with the budget",
    ])

    cmap = get_grid_cmap(N_QUBITS)
    qc = build_dense_pair_blocks_circuit(N_QUBITS, seed=0)
    print(f"grid = {cmap.size()} physical, circuit = {N_QUBITS} qubits\n")

    rows = []
    print(f"{'seed':>5} {'call_limit':>12} {'found':>6} {'min ms':>10} {'med ms':>10}")
    for seed in args.seeds:
        for cl in args.limits:
            mn, md, ok = timed(lambda cl=cl, s=seed: run_once(qc, cmap, s, cl),
                                args.reps)
            print(f"{seed:>5} {cl:>12,} {str(ok):>6} {mn*1000:10.2f} {md*1000:10.2f}")
            rows.append(dict(Seed=seed, CallLimit=cl, Found=ok, Qubits=N_QUBITS,
                             Physical=cmap.size(), Time_min_s=mn, Time_median_s=md))

    df = write_csv(rows, args.out or default_out("vf2_call_limit_sweep"), environment())

    print("\n" + "=" * 78)
    print("Verdict -- min time (ms) by seed, across call_limit")
    print("=" * 78)
    piv = df.pivot_table(index="CallLimit", columns="Seed", values="Time_min_s") * 1000
    print(piv.round(2).to_string())
    print("\n  -> If time increases smoothly, roughly in proportion to call_limit within a "
          "column, the path did not change -- only the cutoff point differs. A "
          "discontinuous jump partway through supports P1: the size of the budget "
          "changes the path itself.")


if __name__ == "__main__":
    main()
