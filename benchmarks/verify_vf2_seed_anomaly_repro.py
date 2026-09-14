#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 2/4) Does the anomalous call_limit-tuple behaviour for seed 3 / seed 4
reproduce?

## Why this is needed

In the 2026-09-14 `verify_vf2_call_limit_tuple.py` run (reps=3), two failing
seeds showed an unexplained, asymmetric anomaly.

  - seed 3: the `scalar` arm is bimodal within one configuration (min 0.358s,
    median 1.013s -- some of the 3 repetitions run nearly 3x slower than
    others). The tuple arms are, conversely, stably fast (~0.33s).
  - seed 4: `scalar`/`max_trials1` are normal (~0.34s), yet **both**
    `tuple_10k` and `tuple_full` are stably slow (0.86-1.16s, nearly 3x).

With reps=3 there is no way to distinguish "coincidence" from "systematic".
Here the same four arms are run for seed 3 and seed 4 only, with more
repetitions (default 20), recording each repetition's raw time and wall-clock
timestamp, to see (a) whether it reproduces, and (b) whether it correlates with
external load (other processes on the system). If `psutil` is available, CPU
usage before each repetition is also recorded (silently skipped if not --
not made a hard dependency).

## Pre-registered predictions

(P1) Seed 3's `scalar`-arm bimodality persists with more repetitions (some
     repetitions cluster clearly near 1 second, the rest near 0.35 seconds; a
     continuous spread across the middle would mean it is not bimodal, just
     high variance).
(P2) Seed 4's `tuple_10k`/`tuple_full` slowness is consistently slow with more
     repetitions (the 2026-09-14 3-repetition result was not a coincidence, and
     stays at the same level over 20 repetitions).
(P3) Where `psutil` is available, slower repetitions correlate with higher CPU
     usage measured just before them (direct support for an external-load
     cause). If `psutil` is unavailable, this prediction is explicitly marked
     undecidable and skipped.

**If this fails**: if (P1) and (P2) fail and the spread across repetitions
settles into something like a normal distribution, the 2026-09-14 anomaly was a
coincidence of reps=3 (one outlier moved the median), and the "unresolved
anomaly" note in addendum-5 can be withdrawn.

## Usage

    python verify_vf2_seed_anomaly_repro.py
    python verify_vf2_seed_anomaly_repro.py --seeds 3 4 --reps 20
"""
from __future__ import annotations

import argparse
import time

from vf2_probe_common import (banner, build_dense_pair_blocks_circuit, default_out,
                              environment, get_grid_cmap, write_csv)

N_QUBITS = 42
BASE_LIMIT = 3_000_000

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


def run_once(qc, cmap, seed, call_limit, max_trials):
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import PropertySet
    from qiskit.transpiler.passes import VF2Layout
    dag = circuit_to_dag(qc)
    p = VF2Layout(coupling_map=cmap, seed=seed, call_limit=call_limit,
                  max_trials=max_trials)
    ps = PropertySet()
    p.property_set = ps
    p.run(dag)
    return str(ps.get("VF2Layout_stop_reason"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 2/4) seed 3 / seed 4 call_limit-tuple anomaly -- does it reproduce?", [
        "P1 seed 3's scalar-arm bimodality persists with more repetitions",
        "P2 seed 4's tuple_10k/tuple_full slowness is consistently slow with more repetitions",
        "P3 (if psutil is available) slower repetitions have higher CPU usage just before them",
    ])
    if not _HAVE_PSUTIL:
        print("Note: psutil not found, so P3 is undecidable (the CPU column will be empty).\n")

    arms = [
        ("scalar",      BASE_LIMIT,               None),
        ("max_trials1", BASE_LIMIT,               1),
        ("tuple_10k",   (BASE_LIMIT, 10_000),     None),
        ("tuple_full",  (BASE_LIMIT, BASE_LIMIT), None),
    ]

    cmap = get_grid_cmap(N_QUBITS)
    qc = build_dense_pair_blocks_circuit(N_QUBITS, seed=0)
    print(f"grid = {cmap.size()} physical, circuit = {N_QUBITS} qubits, "
          f"reps = {args.reps} per arm\n")

    rows = []
    for seed in args.seeds:
        for arm, cl, mt in arms:
            # One warm-up call, outside the timer and the record.
            run_once(qc, cmap, seed, cl, mt)
            times = []
            for rep in range(args.reps):
                cpu_before = psutil.cpu_percent(interval=0.05) if _HAVE_PSUTIL else None
                t0 = time.perf_counter()
                reason = run_once(qc, cmap, seed, cl, mt)
                el = time.perf_counter() - t0
                times.append(el)
                rows.append(dict(Seed=seed, Arm=arm, Rep=rep, Time_s=el,
                                 StopReason=reason, CPU_before_pct=cpu_before,
                                 Wallclock=time.time()))
            import numpy as np
            arr = np.array(times)
            print(f"seed={seed} arm={arm:<12} min={arr.min()*1000:8.1f}ms "
                  f"median={np.median(arr)*1000:8.1f}ms max={arr.max()*1000:8.1f}ms "
                  f"std={arr.std()*1000:7.1f}ms")

    df = write_csv(rows, args.out or default_out("vf2_seed_anomaly_repro"), environment())

    print("\n" + "=" * 78)
    print("Verdict -- shape of the per-arm distribution (bimodality check)")
    print("=" * 78)
    for seed in args.seeds:
        for arm, _, _ in arms:
            sub = df[(df.Seed == seed) & (df.Arm == arm)].Time_s.values
            lo, hi = sub.min(), sub.max()
            mid = (lo + hi) / 2
            below = (sub < mid).sum()
            above = (sub >= mid).sum()
            spread = hi / lo if lo > 0 else float("inf")
            tag = "possibly bimodal" if spread > 1.5 and min(below, above) >= 2 else "unimodal"
            print(f"  seed={seed} arm={arm:<12} spread={spread:5.2f}x "
                  f"(<mid: {below}, >=mid: {above})  -> {tag}")

    if _HAVE_PSUTIL:
        print("\nCorrelation between CPU usage and time (Pearson r, per arm):")
        for seed in args.seeds:
            for arm, _, _ in arms:
                sub = df[(df.Seed == seed) & (df.Arm == arm)]
                if sub.CPU_before_pct.notna().sum() >= 3:
                    r = sub[["CPU_before_pct", "Time_s"]].corr().iloc[0, 1]
                    print(f"  seed={seed} arm={arm:<12} r={r:+.2f}")
    else:
        print("\nP3: undecidable without psutil.")


if __name__ == "__main__":
    main()
