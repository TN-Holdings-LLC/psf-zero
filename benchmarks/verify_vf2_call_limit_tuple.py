#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(1/6) Does the second element of the `call_limit` 2-tuple stop the post-match
improvement search?

## Why this is needed

`verify_vf2_max_trials.py` passed `call_limit` as a **scalar**. So after the first
match, the improvement search consumed the entire remaining budget: seed 1 found a
layout in 3.5 ms but the whole pass still took 343 ms. From that, this project
concluded "ordering and the trial loop are two independent costs".

But the source shows the preset passes a **2-tuple**
(`DefaultLayoutPassManager`):

    level 1: call_limit=(50_000,     1_000)
    level 2: call_limit=(5_000_000,  10_000)
    level 3: call_limit=(30_000_000, 100_000)

The docstring explains the second element:

  "the limit starts as the first item, and swaps to the second after the first
   match is found ... terminate quickly with a small extension budget if one is found"

So **the trial loop may already be bounded in the preset**. If so, "two independent
costs" is a statement about the scalar configuration only.

## Pre-registered predictions

(P1) `(3_000_000, 10_000)` behaves nearly identically to `max_trials=1`.
     seed 1 ~= 3.5 ms, seeds 25/29 ~= 34 ms, seed 8 and the 26 failures ~= 330 ms
     (unchanged).
(P2) The stop reason is identical across all three arms.
(P3) Increasing the second element `(3_000_000, 3_000_000)` reverts to the scalar
     behaviour (~340 ms). If this holds, it is direct confirmation that the
     second element is doing something.

**If this fails**: the second element does not behave as documented. In that case
`spare-qubit-cliff.md`'s "two independent costs" stands unmodified, for the scalar
setting and for the preset alike, and this section needs no correction.

## Usage

    python verify_vf2_call_limit_tuple.py
    python verify_vf2_call_limit_tuple.py --seeds 1 8 25 29 0 2 --reps 3
"""
from __future__ import annotations

import argparse

from vf2_probe_common import (banner, build_dense_pair_blocks_circuit, default_out,
                              environment, get_grid_cmap, timed, write_csv)

N_QUBITS = 42          # saturates the 6x7 grid
BASE_LIMIT = 3_000_000  # same as the existing measurement: 1/10 of level 3


def run_once(qc, cmap, seed, call_limit, max_trials):
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import PropertySet
    from qiskit.transpiler.passes import VF2Layout
    dag = circuit_to_dag(qc)
    p = VF2Layout(coupling_map=cmap, seed=seed,
                  call_limit=call_limit, max_trials=max_trials)
    ps = PropertySet()   # a plain dict raises KeyError on an unset key
    p.property_set = ps
    p.run(dag)
    return str(ps.get("VF2Layout_stop_reason"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[1, 8, 25, 29, 0, 2, 3, 4])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(1/6) Does the call_limit 2-tuple stop the trial loop?", [
        "P1 (3_000_000, 10_000) matches max_trials=1: seed1 ~3.5ms, 25/29 ~34ms, 8 and failures ~330ms",
        "P2 stop reason identical across all three arms",
        "P3 (3_000_000, 3_000_000) reverts to the scalar behaviour (~340ms)",
    ])

    arms = [
        ("scalar",       BASE_LIMIT,                    None),
        ("max_trials1",  BASE_LIMIT,                    1),
        ("tuple_10k",    (BASE_LIMIT, 10_000),          None),
        ("tuple_full",   (BASE_LIMIT, BASE_LIMIT),      None),
    ]

    cmap = get_grid_cmap(N_QUBITS)
    qc = build_dense_pair_blocks_circuit(N_QUBITS, seed=0)
    print(f"grid = {cmap.size()} physical, circuit = {N_QUBITS} qubits, "
          f"spare = {cmap.size() - N_QUBITS}\n")

    rows = []
    print(f"{'seed':>5} {'arm':<12} {'min ms':>9} {'med ms':>9}  stop_reason")
    for seed in args.seeds:
        for arm, cl, mt in arms:
            mn, md, reason = timed(
                lambda cl=cl, mt=mt, s=seed: run_once(qc, cmap, s, cl, mt),
                args.reps)
            print(f"{seed:>5} {arm:<12} {mn*1000:9.1f} {md*1000:9.1f}  {reason}")
            rows.append(dict(Seed=seed, Arm=arm, CallLimit=str(cl), MaxTrials=mt,
                             Qubits=N_QUBITS, Physical=cmap.size(),
                             Time_min_s=mn, Time_median_s=md, StopReason=reason))

    df = write_csv(rows, args.out or default_out("vf2_call_limit_tuple"), environment())

    print("\n" + "=" * 78)
    print("Verdict -- min time by arm (ms)")
    print("=" * 78)
    piv = df.pivot_table(index="Seed", columns="Arm", values="Time_min_s") * 1000
    print(piv.round(1).to_string())
    if {"max_trials1", "tuple_10k"} <= set(piv.columns):
        r = (piv["tuple_10k"] / piv["max_trials1"]).round(2)
        print("\ntuple_10k / max_trials1 (near 1.0 supports P1):")
        print(r.to_string())
        print("\n  -> Near 1.0 on the successful seeds supports P1. Near the scalar value rejects P1.")
    if {"scalar", "tuple_full"} <= set(piv.columns):
        print("\ntuple_full / scalar (near 1.0 supports P3):")
        print((piv["tuple_full"] / piv["scalar"]).round(2).to_string())


if __name__ == "__main__":
    main()
