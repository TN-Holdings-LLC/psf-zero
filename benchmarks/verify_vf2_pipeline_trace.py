#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 10) Trace the whole pipeline -- does `transpile()`'s preset pass
manager actually reach `VF2Layout`/`VF2PostLayout` the way this project assumes,
end to end?

## Why this is needed

Every experiment in this batch so far has called `VF2Layout` or
`rustworkx.vf2_mapping` **directly**. That measures the pass itself accurately,
but it has never been directly confirmed that `transpile(optimization_level=...)`
**actually runs these same passes, with these same arguments, inside its real
pipeline** for the circuit/coupling-map combination used throughout this
project. It has been read from source and assumed based on that reading, but
not traced at runtime.

Also, `verify_qiskit_source_2_5_2.py` (part 6 of the existing batch) already
mechanically checks the *source*, but it does not check what actually **runs**
in a real `transpile()` call. This closes that final gap using
`transpile()`'s own `callback=` argument to record, for a real, complete call
with the saturated grid used throughout, (1) the full list of passes that ran
and their order, (2) which pass VF2Layout/VF2PostLayout land at within that
order, (3) each pass's individual execution time, and (4) the property-set
values immediately after VF2Layout/VF2PostLayout finish (stop reason, whether a
layout was produced).

## Pre-registered predictions

(P1) VF2Layout and VF2PostLayout both appear in the callback's pass list (in the
     real pipeline too, not skipped or replaced by another pass).
(P2) VF2Layout's post-run property set shows `NO_SOLUTION_FOUND`
     (reproducing, within the full pipeline, this project's core existing
     finding that the saturated case fails).
(P3) Of the total call time, the two VF2 passes account for the largest share
     (consistent with the existing finding of "99.9% of the time in two
     passes").
(P4) The set and order of passes that ran matches what would be predicted from
     reading `generate_preset_pass_manager`'s source (no undocumented extra
     pass appears).

**If this fails**: if (P1) or (P2) fails, this project's `VF2Layout` calls
(made directly, bypassing the pipeline) might not accurately represent what a
real `transpile()` call does, and every experiment so far would need to be
re-examined for how it corresponds to the real pipeline.

## Usage

    python verify_vf2_pipeline_trace.py
    python verify_vf2_pipeline_trace.py --qubits 42 --level 3
"""
from __future__ import annotations

import argparse
import time

from vf2_probe_common import (BASIS_GATES, banner, build_dense_pair_blocks_circuit,
                              default_out, environment, get_grid_cmap, write_csv)


def trace_transpile(qc, cmap, level):
    """Times transpile() via the callback, and also records the property set
    right after VF2Layout/VF2PostLayout finish."""
    from qiskit import transpile

    trace = []
    captured = {}

    def cb(**kwargs):
        pass_ = kwargs["pass_"]
        name = type(pass_).__name__
        el = kwargs["time"]
        ps = kwargs["property_set"]
        trace.append((name, el))
        if name in ("VF2Layout", "VF2PostLayout"):
            captured[name] = {
                "stop_reason": str(ps.get(f"{name}_stop_reason")),
                "has_layout": ps.get("layout") is not None,
            }

    t0 = time.perf_counter()
    out = transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
                    optimization_level=level, seed_transpiler=0, callback=cb)
    total = time.perf_counter() - t0
    return out, total, trace, captured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=42)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(followup 10) Trace the whole transpile() pipeline", [
        "P1 both VF2Layout and VF2PostLayout appear in the callback's pass list",
        "P2 VF2Layout's post-run property set shows NO_SOLUTION_FOUND (the saturated case)",
        "P3 the two VF2 passes account for the largest share of the total call time",
        "P4 the set and order of passes that ran matches the source-reading prediction "
        "(no undocumented extra pass appears)",
    ])

    cmap = get_grid_cmap(args.qubits)
    qc = build_dense_pair_blocks_circuit(args.qubits, seed=0)
    print(f"grid = {cmap.size()} physical, circuit = {args.qubits} qubits "
          f"(spare = {cmap.size() - args.qubits}), optimization_level={args.level}\n")

    out, total, trace, captured = trace_transpile(qc, cmap, args.level)

    print(f"{'#':>4} {'pass':<32} {'ms':>10}")
    for i, (name, el) in enumerate(trace):
        marker = "  <-- VF2" if name in ("VF2Layout", "VF2PostLayout") else ""
        print(f"{i:>4} {name:<32} {el*1000:10.3f}{marker}")

    print(f"\ntotal call time: {total*1000:.1f} ms")
    print(f"sum of per-pass times from the callback: {sum(e for _, e in trace)*1000:.1f} ms")

    print("\nProperty set immediately after VF2Layout/VF2PostLayout:")
    for name in ("VF2Layout", "VF2PostLayout"):
        if name in captured:
            print(f"  {name}: {captured[name]}")
        else:
            print(f"  {name}: did not run (not found in the callback)")

    rows = []
    for i, (name, el) in enumerate(trace):
        rows.append(dict(Order=i, Pass=name, Time_s=el,
                         Qubits=args.qubits, Physical=cmap.size(),
                         OptimizationLevel=args.level,
                         VF2Layout_stop_reason=captured.get("VF2Layout", {}).get("stop_reason"),
                         VF2PostLayout_stop_reason=captured.get("VF2PostLayout", {}).get("stop_reason"),
                         TotalCallTime_s=total))
    df = write_csv(rows, args.out or default_out("vf2_pipeline_trace"), environment())

    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    vf2_names = ("VF2Layout", "VF2PostLayout")
    p1 = all(n in captured for n in vf2_names)
    print(f"P1 (both VF2 passes appear): {'HELD' if p1 else 'FAILED'}")

    if "VF2Layout" in captured:
        p2 = captured["VF2Layout"]["stop_reason"].endswith("NO_SOLUTION_FOUND")
        print(f"P2 (VF2Layout is NO_SOLUTION_FOUND): "
              f"{'HELD' if p2 else 'FAILED'} ({captured['VF2Layout']['stop_reason']})")

    vf2_time = sum(e for n, e in trace if n in vf2_names)
    other_time = sum(e for n, e in trace if n not in vf2_names)
    total_named = vf2_time + other_time
    if total_named > 0:
        print(f"P3 (the two VF2 passes are the largest share): "
              f"VF2 total={vf2_time*1000:.1f}ms ({100*vf2_time/total_named:.1f}%), "
              f"other passes total={other_time*1000:.1f}ms")

    print(f"\nList of passes that ran ({len(trace)} total):")
    print("  " + " -> ".join(n for n, _ in trace))
    print("\n  -> Compare this list and order against the source reading in the addendum. "
          "An unexpected extra pass rejects P4.")


if __name__ == "__main__":
    main()
