#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(followup 14) Prototype vs. default pipeline -- an end-to-end race for total time, PSF-Zero included

## Why this is needed

The layout-search prototype built in addendum-13, `psf_smart_layout.py`, was
confirmed in the sandbox to reach the point of "can find a valid layout even
under conditions where Qiskit's default pipeline fails (an 8x8 grid or line
at spare=0)." But **whether it is actually faster has never been measured
once.**

And as found in addendum-13, the search also fails on brick and
diluted_p0.25/0.5. When it fails, the time spent searching (2 seconds at
default) becomes **an added cost in full**, and the default pipeline ends up
running anyway afterward. In other words, **this is a contest the prototype
can lose**, and both winning and losing need to be counted on the same
footing.

"End to end" means measuring the total time from layout search through to
final circuit generation, in the actual usage form that includes PSF-Zero
(`compile_for_hardware`).

## What is measured (the arms)

| Arm | Contents |
|---|---|
| `qiskit_optN` | plain `transpile(optimization_level=N)` (the cliff's baseline) |
| `qiskit_optN_smart` | `smart_vf2_layout()` -> if found, `transpile()` with `initial_layout` specified; if not found, plain `transpile()`. **Search time is always included in the total** |
| `psf_rlN` | `compile_for_hardware(routing_optimization_level=N)` (PSF-Zero's current flow) |
| `psf_rlN_smart` | the same, but passing `smart_vf2_layout()`'s result as `initial_layout`. **Search time is always included in the total** |

The `qiskit_*` pair is included so that, if a difference appears, it can be
separated into "thanks to the layout search" versus "something specific to
PSF-Zero's side."

## Pre-registered predictions (**written before measuring**. Not moved afterward.)

(P1) On tight (spare=0) `grid`, `line`, and `diluted_p0.75`, the smart arm is
     **faster** (wins on total time). Addendum-13 confirmed the prototype
     can find a solution in a few milliseconds to 0.13 seconds on these
     three, while the default pipeline spends 0.6-4.3 seconds on VF2's
     failure on real hardware at L2 (except diluted_p0.75, where the
     default also succeeds, so the gap should be small).
(P2) On tight `brick`, `diluted_p0.25`, and `diluted_p0.5`, the smart arm
     **loses**. The margin of loss should be roughly equal to the search's
     time budget (2.0 seconds by default) -- because the search fails
     entirely before the default pipeline is run anyway.
(P3) At loose (spare=40, n=24), the difference is small across every
     topology. Since VF2Layout succeeds instantly even by default, the
     smart arm is only slightly slower due to the extra pre-check and first
     search attempt.
(P4) Output circuit quality (2-qubit gate count, depth) is **equal to or
     better than** the default arm in cases where the smart arm wins. There
     is no reason for a complete layout found by VF2 to be worse than
     Sabre's compromise layout.

**If this fails**:
- If P2 fails and smart wins even on `brick`, that means the failure-case
  cost is smaller than expected, and it would be worth raising
  `--time-budget` further.
- If P4 fails and quality drops (2-qubit gate count or depth increases),
  that would mean "faster but lower quality," and this whole direction
  would need reconsidering. In particular, if the effect of passing
  `initial_layout` skipping `VF2PostLayout` entirely (see "Caveats" below)
  shows up anywhere, it would show up here.

## Caveats (must be considered when reading the results)

1. **Passing `initial_layout` skips the entire layout stage.** Confirmed in
   the sandbox using a callback:
   - default: `SetLayout, VF2Layout, SabreLayout, VF2PostLayout`
   - with `initial_layout` specified: only `SetLayout, ApplyLayout`
   So `VF2PostLayout` is also skipped. This project's experiments only pass
   `coupling_map` and carry no error rates, so this should have no effect
   here, but **when using a real hardware `Target` (with error rates),
   skipping `VF2PostLayout` could lower fidelity**. This is outside this
   measurement's scope.
2. Including the search time in the total is deliberate, so the cost of a
   failure is not hidden.
3. `reps` defaults to 1 (to stay within a 5-minute budget). See addendum-10
   for the existing caveat about single-measurement variance. `--reps 2`
   should still fit within budget.

## Usage

    python benchmark_smart_layout_vs_default.py
    python benchmark_smart_layout_vs_default.py --reps 2
    python benchmark_smart_layout_vs_default.py --psf-rl 1      # if rl2 fails
    python benchmark_smart_layout_vs_default.py --arms qiskit_opt2,qiskit_opt2_smart

In an environment where `psf_compile` cannot be imported (e.g. the
sandbox), the PSF arms are automatically skipped and only the Qiskit arms
run.
"""
from __future__ import annotations

import argparse
import inspect
import time

import numpy as np

from vf2_probe_common import (BASIS_GATES, banner, build_dense_pair_blocks_circuit,
                              default_out, environment, has_perfect_matching, write_csv)
from verify_vf2_sparse_topology import topologies as sparse_topologies
from psf_smart_layout import smart_vf2_layout


# ---------------------------------------------------------------- fixture
def count_coupling_violations(qc, cmap):
    """Copied verbatim from phase3_v5_spare_qubits.py."""
    edges = set()
    for a, b in cmap.get_edges():
        edges.add((a, b))
        edges.add((b, a))
    total_2q = 0
    violations = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            total_2q += 1
            qi = qc.find_bit(inst.qubits[0]).index
            qj = qc.find_bit(inst.qubits[1]).index
            if (qi, qj) not in edges:
                violations += 1
    return total_2q, violations


def layout_map_to_list(layout_map, num_logical, num_physical):
    """Converts {logical: physical} into the form
    `transpile(initial_layout=...)` expects -- a list of physical indices
    ordered by virtual-qubit index. If a logical qubit has no assignment
    (e.g. the remainder from an odd qubit count), unused physical qubits
    are filled in, in order."""
    used = set(layout_map.values())
    spare_iter = (p for p in range(num_physical) if p not in used)
    out = []
    for q in range(num_logical):
        if q in layout_map:
            out.append(layout_map[q])
        else:
            out.append(next(spare_iter))
    return out


# ---------------------------------------------------------------- PSF support status
def psf_available():
    try:
        from psf_compile import compile_for_hardware  # noqa: F401
        return True
    except Exception:
        return False


def psf_supports_initial_layout():
    """Determines from its signature whether `compile_for_hardware` can
    accept `initial_layout`. Even if it cannot, it may still pass through
    via **kwargs, so that possibility is also reported."""
    try:
        from psf_compile import compile_for_hardware
    except Exception:
        return None, "cannot import psf_compile"
    try:
        sig = inspect.signature(compile_for_hardware)
    except Exception as e:  # noqa: BLE001
        return None, f"could not get the signature: {e}"
    params = sig.parameters
    if "initial_layout" in params:
        return True, f"explicitly supported: {sig}"
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return "maybe", f"has **kwargs (need to run it to see if it passes through): {sig}"
    return False, f"does not accept initial_layout: {sig}"


# ---------------------------------------------------------------- arms
def make_runner(arm, cmap, level, psf_rl, psf_init_layout_mode):
    """Returns a run function. The run function takes an initial_layout
    (a list, or None) and returns a circuit."""
    from qiskit import transpile

    if arm.startswith("qiskit_"):
        def run(qc, initial_layout):
            kw = {} if initial_layout is None else {"initial_layout": initial_layout}
            return transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
                             optimization_level=level, seed_transpiler=0, **kw)
        return run

    from psf_compile import compile_for_hardware

    def run(qc, initial_layout):
        kw = dict(coupling_map=cmap, basis_gates=BASIS_GATES,
                  routing_optimization_level=psf_rl, verify=False,
                  entangling_basis="cx")
        if initial_layout is not None:
            # Pass it through deliberately even if the signature check said
            # it cannot be accepted, letting a TypeError surface. Silently
            # dropping it would measure "an arm where the search was
            # pointless," which would misrepresent the result. The
            # TypeError is caught on the measure_arm side and recorded in
            # the CSV.
            kw["initial_layout"] = initial_layout
        return compile_for_hardware(qc, **kw)
    return run


def measure_arm(arm, run, qc, cmap, pairs, n, reps, time_budget_s,
                per_attempt_call_limit):
    """Measures one condition x one arm. For a smart arm, the search time is
    always included in the total.

    `_smart1` is a variant that uses only stage 1 (6 BFS orderings +
    id_order=True). In a 2026-09-14 real-hardware run, stage 2
    (id_order=False) spent 0.9-2.7 seconds and only caught one case, and even
    on that one case (diluted_p0.75) it still lost to the default -- this arm
    directly measures the hypothesis that "with only the cheap stage, the
    loss margin is smaller."
    """
    is_smart = arm.endswith("_smart") or arm.endswith("_smart1")
    use_fallback = not arm.endswith("_smart1")
    rec = dict(SmartSearch_s=np.nan, SmartFound=None, SmartPhase=None,
               SmartOrder=None, SmartOrderingsTried=np.nan)
    times = []
    out = None
    status = "success"
    for _ in range(reps):
        t0 = time.perf_counter()
        initial_layout = None
        if is_smart:
            layout_map, info = smart_vf2_layout(
                cmap, pairs, n, per_attempt_call_limit=per_attempt_call_limit,
                time_budget_s=time_budget_s, use_fallback=use_fallback)
            rec.update(SmartSearch_s=info["elapsed_s"], SmartFound=info["found"],
                       SmartPhase=info["phase"], SmartOrder=info["order_name"],
                       SmartOrderingsTried=info["orderings_tried"])
            if layout_map is not None:
                initial_layout = layout_map_to_list(layout_map, qc.num_qubits,
                                                    cmap.size())
        try:
            out = run(qc, initial_layout)
        except TypeError as e:
            # Reached here if the PSF side did not accept initial_layout.
            return None, {**rec, "Status": f"TypeError: {e}"}
        times.append(time.perf_counter() - t0)

    total_2q, viol = count_coupling_violations(out, cmap)
    if viol:
        status = f"invalid({viol})"
    rec.update(Status=status, Time_min_s=min(times),
               Time_median_s=float(np.median(times)), Time_max_s=max(times),
               # A 2026-09-14 L3 run found that repeated measurements of the
               # exact same condition swung by 0.34x-2.03x (roughly a 6x
               # range). Looking only at min would not catch this, so the
               # variance itself is kept as a column.
               Spread_max_over_min=(max(times) / min(times)) if min(times) > 0 else float("nan"),
               Final_2Q_Gates=total_2q, Final_Depth=out.depth(),
               Coupling_Violations=viol)
    return out, rec


def warmup(arm, run):
    """Warm-up is outside the timer. Following the same policy as
    phase3_v5, the same code path is run once on a small circuit."""
    from qiskit.transpiler import CouplingMap
    warm_cmap = CouplingMap.from_grid(2, 2)
    warm_qc = build_dense_pair_blocks_circuit(4, gates_per_pair=2, seed=0)
    from qiskit import transpile
    if arm.startswith("qiskit_"):
        transpile(warm_qc, coupling_map=warm_cmap, basis_gates=BASIS_GATES,
                  optimization_level=int(arm.split("opt")[1][0]), seed_transpiler=0)
    else:
        from psf_compile import compile_for_hardware
        compile_for_hardware(warm_qc, coupling_map=warm_cmap, basis_gates=BASIS_GATES,
                             routing_optimization_level=1, verify=False,
                             entangling_basis="cx")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="8x8")
    ap.add_argument("--level", type=int, default=2,
                    help="the Qiskit arm's optimization_level (default 2; L2 only, "
                         "to stay within the 5-minute budget)")
    ap.add_argument("--psf-rl", type=int, default=None,
                    help="the PSF arm's routing_optimization_level. Defaults to "
                         "following --level. In a 2026-09-14 L3 run, --level 3 was "
                         "given but the PSF arm stayed at rl2, ending up measuring "
                         "opt3 vs rl2, a combination that was never a valid "
                         "comparison. (If rl3 fails, pass --psf-rl 1 or 2 explicitly.)")
    ap.add_argument("--spares", type=int, nargs="+", default=[0, 40],
                    help="the tight (0) vs loose (40) contrast")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=2.0,
                    help="smart_vf2_layout's search time budget (seconds)")
    ap.add_argument("--per-attempt-call-limit", type=int, default=50_000)
    ap.add_argument("--arms", default=None,
                    help="comma-separated explicit list (default is automatic: "
                         "4 arms if PSF is available, 2 if not)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lv = args.level
    rl = args.psf_rl if args.psf_rl is not None else lv
    if rl != lv:
        print(f"  [note] The Qiskit arm is optimization_level={lv}, and the PSF arm "
              f"is routing_optimization_level={rl}. **These two cannot be compared "
              f"directly.**")
    psf_ok = psf_available()
    psf_mode, psf_note = psf_supports_initial_layout()

    # PSF's smart arm is pointless to measure if compile_for_hardware cannot
    # accept initial_layout (the search result would be discarded and it
    # would just run plain PSF, mixing in a "row that only lost the search
    # time" and having it counted as a success -- this actually happened in
    # a 2026-09-14 real-hardware run). If it is already known not to accept
    # it, exclude it from the start.
    psf_smart_possible = psf_ok and psf_mode is not False

    if args.arms:
        arms = [a for a in args.arms.split(",") if a]
    else:
        arms = [f"qiskit_opt{lv}", f"qiskit_opt{lv}_smart1", f"qiskit_opt{lv}_smart"]
        if psf_ok:
            arms.append(f"psf_rl{rl}")
            if psf_smart_possible:
                arms += [f"psf_rl{rl}_smart1", f"psf_rl{rl}_smart"]

    banner("(followup 14) Prototype vs. default pipeline -- a total-time contest including PSF-Zero", [
        "P1 on tight grid/line/diluted_p0.75, the smart arm wins",
        "P2 on tight brick/diluted_p0.25/diluted_p0.5, the smart arm loses "
        "(the loss margin is roughly the search's time budget)",
        "P3 at loose (spare=40), the difference is small",
        "P4 output circuit quality (2q gate count, depth) is equal or better in cases where smart wins",
    ])
    print(f"Arms: {arms}")
    print(f"PSF-Zero: {'available' if psf_ok else '**cannot import -> PSF arms are skipped**'}")
    print(f"PSF's initial_layout support: {psf_mode} -- {psf_note}")
    print("**Even when the search fails, the time spent searching is included in the total.**\n")

    rows, cols = (int(x) for x in args.grid.lower().split("x"))
    rows_out = []
    circuits = {}

    hdr = (f"{'topology':<14} {'spare':>5} {'n':>4} {'arm':<20} {'total ms':>10} "
           f"{'search ms':>10} {'found':>6} {'2q':>7} {'depth':>7} {'viol':>5}")
    print(hdr)
    print("-" * len(hdr))

    runners = {}
    for name, cmap, deg, _ in sparse_topologies(rows, cols):
        phys = cmap.size()
        for arm in arms:
            if arm.startswith("psf_") and not psf_ok:
                continue
            if (arm.startswith("psf_") and arm.endswith(("_smart", "_smart1"))
                    and not psf_smart_possible):
                continue
            key = (arm, name)
            runners[key] = make_runner(arm, cmap, lv, rl, psf_mode)
            try:
                warmup(arm, runners[key])
            except Exception as e:  # noqa: BLE001
                print(f"  [warning] warm-up failed for {arm}: {type(e).__name__}: {e}")

        for spare in args.spares:
            n = phys - spare
            if n < 4 or n % 2:
                continue
            pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
            if not has_perfect_matching(cmap, len(pairs)):
                print(f"{name:<14} {spare:>5} {n:>4} no perfect matching exists -> skipped")
                continue
            if n not in circuits:
                circuits[n] = build_dense_pair_blocks_circuit(n, seed=0)
            qc = circuits[n]

            for arm in arms:
                if arm.startswith("psf_") and not psf_ok:
                    continue
                out, rec = measure_arm(arm, runners[(arm, name)], qc, cmap, pairs,
                                       n, args.reps, args.time_budget,
                                       args.per_attempt_call_limit)
                if out is None:
                    print(f"{name:<14} {spare:>5} {n:>4} {arm:<20} {rec['Status']}")
                    rows_out.append(dict(Topology=name, AvgDegree=deg, Physical=phys,
                                         Spare=spare, Qubits=n, Arm=arm,
                                         Level=lv, PSF_RL=rl, Reps=args.reps,
                                         TimeBudget_s=args.time_budget,
                                         PerAttemptCallLimit=args.per_attempt_call_limit,
                                         PSFInitialLayoutMode=str(psf_mode), **rec))
                    continue
                srch = rec["SmartSearch_s"]
                print(f"{name:<14} {spare:>5} {n:>4} {arm:<20} "
                      f"{rec['Time_min_s']*1000:10.1f} "
                      f"{(srch*1000 if srch == srch else float('nan')):10.1f} "
                      f"{str(rec['SmartFound']):>6} {rec['Final_2Q_Gates']:>7} "
                      f"{rec['Final_Depth']:>7} {rec['Coupling_Violations']:>5}",
                      flush=True)
                rows_out.append(dict(Topology=name, AvgDegree=deg, Physical=phys,
                                     Spare=spare, Qubits=n, Arm=arm,
                                     Level=lv, PSF_RL=rl, Reps=args.reps,
                                     TimeBudget_s=args.time_budget,
                                     PerAttemptCallLimit=args.per_attempt_call_limit,
                                     PSFInitialLayoutMode=str(psf_mode), **rec))

    df = write_csv(rows_out, args.out or default_out("smart_layout_vs_default"),
                   environment())

    # ------------------------------------------------------------ verdicts
    print("\n" + "=" * 78)
    print("Verdict 1 -- win/loss table (default arm / smart arm total-time ratio. Above 1 means smart wins)")
    print("=" * 78)
    ok = df[df.Status == "success"] if "Status" in df.columns else df
    bases = [a for a in arms if not a.endswith(("_smart", "_smart1"))]
    for base, smart in [(b, b + sfx) for b in bases for sfx in ("_smart1", "_smart")]:
        if smart not in set(df.Arm):
            continue
        print(f"\n{base} vs {smart}")
        for spare in sorted(set(ok.Spare)):
            print(f"  spare={spare}")
            for name in ok.Topology.unique():
                b = ok[(ok.Arm == base) & (ok.Topology == name) & (ok.Spare == spare)]
                s = ok[(ok.Arm == smart) & (ok.Topology == name) & (ok.Spare == spare)]
                if not len(b) or not len(s):
                    continue
                bt = b.Time_min_s.iloc[0]
                st = s.Time_min_s.iloc[0]
                ratio = bt / st
                verdict = "smart wins" if ratio > 1.05 else (
                    "smart loses" if ratio < 0.95 else "tie")
                found = s.SmartFound.iloc[0]
                # When the search fails, the smart arm is nothing but "the
                # default pipeline + search time," so it is **logically
                # impossible for it to win**. If it still looks like a win,
                # that is variance (measurement noise) on the default side.
                # A 2026-09-14 L3 run actually produced "wins" of 2.41x and
                # 1.21x this way.
                noise = ""
                if (found is False or found == "False") and ratio > 1.05:
                    noise = "  <- **NOISE** (a win is impossible when the search failed)"
                    verdict = "undecidable"
                print(f"    {name:<14} {bt*1000:9.1f} ms / {st*1000:9.1f} ms = "
                      f"{ratio:6.2f}x  {verdict:<12} (search succeeded={found}){noise}")

    print("\n" + "=" * 78)
    print("Verdict 2 -- output circuit quality (P4): 2-qubit gate count and depth")
    print("=" * 78)
    bases = [a for a in arms if not a.endswith(("_smart", "_smart1"))]
    for base, smart in [(b, b + sfx) for b in bases for sfx in ("_smart1", "_smart")]:
        if smart not in set(df.Arm):
            continue
        print(f"\n{base} vs {smart}")
        for spare in sorted(set(ok.Spare)):
            for name in ok.Topology.unique():
                b = ok[(ok.Arm == base) & (ok.Topology == name) & (ok.Spare == spare)]
                s = ok[(ok.Arm == smart) & (ok.Topology == name) & (ok.Spare == spare)]
                if not len(b) or not len(s):
                    continue
                print(f"  spare={spare} {name:<14} "
                      f"2q {int(b.Final_2Q_Gates.iloc[0]):>6} -> "
                      f"{int(s.Final_2Q_Gates.iloc[0]):>6}   "
                      f"depth {int(b.Final_Depth.iloc[0]):>6} -> "
                      f"{int(s.Final_Depth.iloc[0]):>6}")
    print("\n  -> Check whether 2q gate count / depth got worse in cases where smart won.")
    print("     If it got worse, P4 is rejected, meaning \"faster but lower quality.\"")

    print("\n" + "=" * 78)
    print("Verdict 3 -- were any invalid circuits produced (coupling violations)?")
    print("=" * 78)
    bad = df[(df.Coupling_Violations.notna()) & (df.Coupling_Violations > 0)]
    if len(bad):
        print("  **Output(s) with coupling violations were found. Treat this result as invalid.**")
        print(bad[["Topology", "Spare", "Arm", "Coupling_Violations"]].to_string(index=False))
    else:
        print("  0 coupling violations across every arm and every condition. The output circuits are valid.")


if __name__ == "__main__":
    main()
