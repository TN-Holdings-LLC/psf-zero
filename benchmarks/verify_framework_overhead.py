"""benchmarks/verify_framework_overhead.py

Where does Qiskit's and TKET's per-call time actually go, for the single fixed
2-qubit unitary used in verify_determinism_variance.py?

verify_determinism_variance.py measured only each engine's total time per call. The
claim that "most of it is DAG construction, pass-eligibility checks, and format
conversion, not the 2-qubit decomposition itself" is plausible but was not measured
there. This script measures it directly instead of arguing from how pass managers are
documented to work.

Method:
  - Qiskit: generate_preset_pass_manager(...).run(qc, callback=...) times every
    individual pass. Summing gives a per-pass breakdown of the same total
    verify_determinism_variance.py reported.
  - TKET: DecomposeBoxes and FullPeepholeOptimise are timed as separate calls
    (this project's TKET harnesses already call them separately; here they're just
    each wrapped in their own timer instead of one combined block), plus the
    Qiskit<->TKET circuit conversion steps (qiskit_to_tk, tk_to_qiskit), which
    verify_determinism_variance.py's combined timer folded into "TKET time" without
    separating conversion from optimization.
  - PSF-Zero: for comparison, re-confirms section 4's own finding (~87% of
    per-block time is the Operator() self-check) on this exact circuit, rather than
    assuming it transfers unchanged from a different circuit family.

Pre-registered predictions, written before running:
  P1  For Qiskit, layout/routing-related passes (Layout, ApplyLayout, routing) will
      be a large share even though this circuit has no coupling map and needs no
      routing -- these passes still run and check applicability.
  P2  The actual 2-qubit synthesis pass (UnitarySynthesis / Optimize1qGatesDecomposition
      equivalent) will be a small fraction of the Qiskit total, similar in spirit to
      section 4's finding that PSF-Zero's own decomposition is a small fraction of
      its per-block time.
  P3  For TKET, circuit format conversion (qiskit_to_tk + tk_to_qiskit) will be a
      non-trivial fraction of the combined time verify_determinism_variance.py
      reported as "TKET time" -- i.e. some of the 86.3x is conversion overhead, not
      the optimization pass itself.

Usage:
    python verify_framework_overhead.py
Writes framework_overhead_2026-09-13.csv.
"""

import collections
import csv
import platform
import statistics
import sys
import time

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import DecomposeBoxes, FullPeepholeOptimise

from psf_compile import compile as psf_compile

N_ITERATIONS = 200

# On Windows, individual pass times reported by Qiskit's transpile callback land far
# below the platform's effective timer resolution -- measured directly on a first
# run of this script: 97.9% of per-pass readings came back exactly 0.0, and the
# nonzero ones clustered near 0.5 ms and 1.0 ms rather than forming a real
# distribution, which is the signature of a clock quantum, not real pass costs.
# Summing per-call medians in that regime is meaningless (it summed to 0.0 ms against
# a measured total of 0.63 ms in that run). Instead, per-pass time is accumulated
# across all N_ITERATIONS calls and reported as a share of the accumulated total --
# still limited by the same clock resolution per reading, but the accumulated sum
# converges as the number of readings grows, unlike a median of mostly-zero samples.

ENV = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python": sys.version.split()[0],
    "qiskit": qiskit.__version__,
}


def make_target_unitary():
    """Same construction as verify_determinism_variance.py: one fixed SU(4)."""
    u = random_unitary(4, seed=42).data
    u = u / np.linalg.det(u) ** 0.25
    return u


def qiskit_breakdown(qc_base, pm):
    """One transpile call, timed per pass via callback. Returns (total_ms, per_pass_ms)."""
    per_pass = collections.defaultdict(float)

    def cb(**kwargs):
        name = type(kwargs["pass_"]).__name__
        per_pass[name] += kwargs["time"] * 1000.0

    t0 = time.perf_counter()
    pm.run(qc_base, callback=cb)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, dict(per_pass)


def tket_breakdown(qc_base):
    """One TKET call, with conversion and each pass timed separately."""
    t0 = time.perf_counter()
    tk_circ = qiskit_to_tk(qc_base)
    t_to_tk = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    DecomposeBoxes().apply(tk_circ)
    t_decompose = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    FullPeepholeOptimise().apply(tk_circ)
    t_peephole = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    tk_to_qiskit(tk_circ)
    t_from_tk = (time.perf_counter() - t0) * 1000.0

    total_ms = t_to_tk + t_decompose + t_peephole + t_from_tk
    return total_ms, {
        "qiskit_to_tk": t_to_tk,
        "DecomposeBoxes": t_decompose,
        "FullPeepholeOptimise": t_peephole,
        "tk_to_qiskit": t_from_tk,
    }


def main():
    print(ENV)
    print(f"{N_ITERATIONS} iterations, one fixed SU(4) unitary (seed=42)\n")

    u = make_target_unitary()
    qc_base = QuantumCircuit(2)
    qc_base.unitary(u, [0, 1])

    qiskit_pm = generate_preset_pass_manager(
        optimization_level=3, basis_gates=["rz", "sx", "x", "cx"]
    )

    rows = []

    print("--- Qiskit: per-pass breakdown (accumulated over all calls) ---")
    qiskit_totals, qiskit_pass_totals = [], collections.defaultdict(list)
    qiskit_pass_accum = collections.defaultdict(float)
    qiskit_pass_nonzero_count = collections.defaultdict(int)
    for i in range(N_ITERATIONS):
        total_ms, per_pass = qiskit_breakdown(qc_base, qiskit_pm)
        qiskit_totals.append(total_ms)
        for name, ms in per_pass.items():
            qiskit_pass_totals[name].append(ms)
            qiskit_pass_accum[name] += ms
            if ms > 0:
                qiskit_pass_nonzero_count[name] += 1
        rows.append({"engine": "Qiskit_L3", "iteration": i, "component": "TOTAL",
                     "time_ms": round(total_ms, 7), **ENV})
        for name, ms in per_pass.items():
            rows.append({"engine": "Qiskit_L3", "iteration": i, "component": name,
                         "time_ms": round(ms, 7), **ENV})

    print("--- TKET: conversion + pass breakdown ---")
    tket_totals, tket_component_totals = [], collections.defaultdict(list)
    for i in range(N_ITERATIONS):
        total_ms, components = tket_breakdown(qc_base)
        tket_totals.append(total_ms)
        for name, ms in components.items():
            tket_component_totals[name].append(ms)
        rows.append({"engine": "TKET", "iteration": i, "component": "TOTAL",
                     "time_ms": round(total_ms, 7), **ENV})
        for name, ms in components.items():
            rows.append({"engine": "TKET", "iteration": i, "component": name,
                         "time_ms": round(ms, 7), **ENV})

    print("--- PSF-Zero: re-confirming section 4's phase split on this circuit ---")
    # Mirrors profile_synthesize_breakdown.py's phases at the compile() call level:
    # this project's compile() call itself vs its own internal verify step, measured
    # by calling with verify=True and verify=False and taking the difference as an
    # upper bound on what the check costs (not a true sub-call breakdown, since
    # compile() does not expose per-phase callbacks the way Qiskit's PassManager does).
    psf_false_times, psf_true_times = [], []
    for i in range(N_ITERATIONS):
        t0 = time.perf_counter()
        psf_compile(qc_base, verify=False)
        psf_false_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        psf_compile(qc_base, verify=True)
        psf_true_times.append((time.perf_counter() - t0) * 1000.0)
        rows.append({"engine": "PSF-Zero", "iteration": i, "component": "verify=False",
                     "time_ms": round(psf_false_times[-1], 4), **ENV})
        rows.append({"engine": "PSF-Zero", "iteration": i, "component": "verify=True",
                     "time_ms": round(psf_true_times[-1], 4), **ENV})

    # ---- summary ----
    print(f"\n=== Qiskit per-pass share, accumulated over {N_ITERATIONS} calls ===")
    qiskit_total_accum = sum(qiskit_totals)  # sum of per-call totals, ms
    qiskit_pass_sum = sum(qiskit_pass_accum.values())
    for name, accum in sorted(qiskit_pass_accum.items(), key=lambda kv: -kv[1]):
        share = 100 * accum / qiskit_total_accum if qiskit_total_accum else 0.0
        nz = qiskit_pass_nonzero_count[name]
        print(f"  {name:<35} {accum:9.3f} ms total  ({share:5.1f}%)  "
              f"nonzero in {nz:>3}/{N_ITERATIONS} calls")
    print(f"  {'TOTAL (sum of passes, accumulated)':<35} {qiskit_pass_sum:9.3f} ms")
    print(f"  {'TOTAL (measured, accumulated)':<35} {qiskit_total_accum:9.3f} ms")
    unaccounted = qiskit_total_accum - qiskit_pass_sum
    print(f"  {'unaccounted (below callback resolution)':<35} {unaccounted:9.3f} ms  "
          f"({100*unaccounted/qiskit_total_accum:5.1f}%)")
    print("  ('nonzero in N/200 calls' close to 200 means the pass usually costs at")
    print("   least one clock tick; a low count means most calls under-resolved to")
    print("   zero and the accumulated share for that pass is a noisier estimate.)")

    print("\n=== TKET breakdown, median (ms), share of total ===")
    tket_total_median = statistics.median(tket_totals)
    for name, vals in tket_component_totals.items():
        med = statistics.median(vals)
        share = 100 * med / tket_total_median
        print(f"  {name:<25} {med:8.4f} ms  ({share:5.1f}%)")
    conversion = statistics.median(tket_component_totals["qiskit_to_tk"]) + \
                 statistics.median(tket_component_totals["tk_to_qiskit"])
    print(f"  {'conversion (to+from tk)':<25} {conversion:8.4f} ms  "
          f"({100*conversion/tket_total_median:5.1f}%)")

    print("\n=== PSF-Zero: verify cost on this exact circuit ===")
    f_med = statistics.median(psf_false_times)
    t_med = statistics.median(psf_true_times)
    print(f"  verify=False median: {f_med:.4f} ms")
    print(f"  verify=True  median: {t_med:.4f} ms")
    print(f"  verify cost: {t_med - f_med:.4f} ms ({100*(t_med-f_med)/t_med:.1f}% of verify=True total)")

    print("\n--- Predictions ---")
    layout_related = sum(v for k, v in qiskit_pass_accum.items()
                         if any(s in k for s in ("Layout", "Route", "Sabre")))
    print(f"P1 (layout/routing passes are a large share despite no coupling map): "
          f"{layout_related:.3f} ms = "
          f"{100*layout_related/qiskit_total_accum:.1f}% of accumulated Qiskit total")
    synth_related = sum(v for k, v in qiskit_pass_accum.items()
                        if any(s in k for s in ("UnitarySynthesis", "Synthesis", "Optimize1q")))
    print(f"P2 (2-qubit synthesis pass itself is a small fraction): "
          f"{synth_related:.3f} ms = "
          f"{100*synth_related/qiskit_total_accum:.1f}% of accumulated Qiskit total")
    print(f"P3 (TKET conversion is a non-trivial fraction of combined TKET time): "
          f"{100*conversion/tket_total_median:.1f}%")
    print(f"\nNote: {100*unaccounted/qiskit_total_accum:.1f}% of Qiskit's accumulated total")
    print("is not attributed to any named pass -- below the callback's clock")
    print("resolution on this fast a circuit. Treat P1/P2 percentages above as lower")
    print("bounds on each category's true share, not exact figures.")

    out_path = "framework_overhead_2026-09-13.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()