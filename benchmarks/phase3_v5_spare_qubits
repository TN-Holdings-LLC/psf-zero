#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase3_v5_spare_qubits.py -- Controlled experiment for the "0 spare qubits cause the collapse" hypothesis
=========================================================================================================

Why this script is needed
-------------------------
`phase3-hardware-routing-regression.md` reported that qiskit's optimization_level 2/3
becomes orders of magnitude slower exclusively at 100q and 156q, generating the
hypothesis sent to the Roadmap that "this might only happen when the grid has
exactly 0 spare qubits."

Since then, this phenomenon has been reproduced in 3 independent environments
(Linux sandbox / AMD machine / Intel machine). However, **reproduction is not a
controlled test**. The 4 points measured by phase3_v4:

    n= 50 -> grid 7x8 = 56  (6 spare)   Fast
    n=100 -> grid 10x10=100 (0 spare)   Slow
    n=156 -> grid 12x13=156 (0 spare)   Slow
    n=300 -> grid 17x18=306 (6 spare)   Fast

These completely confound "having 0 spare qubits" with "n being 100/156".
No matter how many machines we test these same 4 points on, this confound
won't diminish by a single millimeter.

This script breaks the confound. Across 3 axes, we vary **only the number
of spare qubits**.

Pre-registered predictions (written before measuring)
-----------------------------------------------------
Axis C (Strongest test): Vary only the circuit qubit count on **the identical coupling map**.
    n=50 (6 spare) vs n=56 (0 spare) on the 7x8=56 map
    n=60 (4 spare) vs n=64 (0 spare) on the 8x8=64 map
    n=66 (6 spare) vs n=72 (0 spare) on the 8x9=72 map
    n=38 (4 spare) vs n=42 (0 spare) on the 6x7=42 map
  If the hypothesis is correct: Only the latter (0 spare) in each pair will
  become orders of magnitude slower. This predicts an inversion ("smaller circuits
  are slower than larger ones") that has never appeared in existing data. If this
  happens, "scale is the cause" is definitively refuted.
  If the hypothesis is incorrect: The two points in each pair will remain in
  the same order of magnitude.

Axis A: Fix the coupling map at 10x10=100, and use circuits n=100,98,96,92 (spare 0,2,4,8).
  If the hypothesis is correct, only n=100 will be slow.

Axis B: Fix the circuit at n=100, and expand the grid: 10x10(100)/10x11(110)/11x11(121).
  If the hypothesis is correct, only 10x10 will be slow.

Anchor (verification that the fixture matches phase3_v4)
--------------------------------------------------------
Axis A's n=100/grid100 should be identical to phase3_v4's dense 100q condition.
The measured baselines are opt2 at 1244ms (Linux sandbox) / 1015ms (AMD) / 894ms (Intel).
If we deviate by an order of magnitude from here, the fixture is different
(= this experiment is invalid). The script ensures this check is displayed at the end.

Fixture origins (verbatim copies from existing project files, not guesses)
--------------------------------------------------------------------------
- `get_grid_cmap()`            : Verbatim from phase3_v3.py
- `build_dense_pair_blocks_circuit()` : Verbatim from test_scale_explosion_war2.py
- gates_per_pair=20            : Confirmed by reverse-calculating from phase3_v4's output
                                 (n=50: opt1's 2q=1500 = 25 pairs x 20 x 3CX)
- basis_gates                  : ["rz","sx","x","cx"]
Since we don't have phase3_v4.py itself at hand, we verify it matches via the anchor.

Regarding the PSF arm (psf_rl1_cx)
----------------------------------
Since this hypothesis is about qiskit's internal transpiler, the qiskit arms are
sufficient for the test. In environments that can build psf_zero_core, appending
`--with-psf` allows measuring it simultaneously as a control.

    python3 phase3_v5_spare_qubits.py --axis C
    python3 phase3_v5_spare_qubits.py --axis all --with-psf
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import platform
import sys
import time

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap

BASIS_GATES = ["rz", "sx", "x", "cx"]
GATES_PER_PAIR = 20
TIMEOUT_SECONDS = 900
OUTPUT_CSV = "phase3_v5_spare_qubits_results.csv"


# ---------------------------------------------------------------- fixture
# Verbatim copy from phase3_v3.py
def get_grid_cmap(num_qubits):
    """Generates the closest 2D grid (mimicking real physical device layouts) for a given number of qubits"""
    cols = int(math.ceil(math.sqrt(num_qubits)))
    rows = int(math.ceil(num_qubits / cols))
    return CouplingMap.from_grid(rows, cols)


def grid_shape(num_qubits):
    cols = int(math.ceil(math.sqrt(num_qubits)))
    rows = int(math.ceil(num_qubits / cols))
    return rows, cols


# Verbatim copy from test_scale_explosion_war2.py (docstring summarized)
def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed=0):
    """Circuit stacking deep 2-qubit interactions on adjacent pairs. Has a structure
    PSF-Zero can actually compress (already decompose()d)."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    for (a, b) in pairs:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


def count_coupling_violations(qc, cmap):
    """Verbatim copy from phase3_v3.py"""
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


# ---------------------------------------------------------------- workers
def _worker(arm, num_qubits, rows, cols, seed, reps, q):
    """Child process. Warm-up is outside the timer. Measured one by one for min/median."""
    try:
        cmap = CouplingMap.from_grid(rows, cols)
        qc = build_dense_pair_blocks_circuit(num_qubits, GATES_PER_PAIR, seed=seed)

        if arm.startswith("qiskit_opt"):
            level = int(arm[-1])

            def run(c):
                return transpile(c, coupling_map=cmap, basis_gates=BASIS_GATES,
                                 optimization_level=level)
        elif arm.startswith("psf_rl"):
            from psf_compile import compile_for_hardware
            level = int(arm.split("rl")[1][0])

            def run(c):
                return compile_for_hardware(
                    c, coupling_map=cmap, basis_gates=BASIS_GATES,
                    routing_optimization_level=level, verify=False,
                    entangling_basis="cx")
        else:
            raise ValueError(arm)

        # warm-up (outside timer): Pass through the same path once with a small circuit
        warm_cmap = CouplingMap.from_grid(2, 2)
        warm = build_dense_pair_blocks_circuit(4, 2, seed=0)
        if arm.startswith("qiskit_opt"):
            transpile(warm, coupling_map=warm_cmap, basis_gates=BASIS_GATES,
                      optimization_level=int(arm[-1]))
        else:
            from psf_compile import compile_for_hardware as _cfh
            _cfh(warm, coupling_map=warm_cmap, basis_gates=BASIS_GATES,
                 routing_optimization_level=int(arm.split("rl")[1][0]),
                 verify=False, entangling_basis="cx")

        times = []
        out = None
        for _ in range(reps):
            t0 = time.perf_counter()
            out = run(qc)
            times.append(time.perf_counter() - t0)

        total_2q, viol = count_coupling_violations(out, cmap)
        q.put({
            "status": "success" if viol == 0 else f"invalid({viol})",
            "t_min": min(times), "t_med": float(np.median(times)),
            "t_max": max(times), "final_2q": total_2q, "depth": out.depth(),
            "viol": viol,
        })
    except Exception as e:  # noqa: BLE001
        q.put({"status": f"error: {type(e).__name__}: {e}", "t_min": np.nan,
               "t_med": np.nan, "t_max": np.nan, "final_2q": -1, "depth": -1,
               "viol": -1})


def measure(arm, num_qubits, rows, cols, seed, reps, ctx):
    q = ctx.Queue()
    p = ctx.Process(target=_worker,
                    args=(arm, num_qubits, rows, cols, seed, reps, q))
    p.start()
    p.join(TIMEOUT_SECONDS)
    if p.is_alive():
        p.terminate(); p.join()
        return {"status": "timeout", "t_min": np.nan, "t_med": np.nan,
                "t_max": np.nan, "final_2q": -1, "depth": -1, "viol": -1}
    return q.get() if not q.empty() else {
        "status": "no result", "t_min": np.nan, "t_med": np.nan,
        "t_max": np.nan, "final_2q": -1, "depth": -1, "viol": -1}


# ---------------------------------------------------------------- plans
def axis_c_plan():
    """Pairs varying only the spare qubits on the same map (strongest test)"""
    out = []
    for lo, hi in ((38, 42), (50, 56), (60, 64), (66, 72)):
        rl, cl = grid_shape(lo)
        rh, ch = grid_shape(hi)
        assert (rl, cl) == (rh, ch), (lo, hi, (rl, cl), (rh, ch))
        for n in (lo, hi):
            out.append(("C", n, rl, cl))
    return out


def axis_a_plan():
    """Fix coupling map to 10x10, vary only the circuit qubit count"""
    plan = []
    for n in (100, 98, 96, 92):
        r, c = grid_shape(n)
        assert (r, c) == (10, 10), (n, r, c)
        plan.append(("A", n, 10, 10))
    return plan


def axis_b_plan():
    """Fix circuit to n=100, expand only the grid"""
    return [("B", 100, 10, 10), ("B", 100, 10, 11), ("B", 100, 11, 11)]


def axis_t_plan():
    """Pinning down the threshold location. Since Axis A showed 'spare 2 is slow / spare 4 is fast', 
    we line up the same spare counts on two maps (large and small) to see if the threshold is an 
    absolute count or a ratio of the grid size."""
    plan = []
    for n in (100, 98, 96, 94, 92):          # On 10x10 = 100, spare 0,2,4,6,8
        r, c = grid_shape(n)
        assert (r, c) == (10, 10), (n, r, c)
        plan.append(("T", n, 10, 10))
    for n in (72, 70, 68, 66):               # On 8x9 = 72, spare 0,2,4,6
        r, c = grid_shape(n)
        assert (r, c) == (8, 9), (n, r, c)
        plan.append(("T", n, 8, 9))
    return plan


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="C", choices=["A", "B", "C", "T", "all"])
    ap.add_argument("--arms", default="qiskit_opt2,qiskit_opt3")
    ap.add_argument("--with-psf", action="store_true")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default=OUTPUT_CSV)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    if args.with_psf:
        arms.append("psf_rl1_cx")
    seeds = [int(s) for s in args.seeds.split(",")]

    plan = []
    if args.axis in ("C", "all"):
        plan += axis_c_plan()
    if args.axis in ("A", "all"):
        plan += axis_a_plan()
    if args.axis in ("B", "all"):
        plan += axis_b_plan()
    if args.axis in ("T", "all"):
        plan += axis_t_plan()

    env = {
        "Platform": platform.platform(), "Python": platform.python_version(),
        "Qiskit": __import__("qiskit").__version__, "StartMethod": "spawn",
        "CPU": platform.processor(),
    }
    print("environment:", env)
    print(f"arms={arms} seeds={seeds} reps={args.reps}")
    print("\nPre-registered prediction: If only the 0-spare-qubit points become orders of magnitude slower, the hypothesis is supported."
          "\nIf there is no difference between pairs on the same map, the hypothesis is refuted.\n")

    ctx = multiprocessing.get_context("spawn")
    rows_out = []
    for axis, n, r, c in plan:
        spare = r * c - n
        print(f"=== axis {axis}: n={n:>3}  grid={r}x{c}={r*c}  Spare={spare:>2} ===")
        for arm in arms:
            for seed in seeds:
                res = measure(arm, n, r, c, seed, args.reps, ctx)
                print(f"    [{arm:<12} seed{seed}] {res['status']:<10} "
                      f"min {res['t_min']*1000:9.1f} ms  "
                      f"2q={res['final_2q']:<7} depth={res['depth']:<6} viol={res['viol']}")
                rows_out.append({
                    "Axis": axis, "Qubits": n, "Grid_Rows": r, "Grid_Cols": c,
                    "Grid_Qubits": r * c, "Spare_Qubits": spare, "Arm": arm,
                    "Seed": seed, "Reps": args.reps, "Status": res["status"],
                    "Time_min_s": res["t_min"], "Time_median_s": res["t_med"],
                    "Time_max_s": res["t_max"], "Final_2Q_Gates": res["final_2q"],
                    "Final_Depth": res["depth"], "Coupling_Violations": res["viol"],
                    **env,
                })

    df = pd.DataFrame(rows_out)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 78)
    print("SUMMARY -- min time (ms), median across seeds")
    print("=" * 78)
    piv = df.pivot_table(index=["Axis", "Qubits", "Grid_Qubits", "Spare_Qubits"],
                         columns="Arm", values="Time_min_s", aggfunc="median") * 1000
    print(piv.round(1))

    anchor = df[(df.Qubits == 100) & (df.Grid_Qubits == 100) & (df.Arm == "qiskit_opt2")]
    if len(anchor):
        v = anchor["Time_min_s"].median() * 1000
        print(f"\nFixture anchor: qiskit_opt2 for n=100/grid100 = {v:.1f} ms")
        print("  Measured baselines: 1244ms (Linux sandbox) / 1015ms (AMD machine) / 894ms (Intel machine)")
        print("  -> If the order of magnitude matches, the fixture aligns with phase3_v4. If it deviates widely, discard this experiment as invalid.")


if __name__ == "__main__":
    sys.exit(main())
