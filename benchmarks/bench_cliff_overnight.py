#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overnight batch: Qiskit vs. PSF-Zero across the spare-qubit cliff (tight
to flat), repeated over several rounds with fixed seeds, meant to be
started before going to sleep and checked the next morning.

## Why this exists

Two things motivated this specific design:

1. The cliff's speed comparison (`bench_cliff_1v1.py`, single run) only
   covered a few spare values with a handful of seeds. A longer,
   multi-round sweep from the cliff's worst point (spare=0) out to a
   clearly-flat region gives a fuller picture of how the gap changes
   across that whole range, not just a few sample points.
2. Two apparent outliers were found by hand while re-running
   `bench_cliff_1v1.py` manually: `psf_zero` with `layout_search=False`
   at **the same seed (seed=1)** came back anomalously slow **twice, in
   two independent runs, at two different spare values** (144.202ms both
   times at spare=2; ~27ms at spare=0, also elevated vs. that condition's
   other seeds). Two independent reproductions of the same seed producing
   an outlier is a real pattern worth investigating, not run-to-run noise
   -- this project's own addenda 20-23 found and chased down comparable
   seed/period-specific effects before. Fixing the seed set across rounds
   (rather than drawing fresh ones each round) is deliberate: it lets
   "does this specific seed's circuit reproduce as an outlier" be checked
   directly, round over round, exactly as addendum 20's approach did.

## Design for being left running unattended

  - **Every condition's result is appended to disk immediately**, not
    held in memory until the end -- if the process is killed or the
    machine sleeps partway through, everything completed so far is on
    disk already, not lost.
  - **A failure in one condition does not stop the whole run** -- recorded
    as `Status=failed` with the actual exception, the loop continues.
  - **No fixed output filename** -- see this project's own
    `provenance-map.md` for why silently overwriting a previous night's
    data is a real, previously-hit failure mode in this project.
  - **A running progress estimate is printed after each round** so a
    glance at the terminal (or a redirected log file) shows how far along
    an unattended run is.

## Usage

    python bench_cliff_overnight.py
    python bench_cliff_overnight.py --grid 6x7 --spares 0 1 2 4 8 16 24 --rounds 5
    python bench_cliff_overnight.py --seeds 0 1 2      # fixed seed set, matching
                                                        # the seed=1 outlier found by hand

To actually run overnight unattended and keep a log even if the terminal
is closed:

    python bench_cliff_overnight.py > overnight_log_2026-09-17.txt 2>&1
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import time
import traceback
from datetime import date, datetime

import numpy as np


# ---------------------------------------------------------------- circuits
def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for (a, b) in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


BASIS_GATES = ["rz", "sx", "x", "cx"]


# ---------------------------------------------------------------- fidelity
# Copied from bench_cliff_1v1.py (post-fix: the n_new == n_orig special
# case that produced false exact_FAIL rows on layout-searched circuits has
# been removed there and here alike -- see that script's own docstring
# for the full account of that bug).
def exact_fidelity_check(qc_orig, qc_new):
    """Exact unitary equivalence, up to global phase. Returns
    (passed: bool | None, infidelity: float | None, detail: str)."""
    from qiskit.quantum_info import Operator

    op_orig = Operator(qc_orig)
    n_orig = qc_orig.num_qubits
    n_new = qc_new.num_qubits

    used_qubits = set()
    for inst in qc_new.data:
        for q in inst.qubits:
            used_qubits.add(qc_new.find_bit(q).index)

    if len(used_qubits) != n_orig:
        return None, None, (
            f"qc_new has {n_new} qubits, {len(used_qubits)} touched, but "
            f"qc_orig has {n_orig}. Falling back to structural check."
        )

    order = None
    layout = getattr(qc_new, "layout", None)
    if layout is not None:
        try:
            index_layout = layout.final_index_layout(filter_ancillas=True)
            if sorted(index_layout) == sorted(used_qubits):
                order = list(index_layout)
        except Exception:  # noqa: BLE001
            order = None
    if order is None:
        if n_new == n_orig:
            order = list(range(n_orig))
            order_source = "identity order (no .layout, counts matched)"
        else:
            order = sorted(used_qubits)
            order_source = "ascending-index fallback (no usable .layout)"
    else:
        order_source = "qc_new.layout.final_index_layout"

    from qiskit import QuantumCircuit
    phys_to_dense = {phys: dense for dense, phys in enumerate(order)}
    reduced = QuantumCircuit(n_orig)
    for inst in qc_new.data:
        phys_indices = [qc_new.find_bit(q).index for q in inst.qubits]
        dense_indices = [phys_to_dense[p] for p in phys_indices]
        reduced.append(inst.operation, dense_indices)

    try:
        op_new = Operator(reduced)
    except Exception as e:  # noqa: BLE001
        return None, None, (
            f"Built a {n_orig}-qubit reduced circuit (order source: "
            f"{order_source}) but Operator() raised {type(e).__name__}: "
            f"{e}. Falling back to structural check."
        )

    dim = op_orig.dim[0]
    overlap = np.abs(np.trace(op_orig.data.conj().T @ op_new.data)) / dim
    infidelity = 1.0 - overlap
    return infidelity < 1e-6, float(infidelity), f"order_source={order_source}"


def structural_check(qc, cmap):
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
    return dict(two_qubit_gates=total_2q, coupling_violations=violations,
               depth=qc.depth())


# ---------------------------------------------------------------- engines
def run_qiskit(qc, cmap, level, seed_transpiler):
    from qiskit import transpile
    return transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
                     optimization_level=level, seed_transpiler=seed_transpiler)


def run_psf(qc, cmap, routing_optimization_level, layout_search):
    import psf_compile
    return psf_compile.compile_for_hardware(
        qc, coupling_map=cmap, basis_gates=BASIS_GATES,
        routing_optimization_level=routing_optimization_level,
        verify=True, entangling_basis="canonical",
        layout_search=layout_search,
    )


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="6x7")
    ap.add_argument("--spares", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 24],
                    help="from the cliff's worst point (0) out to a clearly "
                         "flat region")
    ap.add_argument("--gates-per-pair", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="fixed seed set, reused identically every round -- "
                         "includes 1, since that seed produced a reproduced "
                         "outlier (144.202ms, spare=2, layout_search=False, "
                         "twice) when this was run by hand; keeping it fixed "
                         "lets that specific circuit be checked again here")
    ap.add_argument("--rounds", type=int, default=5,
                    help="how many times to repeat the entire seed x spare "
                         "x arm sweep")
    ap.add_argument("--exact-verify-max-qubits", type=int, default=12,
                    help="kept modest -- exact verification cost grows as "
                         "4**n, and most spare values here will exceed this "
                         "and fall back to the structural check, which is "
                         "expected: this run's main purpose is the speed "
                         "sweep, not re-proving correctness (already "
                         "established separately for 4-12 qubits)")
    ap.add_argument("--qiskit-level", type=int, default=3)
    ap.add_argument("--routing-optimization-level", type=int, default=1)
    ap.add_argument("--layout-search-modes", nargs="+", default=["false", "true"],
                    choices=["true", "false"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, cols = (int(x) for x in args.grid.lower().split("x"))
    from qiskit.transpiler import CouplingMap
    cmap = CouplingMap.from_grid(rows, cols)
    n_physical = cmap.size()

    n_conditions = sum(1 for s in args.spares if (n_physical - s) >= 4
                       and (n_physical - s) % 2 == 0)
    n_arms = 1 + len(args.layout_search_modes)
    total_calls = args.rounds * n_conditions * len(args.seeds) * n_arms

    out_path = args.out or f"bench_cliff_overnight_{date.today().isoformat()}.csv"
    fieldnames = ["Round", "Spare", "Qubits", "LayoutSearch", "Seed", "Arm",
                 "Time_s", "Status", "Error", "FidelityCheck", "FidelityDetail",
                 "Infidelity", "TwoQubitGates", "CouplingViolations", "Depth",
                 "QiskitLevel", "RoutingOptimizationLevel", "Grid", "Timestamp"]

    print("=" * 78)
    print(f"Overnight batch starting {datetime.now().isoformat()}")
    print(f"Grid: {args.grid} ({n_physical} physical qubits)")
    print(f"Spares: {args.spares}")
    print(f"Seeds (fixed across all rounds): {args.seeds}")
    print(f"Rounds: {args.rounds}")
    print(f"layout_search modes: {args.layout_search_modes}")
    print(f"Total conditions to run: {total_calls}")
    print(f"Output (appended incrementally, never overwritten mid-run): {out_path}")
    print("=" * 78 + "\n")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    try:
        import psf_compile  # noqa: F401
    except ImportError:
        print("[fatal] psf_compile not importable. Aborting before any work is done.")
        return

    t_start = time.perf_counter()
    n_done = 0
    n_failed = 0

    for round_idx in range(1, args.rounds + 1):
        print(f"\n{'='*78}\nRound {round_idx}/{args.rounds} -- "
              f"{datetime.now().isoformat()}\n{'='*78}")

        for spare in args.spares:
            n = n_physical - spare
            if n < 4 or n % 2:
                continue

            for seed in args.seeds:
                qc = build_dense_pair_blocks_circuit(n, args.gates_per_pair, seed)
                exact_ok = n <= args.exact_verify_max_qubits

                arms = [("qiskit_opt3", None,
                         lambda: run_qiskit(qc, cmap, args.qiskit_level, seed))]
                for ls_str in args.layout_search_modes:
                    ls = ls_str == "true"
                    arms.append((f"psf_zero_ls{int(ls)}", ls,
                                lambda ls=ls: run_psf(qc, cmap,
                                                      args.routing_optimization_level, ls)))

                for arm_name, ls_flag, fn in arms:
                    row = dict(Round=round_idx, Spare=spare, Qubits=n,
                              LayoutSearch=("-" if ls_flag is None else str(ls_flag)),
                              Seed=seed, Arm=arm_name,
                              QiskitLevel=args.qiskit_level,
                              RoutingOptimizationLevel=args.routing_optimization_level,
                              Grid=args.grid, Timestamp=datetime.now().isoformat())
                    try:
                        t0 = time.perf_counter()
                        out = fn()
                        elapsed = time.perf_counter() - t0
                        row["Time_s"] = elapsed
                        row["Status"] = "success"
                        row["Error"] = ""

                        fid_check = "none"
                        infidelity = None
                        fid_detail = ""
                        if exact_ok:
                            try:
                                passed, infidelity, fid_detail = exact_fidelity_check(qc, out)
                                fid_check = ("exact_pass" if passed else "exact_FAIL"
                                            if passed is not None
                                            else "dim_mismatch_structural_only")
                            except Exception as e:  # noqa: BLE001
                                fid_check = f"exact_error({type(e).__name__})"
                                fid_detail = f"{type(e).__name__}: {e}"
                        else:
                            fid_check = "structural_only"
                        struct = structural_check(out, cmap)

                        row.update(FidelityCheck=fid_check, FidelityDetail=fid_detail,
                                  Infidelity=infidelity,
                                  TwoQubitGates=struct["two_qubit_gates"],
                                  CouplingViolations=struct["coupling_violations"],
                                  Depth=struct["depth"])
                        n_done += 1
                        flag = ""
                        if elapsed > 1.0 and "psf_zero" in arm_name:
                            flag = "  <-- SLOW psf_zero (compare to other seeds at this condition)"
                        print(f"  round={round_idx} spare={spare:>3} n={n:>3} "
                              f"ls={row['LayoutSearch']:>5} seed={seed} "
                              f"{arm_name:<12} {elapsed*1000:9.2f}ms {fid_check}{flag}",
                              flush=True)
                    except Exception as e:  # noqa: BLE001
                        row.update(Time_s=None, Status="failed",
                                  Error=f"{type(e).__name__}: {e}",
                                  FidelityCheck="none", FidelityDetail="",
                                  Infidelity=None, TwoQubitGates=None,
                                  CouplingViolations=None, Depth=None)
                        n_failed += 1
                        print(f"  round={round_idx} spare={spare:>3} n={n:>3} "
                              f"{arm_name:<12} FAILED: {type(e).__name__}: {e}", flush=True)
                        traceback.print_exc()

                    with open(out_path, "a", newline="", encoding="utf-8") as f:
                        w = csv_mod.DictWriter(f, fieldnames=fieldnames)
                        w.writerow(row)

        elapsed_total = time.perf_counter() - t_start
        done_so_far = round_idx * n_conditions * len(args.seeds) * n_arms
        rate = elapsed_total / max(done_so_far, 1)
        remaining = (total_calls - done_so_far) * rate
        print(f"\n  [progress] {done_so_far}/{total_calls} conditions done, "
              f"{elapsed_total/60:.1f} min elapsed, "
              f"~{remaining/60:.1f} min remaining (rough estimate)")

    print(f"\n{'='*78}")
    print(f"Done: {datetime.now().isoformat()}")
    print(f"{n_done} succeeded, {n_failed} failed, out of {total_calls} planned")
    print(f"Results: {out_path}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
