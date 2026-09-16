#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qiskit vs. PSF-Zero, 1-on-1, across the spare-qubit cliff -- with exact
unitary-equivalence verification, not just structural checks.

## Why this exists

This project's spare-qubit-cliff investigation (addenda 9-26) established
the cliff's mechanism and measured its size many times, but never ran the
exact `Operator`-based fidelity check this session's `bench_qiskit_tket_psf.py`
built -- `test_cliff_sniper_corrected.py` and its successors used only
structural checks (gate count, coupling adherence) at cliff-relevant sizes.
This script re-runs the same cliff crossing (tight -> loose, spare
0..N) but with PSF-Zero specifically, one-on-one against Qiskit
`optimization_level=3`, at a size small enough for exact verification, and
reuses this session's `exact_fidelity_check` / `structural_check`
machinery directly rather than reimplementing it.

TKET is deliberately excluded here -- this script is scoped to the
Qiskit/PSF-Zero comparison this project's own findings are built around,
not a three-way comparison. (The TKET arm in `bench_qiskit_tket_psf.py`
also currently has an open, unresolved fidelity-check limitation: its
`DefaultMappingPass` output has no Qiskit `.layout` to recover the true
qubit correspondence from, so this session's exact check falls back to an
unreliable ascending-index guess for it and produced spurious `exact_FAIL`
rows. That is a separate problem from this script's scope and is not
addressed here.)

## What this measures

For each `spare` value in `--spares` (0 = tight/cliff, increasing = further
from the cliff) and each `layout_search` setting in `--layout-search-modes`:

  - `qiskit_opt3`: plain `transpile(optimization_level=3)`
  - `psf_zero`: `compile_for_hardware(routing_optimization_level=...,
    layout_search=...)`

Both are checked for exact unitary equivalence against the original
circuit (feasible here since qubit counts stay small -- see
`--exact-verify-max-qubits`), not just structural adherence, using this
session's `exact_fidelity_check` (circuit-level qubit remapping via
`qc_new.layout`, avoiding hand-rolled tensor/axis-order arithmetic).

## Usage

    python bench_cliff_1v1.py
    python bench_cliff_1v1.py --grid 6x7 --spares 0 1 2 4 --seeds 3
    python bench_cliff_1v1.py --layout-search-modes true false
"""
from __future__ import annotations

import argparse
import time
import traceback
from datetime import date

import numpy as np


# ---------------------------------------------------------------- circuits
def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    """Same generator this project's other cliff scripts use (dense,
    adjacent-pair blocks) -- kept identical rather than reinvented, since
    the cliff's reproduction depends on this specific circuit structure
    (addendum 19's own coupling-map-free comparison, which used a
    different circuit family entirely, is explicitly NOT the same
    measurement as this one)."""
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
# Copied verbatim from this session's bench_qiskit_tket_psf.py rather than
# imported, since that script is a standalone CLI tool, not a library --
# see this project's own conventions elsewhere for why duplication is
# preferred here over a fragile cross-script import.
def exact_fidelity_check(qc_orig, qc_new):
    """Exact unitary equivalence, up to global phase. Only feasible for
    small qubit counts -- the caller is responsible for gating this by
    size. Returns (passed: bool | None, infidelity: float | None, detail: str).

    An earlier version of this function special-cased `n_new == n_orig`
    as "no remapping needed, compare directly." That assumption is wrong:
    equal qubit *counts* does not imply qubit *i* still holds logical
    qubit *i*'s state -- layout search and Sabre routing can both permute
    qubits while leaving the total count unchanged. Checked directly
    against real output: on a 3x4 grid with spare=0 (qubit count exactly
    equal to the physical qubit count, so this branch was taken),
    `qiskit_opt3` and `psf_zero` with `layout_search=True` both came back
    with infidelity ~0.995-0.9999 (i.e. almost completely different
    operators) using the old direct-comparison code, while
    `layout_search=False` (which, per its design, does not invoke a
    search that could reorder qubits) passed exactly. That contrast is
    itself strong evidence the direct-comparison branch, not the
    circuits being compared, was the bug. Fixed by removing the special
    case entirely: every comparison now goes through the same
    qubit-correspondence step regardless of whether the counts happen to
    match.
    """
    from qiskit.quantum_info import Operator

    op_orig = Operator(qc_orig)
    n_orig = qc_orig.num_qubits
    n_new = qc_new.num_qubits

    used_qubits = set()
    for inst in qc_new.data:
        for q in inst.qubits:
            used_qubits.add(qc_new.find_bit(q).index)
    # A circuit with n_new == n_orig may still have some of its qubits
    # entirely idle if, e.g., a layout search placed the logical circuit
    # on a strict subset of the physical register even though the count
    # happens to match some other reference size -- fall through to the
    # same touched-qubit-count check used for the n_new > n_orig case,
    # rather than assuming "count matches" means "no remapping needed."
    if len(used_qubits) != n_orig:
        return None, None, (
            f"qc_new has {n_new} qubits, {len(used_qubits)} touched, but "
            f"qc_orig has {n_orig} -- ancillas may have been used as "
            f"routing waypoints. Falling back to structural check."
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
            # No layout to consult, but the qubit count matches -- most
            # likely no permutation was needed (e.g. no coupling map was
            # even involved), so identity order is a reasonable, clearly
            # labeled guess rather than the previously-implicit
            # unlabeled assumption.
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
    ap.add_argument("--grid", default="6x7",
                    help="grid size, matching this project's earlier cliff "
                         "measurements (addenda 24-26 used 6x7)")
    ap.add_argument("--spares", type=int, nargs="+", default=[0, 1, 2, 4],
                    help="0 = tight (on the cliff), increasing = further "
                         "from it, matching test_cliff_sniper_corrected.py's "
                         "own convention")
    ap.add_argument("--gates-per-pair", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=1,
                    help="kept low by default since tight-condition Qiskit "
                         "L3 compiles can take seconds each; raise if "
                         "reproducibility (per addendum 18/20) matters more "
                         "than wall-clock budget for a given run")
    ap.add_argument("--exact-verify-max-qubits", type=int, default=14,
                    help="above this, only the structural check runs; kept "
                         "modest since exact verification cost grows as "
                         "4**n")
    ap.add_argument("--qiskit-level", type=int, default=3)
    ap.add_argument("--routing-optimization-level", type=int, default=1)
    ap.add_argument("--layout-search-modes", nargs="+", default=["false", "true"],
                    choices=["true", "false"],
                    help="which layout_search settings to test PSF-Zero "
                         "under, per addendum 26's finding that this option "
                         "collapses PSF-Zero's own cliff")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, cols = (int(x) for x in args.grid.lower().split("x"))
    from qiskit.transpiler import CouplingMap
    cmap = CouplingMap.from_grid(rows, cols)
    n_physical = cmap.size()
    print(f"Coupling map: {rows}x{cols} grid, {n_physical} physical qubits")
    print(f"Spare values: {args.spares} (qubit counts: "
          f"{[n_physical - s for s in args.spares]})")
    print(f"layout_search modes: {args.layout_search_modes}")
    print(f"Seeds per condition: {args.seeds}\n")

    try:
        import psf_compile  # noqa: F401
    except ImportError:
        print("[fatal] psf_compile not importable -- this script needs it. Aborting.")
        return

    rows_out = []
    hdr = (f"{'spare':>6} {'n':>4} {'ls':>5} {'seed':>5} {'arm':<12} "
           f"{'time_ms':>10} {'status':<8} {'fid_check':<12} {'2q':>6} {'viol':>5}")
    print(hdr)
    print("-" * len(hdr))

    for spare in args.spares:
        n = n_physical - spare
        if n < 4 or n % 2:
            print(f"spare={spare}: n={n} invalid (need >=4, even) -- skipped")
            continue
        exact_ok = n <= args.exact_verify_max_qubits

        for seed in range(args.seeds):
            qc = build_dense_pair_blocks_circuit(n, args.gates_per_pair, seed)

            arms = [("qiskit_opt3", None,
                     lambda: run_qiskit(qc, cmap, args.qiskit_level, seed))]
            for ls_str in args.layout_search_modes:
                ls = ls_str == "true"
                arms.append((f"psf_zero_ls{int(ls)}", ls,
                            lambda ls=ls: run_psf(qc, cmap,
                                                  args.routing_optimization_level, ls)))

            for arm_name, ls_flag, fn in arms:
                if seed == 0:
                    try:
                        fn()
                    except Exception:
                        pass  # warm-up only; the timed call below records failures properly

                times = []
                out = None
                status = "success"
                error_text = ""
                try:
                    for _ in range(args.reps):
                        t0 = time.perf_counter()
                        out = fn()
                        times.append(time.perf_counter() - t0)
                except Exception as e:  # noqa: BLE001
                    status = "failed"
                    error_text = f"{type(e).__name__}: {e}"
                    traceback.print_exc()

                fid_check = "none"
                infidelity = None
                fid_detail = ""
                struct = dict(two_qubit_gates=None, coupling_violations=None, depth=None)
                if status == "success":
                    if exact_ok:
                        try:
                            passed, infidelity, fid_detail = exact_fidelity_check(qc, out)
                            fid_check = ("exact_pass" if passed else "exact_FAIL"
                                        if passed is not None else "dim_mismatch_structural_only")
                        except Exception as e:  # noqa: BLE001
                            fid_check = f"exact_error({type(e).__name__})"
                            fid_detail = f"{type(e).__name__}: {e}"
                    else:
                        fid_check = "structural_only"
                    struct = structural_check(out, cmap)

                t_ms = min(times) * 1000 if times else float("nan")
                ls_disp = "-" if ls_flag is None else str(ls_flag)
                print(f"{spare:>6} {n:>4} {ls_disp:>5} {seed:>5} {arm_name:<12} "
                      f"{t_ms:10.3f} {status:<8} {fid_check:<12} "
                      f"{str(struct['two_qubit_gates']):>6} "
                      f"{str(struct['coupling_violations']):>5}", flush=True)

                rows_out.append(dict(
                    Spare=spare, Qubits=n, LayoutSearch=ls_disp, Seed=seed,
                    Arm=arm_name, Reps=args.reps,
                    Time_min_s=min(times) if times else None,
                    Time_median_s=float(np.median(times)) if times else None,
                    Status=status, Error=error_text,
                    FidelityCheck=fid_check, FidelityDetail=fid_detail,
                    Infidelity=infidelity,
                    TwoQubitGates=struct["two_qubit_gates"],
                    CouplingViolations=struct["coupling_violations"],
                    Depth=struct["depth"],
                    QiskitLevel=args.qiskit_level,
                    RoutingOptimizationLevel=args.routing_optimization_level,
                    Grid=args.grid,
                ))

    import csv as csv_mod
    out_path = args.out or f"bench_cliff_1v1_{date.today().isoformat()}.csv"
    if rows_out:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    print(f"\nWrote {out_path} ({len(rows_out)} rows)")

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 78)
    print("Summary -- median time (ms) per condition, successful rows only")
    print("=" * 78)
    arms_seen = sorted({r["Arm"] for r in rows_out})
    print(f"{'spare':>6} " + " ".join(f"{a:>16}" for a in arms_seen))
    for spare in args.spares:
        line = f"{spare:>6} "
        for a in arms_seen:
            vals = [r["Time_min_s"] for r in rows_out
                    if r["Spare"] == spare and r["Arm"] == a and r["Status"] == "success"]
            cell = f"{np.median(vals)*1000:12.3f}ms" if vals else "            --"
            line += f"{cell:>16} "
        print(line)

    print("\n" + "=" * 78)
    print("Fidelity summary")
    print("=" * 78)
    for a in arms_seen:
        exact_p = sum(1 for r in rows_out if r["Arm"] == a and r["FidelityCheck"] == "exact_pass")
        exact_f = sum(1 for r in rows_out if r["Arm"] == a and r["FidelityCheck"] == "exact_FAIL")
        other = sum(1 for r in rows_out if r["Arm"] == a
                    and r["FidelityCheck"] not in ("exact_pass", "exact_FAIL"))
        print(f"  {a}: exact_pass={exact_p} exact_FAIL={exact_f} other/fallback={other}")
        if exact_f:
            print(f"    **{exact_f} exact fidelity failure(s) -- inspect FidelityDetail/Infidelity in the CSV**")

    failed = [r for r in rows_out if r["Status"] == "failed"]
    if failed:
        print(f"\n**{len(failed)} row(s) failed outright** -- not counted above, not replaced with a placeholder.")


if __name__ == "__main__":
    main()
