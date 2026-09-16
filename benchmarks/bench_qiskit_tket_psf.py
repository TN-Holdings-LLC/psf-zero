#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A fair, honest three-way benchmark: Qiskit L3, TKET (FullPeepholeOptimise),
and PSF-Zero (`compile_for_hardware`, optionally with `layout_search=True`).

## Why this exists, and what "fair" means here

This project found, earlier in the same session, that an older benchmark
script for this exact comparison had a critical bug: every failed PSF-Zero
call was silently caught and replaced with a fabricated placeholder number
(0.5 ms), making every reported "speedup" fictitious. This script is built
to avoid that and the other failure modes found across today's addenda:

  - **No exception is ever caught and replaced with a number.** A failed
    compile is recorded as `Status=failed` with the actual exception text,
    never silently substituted.
  - **Warm-up is outside the timer**, once per engine per circuit size,
    matching the convention used throughout this project's other
    benchmarks (`vf2_probe_common.py`, `test_cumulative_compile_scale.py`).
  - **Multiple seeds are used and the median is reported**, not a single
    run -- Addendum 16 and 26 both found real run-to-run variance in this
    kind of measurement, and reporting one number without repetition would
    hide it.
  - **Fidelity is checked at the strength the circuit size actually
    allows, and the check performed is stated explicitly, not implied.**
    Small circuits get exact unitary-equivalence verification
    (`Operator` comparison, up to global phase). Larger circuits get a
    structural sanity check only (gate count, that every 2-qubit gate
    lands on a coupled edge for the hardware-aware arms) -- and the CSV
    records which check each row actually received, so a reader is never
    left assuming "success" means the same thing at every size.
  - **No fixed output filename.** Every run writes a dated file and never
    silently overwrites a previous run's data (the same fixed-filename bug
    this project's own `provenance-map.md` documents for `test1_v3.py`).
  - **The TKET arm is not penalized or flattered by a hidden conversion
    cost.** `qiskit_to_tk` / `tk_to_qiskit` conversion is timed separately
    from `FullPeepholeOptimise` itself and both are reported, so a reader
    can see how much of TKET's total time is conversion overhead versus
    the optimization pass -- this project's own addenda already found,
    for a different comparison, that conversion overhead was NOT what
    explained a large TKET/PSF gap (worth checking directly here too,
    rather than assuming the same finding transfers).

## What this does NOT claim

  - This is not a scaling study. Circuit sizes here are chosen to keep
    exact fidelity verification feasible; see `--qubits` and
    `--exact-verify-max-qubits`.
  - TKET's `FullPeepholeOptimise` is one specific TKET pass among several
    it offers; this is not a claim that it represents TKET's best possible
    result for this workload.
  - "PSF-Zero" here means `compile_for_hardware()` specifically (the
    hardware-aware entry point, matching what the other two arms are
    actually doing -- targeting a coupling map). Plain `compile()` (no
    coupling map, no routing) is a different comparison and is out of
    scope for this script.

## Usage

    python bench_qiskit_tket_psf.py
    python bench_qiskit_tket_psf.py --qubits 6 8 10 --seeds 5
    python bench_qiskit_tket_psf.py --layout-search --routing-optimization-level 2
    python bench_qiskit_tket_psf.py --grid 6x7 --exact-verify-max-qubits 8
"""
from __future__ import annotations

import argparse
import time
import traceback
from datetime import date

import numpy as np


# ---------------------------------------------------------------- circuits
def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    """Copied from this project's own `vf2_probe_common.py` /
    `test_cumulative_compile_scale.py` generator, for consistency with
    every other benchmark in this project rather than inventing a new
    circuit family."""
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


def get_grid_cmap(rows, cols):
    from qiskit.transpiler import CouplingMap
    return CouplingMap.from_grid(rows, cols)


BASIS_GATES = ["rz", "sx", "x", "cx"]


# ---------------------------------------------------------------- fidelity
def exact_fidelity_check(qc_orig, qc_new):
    """Exact unitary equivalence, up to global phase. Only feasible for
    small qubit counts -- the caller is responsible for gating this by
    size. Returns (passed: bool | None, infidelity: float | None, detail: str).

    `qc_new` (post-transpile) commonly has more qubits than `qc_orig` (the
    coupling map's full physical qubit count, not just qc_orig's logical
    ones). An earlier version of this function tried
    `qiskit.quantum_info.Operator.from_circuit()` expecting it to reduce
    the dimension down to qc_orig's qubit count -- checked directly
    against real output, it does not: it only reorders qubits into the
    circuit's original input ordering, and still returns the full
    2**n_new-dimensional operator, ancillas included. That was caught
    before being trusted further (the whole point of this function is not
    producing a number that looks right but silently isn't).

    The approach here instead works at the **circuit** level, which
    avoids hand-rolling any qubit-order bookkeeping in numpy (a strong
    source of exactly the kind of subtle, hard-to-notice bug this project
    has hit more than once with unverified index/axis assumptions):

      1. Determine which physical qubits `qc_new` actually uses (any
         qubit touched by at least one instruction).
      2. If that set's size doesn't match `qc_orig`'s qubit count, this is
         reported honestly rather than guessed at -- it means either
         ancilla qubits were non-trivially involved (e.g. used as a swap
         waypoint) or something else unexpected, and the assumption this
         function relies on (idle ancillas factor out cleanly) does not
         hold.
      3. If it does match, a new `n_orig`-qubit circuit is built by
         copying every instruction from `qc_new` with its qubits remapped
         from physical index to a dense 0..n_orig-1 index (in the order
         given by `qc_new.layout`, when present, so the qubit
         correspondence to `qc_orig` is the transpiler's own, not
         assumed). `Operator()` is then computed on this reduced circuit
         directly -- letting Qiskit's own, tested tensor-product handling
         do the actual linear algebra, rather than a hand-written
         reshape/slice.

    A second bug was found later, in a sibling script
    (`bench_cliff_1v1.py`) that copied this function: the `n_new ==
    n_orig` case above used to skip straight to a direct comparison,
    on the assumption that equal qubit *counts* meant no remapping was
    needed. That is false -- layout search and Sabre routing can both
    permute qubits while leaving the total count unchanged. Checked
    directly: on a grid where the circuit's qubit count exactly equals
    the physical qubit count (so this branch was taken), both Qiskit
    L3 and PSF-Zero with `layout_search=True` came back with
    infidelity ~0.995-0.9999 against the direct-comparison code, while
    `layout_search=False` (whose design does not reorder qubits)
    passed exactly -- strong evidence the comparison branch, not the
    circuits, was wrong. Fixed here too, by removing the special case:
    every comparison now goes through the same touched-qubit /
    layout-based correspondence step regardless of whether the counts
    happen to match.
    """
    from qiskit.quantum_info import Operator

    op_orig = Operator(qc_orig)
    n_orig = qc_orig.num_qubits
    n_new = qc_new.num_qubits

    # Find which physical qubits are actually touched by any
    # instruction -- done unconditionally now, even when n_new ==
    # n_orig, since qubit count matching does not imply qubit i still
    # holds logical qubit i's state (see the n_new == n_orig bug noted
    # above).
    used_qubits = set()
    for inst in qc_new.data:
        for q in inst.qubits:
            used_qubits.add(qc_new.find_bit(q).index)

    if len(used_qubits) != n_orig:
        return None, None, (
            f"qc_new has {n_new} qubits, {len(used_qubits)} of them "
            f"touched by at least one instruction, but qc_orig has "
            f"{n_orig} -- the \"extra qubits are provably idle\" "
            f"assumption this check relies on does not hold cleanly here "
            f"(ancillas may have been used as routing waypoints). Falling "
            f"back to structural check rather than guessing at a mapping."
        )

    # Determine the used physical qubits' order. Prefer the transpiler's
    # own layout (the correspondence it actually chose between qc_orig's
    # virtual qubits and qc_new's physical ones) over an assumed identity
    # mapping, since layout_search / Sabre routing do not generally
    # preserve qubit i -> qubit i.
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
            # labeled guess rather than the previously-implicit unlabeled
            # assumption the n_new == n_orig special case used to make.
            order = list(range(n_orig))
            order_source = "identity order (no .layout, counts matched)"
        else:
            # No usable layout -- fall back to the touched qubits in
            # ascending physical-index order. This is not guaranteed to
            # match qc_orig's own qubit order, so this path is less
            # trustworthy; noted in the returned detail string when taken.
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
    """Gate count and coupling-map adherence only -- NOT a substitute for
    exact_fidelity_check. Used for circuit sizes where exact verification
    is not feasible, and the CSV records this distinction explicitly."""
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


def run_tket(qc, cmap):
    """Returns (result_circuit, convert_in_s, optimize_s, route_s,
    convert_out_s).

    An earlier version of this function ran `FullPeepholeOptimise` alone
    and stopped there. `FullPeepholeOptimise` does not attempt to satisfy
    any qubit-connectivity constraint at all -- checked directly, every
    TKET row this produced had `CouplingViolations` equal to its entire
    2-qubit gate count, meaning the output did not respect `cmap` in any
    way. Comparing that against the Qiskit and PSF-Zero arms, which are
    both held to `cmap`, was not a fair speed comparison: TKET was doing
    strictly less work. `DefaultMappingPass`, applied here after
    optimization (mirroring the order used in TKET's own manual
    compilation example -- optimize the logical circuit first, then map
    and route it to hardware), adds placement and SWAP-based routing so
    TKET's output is held to the same constraint as the other two arms.
    Whether this actually produces zero violations is not assumed here --
    it is checked directly, the same way as for the other two arms, via
    this script's own `structural_check`.
    """
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    from pytket.passes import FullPeepholeOptimise, DefaultMappingPass
    from pytket.architecture import Architecture

    # Built directly from cmap's edge list as [[node, node], ...] pairs of
    # plain ints. This matches pytket's own description of what
    # Architecture expects ("a list of pairs of qubits"), but has not
    # been run against a real pytket install in this session (sandbox has
    # no network access to install pytket) -- if this raises on the first
    # real run, the fix is almost certainly in how this list is built,
    # not in the surrounding logic.
    arch = Architecture([[a, b] for a, b in cmap.get_edges()])

    t0 = time.perf_counter()
    tk_circ = qiskit_to_tk(qc)
    t1 = time.perf_counter()
    FullPeepholeOptimise().apply(tk_circ)
    t2 = time.perf_counter()
    DefaultMappingPass(arch).apply(tk_circ)
    t3 = time.perf_counter()
    qc_out = tk_to_qiskit(tk_circ)
    t4 = time.perf_counter()
    return qc_out, (t1 - t0), (t2 - t1), (t3 - t2), (t4 - t3)


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
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8],
                    help="qubit counts to test. Kept small by default so "
                         "exact fidelity verification stays feasible -- see "
                         "--exact-verify-max-qubits")
    ap.add_argument("--gates-per-pair", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of independent circuit seeds per qubit count; "
                         "the median across seeds is what gets reported")
    ap.add_argument("--reps", type=int, default=3,
                    help="repeated timing measurements per (seed, engine), "
                         "min of these is used as that seed's time")
    ap.add_argument("--exact-verify-max-qubits", type=int, default=10,
                    help="circuits at or below this size get exact Operator-"
                         "based fidelity verification; above it, only the "
                         "structural check (gate count, coupling adherence) "
                         "is performed and the CSV says so explicitly")
    ap.add_argument("--qiskit-level", type=int, default=3)
    ap.add_argument("--routing-optimization-level", type=int, default=1,
                    help="PSF-Zero's own routing_optimization_level, passed "
                         "through to compile_for_hardware -- kept separate "
                         "from --qiskit-level since Addendum 25 found these "
                         "are not directly comparable settings")
    ap.add_argument("--layout-search", action="store_true",
                    help="enable compile_for_hardware(layout_search=True)")
    ap.add_argument("--grid", default=None,
                    help="e.g. 4x4 -- if not given, a coupling map sized to "
                         "the largest --qubits value is built automatically")
    ap.add_argument("--skip-tket", action="store_true",
                    help="skip the TKET arm (e.g. if pytket is not installed)")
    ap.add_argument("--skip-psf", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import math
    if args.grid:
        rows, cols = (int(x) for x in args.grid.lower().split("x"))
    else:
        n = max(args.qubits)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    cmap = get_grid_cmap(rows, cols)
    print(f"Coupling map: {rows}x{cols} grid, {cmap.size()} physical qubits")
    print(f"Qubit counts to test: {args.qubits}")
    print(f"Seeds per size: {args.seeds}, reps per seed: {args.reps}")
    print(f"Exact fidelity verification up to {args.exact_verify_max_qubits} qubits\n")

    have_tket = False
    if not args.skip_tket:
        try:
            import pytket  # noqa: F401
            have_tket = True
        except ImportError:
            print("[warning] pytket not importable -- TKET arm will be skipped "
                  "for all rows (not silently treated as 0 or omitted from the "
                  "printed warning)")
    have_psf = False
    if not args.skip_psf:
        try:
            import psf_compile  # noqa: F401
            have_psf = True
        except ImportError:
            print("[warning] psf_compile not importable -- PSF-Zero arm will "
                  "be skipped for all rows")

    rows_out = []
    hdr = (f"{'n':>4} {'seed':>5} {'arm':<10} {'time_ms':>10} {'status':<10} "
           f"{'fid_check':<12} {'2q_gates':>9} {'violations':>11}")
    print(hdr)
    print("-" * len(hdr))

    for n in args.qubits:
        exact_ok = n <= args.exact_verify_max_qubits
        for seed in range(args.seeds):
            qc = build_dense_pair_blocks_circuit(n, args.gates_per_pair, seed)

            arms = [("qiskit", lambda: run_qiskit(qc, cmap, args.qiskit_level, seed))]
            if have_tket:
                arms.append(("tket", lambda: run_tket(qc, cmap)))
            if have_psf:
                arms.append(("psf_zero", lambda: run_psf(
                    qc, cmap, args.routing_optimization_level, args.layout_search)))

            for arm_name, fn in arms:
                # warm-up: same code path, outside the timer, once per
                # (arm, size) rather than once total, since a coupling map
                # of a different size could plausibly warm different code
                if seed == 0:
                    try:
                        fn()
                    except Exception:
                        pass  # a warm-up failure is not recorded; the timed
                              # call below will surface and record it properly

                times = []
                out = None
                status = "success"
                error_text = ""
                extra = dict(convert_in_s=None, optimize_s=None, route_s=None,
                            convert_out_s=None)
                try:
                    for _ in range(args.reps):
                        t0 = time.perf_counter()
                        if arm_name == "tket":
                            out, c_in, opt_s, route_s, c_out = fn()
                            extra.update(convert_in_s=c_in, optimize_s=opt_s,
                                        route_s=route_s, convert_out_s=c_out)
                        else:
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
                            if passed is None:
                                fid_check = "dim_mismatch_structural_only"
                            else:
                                fid_check = "exact_pass" if passed else "exact_FAIL"
                        except Exception as e:  # noqa: BLE001
                            fid_check = f"exact_error({type(e).__name__})"
                            fid_detail = f"{type(e).__name__}: {e}"
                    else:
                        fid_check = "structural_only"
                    struct = structural_check(out, cmap)

                t_ms = min(times) * 1000 if times else float("nan")
                print(f"{n:>4} {seed:>5} {arm_name:<10} {t_ms:10.3f} {status:<10} "
                      f"{fid_check:<12} {str(struct['two_qubit_gates']):>9} "
                      f"{str(struct['coupling_violations']):>11}")

                rows_out.append(dict(
                    Qubits=n, Seed=seed, Arm=arm_name, Reps=args.reps,
                    Time_min_s=min(times) if times else None,
                    Time_median_s=float(np.median(times)) if times else None,
                    Status=status, Error=error_text,
                    FidelityCheck=fid_check, FidelityDetail=fid_detail,
                    Infidelity=infidelity,
                    TwoQubitGates=struct["two_qubit_gates"],
                    CouplingViolations=struct["coupling_violations"],
                    Depth=struct["depth"],
                    **extra,
                    QiskitLevel=args.qiskit_level,
                    RoutingOptimizationLevel=args.routing_optimization_level,
                    LayoutSearch=args.layout_search,
                ))

    import csv as csv_mod
    out_path = args.out or f"bench_qiskit_tket_psf_{date.today().isoformat()}.csv"
    if rows_out:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv_mod.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    print(f"\nWrote {out_path} ({len(rows_out)} rows)")

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 78)
    print("Summary -- median time (ms) by arm, per qubit count (successful rows only)")
    print("=" * 78)
    by_key = {}
    for r in rows_out:
        if r["Status"] != "success":
            continue
        by_key.setdefault((r["Qubits"], r["Arm"]), []).append(r["Time_min_s"])
    arms_seen = sorted({k[1] for k in by_key})
    print(f"{'n':>4} " + " ".join(f"{a:>12}" for a in arms_seen))
    for n in args.qubits:
        line = f"{n:>4} "
        for a in arms_seen:
            vals = by_key.get((n, a))
            cell = f"{np.median(vals)*1000:10.3f}ms" if vals else "        --"
            line += f"{cell:>12} "
        print(line)

    print("\n" + "=" * 78)
    print("Fidelity summary")
    print("=" * 78)
    for a in arms_seen:
        exact_rows = [r for r in rows_out if r["Arm"] == a and r["FidelityCheck"] == "exact_pass"]
        exact_fail = [r for r in rows_out if r["Arm"] == a and r["FidelityCheck"] == "exact_FAIL"]
        struct_rows = [r for r in rows_out if r["Arm"] == a
                      and r["FidelityCheck"] in ("structural_only", "dim_mismatch_structural_only")]
        print(f"  {a}: exact_pass={len(exact_rows)} exact_FAIL={len(exact_fail)} "
              f"structural_only={len(struct_rows)}")
        if exact_fail:
            print(f"    **{len(exact_fail)} exact fidelity failure(s) -- see CSV Error/Infidelity columns**")

    failed = [r for r in rows_out if r["Status"] == "failed"]
    if failed:
        print(f"\n**{len(failed)} row(s) failed outright** (see CSV Error column) -- "
              f"these are NOT counted in any average above, and are not replaced "
              f"with a placeholder value.")


if __name__ == "__main__":
    main()
