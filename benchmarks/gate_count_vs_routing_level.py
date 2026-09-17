"""gate_count_vs_routing_level.py -- does raising compile_for_hardware()'s
existing `routing_optimization_level` knob close PSF-Zero's 2-qubit
gate-count gap on this project's canonical circuit family, before any new
peephole/cancellation pass is written?

Motivation, from Addendum 29's 2026-09-17 update: on this project's
canonical 6x7 dense-pair-blocks grid circuit, PSF-Zero's own 2-qubit gate
count (`layout_search=False` or `True`, `routing_optimization_level=1`,
the default) is **exactly 2.00x** Qiskit L3's, uniformly from spare=0 to
spare=24 -- not a cliff-specific effect. Separately, `psf_compile.py`'s
own `compile_for_hardware()` docstring claims that raising
`routing_optimization_level` to 2 already produces "the same 2-qubit gate
count as Qiskit's optimization_level 2 and 3 ... for 1/20th to 1/59th of
their time" -- but that claim was measured on a *different* benchmark
(100-156 qubit dense-pair-blocks over an unconstrained/non-saturated
topology), never on this project's 6x7-grid cliff-sniper family. This
script tests directly, on the SAME circuit family the 2.00x gap was
found in, whether `routing_optimization_level=2` or `3` closes it, before
any new hand-written gate-cancellation pass is designed.

Time cost at each level is *not* what this script is measuring -- it is
already on record from Addendum 25 (rl=1: 88.148ms, rl=2: 554.557ms,
rl=3: 6696.621ms, all at this project's spare=0/6x7 scenario). This
script's only new contribution is TwoQubitGates and Depth at each level,
which Addendum 25 never recorded.

Repeats exist for a specific reason, not just statistical caution:
Addendum 29 found that at spare=0 specifically, `layout_search=False`'s
own OUTPUT CIRCUIT (not merely its timing) varies from call to call on
IDENTICAL input -- 132, 129, and 126 two-qubit gates across three calls
in one round, despite a pinned `seed_transpiler` -- consistent with
Qiskit's own previously-documented VF2Layout nondeterminism at a
saturated coupling map (Addendum 9). Reporting a single call per point
here would risk reporting that noise as if it were "the" gate count for
a given (spare, routing_optimization_level) combination. Every call is
individually recorded below, never averaged away, exactly like
spare0_outlier_hunt.py's convention -- this script asks the same
question of gate count that that one asked of timing.

Pre-registered predictions: see
spare-qubit-cliff-addendum-30-preregistration-2026-09-17.md, written
BEFORE this script is ever run against the real psf_zero_core. Do not
adjust that document after seeing this script's output.

Usage:
    python gate_count_vs_routing_level.py --rows 6 --cols 7 \\
        --spares 0,2,4,8,16,24 --levels 1,2,3 --seeds 3 --repeats 3

Expect this to take several minutes, not seconds: at spare=0,
routing_optimization_level=3 alone costs several seconds PER CALL
(Addendum 25: ~6.7s median), and this script calls it
(seeds x repeats x 2 PSF-Zero arms) + warm-ups times at that one
(spare, level) cell alone. The Qiskit L3 baseline at spare=0 is also
multi-second per call (Addenda 24-27). Every other (spare, level) cell
is fast (tens to low hundreds of ms). This is a deliberate design choice
-- see the module docstring above -- not a runaway loop; if it is running
far longer than a few minutes, check that `--spares` was not left at a
much wider default than intended.

Requires the real `psf_zero_core` Rust extension (as always in this
project) and this session's current `psf_compile.py` (any version with
`routing_optimization_level` -- this parameter is not new; item 13's
`callback` addition is not required by this script and is not used).
"""
from __future__ import annotations

import argparse
import os
import platform
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.transpiler import CouplingMap

from psf_compile import compile as psf_compile_only
from psf_compile import compile_for_hardware

BASIS_GATES = ["rz", "sx", "x", "cx"]
GATES_PER_PAIR = 20
SEED_TRANSPILER = 42


def build_dense_pair_blocks_circuit(num_qubits: int, gates_per_pair: int = GATES_PER_PAIR,
                                     seed: int = 0) -> QuantumCircuit:
    """Unchanged from the rest of this project's cliff-sniper family."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for a, b in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


def check_routing_validity(qc: QuantumCircuit, cm: CouplingMap) -> tuple[bool, int]:
    edges = set(map(tuple, cm.get_edges()))
    violations = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            i = qc.find_bit(inst.qubits[0]).index
            j = qc.find_bit(inst.qubits[1]).index
            if (i, j) not in edges and (j, i) not in edges:
                violations += 1
    return violations == 0, violations


def small_scale_correctness_precheck(seed: int) -> None:
    print("--- Small-scale correctness pre-check (n=6, no coupling map) ---")
    qc = build_dense_pair_blocks_circuit(6, seed=seed)
    target = Operator(qc)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = psf_compile_only(qc, verify=True)
    ok = Operator(out).equiv(target)
    print(f"  Operator(out).equiv(Operator(in)) == {ok}")
    for w in caught:
        print(f"  [fallback warning] {w.message}")
    if not ok:
        raise RuntimeError(
            "Small-scale correctness pre-check FAILED -- refusing to run "
            "this sweep against a build that fails its own equivalence check."
        )
    print()


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--spares", type=str, default="0,2,4,8,16,24")
    ap.add_argument("--levels", type=str, default="1,2,3",
                     help="routing_optimization_level values to test on the "
                          "PSF-Zero side (Qiskit L3 baseline is unaffected "
                          "by this and is measured once per spare/seed)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3,
                     help="individually-recorded repeats per point, to "
                          "surface the spare=0 output-circuit "
                          "nondeterminism Addendum 29 found rather than "
                          "average over it")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    grid_size = args.rows * args.cols
    spares = [int(s) for s in args.spares.split(",")]
    levels = [int(l) for l in args.levels.split(",")]
    seeds = list(range(args.seeds))

    small_scale_correctness_precheck(seed=0)

    print("=" * 100)
    print(f"GATE COUNT vs ROUTING LEVEL: {args.rows}x{args.cols} grid, spares={spares}, "
          f"routing_optimization_level in {levels}, seeds={seeds}, repeats={args.repeats}")
    print(f"Qiskit L3 baseline measured once per (spare, seed), independent of level.")
    print("=" * 100)

    rows_out = []
    t_start = time.perf_counter()

    for spare in spares:
        n = grid_size - spare
        cm = CouplingMap.from_grid(args.rows, args.cols)

        for seed in seeds:
            qc = build_dense_pair_blocks_circuit(n, seed=seed)

            # --- Qiskit L3 baseline: once per (spare, seed), independent of level ---
            def run_qiskit(qc=qc, cm=cm):
                out = transpile(qc, coupling_map=cm, basis_gates=BASIS_GATES,
                                 optimization_level=3, seed_transpiler=SEED_TRANSPILER)
                ok, violations = check_routing_validity(out, cm)
                if not ok:
                    raise RuntimeError(f"qiskit_opt3 output has {violations} coupling violation(s)")
                return out

            try:
                run_qiskit()  # warm-up, discarded
            except Exception as exc:
                print(f"  spare={spare:2d} seed={seed} qiskit_opt3 WARM-UP FAILED -- {type(exc).__name__}: {exc}")

            for rep in range(args.repeats):
                t0 = time.perf_counter()
                try:
                    out = run_qiskit()
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    rows_out.append(dict(
                        spare=spare, n=n, seed=seed, arm="qiskit_opt3",
                        routing_optimization_level=None, repeat=rep,
                        time_ms=elapsed_ms, two_qubit_gates=two_qubit_gate_count(out),
                        depth=out.depth(), error="",
                        elapsed_since_start_s=time.perf_counter() - t_start,
                    ))
                except Exception as exc:
                    rows_out.append(dict(
                        spare=spare, n=n, seed=seed, arm="qiskit_opt3",
                        routing_optimization_level=None, repeat=rep,
                        time_ms=None, two_qubit_gates=None, depth=None,
                        error=f"{type(exc).__name__}: {exc}",
                        elapsed_since_start_s=time.perf_counter() - t_start,
                    ))

            # --- PSF-Zero: both arms, every requested routing_optimization_level ---
            for level in levels:
                for arm_name, layout_search in (("psf_zero_ls0", False), ("psf_zero_ls1", True)):

                    def run_psf(qc=qc, cm=cm, level=level, layout_search=layout_search):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            out = compile_for_hardware(
                                qc, coupling_map=cm, basis_gates=BASIS_GATES,
                                routing_optimization_level=level,
                                entangling_basis="cx", seed_transpiler=SEED_TRANSPILER,
                                layout_search=layout_search,
                            )
                        ok, violations = check_routing_validity(out, cm)
                        if not ok:
                            raise RuntimeError(f"{arm_name} output has {violations} coupling violation(s)")
                        return out, len(caught)

                    try:
                        run_psf()  # warm-up, discarded
                    except Exception as exc:
                        print(f"  spare={spare:2d} seed={seed} rl={level} {arm_name} WARM-UP FAILED -- {type(exc).__name__}: {exc}")

                    for rep in range(args.repeats):
                        t0 = time.perf_counter()
                        try:
                            out, fallback_count = run_psf()
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0
                            rows_out.append(dict(
                                spare=spare, n=n, seed=seed, arm=arm_name,
                                routing_optimization_level=level, repeat=rep,
                                time_ms=elapsed_ms, two_qubit_gates=two_qubit_gate_count(out),
                                depth=out.depth(), error="",
                                fallback_count=fallback_count,
                                elapsed_since_start_s=time.perf_counter() - t_start,
                            ))
                        except Exception as exc:
                            rows_out.append(dict(
                                spare=spare, n=n, seed=seed, arm=arm_name,
                                routing_optimization_level=level, repeat=rep,
                                time_ms=None, two_qubit_gates=None, depth=None,
                                error=f"{type(exc).__name__}: {exc}",
                                elapsed_since_start_s=time.perf_counter() - t_start,
                            ))

            elapsed_total = time.perf_counter() - t_start
            print(f"  spare={spare:2d} seed={seed} done  (elapsed so far: {elapsed_total:6.1f}s)")

    df = pd.DataFrame(rows_out)
    df["cpu"] = platform.processor()
    df["python_version"] = platform.python_version()
    df["qiskit_version"] = qiskit.__version__
    df["grid_rows"] = args.rows
    df["grid_cols"] = args.cols
    df["seed_transpiler"] = SEED_TRANSPILER

    print("=" * 100)
    print("Summary: median two_qubit_gates by (spare, arm, routing_optimization_level)")
    summary = df[df.error == ""].pivot_table(
        index=["spare"], columns=["arm", "routing_optimization_level"],
        values="two_qubit_gates", aggfunc="median",
    )
    print(summary)
    print()
    print("Per-point spread (max-min) in two_qubit_gates, to surface any "
          "call-to-call nondeterminism (Addendum 29's spare=0 finding):")
    spread = df[df.error == ""].groupby(
        ["spare", "arm", "routing_optimization_level"]
    )["two_qubit_gates"].agg(lambda s: s.max() - s.min())
    print(spread[spread > 0] if (spread > 0).any() else "  (none -- every point's repeats agreed exactly)")
    print("=" * 100)

    cpu_tag = (platform.processor() or "unknown_cpu").replace(" ", "_").replace(",", "")
    base_name = f"gate_count_vs_routing_level_{args.rows}x{args.cols}_{cpu_tag}_{date.today().isoformat()}"
    out_path = os.path.join(args.out_dir, base_name + ".csv")
    if os.path.exists(out_path):
        i = 2
        while os.path.exists(os.path.join(args.out_dir, f"{base_name}_run{i}.csv")):
            i += 1
        out_path = os.path.join(args.out_dir, f"{base_name}_run{i}.csv")

    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df)} rows)")
    print(
        "Before pasting this file's contents anywhere outside this machine: "
        "check it for a local file path (e.g. C:\\Users\\...) or any other "
        "machine-identifying string beyond the CPU signature recorded above, "
        "per this project's standing record-keeping rules."
    )


if __name__ == "__main__":
    main()
