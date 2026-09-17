"""occupancy_sweep.py -- where exactly is the coupling-map occupancy
threshold, and does VF2Layout's stop reason move with it?

Addenda 4-32 of this project all compare a saturated coupling map
(spare=0) against a comfortably padded one (spare=4, 6, 24). That
establishes *that* a compile-time cliff exists. It does not locate the
edge, and it does not separate two explanations that predict the same
coarse result:

  (A) budget exhaustion near a constraint-satisfaction threshold -- a
      valid embedding still exists at spare=0, but so few remain that
      VF2Layout's bounded search (`call_limit`) runs out before finding
      one and Qiskit falls back to the slower SabreLayout; or
  (B) something that scales smoothly with occupancy, with no threshold.

Under (A), the timing cliff must coincide with the point where
`property_set["VF2Layout_stop_reason"]` flips between NO_SOLUTION_FOUND
and SOLUTION_FOUND. Under (B), the stop reason should not track the
timing at all. This script records both, per run, so the question is
decided by data rather than by argument.

An external profiling study (arXiv 2504.15141) independently found
VF2Layout consuming >99% of compile time (61.6s) on a 100-qubit circuit,
confirming the phenomenon is not peculiar to this project's setup -- but
it held qubit count fixed on a single backend and never varied device
occupancy, so it cannot distinguish (A) from (B). That is the gap here.

**This script does not import or exercise PSF-Zero.** It measures
Qiskit's own behaviour, and needs only qiskit / numpy / pandas /
networkx. The circuit builder is copied verbatim from this project's
cliff-sniper family so that results are comparable with the existing
addenda.

Pre-registered predictions: see
`spare-qubit-cliff-addendum-34-preregistration-2026-09-17.md`, written
BEFORE this script was run at full scale. Do not edit that document
after seeing this script's output.

Usage:
    python occupancy_sweep.py --rows 6 --cols 7 \\
        --spares 0,1,2,3,4,5,6,8,10,12,16,20,24 \\
        --levels 1,3 --seeds 3 --repeats 3

Expect this to be slow, and unevenly so: on this project's machines
spare=0 at optimization_level=3 costs several seconds per call while
spare>=6 costs tens of milliseconds, so nearly all of the wall-clock
time is spent on the first two or three spare values. That is the
phenomenon being measured, not a runaway loop. `--levels 1` alone
finishes in well under a minute if you want to check the plumbing first.
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from datetime import date

import networkx as nx
import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap

BASIS_GATES = ["rz", "sx", "x", "cx"]
GATES_PER_PAIR = 20
SEED_TRANSPILER = 42

# Passes whose combined time is the "layout stage" for prediction P3.
LAYOUT_PASSES = ("VF2Layout", "SabreLayout", "VF2PostLayout", "TrivialLayout",
                 "DenseLayout", "SetLayout", "ApplyLayout", "FullAncillaAllocation",
                 "EnlargeWithAncilla")
ROUTING_PASSES = ("SabreSwap", "StochasticSwap", "BasicSwap", "LookaheadSwap")


def build_dense_pair_blocks_circuit(num_qubits: int, gates_per_pair: int = GATES_PER_PAIR,
                                    seed: int = 0) -> QuantumCircuit:
    """Verbatim from this project's cliff-sniper family -- do not 'improve'.

    Note that for odd `num_qubits` the final qubit is left unpaired; the
    number of pairs actually built is recorded as `n_pairs` in the output
    so this is visible in the data rather than hidden.
    """
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


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


def matching_feasibility(cm: CouplingMap, n_pairs: int) -> dict:
    """Independent (non-Qiskit) check of whether the circuit's interaction
    graph -- `n_pairs` disjoint edges -- can be embedded in the coupling
    graph at all.

    This is prediction P4's evidence: if a perfect matching exists and
    VF2Layout still reports NO_SOLUTION_FOUND, the failure is a search-
    budget failure and not an infeasibility, which is the whole basis of
    explanation (A).
    """
    g = nx.Graph()
    g.add_nodes_from(range(cm.size()))
    g.add_edges_from([tuple(e) for e in cm.get_edges()])
    m = nx.max_weight_matching(g, maxcardinality=True)
    max_matching = len(m)
    return dict(
        max_matching_size=max_matching,
        n_pairs_required=n_pairs,
        embedding_feasible=bool(max_matching >= n_pairs),
        perfect_matching_exists=bool(max_matching * 2 == cm.size()),
        matching_slack=max_matching - n_pairs,
    )


def timed_transpile(qc: QuantumCircuit, cm: CouplingMap, level: int) -> dict:
    """One transpile, instrumented per pass.

    `transpile(callback=...)` fires once per pass with that pass's own
    execution time, so summing by pass name attributes the total. The
    live `property_set` is also handed to the callback, which is how
    VF2Layout's stop reason is captured -- `transpile()` itself does not
    return the property set.
    """
    per_pass: dict[str, float] = {}
    state = {"stop_reason": None}

    def cb(pass_, dag, time, property_set, count, **_):
        name = type(pass_).__name__
        per_pass[name] = per_pass.get(name, 0.0) + float(time)
        reason = property_set.get("VF2Layout_stop_reason")
        if reason is not None:
            state["stop_reason"] = getattr(reason, "value", str(reason))

    t0 = time.perf_counter()
    out = transpile(qc, coupling_map=cm, basis_gates=BASIS_GATES,
                    optimization_level=level, seed_transpiler=SEED_TRANSPILER,
                    callback=cb)
    total_ms = (time.perf_counter() - t0) * 1000.0

    layout_ms = sum(v for k, v in per_pass.items() if k in LAYOUT_PASSES) * 1000.0
    routing_ms = sum(v for k, v in per_pass.items() if k in ROUTING_PASSES) * 1000.0
    accounted_ms = sum(per_pass.values()) * 1000.0
    slowest = max(per_pass.items(), key=lambda kv: kv[1]) if per_pass else ("", 0.0)

    return dict(
        out=out,
        total_ms=total_ms,
        layout_ms=layout_ms,
        routing_ms=routing_ms,
        accounted_ms=accounted_ms,
        layout_frac=(layout_ms / total_ms) if total_ms > 0 else None,
        vf2layout_ms=per_pass.get("VF2Layout", 0.0) * 1000.0,
        sabrelayout_ms=per_pass.get("SabreLayout", 0.0) * 1000.0,
        vf2postlayout_ms=per_pass.get("VF2PostLayout", 0.0) * 1000.0,
        sabreswap_ms=per_pass.get("SabreSwap", 0.0) * 1000.0,
        slowest_pass=slowest[0],
        slowest_pass_ms=slowest[1] * 1000.0,
        n_passes=len(per_pass),
        vf2_stop_reason=state["stop_reason"] or "",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--spares", type=str, default="0,1,2,3,4,5,6,8,10,12,16,20,24",
                    help="deliberately fine-grained near zero -- locating the "
                         "edge is the entire point of this script")
    ap.add_argument("--levels", type=str, default="1,3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gates-per-pair", type=int, default=GATES_PER_PAIR)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    grid_size = args.rows * args.cols
    spares = [int(s) for s in args.spares.split(",")]
    levels = [int(l) for l in args.levels.split(",")]
    seeds = list(range(args.seeds))

    print("=" * 100)
    print(f"OCCUPANCY SWEEP: {args.rows}x{args.cols} grid ({grid_size} physical qubits)")
    print(f"spares={spares}  levels={levels}  seeds={seeds}  repeats={args.repeats}")
    print("Measuring Qiskit only -- PSF-Zero is not involved in this experiment.")
    print("=" * 100)

    rows_out = []
    t_start = time.perf_counter()

    for spare in spares:
        n = grid_size - spare
        if n < 2:
            print(f"  spare={spare} skipped (fewer than 2 circuit qubits)")
            continue
        cm = CouplingMap.from_grid(args.rows, args.cols)
        n_pairs = len(range(0, n - 1, 2))
        feas = matching_feasibility(cm, n_pairs)
        occupancy = n / grid_size

        for seed in seeds:
            qc = build_dense_pair_blocks_circuit(n, gates_per_pair=args.gates_per_pair,
                                                 seed=seed)
            for level in levels:
                try:
                    timed_transpile(qc, cm, level)  # warm-up, discarded
                except Exception as exc:
                    print(f"  spare={spare:2d} seed={seed} L{level} WARM-UP FAILED "
                          f"-- {type(exc).__name__}: {exc}")

                for rep in range(args.repeats):
                    base = dict(
                        spare=spare, n=n, occupancy=occupancy, n_pairs=n_pairs,
                        seed=seed, optimization_level=level, repeat=rep,
                        elapsed_since_start_s=time.perf_counter() - t_start,
                        **feas,
                    )
                    try:
                        r = timed_transpile(qc, cm, level)
                        ok, violations = check_routing_validity(r["out"], cm)
                        if not ok:
                            raise RuntimeError(
                                f"output has {violations} coupling violation(s)")
                        rows_out.append(dict(
                            base,
                            time_ms=r["total_ms"],
                            layout_ms=r["layout_ms"],
                            routing_ms=r["routing_ms"],
                            accounted_ms=r["accounted_ms"],
                            layout_frac=r["layout_frac"],
                            vf2layout_ms=r["vf2layout_ms"],
                            sabrelayout_ms=r["sabrelayout_ms"],
                            vf2postlayout_ms=r["vf2postlayout_ms"],
                            sabreswap_ms=r["sabreswap_ms"],
                            slowest_pass=r["slowest_pass"],
                            slowest_pass_ms=r["slowest_pass_ms"],
                            n_passes=r["n_passes"],
                            vf2_stop_reason=r["vf2_stop_reason"],
                            two_qubit_gates=two_qubit_gate_count(r["out"]),
                            depth=r["out"].depth(),
                            error="",
                        ))
                    except Exception as exc:
                        rows_out.append(dict(
                            base, time_ms=None, layout_ms=None, routing_ms=None,
                            accounted_ms=None, layout_frac=None, vf2layout_ms=None,
                            sabrelayout_ms=None, vf2postlayout_ms=None,
                            sabreswap_ms=None, slowest_pass="", slowest_pass_ms=None,
                            n_passes=None, vf2_stop_reason="",
                            two_qubit_gates=None, depth=None,
                            error=f"{type(exc).__name__}: {exc}",
                        ))

            print(f"  spare={spare:2d} (occupancy {occupancy:5.1%}, n={n:3d}) seed={seed} "
                  f"done  (elapsed {time.perf_counter() - t_start:7.1f}s)")

    df = pd.DataFrame(rows_out)
    df["cpu"] = platform.processor()
    df["platform"] = platform.platform()
    df["python_version"] = platform.python_version()
    df["qiskit_version"] = qiskit.__version__
    df["grid_rows"] = args.rows
    df["grid_cols"] = args.cols
    df["grid_size"] = grid_size
    df["gates_per_pair"] = args.gates_per_pair
    df["seed_transpiler"] = SEED_TRANSPILER

    good = df[df.error == ""]
    if not good.empty:
        print("=" * 100)
        print("Median total compile time (ms) by (spare, optimization_level):")
        print(good.pivot_table(index="spare", columns="optimization_level",
                               values="time_ms", aggfunc="median").round(2))
        print()
        print("Median layout-stage fraction of total time:")
        print(good.pivot_table(index="spare", columns="optimization_level",
                               values="layout_frac", aggfunc="median").round(3))
        print()
        print("VF2Layout stop reason by (spare, optimization_level) -- the mechanism:")
        print(good.groupby(["spare", "optimization_level"])["vf2_stop_reason"]
              .agg(lambda s: "/".join(sorted(set(x for x in s if x)) or ["(not run)"])))
        print()
        print("Step-to-step ratio of median time (this is where the cliff shows up):")
        for level in sorted(good.optimization_level.unique()):
            sub = (good[good.optimization_level == level]
                   .groupby("spare")["time_ms"].median().sort_index())
            print(f"  optimization_level={level}")
            prev_spare, prev_val = None, None
            for sp, val in sub.items():
                if prev_val is not None and val > 0:
                    print(f"    spare {prev_spare:2d} -> {sp:2d}: "
                          f"{prev_val:9.2f} -> {val:9.2f} ms   "
                          f"ratio {prev_val / val:7.2f}x")
                prev_spare, prev_val = sp, val
        print()
        print("Feasibility (independent of Qiskit -- prediction P4):")
        print(good.groupby("spare")[["n_pairs_required", "max_matching_size",
                                     "embedding_feasible", "matching_slack"]].first())
        print("=" * 100)

    cpu_tag = (platform.processor() or "unknown_cpu").replace(" ", "_").replace(",", "")
    base_name = (f"occupancy_sweep_{args.rows}x{args.cols}_{cpu_tag}_"
                 f"{date.today().isoformat()}")
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
        "check it for a local file path or any other machine-identifying "
        "string beyond the CPU signature recorded above, per this project's "
        "standing record-keeping rules."
    )


if __name__ == "__main__":
    main()
