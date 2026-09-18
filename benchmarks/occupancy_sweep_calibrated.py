"""occupancy_sweep_calibrated.py -- does `VF2PostLayout` ever return
"solution found" when the target actually carries error rates?

Pre-registered in `spare-qubit-cliff-addendum-50-preregistration-2026-09-18.md`.

## The question

Across 171 instrumented rows on three topology families (square grid,
heavy-hex, synthetic sparse balanced -- Addenda 46-49), `VF2PostLayout`
returned `"no better solution found"` every single time, never once
`"solution found"`. But every one of those runs passed a bare
`CouplingMap`, from which Qiskit builds a `Target` carrying **no error
rates**. `VF2PostLayout`'s entire purpose is finding a *lower-error*
layout, so a device where every qubit and link is equally (and
unknown-ly) good gives it nothing to prefer. **"It never improves
anything" may simply mean "it was never given anything to optimise."**

## Design

Two arms over the same device, `FakeTorino` (133-qubit IBM Heron
snapshot with real calibration data -- also the target Benchpress used,
so this closes that comparison's outstanding gap too):

  - `with_errors`: `transpile(qc, target=backend.target, ...)`, full
    calibration.
  - `no_errors`: `transpile(qc, coupling_map=..., basis_gates=..., ...)`
    on the *same topology and basis*, error rates stripped. This control
    isolates error rates as the only variable -- without it, any
    difference could be blamed on the topology being new.

`spare` is relative to maximum matching capacity (heavy-hex admits no
perfect matching -- Addendum 39). Everything else, including the
per-pass callback instrumentation and the
`vf2post_stop_reason` / `vf2postlayout_ran` columns, is unchanged from
`occupancy_sweep_heavy_hex.py`.

Requires `qiskit-ibm-runtime` (a separate package from `qiskit`). If it
is absent the script exits with an explanation rather than silently
substituting `GenericBackendV2`, whose randomly-generated noise is NOT
equivalent to a real snapshot and would need its own pre-registration.

Usage:
    python occupancy_sweep_calibrated.py --spare-pairs 0,1,2,4,8 \
        --levels 3 --seeds 3 --repeats 2
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

LAYOUT_PASSES = ("VF2Layout", "SabreLayout", "VF2PostLayout", "TrivialLayout",
                 "DenseLayout", "SetLayout", "ApplyLayout", "FullAncillaAllocation",
                 "EnlargeWithAncilla")
ROUTING_PASSES = ("SabreSwap", "StochasticSwap", "BasicSwap", "LookaheadSwap")


def build_dense_pair_blocks_circuit(num_qubits: int, gates_per_pair: int = GATES_PER_PAIR,
                                    seed: int = 0) -> QuantumCircuit:
    """Verbatim from occupancy_sweep.py / this project's cliff-sniper family."""
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


def heavy_hex_matching_structure(cm: CouplingMap) -> dict:
    """Independent (non-Qiskit) characterization of this topology's
    matching capacity -- the Tier-0 check, recomputed every run rather
    than hard-coded, so a change in Qiskit's heavy-hex generator would be
    caught rather than silently assumed unchanged."""
    n = cm.size()
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([tuple(e) for e in cm.get_edges()])
    is_bipartite = nx.is_bipartite(g)
    if is_bipartite:
        part_a, part_b = nx.bipartite.sets(g)
        size_a, size_b = len(part_a), len(part_b)
    else:
        size_a = size_b = None
    m = nx.max_weight_matching(g, maxcardinality=True)
    max_matching = len(m)
    return dict(
        device_qubits=n,
        is_bipartite=is_bipartite,
        bipartite_size_a=size_a,
        bipartite_size_b=size_b,
        bipartite_imbalance=(abs(size_a - size_b) if is_bipartite else None),
        max_matching_pairs=max_matching,
        perfect_matching_exists=bool(max_matching * 2 == n),
    )


def matching_feasibility(max_matching_pairs: int, n_pairs_required: int) -> dict:
    return dict(
        n_pairs_required=n_pairs_required,
        embedding_feasible=bool(max_matching_pairs >= n_pairs_required),
        matching_relative_slack=max_matching_pairs - n_pairs_required,
    )


def timed_transpile(qc, cm, level, target=None, basis_gates=None) -> dict:
    """Verbatim from occupancy_sweep.py."""
    per_pass: dict[str, float] = {}
    state = {"stop_reason": None, "post_stop_reason": None}
    # Which passes appeared in the callback at all -- separates "the pass
    # was skipped" from "the pass ran and returned in ~0 ms", which an
    # exact 0.0 timing alone cannot distinguish (Addenda 43-44).
    passes_seen: set[str] = set()

    def cb(pass_, dag, time, property_set, count, **_):
        name = type(pass_).__name__
        passes_seen.add(name)
        per_pass[name] = per_pass.get(name, 0.0) + float(time)
        reason = property_set.get("VF2Layout_stop_reason")
        if reason is not None:
            state["stop_reason"] = getattr(reason, "value", str(reason))
        post_reason = property_set.get("VF2PostLayout_stop_reason")
        if post_reason is not None:
            state["post_stop_reason"] = getattr(post_reason, "value", str(post_reason))

    t0 = time.perf_counter()
    if target is not None:
        # Calibrated arm: hand transpile() the full Target, including the
        # error rates VF2PostLayout scores against. Passing `target` and
        # `coupling_map`/`basis_gates` together is redundant (the Target
        # already carries both), so only `target` is passed here.
        out = transpile(qc, target=target,
                        optimization_level=level, seed_transpiler=SEED_TRANSPILER,
                        callback=cb)
    else:
        out = transpile(qc, coupling_map=cm, basis_gates=basis_gates,
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
        vf2post_stop_reason=state["post_stop_reason"] or "",
        vf2postlayout_ran=("VF2PostLayout" in passes_seen),
        vf2layout_ran=("VF2Layout" in passes_seen),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=str, default="with_errors,no_errors",
                    help="comma-separated: with_errors (pass FakeTorino's "
                         "full Target, including calibration data) and/or "
                         "no_errors (same topology and basis, error rates "
                         "stripped -- the control isolating error rates as "
                         "the only variable)")
    ap.add_argument("--spare-pairs", type=str, default="0,1,2,3,4,6,8,10,14,20",
                    help="deliberately fine-grained near zero, in PAIRS "
                         "below the graph's own max-matching capacity -- "
                         "see module docstring for why pairs, not raw "
                         "qubit count, is the right unit here")
    ap.add_argument("--levels", type=str, default="1,3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gates-per-pair", type=int, default=GATES_PER_PAIR)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
    except ImportError:
        print("[fatal] qiskit_ibm_runtime is not installed. This experiment "
              "needs a real calibrated device snapshot; a GenericBackendV2 "
              "with randomly-generated noise is NOT equivalent and would "
              "need its own pre-registration (see Addendum 50 Section 4). "
              "Install with: pip install qiskit-ibm-runtime")
        return
    backend = FakeTorino()
    target = backend.target
    cm = target.build_coupling_map()
    device_basis = sorted(target.operation_names)
    n_device_qubits = target.num_qubits
    struct = heavy_hex_matching_structure(cm)
    max_matching = struct["max_matching_pairs"]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print("=" * 100)
    print(f"OCCUPANCY SWEEP (CALIBRATED DEVICE): FakeTorino, "
          f"{n_device_qubits} physical qubits")
    print(f"arms={arms}  (with_errors = full Target incl. calibration; "
          f"no_errors = same topology/basis, error rates stripped)")
    print(f"device basis gates: {device_basis}")
    print(f"bipartite={struct['is_bipartite']}, parts={struct['bipartite_size_a']}/"
          f"{struct['bipartite_size_b']}, imbalance={struct['bipartite_imbalance']}, "
          f"max_matching_pairs={max_matching}, "
          f"perfect_matching_exists={struct['perfect_matching_exists']}")
    print("Measuring Qiskit only -- PSF-Zero is not involved in this experiment.")
    print("=" * 100)

    spare_pairs_list = [int(s) for s in args.spare_pairs.split(",")]
    levels = [int(l) for l in args.levels.split(",")]
    seeds = list(range(args.seeds))

    rows_out = []
    t_start = time.perf_counter()

    for spare_pairs in spare_pairs_list:
        n_pairs_required = max_matching - spare_pairs
        if n_pairs_required < 1:
            print(f"  spare_pairs={spare_pairs} skipped "
                  f"(exceeds max_matching_pairs={max_matching})")
            continue
        n = 2 * n_pairs_required
        feas = matching_feasibility(max_matching, n_pairs_required)
        occupancy_of_matching_capacity = n_pairs_required / max_matching
        occupancy_of_device = n / struct["device_qubits"]

        for seed in seeds:
            qc = build_dense_pair_blocks_circuit(n, gates_per_pair=args.gates_per_pair,
                                                 seed=seed)
            for level in levels:
              for arm in arms:
                arm_target = target if arm == "with_errors" else None
                arm_basis = None if arm == "with_errors" else device_basis
                try:
                    timed_transpile(qc, cm, level, target=arm_target,
                                    basis_gates=arm_basis)  # warm-up, discarded
                except Exception as exc:
                    print(f"  spare_pairs={spare_pairs:2d} seed={seed} L{level} "
                          f"arm={arm} WARM-UP FAILED -- {type(exc).__name__}: {exc}")

                for rep in range(args.repeats):
                    base = dict(
                        arm=arm,
                        spare_pairs=spare_pairs, spare_qubits=2 * spare_pairs,
                        n=n, n_pairs=n_pairs_required,
                        occupancy_of_matching_capacity=occupancy_of_matching_capacity,
                        occupancy_of_device=occupancy_of_device,
                        seed=seed, optimization_level=level, repeat=rep,
                        elapsed_since_start_s=time.perf_counter() - t_start,
                        **feas, **struct,
                    )
                    try:
                        r = timed_transpile(qc, cm, level, target=arm_target,
                                            basis_gates=arm_basis)
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
                            vf2post_stop_reason=r["vf2post_stop_reason"],
                            vf2postlayout_ran=r["vf2postlayout_ran"],
                            vf2layout_ran=r["vf2layout_ran"],
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
                            vf2post_stop_reason="", vf2postlayout_ran=None,
                            vf2layout_ran=None,
                            two_qubit_gates=None, depth=None,
                            error=f"{type(exc).__name__}: {exc}",
                        ))

            print(f"  spare_pairs={spare_pairs:2d} (n_pairs={n_pairs_required:3d}, "
                  f"n={n:3d}, occ_of_capacity={occupancy_of_matching_capacity:5.1%}) "
                  f"seed={seed} done  (elapsed {time.perf_counter() - t_start:7.1f}s)")

    df = pd.DataFrame(rows_out)
    df["cpu"] = platform.processor() or "unknown_cpu"
    df["cpu_model"] = _cpu_model_string()
    df["platform"] = platform.platform()
    df["python_version"] = platform.python_version()
    df["qiskit_version"] = qiskit.__version__
    df["device"] = "FakeTorino"
    df["device_qubits_total"] = n_device_qubits
    df["gates_per_pair"] = args.gates_per_pair
    df["seed_transpiler"] = SEED_TRANSPILER

    good = df[df.error == ""]
    if not good.empty:
        print("=" * 100)
        print("Median total compile time (ms) by (spare_pairs, optimization_level):")
        print(good.pivot_table(index="spare_pairs", columns="optimization_level",
                               values="time_ms", aggfunc="median").round(2))
        print()
        print("VF2Layout stop reason by (spare_pairs, optimization_level):")
        print(good.groupby(["arm", "spare_pairs", "optimization_level"])["vf2_stop_reason"]
              .agg(lambda s: "/".join(sorted(set(x for x in s if x)) or ["(not run)"])))
        print()
        print("VF2PostLayout stop reason by (spare_pairs, optimization_level):")
        print(good.groupby(["arm", "spare_pairs", "optimization_level"])["vf2post_stop_reason"]
              .agg(lambda s: "/".join(sorted(set(x for x in s if x)) or ["(no reason set)"])))
        print()
        print("Did VF2PostLayout run at all (appeared in the pass callback)?")
        print(good.groupby(["arm", "spare_pairs", "optimization_level"])["vf2postlayout_ran"]
              .agg(lambda s: "/".join(sorted(set(str(x) for x in s)))))
        print()
        print("Step-to-step ratio of median time:")
        for level in sorted(good.optimization_level.unique()):
            for arm_name in sorted(good.arm.unique()):
                # Iterate within an arm, not across the (arm, spare_pairs)
                # MultiIndex as a whole -- a step from the last spare value
                # of one arm to the first of the next is not a step at all.
                sub = (good[(good.optimization_level == level)
                            & (good.arm == arm_name)]
                       .groupby("spare_pairs")["time_ms"].median().sort_index())
                print(f"  optimization_level={level}, arm={arm_name}")
                prev_sp, prev_val = None, None
                for sp, val in sub.items():
                    if prev_val is not None and val > 0:
                        print(f"    spare_pairs {prev_sp:2d} -> {sp:2d}: "
                              f"{prev_val:9.2f} -> {val:9.2f} ms   "
                              f"ratio {prev_val / val:7.2f}x")
                    prev_sp, prev_val = sp, val
        print("=" * 100)

    cpu_tag = (df["cpu_model"].iloc[0] if not df.empty else "unknown_cpu")
    cpu_tag = cpu_tag.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    base_name = (f"occupancy_sweep_calibrated_torino_{cpu_tag}_"
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


def _cpu_model_string() -> str:
    """A more informative CPU string than platform.processor(), which
    returns only 'x86_64' on this project's Linux sandbox environments.
    Falls back to platform.processor() if /proc/cpuinfo is unavailable
    (e.g. non-Linux)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown_cpu"


if __name__ == "__main__":
    main()
