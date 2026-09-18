"""synthetic_sparse_balanced_cliff.py -- isolates degree/sparsity from the
"never reached 100% device occupancy" confound in Addendum 40's heavy-hex
null result.

**Motivation (Addendum 41 pre-registration, written BEFORE this script was
run at full scale).** Real heavy-hex, per Addendum 39, cannot admit a
perfect matching at any tested size (its bipartite parts are unbalanced),
so the dense-pair-blocks circuit family used throughout this project can
never actually saturate 100% of a heavy-hex device -- the hardest
achievable case (Addendum 40) tops out at ~83-84% device occupancy. Every
square-grid measurement that showed the occupancy cliff (Addenda 4-35),
by contrast, was measured at genuine 100% device occupancy
(`n_circuit == n_device`), and Addendum 34 pinned the cliff to exactly
that single point. Addendum 40's "no cliff on heavy-hex" result is
therefore confounded: it could be because heavy-hex's low degree/sparsity
protects against the cliff mechanism, or it could simply be an artifact
of never having tested the actual danger zone at all.

**This script builds a synthetic control that heavy-hex itself cannot
be**: a random bipartite graph with heavy-hex's own degree distribution
(mostly degree-2, a smaller share of degree-1 and degree-3, max degree 3)
but with BALANCED parts, so a perfect matching -- and therefore a true,
zero-slack `spare=0` at 100% device occupancy -- genuinely exists. If the
cliff appears here at spare=0, heavy-hex's real-world immunity is fully
explained by the occupancy ceiling (Addendum 39), and low degree confers
no independent protection. If it does not appear even here, low degree/
sparsity itself is doing real protective work.

Graph construction: `networkx.bipartite.configuration_model` with random
per-side degree sequences drawn from heavy-hex's own approximate
proportions, retried with new random draws until the result is simple
(no multi-edges/self-loops), connected, and admits a perfect matching.
This is independent of, and does not modify, the real heavy-hex graph.

`build_dense_pair_blocks_circuit`, `check_routing_validity`,
`two_qubit_gate_count`, and `timed_transpile` are copied verbatim from
`occupancy_sweep.py` / `occupancy_sweep_heavy_hex.py` so results are
structurally comparable across all three (grid, heavy-hex,
synthetic-sparse-balanced) experiments.

Pre-registered predictions: see
`spare-qubit-cliff-addendum-41-preregistration-2026-09-18.md`, written
BEFORE this script was run at full scale. Do not edit that document after
seeing this script's output.

## Revision 2026-09-18: VF2PostLayout instrumentation added

Every addendum from 34 onward recorded only `VF2Layout_stop_reason`.
`VF2PostLayout` sets its own, separate property-set key with a different
four-value enum (`solution found`, `no better solution found`,
`nonexistent solution`, `>2q gates in basis` -- see Addendum 44's reading
of `vf2_post_layout.py`). Without it, Addendum 43's three distinct
observed behaviours -- grid (both passes clear together), synthetic sparse
(they clear at different occupancies), heavy-hex (`VF2PostLayout`
constant at 0.2-0.4 ms, never exactly zero) -- had to be inferred from
timing alone, and an exact `0.0` reading was ambiguous between "the pass
was skipped" and "the pass ran and returned instantly."

Three columns are added, none of which change what is measured or how:

  - `vf2post_stop_reason`: the four-value enum above.
  - `vf2postlayout_ran` / `vf2layout_ran`: whether the pass appeared in
    the transpiler callback at all, which separates "did not run" from
    "ran in ~0 ms" directly rather than by inference.

Usage:
    python synthetic_sparse_balanced_cliff.py --n-per-side 58 --degree-seed 42 \\
        --spares 0,1,2,4,8,16 --levels 3 --seeds 5 --repeats 3
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from collections import Counter
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

# Heavy-hex d=7's own approximate per-node degree proportions (Addendum 39):
# degree 1: 2/115 = 1.7%, degree 2: 77/115 = 67.0%, degree 3: 36/115 = 31.3%.
HEAVY_HEX_DEGREE_PROBS = {1: 0.02, 2: 0.65, 3: 0.33}


def build_dense_pair_blocks_circuit(num_qubits: int, gates_per_pair: int = GATES_PER_PAIR,
                                    seed: int = 0) -> QuantumCircuit:
    """Verbatim from occupancy_sweep.py / occupancy_sweep_heavy_hex.py."""
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


def make_synthetic_sparse_balanced(n_per_side: int, seed: int, max_tries: int = 5000,
                                    max_degree: int = 3):
    """Build a balanced bipartite graph matching heavy-hex's own degree
    distribution (HEAVY_HEX_DEGREE_PROBS), retried until the result is
    simple, connected, and admits a perfect matching. Returns
    (networkx.Graph, avg_degree, n_tries, degree_distribution_dict) or
    (None, None, max_tries, None) on failure."""
    rng = np.random.default_rng(seed)
    probs = [HEAVY_HEX_DEGREE_PROBS[d] for d in (1, 2, 3)]
    for attempt in range(max_tries):
        deg_a = rng.choice([1, 2, 3], size=n_per_side, p=probs)
        deg_b = rng.choice([1, 2, 3], size=n_per_side, p=probs)
        diff = int(deg_a.sum()) - int(deg_b.sum())
        if diff != 0:
            idx = rng.integers(0, n_per_side)
            if diff > 0:
                deg_b[idx] = min(max_degree, deg_b[idx] + abs(diff))
            else:
                deg_a[idx] = min(max_degree, deg_a[idx] + abs(diff))
        if int(deg_a.sum()) != int(deg_b.sum()):
            continue  # nudge above didn't converge (degree cap hit); retry fresh draw
        try:
            g = nx.bipartite.configuration_model(
                list(deg_a), list(deg_b), seed=int(rng.integers(0, 2**31)))
            g = nx.Graph(g)
            g.remove_edges_from(nx.selfloop_edges(g))
        except Exception:
            continue
        if g.number_of_nodes() != 2 * n_per_side:
            continue
        if not nx.is_connected(g):
            continue
        m = nx.bipartite.maximum_matching(g, top_nodes=range(n_per_side))
        matching_size = len(m) // 2
        if matching_size != n_per_side:
            continue
        avg_deg = 2 * g.number_of_edges() / g.number_of_nodes()
        deg_dist = dict(Counter(dict(g.degree()).values()))
        return g, avg_deg, attempt, deg_dist
    return None, None, max_tries, None


def timed_transpile(qc: QuantumCircuit, cm: CouplingMap, level: int) -> dict:
    """Verbatim from occupancy_sweep.py."""
    per_pass: dict[str, float] = {}
    state = {"stop_reason": None, "post_stop_reason": None}
    # Which passes actually appeared in the callback at all. A pass that
    # never appears here did not run; a pass that appears with ~0.0 time
    # ran and returned immediately. Addenda 42-43 could not tell those two
    # cases apart -- an exact 0.0 in `vf2postlayout_ms` was ambiguous
    # between "skipped" and "ran, returned instantly" -- because only
    # accumulated times were recorded, never the fact of invocation.
    passes_seen: set[str] = set()

    def cb(pass_, dag, time, property_set, count, **_):
        name = type(pass_).__name__
        passes_seen.add(name)
        per_pass[name] = per_pass.get(name, 0.0) + float(time)
        reason = property_set.get("VF2Layout_stop_reason")
        if reason is not None:
            state["stop_reason"] = getattr(reason, "value", str(reason))
        # VF2PostLayout sets its own, separate property-set key with a
        # four-value enum (SOLUTION_FOUND, NO_BETTER_SOLUTION_FOUND,
        # NO_SOLUTION_FOUND, MORE_THAN_2Q) -- see Addendum 44. Every
        # addendum from 34 onward recorded only VF2Layout's key, which is
        # why the three distinct behaviours catalogued in Addendum 43 had
        # to be inferred from timing alone.
        post_reason = property_set.get("VF2PostLayout_stop_reason")
        if post_reason is not None:
            state["post_stop_reason"] = getattr(post_reason, "value", str(post_reason))

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
        out=out, total_ms=total_ms, layout_ms=layout_ms, routing_ms=routing_ms,
        accounted_ms=accounted_ms,
        layout_frac=(layout_ms / total_ms) if total_ms > 0 else None,
        vf2layout_ms=per_pass.get("VF2Layout", 0.0) * 1000.0,
        sabrelayout_ms=per_pass.get("SabreLayout", 0.0) * 1000.0,
        vf2postlayout_ms=per_pass.get("VF2PostLayout", 0.0) * 1000.0,
        sabreswap_ms=per_pass.get("SabreSwap", 0.0) * 1000.0,
        slowest_pass=slowest[0], slowest_pass_ms=slowest[1] * 1000.0,
        n_passes=len(per_pass), vf2_stop_reason=state["stop_reason"] or "",
        vf2post_stop_reason=state["post_stop_reason"] or "",
        vf2postlayout_ran=("VF2PostLayout" in passes_seen),
        vf2layout_ran=("VF2Layout" in passes_seen),
    )


def _cpu_model_string() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown_cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-side", type=int, default=58,
                    help="nodes per bipartite side; total device qubits = 2x this")
    ap.add_argument("--degree-seed", type=int, default=42,
                    help="seed for the synthetic graph's own construction "
                         "(independent of the circuit seeds below)")
    ap.add_argument("--spares", type=str, default="0,1,2,4,8,16")
    ap.add_argument("--levels", type=str, default="3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gates-per-pair", type=int, default=GATES_PER_PAIR)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    g, avg_deg, n_tries, deg_dist = make_synthetic_sparse_balanced(
        args.n_per_side, args.degree_seed)
    if g is None:
        raise SystemExit(f"Failed to construct a valid synthetic graph after "
                          f"{n_tries} tries -- try a different --degree-seed.")

    n_device = g.number_of_nodes()
    cm = CouplingMap(list(g.edges()) + [(b, a) for a, b in g.edges()])

    print("=" * 100)
    print(f"SYNTHETIC SPARSE-BALANCED CLIFF TEST: n_per_side={args.n_per_side}, "
          f"{n_device} device qubits, degree_seed={args.degree_seed}")
    print(f"avg_degree={avg_deg:.3f} (heavy-hex d=7: 2.296, d=5: 2.246), "
          f"degree_dist={deg_dist}, connected=True, perfect_matching=True "
          f"(verified during construction), construction_tries={n_tries}")
    print("This graph is NOT heavy-hex; it is an independent synthetic control "
          "matched on size and degree statistics, with BALANCED bipartite parts "
          "so a true spare=0 (100% device occupancy) is achievable, unlike real "
          "heavy-hex (Addendum 39).")
    print("Measuring Qiskit only -- PSF-Zero is not involved in this experiment.")
    print("=" * 100)

    spares = [int(s) for s in args.spares.split(",")]
    levels = [int(l) for l in args.levels.split(",")]
    seeds = list(range(args.seeds))

    rows_out = []
    t_start = time.perf_counter()

    for spare in spares:
        n = n_device - spare
        if n < 2 or n % 2 != 0:
            print(f"  spare={spare} skipped (n={n} invalid -- must be even and >=2)")
            continue
        occupancy = n / n_device

        for seed in seeds:
            qc = build_dense_pair_blocks_circuit(n, gates_per_pair=args.gates_per_pair,
                                                 seed=seed)
            for level in levels:
                try:
                    timed_transpile(qc, cm, level)  # warm-up, discarded
                except Exception as exc:
                    print(f"  spare={spare:2d} seed={seed} L{level} "
                          f"WARM-UP FAILED -- {type(exc).__name__}: {exc}")

                for rep in range(args.repeats):
                    base = dict(
                        spare=spare, n=n, n_device=n_device, occupancy=occupancy,
                        avg_degree=avg_deg, degree_seed=args.degree_seed,
                        n_construction_tries=n_tries,
                        seed=seed, optimization_level=level, repeat=rep,
                        elapsed_since_start_s=time.perf_counter() - t_start,
                    )
                    try:
                        r = timed_transpile(qc, cm, level)
                        ok, violations = check_routing_validity(r["out"], cm)
                        if not ok:
                            raise RuntimeError(
                                f"output has {violations} coupling violation(s)")
                        rows_out.append(dict(
                            base, time_ms=r["total_ms"], layout_ms=r["layout_ms"],
                            routing_ms=r["routing_ms"], accounted_ms=r["accounted_ms"],
                            layout_frac=r["layout_frac"], vf2layout_ms=r["vf2layout_ms"],
                            sabrelayout_ms=r["sabrelayout_ms"],
                            vf2postlayout_ms=r["vf2postlayout_ms"],
                            sabreswap_ms=r["sabreswap_ms"], slowest_pass=r["slowest_pass"],
                            slowest_pass_ms=r["slowest_pass_ms"], n_passes=r["n_passes"],
                            vf2_stop_reason=r["vf2_stop_reason"],
                            vf2post_stop_reason=r["vf2post_stop_reason"],
                            vf2postlayout_ran=r["vf2postlayout_ran"],
                            vf2layout_ran=r["vf2layout_ran"],
                            two_qubit_gates=two_qubit_gate_count(r["out"]),
                            depth=r["out"].depth(), error="",
                        ))
                    except Exception as exc:
                        rows_out.append(dict(
                            base, time_ms=None, layout_ms=None, routing_ms=None,
                            accounted_ms=None, layout_frac=None, vf2layout_ms=None,
                            sabrelayout_ms=None, vf2postlayout_ms=None,
                            sabreswap_ms=None, slowest_pass="", slowest_pass_ms=None,
                            n_passes=None, vf2_stop_reason="",
                            vf2post_stop_reason="", vf2postlayout_ran=None,
                            vf2layout_ran=None, two_qubit_gates=None,
                            depth=None, error=f"{type(exc).__name__}: {exc}",
                        ))

            print(f"  spare={spare:2d} (n={n:3d}, occupancy={occupancy:5.1%}) "
                  f"seed={seed} done  (elapsed {time.perf_counter() - t_start:7.1f}s)")

    df = pd.DataFrame(rows_out)
    df["cpu"] = platform.processor() or "unknown_cpu"
    df["cpu_model"] = _cpu_model_string()
    df["platform"] = platform.platform()
    df["python_version"] = platform.python_version()
    df["qiskit_version"] = qiskit.__version__
    df["n_per_side"] = args.n_per_side
    df["gates_per_pair"] = args.gates_per_pair
    df["seed_transpiler"] = SEED_TRANSPILER

    good = df[df.error == ""] if ("error" in df.columns and not df.empty) else df.iloc[0:0]
    if not good.empty:
        print("=" * 100)
        print("Median total compile time (ms) by (spare, optimization_level):")
        print(good.pivot_table(index="spare", columns="optimization_level",
                               values="time_ms", aggfunc="median").round(2))
        print()
        print("VF2Layout stop reason by (spare, optimization_level):")
        print(good.groupby(["spare", "optimization_level"])["vf2_stop_reason"]
              .agg(lambda s: "/".join(sorted(set(x for x in s if x)) or ["(not run)"])))
        print()
        print("VF2PostLayout stop reason by (spare, optimization_level):")
        print(good.groupby(["spare", "optimization_level"])["vf2post_stop_reason"]
              .agg(lambda s: "/".join(sorted(set(x for x in s if x)) or ["(no reason set)"])))
        print()
        print("Did VF2PostLayout run at all (appeared in the pass callback)?")
        print(good.groupby(["spare", "optimization_level"])["vf2postlayout_ran"]
              .agg(lambda s: "/".join(sorted(set(str(x) for x in s)))))
        print()
        print("Step-to-step ratio of median time:")
        for level in sorted(good.optimization_level.unique()):
            sub = (good[good.optimization_level == level]
                   .groupby("spare")["time_ms"].median().sort_index())
            print(f"  optimization_level={level}")
            prev_sp, prev_val = None, None
            for sp, val in sub.items():
                if prev_val is not None and val > 0:
                    print(f"    spare {prev_sp:2d} -> {sp:2d}: "
                          f"{prev_val:9.2f} -> {val:9.2f} ms   "
                          f"ratio {prev_val / val:7.2f}x")
                prev_sp, prev_val = sp, val
        print("=" * 100)

    cpu_tag = (df["cpu_model"].iloc[0] if not df.empty else "unknown_cpu")
    cpu_tag = cpu_tag.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    base_name = (f"synthetic_sparse_balanced_cliff_n{n_device}_{cpu_tag}_"
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
