"""cross_compiler_cliff.py -- is the coupling-map saturation cliff a
Qiskit bug, or a property of placement-by-bounded-subgraph-isomorphism?

This is the highest-value open question left by Addenda 4-34. If only
Qiskit degrades as device occupancy approaches 100%, the finding is an
issue report about one codebase (and is already partly filed upstream as
#7705 / #8667). If TKET degrades in the same place, the finding is about
the *technique*, and applies to any compiler that places qubits by
searching for an embedding of the circuit's interaction graph into the
device graph.

The comparison is like-for-like because both tools do structurally the
same thing at this stage, under different budget shapes:

  Qiskit  VF2Layout      -- VF2 subgraph isomorphism, bounded by a
                            CALL COUNT (`call_limit`), no wall-clock cap
  TKET    GraphPlacement -- subgraph monomorphism, bounded by BOTH
                            `maximum_matches` (default 1000) and a
                            wall-clock `timeout` (default 1000 ms)

That difference in budget shape is itself a pre-registered prediction
(Addendum 35, P3): a wall-clock timeout caps the damage, a call limit
does not, because individual calls get slower exactly when the instance
gets hard.

Placement and routing are timed **separately** on the TKET side, so
TKET's placement stage can be compared against Qiskit's layout stage
rather than against Qiskit's entire pipeline. Heavy peephole
optimisation (`FullPeepholeOptimise`) is deliberately OFF by default --
this experiment is about the placement stage, and folding in unrelated
circuit rewriting would make the comparison meaningless. `--with-peephole`
enables it for a separate, clearly-labelled arm if wanted.

Gate counts are not directly comparable across tools unless both outputs
are in the same gate set, so the TKET arms rebase to the Qiskit basis
in a separately-timed step (`rebase_ms`) that is excluded from the
placement/routing comparison.

**This script does not import or exercise PSF-Zero.** It measures Qiskit
and TKET. Requires qiskit, pytket, pytket-qiskit, numpy, pandas.

Pre-registered predictions: see
`spare-qubit-cliff-addendum-35-preregistration-2026-09-17.md`, written
BEFORE this script was run at full scale. Do not edit that document
after seeing this script's output.

Usage:
    python cross_compiler_cliff.py --rows 6 --cols 7 \\
        --spares 0,1,2,4,8,16 --seeds 3 --repeats 3
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from datetime import date

import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap

import pytket
from pytket.architecture import Architecture
from pytket.circuit import OpType
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.passes import (AutoRebase, DefaultMappingPass, FullPeepholeOptimise,
                           PlacementPass, RoutingPass)
from pytket.placement import GraphPlacement

BASIS_GATES = ["rz", "sx", "x", "cx"]
TKET_BASIS = {OpType.Rz, OpType.SX, OpType.X, OpType.CX}
GATES_PER_PAIR = 20
SEED_TRANSPILER = 42


def build_dense_pair_blocks_circuit(num_qubits: int, gates_per_pair: int = GATES_PER_PAIR,
                                    seed: int = 0) -> QuantumCircuit:
    """Verbatim from this project's cliff-sniper family -- do not 'improve'."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for a, b in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


def qiskit_two_qubit_gates(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


def check_qiskit_routing(qc: QuantumCircuit, cm: CouplingMap) -> int:
    edges = set(map(tuple, cm.get_edges()))
    violations = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            i = qc.find_bit(inst.qubits[0]).index
            j = qc.find_bit(inst.qubits[1]).index
            if (i, j) not in edges and (j, i) not in edges:
                violations += 1
    return violations


def _node_index(q) -> int | None:
    """TKET qubits carry an index list, e.g. node[3] -> [3]."""
    try:
        return int(q.index[0])
    except Exception:
        return None


def check_tket_routing(circ, cm: CouplingMap) -> int:
    """Count 2-qubit commands that land off a coupling-map edge.

    Returns -1 if the circuit's qubits could not be resolved to physical
    indices at all, which is itself a result worth recording rather than
    silently treating as 'valid'.
    """
    edges = set(map(tuple, cm.get_edges()))
    violations = 0
    for cmd in circ:
        qs = cmd.qubits
        if len(qs) == 2:
            i, j = _node_index(qs[0]), _node_index(qs[1])
            if i is None or j is None:
                return -1
            if (i, j) not in edges and (j, i) not in edges:
                violations += 1
    return violations


def tket_two_qubit_gates(circ) -> int:
    return sum(1 for cmd in circ if len(cmd.qubits) == 2)


def run_qiskit(qc: QuantumCircuit, cm: CouplingMap, level: int) -> dict:
    t0 = time.perf_counter()
    out = transpile(qc, coupling_map=cm, basis_gates=BASIS_GATES,
                    optimization_level=level, seed_transpiler=SEED_TRANSPILER)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return dict(total_ms=total_ms, placement_ms=None, routing_ms=None,
                rebase_ms=None, two_qubit_gates=qiskit_two_qubit_gates(out),
                depth=out.depth(), violations=check_qiskit_routing(out, cm))


def run_tket_placement_routing(qc: QuantumCircuit, cm: CouplingMap, arch: Architecture,
                               timeout_ms: int, max_matches: int,
                               with_peephole: bool) -> dict:
    circ = qiskit_to_tk(qc)
    t_peep = 0.0
    if with_peephole:
        t0 = time.perf_counter()
        FullPeepholeOptimise().apply(circ)
        t_peep = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    PlacementPass(GraphPlacement(arch, maximum_matches=max_matches,
                                 timeout=timeout_ms)).apply(circ)
    placement_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    RoutingPass(arch).apply(circ)
    routing_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    AutoRebase(TKET_BASIS).apply(circ)
    rebase_ms = (time.perf_counter() - t0) * 1000.0

    return dict(total_ms=placement_ms + routing_ms, placement_ms=placement_ms,
                routing_ms=routing_ms, rebase_ms=rebase_ms, peephole_ms=t_peep,
                two_qubit_gates=tket_two_qubit_gates(circ), depth=circ.depth(),
                violations=check_tket_routing(circ, cm))


def run_tket_default(qc: QuantumCircuit, cm: CouplingMap, arch: Architecture) -> dict:
    circ = qiskit_to_tk(qc)
    t0 = time.perf_counter()
    DefaultMappingPass(arch).apply(circ)
    total_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    AutoRebase(TKET_BASIS).apply(circ)
    rebase_ms = (time.perf_counter() - t0) * 1000.0
    return dict(total_ms=total_ms, placement_ms=None, routing_ms=None,
                rebase_ms=rebase_ms, two_qubit_gates=tket_two_qubit_gates(circ),
                depth=circ.depth(), violations=check_tket_routing(circ, cm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--spares", type=str, default="0,1,2,4,8,16")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gates-per-pair", type=int, default=GATES_PER_PAIR)
    ap.add_argument("--tket-timeout", type=int, default=1000,
                    help="GraphPlacement wall-clock timeout in ms (pytket "
                         "default is 1000; prediction P3 is about this cap)")
    ap.add_argument("--tket-max-matches", type=int, default=1000)
    ap.add_argument("--with-peephole", action="store_true",
                    help="add FullPeepholeOptimise to the TKET arm, timed "
                         "separately; OFF by default because this experiment "
                         "is about the placement stage, not optimisation")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    grid_size = args.rows * args.cols
    spares = [int(s) for s in args.spares.split(",")]
    seeds = list(range(args.seeds))

    print("=" * 100)
    print(f"CROSS-COMPILER CLIFF: {args.rows}x{args.cols} grid ({grid_size} qubits)")
    print(f"spares={spares}  seeds={seeds}  repeats={args.repeats}")
    print(f"TKET GraphPlacement: timeout={args.tket_timeout}ms, "
          f"maximum_matches={args.tket_max_matches}")
    print("Measuring Qiskit and TKET -- PSF-Zero is not involved in this experiment.")
    print("=" * 100)

    rows_out = []
    t_start = time.perf_counter()

    for spare in spares:
        n = grid_size - spare
        if n < 2:
            continue
        cm = CouplingMap.from_grid(args.rows, args.cols)
        arch = Architecture([tuple(e) for e in cm.get_edges()])
        occupancy = n / grid_size

        arms = [
            ("qiskit_opt1", lambda qc: run_qiskit(qc, cm, 1)),
            ("qiskit_opt3", lambda qc: run_qiskit(qc, cm, 3)),
            ("tket_placement_routing",
             lambda qc: run_tket_placement_routing(qc, cm, arch, args.tket_timeout,
                                                   args.tket_max_matches,
                                                   args.with_peephole)),
            ("tket_default_mapping", lambda qc: run_tket_default(qc, cm, arch)),
        ]

        for seed in seeds:
            qc = build_dense_pair_blocks_circuit(n, gates_per_pair=args.gates_per_pair,
                                                 seed=seed)
            for arm_name, fn in arms:
                try:
                    fn(qc)  # warm-up, discarded
                except Exception as exc:
                    print(f"  spare={spare:2d} seed={seed} {arm_name} WARM-UP FAILED "
                          f"-- {type(exc).__name__}: {exc}")

                for rep in range(args.repeats):
                    base = dict(spare=spare, n=n, occupancy=occupancy, seed=seed,
                                arm=arm_name, repeat=rep,
                                elapsed_since_start_s=time.perf_counter() - t_start)
                    try:
                        r = fn(qc)
                        rows_out.append(dict(base, **{k: v for k, v in r.items()},
                                             error=""))
                    except Exception as exc:
                        rows_out.append(dict(
                            base, total_ms=None, placement_ms=None, routing_ms=None,
                            rebase_ms=None, two_qubit_gates=None, depth=None,
                            violations=None,
                            error=f"{type(exc).__name__}: {exc}"))

            print(f"  spare={spare:2d} (occupancy {occupancy:5.1%}) seed={seed} done "
                  f"(elapsed {time.perf_counter() - t_start:7.1f}s)")

    df = pd.DataFrame(rows_out)
    df["cpu"] = platform.processor()
    df["platform"] = platform.platform()
    df["python_version"] = platform.python_version()
    df["qiskit_version"] = qiskit.__version__
    df["pytket_version"] = pytket.__version__
    df["grid_rows"] = args.rows
    df["grid_cols"] = args.cols
    df["gates_per_pair"] = args.gates_per_pair
    df["tket_timeout_ms"] = args.tket_timeout
    df["tket_max_matches"] = args.tket_max_matches
    df["with_peephole"] = args.with_peephole

    good = df[df.error == ""]
    if not good.empty:
        print("=" * 100)
        print("Median total time (ms) by (spare, arm):")
        print(good.pivot_table(index="spare", columns="arm", values="total_ms",
                               aggfunc="median").round(2))
        print()
        print("Median TKET placement time (ms) alone -- the direct analogue of "
              "Qiskit's VF2Layout stage:")
        tk = good[good.placement_ms.notna()]
        if not tk.empty:
            print(tk.pivot_table(index="spare", columns="arm", values="placement_ms",
                                 aggfunc="median").round(2))
        print()
        print("Cliff ratio (spare=0 median / largest-spare median), per arm:")
        lo, hi = min(spares), max(spares)
        for arm in sorted(good.arm.unique()):
            a = good[(good.arm == arm) & (good.spare == lo)]["total_ms"].median()
            b = good[(good.arm == arm) & (good.spare == hi)]["total_ms"].median()
            if a and b:
                print(f"  {arm:24s} spare={lo}: {a:9.2f} ms   spare={hi}: {b:9.2f} ms"
                      f"   ratio {a / b:8.2f}x")
        print()
        print("Median 2-qubit gate count by (spare, arm) -- prediction P4:")
        print(good.pivot_table(index="spare", columns="arm", values="two_qubit_gates",
                               aggfunc="median"))
        bad = good[good.violations != 0]
        print()
        print(f"Coupling-map violations: {len(bad)} row(s) with non-zero "
              f"(note -1 means qubit indices were unresolvable, not zero violations)")
        if not bad.empty:
            print(bad.groupby("arm")["violations"].agg(["count", "min", "max"]))
        print("=" * 100)

    cpu_tag = (platform.processor() or "unknown_cpu").replace(" ", "_").replace(",", "")
    base_name = (f"cross_compiler_cliff_{args.rows}x{args.cols}_{cpu_tag}_"
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
