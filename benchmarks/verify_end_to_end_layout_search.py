"""verify_end_to_end_layout_search.py -- Addendum 101.

Addenda 88-95 fixed and verified `smart_vf2_layout()`, the component.
This measures the FEATURE: `compile_for_hardware(layout_search=True)`,
whose own docstring in `psf_compile.py` still says it has never been
benchmarked. Finding a layout is not the same as the compile being
better -- the layout is handed to `transpile()`, which then routes and
optimizes, and a valid layout can still cost search time without
improving the end result.

For chain-shaped circuits this comparison is also the cleanest possible
isolation of Addendum 89's guard fix: before that fix the guard rejected
such inputs before any search ran, so `layout_search=True` was identical
in effect to `False`. Any difference measured here between the two arms
therefore IS the fix's own end-to-end effect -- no code swap needed.

## Arms (all through real public entry points, no internal helpers)

  qiskit_l3    -- transpile(..., optimization_level=3)
  psf_ls_false -- compile_for_hardware(..., layout_search=False)
  psf_ls_true  -- compile_for_hardware(..., layout_search=True)

## Configurations (all spare=0, verified zero idle qubits)

  6x7 dense_pairs  (control: the guard passed even before the fix)
  6x7 chain-shaped (new capability: 11 bare edges + two 10q chains)
  8x8 dense_pairs  (control)
  8x8 chain-shaped (new capability: 17 bare edges + 10q and 20q chains)

## Requirements

Run from the repository directory, so `psf_compile` and
`psf_smart_layout` are the real, installed ones. Needs `qiskit`,
`rustworkx`, `networkx`, `numpy`.

Usage:
    python verify_end_to_end_layout_search.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import statistics
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap

import psf_compile
from psf_smart_layout import smart_vf2_layout

BASIS_GATES = ["rz", "sx", "x", "cx"]
GATES_PER_PAIR = 20
SEEDS = [0, 1, 2]
REPEATS = 2


# ---------------------------------------------------------------------
# Interaction-graph constructions. Both leave zero idle qubits and have
# an interaction-graph maximum matching exactly equal to the grid's own
# -- i.e. genuine spare=0 saturation, verified numerically before this
# script was written rather than assumed from the parameter choice.
# ---------------------------------------------------------------------
def edges_dense_pairs(n):
    return [(i, i + 1) for i in range(0, n - 1, 2)]


def edges_chain_shaped(n, n_bare, dominant):
    filler = (n - n_bare * 2) - dominant
    if filler < 0 or (0 < filler < 2) or dominant < 2:
        raise ValueError(f"invalid: n={n}, n_bare={n_bare}, dominant={dominant}")
    edges, q = [], 0
    for _ in range(n_bare):
        edges.append((q, q + 1))
        q += 2
    for i in range(dominant - 1):
        edges.append((q + i, q + i + 1))
    q += dominant
    if filler > 0:
        for i in range(filler - 1):
            edges.append((q + i, q + i + 1))
    return edges


CONFIGS = [
    ("6x7 dense_pairs",  6, 7, lambda n: edges_dense_pairs(n)),
    ("6x7 chain-shaped", 6, 7, lambda n: edges_chain_shaped(n, 11, 10)),
    ("8x8 dense_pairs",  8, 8, lambda n: edges_dense_pairs(n)),
    ("8x8 chain-shaped", 8, 8, lambda n: edges_chain_shaped(n, 17, 10)),
]


def build_circuit(n_qubits, edges, seed):
    """One random SU(4) block per edge, GATES_PER_PAIR deep, so every
    edge carries real work rather than a single gate."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for (a, b) in edges:
        for _ in range(GATES_PER_PAIR):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def count_coupling_violations(qc, coupling_map):
    """P3's gate: every 2-qubit instruction in the routed output must sit
    on a physically adjacent pair. Checked against the coupling map's own
    edge set rather than trusting that transpile() succeeded."""
    allowed = set()
    for a, b in coupling_map.get_edges():
        allowed.add((a, b))
        allowed.add((b, a))
    violations = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            i = qc.find_bit(inst.qubits[0]).index
            j = qc.find_bit(inst.qubits[1]).index
            if i != j and (i, j) not in allowed:
                violations += 1
    return violations


def count_2q_gates(qc):
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


def run_arm(arm, qc, cm, seed):
    """Returns (elapsed_s, routed_circuit). PSF-Zero's own debug output is
    suppressed so it does not pollute the timing or the console."""
    if arm == "qiskit_l3":
        t0 = time.perf_counter()
        out = transpile(qc, coupling_map=cm, basis_gates=BASIS_GATES,
                        optimization_level=3, seed_transpiler=seed)
        return time.perf_counter() - t0, out

    layout_search = (arm == "psf_ls_true")
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        out = psf_compile.compile_for_hardware(
            qc,
            coupling_map=cm,
            basis_gates=BASIS_GATES,
            routing_optimization_level=1,
            entangling_basis="cx",
            verify=False,
            seed_transpiler=seed,
            layout_search=layout_search,
        )
        elapsed = time.perf_counter() - t0
    return elapsed, out


def main():
    rows = []

    for label, r, c, edge_fn in CONFIGS:
        n = r * c
        cm = CouplingMap.from_grid(r, c)
        edges = edge_fn(n)

        # P4: did the search itself actually succeed on these pairs? The
        # feature discards its own _search_info, so without this there is
        # no way to distinguish a successful search from a silent
        # fall-through to Qiskit's default layout stage.
        pairs = sorted({(min(a, b), max(a, b)) for (a, b) in edges})
        layout_map, info = smart_vf2_layout(cm, pairs, n)
        search_ok = layout_map is not None

        print("=" * 100)
        print(f"{label}  --  {n} qubits, {len(edges)} interaction edges, spare=0")
        print(f"  search check (P4): found={search_ok}, "
              f"feasible={info.get('feasible')}, "
              f"tried={info.get('orderings_tried')}, "
              f"order={info.get('order_name')}")
        print("=" * 100)
        print(f"{'arm':>14} {'time (ms)':>22} {'2q gates':>10} {'depth':>8} {'violations':>11}")

        # Warm-up, outside every timer, once per configuration per arm.
        warm = build_circuit(n, edges, seed=999)
        for arm in ("qiskit_l3", "psf_ls_false", "psf_ls_true"):
            run_arm(arm, warm, cm, seed=999)

        per_arm = {}
        for arm in ("qiskit_l3", "psf_ls_false", "psf_ls_true"):
            times, gates, depths, viols = [], set(), set(), 0
            for seed in SEEDS:
                qc = build_circuit(n, edges, seed=seed)
                for _ in range(REPEATS):
                    el, out = run_arm(arm, qc, cm, seed)
                    times.append(el)
                    gates.add(count_2q_gates(out))
                    depths.add(out.depth())
                    viols += count_coupling_violations(out, cm)
            med = statistics.median(times)
            per_arm[arm] = med
            g = sorted(gates)
            d = sorted(depths)
            g_str = str(g[0]) if len(g) == 1 else f"{g[0]}-{g[-1]}"
            d_str = str(d[0]) if len(d) == 1 else f"{d[0]}-{d[-1]}"
            print(f"{arm:>14} {med*1000:>15.2f} (n={len(times)}) "
                  f"{g_str:>10} {d_str:>8} {viols:>11}")

            rows.append(dict(
                config=label, arm=arm, search_found=search_ok,
                median_ms=med * 1000, min_ms=min(times) * 1000,
                max_ms=max(times) * 1000, n_runs=len(times),
                gates_2q=g_str, depth=d_str, coupling_violations=viols,
            ))

        t_f, t_t = per_arm["psf_ls_false"], per_arm["psf_ls_true"]
        print(f"  P1: layout_search True vs False -- "
              f"{t_f*1000:.2f}ms vs {t_t*1000:.2f}ms "
              f"({t_f/t_t:.2f}x {'faster with True' if t_t < t_f else 'SLOWER with True'})")
        print()

    out_path = "end_to_end_layout_search_2026-09-20.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total_viol = sum(r["coupling_violations"] for r in rows)
    print("=" * 100)
    print(f"P3 gate: total coupling-map violations across every arm and run "
          f"= {total_viol} (0 required; any other value invalidates that "
          f"arm's timings)")
    print("Note: gate count and depth are shown as a single value when every "
          "seed and repeat agreed,")
    print("and as a range otherwise -- a range where the README's own table "
          "recorded zero spread is")
    print("itself worth noticing.")
    print("=" * 100)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
