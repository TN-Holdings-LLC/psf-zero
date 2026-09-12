"""Does finding a layout early shorten the pass, or does the trial loop burn the
budget anyway?

Background: the seed scan showed `VF2Layout` succeeding on 4 of 30 shuffle seeds on a
saturated 6x7 grid, with elapsed time flat across both outcomes (failures 335.6 ms
median, successes 339.2 ms). Reading `minimize_vf2` in
crates/transpiler/src/passes/vf2_layout.rs suggests why: it does not stop at the first
match. It takes the first mapping, then keeps iterating under `max_trials` (default
`None` -> `15 + max(needle.edge_count(), haystack.edge_count())`) looking for a
better-scoring one. That is a reading of the source, not a measurement.

This script measures it. `max_trials=1` stops after the first match; the default keeps
going. If the trial loop is what costs the time, the successful seeds should get much
faster with `max_trials=1` and the failing seeds should not change (they never find a
first match, so there is nothing to stop after).

Pre-registered predictions, written before running:
  P1  successful seeds (1, 8, 25, 29): max_trials=1 is markedly faster than default.
  P2  failing seeds: max_trials=1 is within noise of default.
  P3  stop reasons are unchanged for every seed under both settings.
If P1 fails, the trial loop is not the explanation and the flat timing needs another
one.

Usage:
    python verify_vf2_max_trials.py
Writes vf2_max_trials_2026-09-11.csv.
"""

import csv
import platform
import sys
import time
from collections import Counter

import qiskit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import VF2Layout

from phase3_v5_spare_qubits import build_dense_pair_blocks_circuit

ROWS, COLS = 6, 7
N_QUBITS, GATES_PER_PAIR = 42, 20
CALL_LIMIT = 3_000_000
REPS = 3

# The four seeds that found a layout in the earlier scan, plus four that did not.
SUCCESS_SEEDS = [1, 8, 25, 29]
FAILURE_SEEDS = [0, 2, 3, 4]

# `None` is the pass default (15 + max edge count); 1 stops after the first match.
MAX_TRIALS_ARMS = [None, 1]

ENV = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python": sys.version.split()[0],
    "qiskit": qiskit.__version__,
}


def run_one(dag, cmap, seed, max_trials):
    """One timed run. Returns (stop_reason, elapsed_ms) using the minimum of REPS."""
    kwargs = {"coupling_map": cmap, "seed": seed, "call_limit": CALL_LIMIT}
    if max_trials is not None:
        kwargs["max_trials"] = max_trials

    best_ms = float("inf")
    reason = None
    for _ in range(REPS):
        vf2 = VF2Layout(**kwargs)
        t0 = time.perf_counter()
        vf2.run(dag)
        best_ms = min(best_ms, (time.perf_counter() - t0) * 1000.0)
        reason = str(vf2.property_set.get("VF2Layout_stop_reason"))
    return reason, best_ms


def main():
    print(ENV)
    print(
        f"grid {ROWS}x{COLS} ({ROWS * COLS} physical), circuit {N_QUBITS} qubits "
        f"(spare {ROWS * COLS - N_QUBITS}), call_limit {CALL_LIMIT:,}, "
        f"min of {REPS} reps\n"
    )

    cmap = CouplingMap.from_grid(ROWS, COLS)
    dag = circuit_to_dag(build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR))

    rows = []
    for group, seeds in (("success", SUCCESS_SEEDS), ("failure", FAILURE_SEEDS)):
        for seed in seeds:
            line = {"group": group, "seed": seed}
            for max_trials in MAX_TRIALS_ARMS:
                reason, ms = run_one(dag, cmap, seed, max_trials)
                label = "default" if max_trials is None else str(max_trials)
                line[f"ms_{label}"] = round(ms, 3)
                line[f"reason_{label}"] = reason
            line["ratio_default_over_1"] = round(line["ms_default"] / line["ms_1"], 3)
            line["reason_changed"] = line["reason_default"] != line["reason_1"]
            print(
                f"{group:>7} seed {seed:>3}: "
                f"default {line['ms_default']:8.2f} ms | "
                f"max_trials=1 {line['ms_1']:8.2f} ms | "
                f"ratio {line['ratio_default_over_1']:6.2f}x"
                + ("   REASON CHANGED" if line["reason_changed"] else "")
            )
            rows.append(
                {
                    **line,
                    "rows": ROWS,
                    "cols": COLS,
                    "circuit_qubits": N_QUBITS,
                    "call_limit": CALL_LIMIT,
                    "reps": REPS,
                    "pass": "VF2Layout",
                    **ENV,
                }
            )

    print("\n--- summary ---")
    for group in ("success", "failure"):
        sub = [r for r in rows if r["group"] == group]
        ratios = sorted(r["ratio_default_over_1"] for r in sub)
        mid = ratios[len(ratios) // 2]
        print(f"{group:>7}: median default/max_trials=1 = {mid:.2f}x  (all: {ratios})")

    changed = [r["seed"] for r in rows if r["reason_changed"]]
    print(f"stop reason changed for seeds: {changed if changed else 'none'}")
    print(
        "reason counts:",
        dict(Counter(r["reason_default"].rsplit(".", 1)[-1] for r in rows)),
    )

    out = "vf2_max_trials_2026-09-11.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
