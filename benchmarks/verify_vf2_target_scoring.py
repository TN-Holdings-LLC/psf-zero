"""Was the 4-in-30 hit rate an artifact of the dummy target?

The standalone scan (verify_vf2_seed.py) passed `coupling_map=` to `VF2Layout`, so the
pass built a target through `_build_dummy_target` — basis `["u", "cx"]`, no error
rates. With no errors anywhere, `build_average_error_map` in
crates/transpiler/src/passes/vf2_layout.rs falls through to its legacy branch and
scores nodes by degree instead.

The preset pass manager passes a real target. Scoring drives the pruning, and pruning
drives the search, so the two runs may not have been solving the same problem — which
would explain 4/30 standalone against 0/30 through the preset without any seed being
involved.

This runs the same seed scan twice, changing only how the target is supplied:

  dummy   VF2Layout(coupling_map=cmap, ...)                  -> _build_dummy_target
  real    VF2Layout(target=Target.from_configuration(...))   -> what the preset uses

Everything else is identical: same circuit, same grid, same call limit, same seeds.

Pre-registered predictions, written before running:
  P1  the dummy arm reproduces the earlier scan: SOLUTION_FOUND on seeds 1, 8, 25, 29.
  P2  if scoring is what separates the two earlier results, the real arm finds a
      layout on far fewer seeds — possibly zero.
  P3  if the real arm also gives 4/30 on the same seeds, scoring is not the
      difference, and the preset's 0/30 needs another explanation.

Runtime: 60 runs at ~335 ms each, under a minute.

Usage:
    python verify_vf2_target_scoring.py
Writes vf2_target_scoring_2026-09-12.csv.
"""

import csv
import platform
import statistics
import sys
import time
from collections import Counter

import qiskit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.passes import VF2Layout

from phase3_v5_spare_qubits import build_dense_pair_blocks_circuit

ROWS, COLS = 6, 7
N_QUBITS, GATES_PER_PAIR = 42, 20
BASIS = ["rz", "sx", "x", "cx"]
CALL_LIMIT = 3_000_000
SEEDS = 30

# Seeds that found a layout in the original scan, for a quick like-for-like check.
KNOWN_HITS = {1, 8, 25, 29}

ENV = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python": sys.version.split()[0],
    "qiskit": qiskit.__version__,
}


def short(reason) -> str:
    return "None" if reason is None else str(reason).rsplit(".", 1)[-1]


def main():
    print(ENV)
    print(
        f"grid {ROWS}x{COLS} ({ROWS * COLS} physical), circuit {N_QUBITS} qubits "
        f"(spare {ROWS * COLS - N_QUBITS}), call_limit {CALL_LIMIT:,}, {SEEDS} seeds\n"
    )

    cmap = CouplingMap.from_grid(ROWS, COLS)
    real_target = Target.from_configuration(
        basis_gates=BASIS, num_qubits=cmap.size(), coupling_map=cmap
    )
    dag = circuit_to_dag(build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR))

    rows = []
    for arm in ("dummy", "real"):
        print(f"--- {arm} target ---")
        for seed in range(SEEDS):
            if arm == "dummy":
                vf2 = VF2Layout(coupling_map=cmap, seed=seed, call_limit=CALL_LIMIT)
            else:
                vf2 = VF2Layout(target=real_target, seed=seed, call_limit=CALL_LIMIT)

            t0 = time.perf_counter()
            vf2.run(dag)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            reason = short(vf2.property_set.get("VF2Layout_stop_reason"))

            print(
                f"  seed {seed:>3}: {reason:<20} {elapsed_ms:9.2f} ms"
                + ("   <-- FOUND" if reason == "SOLUTION_FOUND" else "")
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "stop_reason": reason,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "rows": ROWS,
                    "cols": COLS,
                    "circuit_qubits": N_QUBITS,
                    "call_limit": CALL_LIMIT,
                    "basis_gates": " ".join(BASIS) if arm == "real" else "u cx (dummy)",
                    **ENV,
                }
            )

    print("\n--- summary ---")
    hits = {}
    for arm in ("dummy", "real"):
        sub = [r for r in rows if r["arm"] == arm]
        found = sorted(r["seed"] for r in sub if r["stop_reason"] == "SOLUTION_FOUND")
        hits[arm] = set(found)
        times = [r["elapsed_ms"] for r in sub]
        print(
            f"{arm:>6}: {len(found)}/{SEEDS} found, seeds {found}, "
            f"median {statistics.median(times):.1f} ms"
        )
        print(f"        reasons: {dict(Counter(r['stop_reason'] for r in sub))}")

    print(f"\nP1 (dummy reproduces {sorted(KNOWN_HITS)}): "
          + ("HELD" if hits["dummy"] == KNOWN_HITS else f"FAILED — got {sorted(hits['dummy'])}"))
    if hits["real"] == hits["dummy"]:
        print("P3: the two arms agree — scoring is not what separated the earlier results")
    elif not hits["real"]:
        print("P2: the real target finds nothing — scoring, not the seed, explains the preset's 0/30")
    else:
        print(f"partial: real target hits {sorted(hits['real'])}, dummy hits {sorted(hits['dummy'])}")

    out_path = "vf2_target_scoring_2026-09-12.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()