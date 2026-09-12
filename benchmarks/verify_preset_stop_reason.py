"""Does the preset draw a fresh node ordering on every transpile() call?

The earlier timing test (verify_preset_shuffle.py) could not answer this. A lucky
ordering does not make the pass fast: verify_vf2_max_trials.py measured seed 1 finding
its layout in 3.5 ms while the pass still ran for 343 ms, because the trial loop burns
the budget afterwards. So counting fast calls detects nothing either way.

This reads the outcome directly instead of inferring it from the clock. After each
run the preset's property set carries `VF2Layout_stop_reason` and
`VF2PostLayout_stop_reason`, which say whether a layout was found, regardless of how
long the pass took.

Background on why the question is open: `Vf2PassConfiguration::from_legacy_api` in
crates/transpiler/src/passes/vf2_layout.rs treats a `None` seed as "seed with OS
entropy" and `-1` as "no shuffling". Which of those the preset pass managers pass has
not been checked, and the standalone scan found 4 of 30 shuffle seeds do find a
layout on this input.

Pre-registered predictions, written before running:
  P1  every unpinned call reports NO_SOLUTION_FOUND.
  P2  every pinned call reports NO_SOLUTION_FOUND, identically.
  P3  if P1 fails, the SOLUTION_FOUND fraction is near 4/30 (13%), i.e. ~4 of 30.

Reading the outcomes:
  - No SOLUTION_FOUND in 30 unpinned calls puts the all-miss probability at
    (26/30)^30 = 1.4% if the preset were drawing orderings at the standalone rate.
    That is evidence the preset is not drawing a fresh ordering per call, and the
    cliff is deterministic for a given input.
  - Any SOLUTION_FOUND that still takes ~6.8 s is stronger evidence for the trial
    loop: the layout was found and the time was spent anyway.

Runtime: each call is ~7 s, so 30 unpinned + 10 pinned is roughly 5 minutes.

Usage:
    python verify_preset_stop_reason.py
Writes preset_stop_reason_2026-09-12.csv.
"""

import csv
import platform
import statistics
import sys
import time
from collections import Counter

import qiskit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from phase3_v5_spare_qubits import build_dense_pair_blocks_circuit

ROWS, COLS = 6, 7
N_QUBITS, GATES_PER_PAIR = 42, 20
BASIS = ["rz", "sx", "x", "cx"]
OPT_LEVEL = 3
UNPINNED_CALLS = 30
PINNED_CALLS = 10

ENV = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python": sys.version.split()[0],
    "qiskit": qiskit.__version__,
}


def short(reason) -> str:
    """'VF2LayoutStopReason.NO_SOLUTION_FOUND' -> 'NO_SOLUTION_FOUND'."""
    if reason is None:
        return "None"
    return str(reason).rsplit(".", 1)[-1]


def main():
    print(ENV)
    print(
        f"grid {ROWS}x{COLS} ({ROWS * COLS} physical), circuit {N_QUBITS} qubits "
        f"(spare {ROWS * COLS - N_QUBITS}), optimization_level={OPT_LEVEL}\n"
    )

    cmap = CouplingMap.from_grid(ROWS, COLS)
    qc = build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR)

    # Warm up the preset machinery outside the measured calls.
    warm = generate_preset_pass_manager(
        optimization_level=1, coupling_map=cmap, basis_gates=BASIS
    )
    warm.run(qc)

    rows = []
    arms = [("unpinned", None, UNPINNED_CALLS), ("pinned", 0, PINNED_CALLS)]

    for arm, seed, calls in arms:
        print(f"--- {arm} ({calls} calls) ---")
        for call in range(calls):
            kwargs = {
                "optimization_level": OPT_LEVEL,
                "coupling_map": cmap,
                "basis_gates": BASIS,
            }
            if seed is not None:
                kwargs["seed_transpiler"] = seed
            pm = generate_preset_pass_manager(**kwargs)

            t0 = time.perf_counter()
            out = pm.run(qc)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            layout = short(pm.property_set.get("VF2Layout_stop_reason"))
            post = short(pm.property_set.get("VF2PostLayout_stop_reason"))
            two_q = sum(1 for inst in out.data if inst.operation.num_qubits == 2)

            print(
                f"  call {call:>2}: {elapsed_ms:9.1f} ms  "
                f"VF2Layout {layout:<22} VF2PostLayout {post:<26} "
                f"2q {two_q:>4} depth {out.depth():>4}"
                + ("   <-- FOUND" if layout == "SOLUTION_FOUND" else "")
            )
            rows.append(
                {
                    "arm": arm,
                    "call": call,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "vf2layout_stop_reason": layout,
                    "vf2postlayout_stop_reason": post,
                    "two_qubit_gates": two_q,
                    "depth": out.depth(),
                    "rows": ROWS,
                    "cols": COLS,
                    "circuit_qubits": N_QUBITS,
                    "optimization_level": OPT_LEVEL,
                    "seed_transpiler": "" if seed is None else seed,
                    **ENV,
                }
            )

    print("\n--- summary ---")
    for arm, _, _ in arms:
        sub = [r for r in rows if r["arm"] == arm]
        times = [r["elapsed_ms"] for r in sub]
        print(f"\n{arm}: {len(sub)} calls, median {statistics.median(times):.1f} ms")
        print("  VF2Layout    :", dict(Counter(r["vf2layout_stop_reason"] for r in sub)))
        print("  VF2PostLayout:", dict(Counter(r["vf2postlayout_stop_reason"] for r in sub)))

    unpinned = [r for r in rows if r["arm"] == "unpinned"]
    found = [r for r in unpinned if r["vf2layout_stop_reason"] == "SOLUTION_FOUND"]
    print(
        f"\nP1 (no SOLUTION_FOUND when unpinned): "
        + ("HELD" if not found else f"FAILED — {len(found)}/{len(unpinned)} found")
    )
    if not found:
        print(
            f"  all-miss probability at the standalone 4/30 rate: "
            f"{(26 / 30) ** len(unpinned):.4f}"
        )
    else:
        ft = [r["elapsed_ms"] for r in found]
        others = [r["elapsed_ms"] for r in unpinned if r not in found]
        print(f"  found-call times: {[round(x) for x in ft]} ms")
        print(f"  median of the rest: {statistics.median(others):.1f} ms")
        print("  -> a found layout that still costs seconds points at the trial loop")

    out_path = "preset_stop_reason_2026-09-12.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()