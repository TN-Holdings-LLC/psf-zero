"""Is the cliff deterministic, or does the preset roll a die every call?

Background: the seed scan showed `VF2Layout` finding a layout on 4 of 30 shuffle
seeds on a saturated 6x7 grid. `Vf2PassConfiguration::from_legacy_api` in
crates/transpiler/src/passes/vf2_layout.rs treats a `None` seed as "seed with OS
entropy" and `-1` as "no shuffling":

    None => Some(Pcg64Mcg::try_from_rng(&mut SysRng).unwrap().next_u64()),
    Some(-1) => None,

If the preset pass managers leave the seed unset, every `transpile()` call would draw
a fresh node ordering, and roughly 1 call in 7 should land on a lucky one and finish
fast. Dozens of measurements of the saturated case have never produced a fast one,
which does not fit. This measures it directly instead of arguing from the source.

Two arms, same circuit, same coupling map:
  unpinned   transpile(...) with no seed_transpiler    -> entropy, if the preset uses it
  pinned     transpile(..., seed_transpiler=0)         -> fixed

Pre-registered predictions, written before running:
  P1  the unpinned arm is slow on every call; no run finishes in the tens of ms.
  P2  pinned and unpinned have the same median, within ordinary run-to-run noise.
  P3  if P1 fails and some calls are fast, the fast fraction is near 4/30 (13%).
P1 holding means the preset does not entropy-shuffle, so the cliff is deterministic
for a given input. P1 failing means the cliff is probabilistic and every timing of it
in this project is a sample from a mixture.

Runtime: each slow call is ~7-9 s, so 20 calls per arm is roughly 5 minutes.

Usage:
    python verify_preset_shuffle.py
Writes preset_shuffle_2026-09-11.csv.
"""

import csv
import platform
import statistics
import sys
import time

import qiskit
from qiskit import transpile
from qiskit.transpiler import CouplingMap

from phase3_v5_spare_qubits import build_dense_pair_blocks_circuit

ROWS, COLS = 6, 7
N_QUBITS, GATES_PER_PAIR = 42, 20
BASIS = ["rz", "sx", "x", "cx"]
OPT_LEVEL = 3
CALLS = 20
FAST_THRESHOLD_MS = 1000.0  # anything under a second is "did not hit the cliff"

ENV = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python": sys.version.split()[0],
    "qiskit": qiskit.__version__,
}


def main():
    print(ENV)
    print(
        f"grid {ROWS}x{COLS} ({ROWS * COLS} physical), circuit {N_QUBITS} qubits "
        f"(spare {ROWS * COLS - N_QUBITS}), optimization_level={OPT_LEVEL}, "
        f"{CALLS} calls per arm\n"
    )

    cmap = CouplingMap.from_grid(ROWS, COLS)
    qc = build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR)

    # Warm up the preset pass manager machinery outside the timed calls.
    transpile(qc, coupling_map=cmap, basis_gates=BASIS, optimization_level=1)

    rows = []
    for arm in ("unpinned", "pinned"):
        print(f"--- {arm} ---")
        for call in range(CALLS):
            kwargs = {
                "coupling_map": cmap,
                "basis_gates": BASIS,
                "optimization_level": OPT_LEVEL,
            }
            if arm == "pinned":
                kwargs["seed_transpiler"] = 0

            t0 = time.perf_counter()
            out = transpile(qc, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            two_q = sum(1 for inst in out.data if inst.operation.num_qubits == 2)
            fast = elapsed_ms < FAST_THRESHOLD_MS
            print(
                f"  call {call:>2}: {elapsed_ms:9.1f} ms  "
                f"2q gates {two_q:>4}  depth {out.depth():>4}"
                + ("   <-- FAST" if fast else "")
            )
            rows.append(
                {
                    "arm": arm,
                    "call": call,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "fast": fast,
                    "two_qubit_gates": two_q,
                    "depth": out.depth(),
                    "rows": ROWS,
                    "cols": COLS,
                    "circuit_qubits": N_QUBITS,
                    "optimization_level": OPT_LEVEL,
                    **ENV,
                }
            )

    print("\n--- summary ---")
    for arm in ("unpinned", "pinned"):
        vals = [r["elapsed_ms"] for r in rows if r["arm"] == arm]
        n_fast = sum(1 for r in rows if r["arm"] == arm and r["fast"])
        print(
            f"{arm:>8}: median {statistics.median(vals):9.1f} ms  "
            f"min {min(vals):9.1f}  max {max(vals):9.1f}  "
            f"fast {n_fast}/{len(vals)} ({100 * n_fast / len(vals):.0f}%)"
        )

    unp = [r["elapsed_ms"] for r in rows if r["arm"] == "unpinned"]
    pin = [r["elapsed_ms"] for r in rows if r["arm"] == "pinned"]
    print(
        f"unpinned / pinned median ratio: "
        f"{statistics.median(unp) / statistics.median(pin):.3f}x"
    )
    total_fast = sum(1 for r in rows if r["fast"])
    print(
        "\nP1 (no fast calls in the unpinned arm): "
        + ("HELD" if not any(r["fast"] for r in rows if r["arm"] == "unpinned") else "FAILED")
    )
    print(f"fast calls overall: {total_fast}/{len(rows)}  (4/30 seeds = 13% would be ~2.6/20)")

    out_path = "preset_shuffle_2026-09-11.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()