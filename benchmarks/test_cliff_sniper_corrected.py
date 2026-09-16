"""test_cliff_sniper_corrected.py -- Qiskit L3 vs PSF-Zero across the
spare-qubit cliff, rewritten to fix two problems found in the uploaded
`test_cliff_sniper.py` and its output (both read before this file was
written).

Problem 1: every "PSF-Zero" number on record is a fabricated placeholder,
not a measurement
------------------------------------------------------------------------
The uploaded script's timing loop was:

    if hasattr(psf_compile, 'run_psf_zero'):
        psf_compile.run_psf_zero(qc, cm)
    elif hasattr(psf_compile, 'compile'):
        psf_compile.compile(qc, cm)          # <-- the branch actually taken
    ...
    except Exception:
        p_time = 0.5                          # <-- silently substituted

The current `psf_compile.py` (VERSION 2026-09-16, read from the project
before writing this file) defines exactly one matching entry point,
`compile(qc, block_gate_floor=12, verify=True, entangling_basis="canonical",
on_unsupported="keep", tol=1e-5)`. It has no `coupling_map` parameter, so
`psf_compile.compile(qc, cm)` puts the CouplingMap object into the
`block_gate_floor` slot. The very first comparison inside that call,
`len(block) > block_gate_floor`, then compares an int to a CouplingMap and
raises TypeError -- for every qubit count, on every run. That exception hits
`except Exception: p_time = 0.5`, which is why all five reported "PSF-Zero"
times were exactly 0.50 ms and why every reported "speedup" (19x-3223x) is
an artifact of dividing Qiskit's real time by a constant, not a real
comparison. Confirmed by the source, not just an inference from the flat
output.

There is also no coupling-map-aware entry point being reached at all in that
version of the test: the function that actually does layout + routing is
`compile_for_hardware(qc, coupling_map, basis_gates=None,
routing_optimization_level=1, seed_transpiler=None, ...)`. That is what this
rewrite calls, by name, with no hasattr() guessing.

Problem 2: the circuit under test is not known to reproduce either side of
the comparison
------------------------------------------------------------------------
The uploaded script's circuit hops between many DIFFERENT qubit pairs
(`qc.cx(k, (k + 5) % n)`). `psf_compile.compile()`'s block collection only
consolidates RUNS of more than `block_gate_floor` (12) gates on the SAME
pair, so a hopping pattern like this mostly reports "0/0 blocks" -- there is
nothing here for PSF-Zero's synthesis path to do, fixed call bug or not.
Separately, this project's own spare-qubit-cliff finding
(`docs/findings/spare-qubit-cliff.md`, `README.md`) was established with a
specific circuit shape and states plainly that a different shape does not
reproduce it: "the effect needs the dense adjacent-pair circuit structure as
well as the saturated map -- a gate-count-matched random_circuit() workload
shows no cliff at all." A cx-hopping circuit was never tested against that
claim and has no basis for assuming it lands on the "dense adjacent-pair"
side of that distinction.

This rewrite uses `build_dense_pair_blocks_circuit()`, copied from this
project's own `qiskit-issue-draft.md` / `phase3_v6_workload_control.py`
(the SAME generator used to originally establish the cliff), instead of an
untested ad hoc shape.

What else changed, and why
------------------------------------------------------------------------
- No silent fallback numbers, anywhere. A failed call is printed in full
  (exception type + message) and recorded as a failed row (blank/NaN in the
  output CSV) -- never replaced with a placeholder that looks like data.
- Each point is timed with one discarded warm-up call plus `--repeats`
  (default 5) timed calls, median reported -- this project's own README
  states its convention as "warm-up call outside the timer ... medians, not
  means," and a single perf_counter() call at the few-to-few-hundred-ms
  scale used here is noisy enough on its own to manufacture an apparent
  trend.
- A small-scale (n=6, no coupling map) Operator-equivalence pre-check runs
  once before the sweep and aborts the whole script if it fails, so a
  broken environment produces a loud error instead of a table of numbers
  that merely look plausible. This is also the ONLY full unitary-equivalence
  check this script performs: Operator() at 38-42 qubits is a 2^38..2^42
  matrix and is not computable, so correctness at the sizes actually swept
  rests on (a) that small-n check, (b) `compile_for_hardware`'s own internal
  `verify=True` reconstruction check (the current default, and -- per this
  file's VERSION 2026-09-16 changelog -- now cheap rather than dominant
  cost), and (c) a coupling-map-validity scan of the routed output performed
  below. None of that is a substitute for full unitary verification at
  scale, and this script does not claim otherwise.
- PSF-Zero fallback/degenerate-block warnings are captured and printed
  per point instead of being left to print to stderr unlabeled or pass
  unnoticed.
- Output is written to a filename that encodes the grid shape, CPU
  signature (`platform.processor()`) and today's date, and the script
  refuses to silently overwrite a same-named file from an earlier run --
  per this project's standing rule against ambiguous/fixed output
  filenames (see this project's Addendum 23, Section 3, for a concrete
  account of what that rule exists to prevent: a period-check script picked
  up an old, differently-conditioned file by default instead of the
  intended one).

Pre-registered prediction (written before this script is ever run against
the real `psf_zero_core`)
------------------------------------------------------------------------
On a saturated coupling map, Qiskit L3's compile time is expected to jump
sharply (this project's own prior measurements range from ~20x to several
hundred x depending on grid size) between 2 spare qubits and 0 spare qubits,
consistent with the VF2Layout -> SabreLayout fallback this project already
documented. Whether PSF-Zero's `compile_for_hardware()` shows the same
cliff, a smaller one, or none at all is an OPEN question this script does
not assume an answer to going in: `compile_for_hardware()` calls the same
Qiskit layout/routing code internally, but at `routing_optimization_level=1`
by default rather than 3, and this project's README states that level 1
"never enters the regime at all" for a related (not identical) blow-up --
that is a claim about a different pass configuration and is not assumed to
transfer here without measuring it. This paragraph is written before the
sweep runs so the prediction cannot be quietly adjusted after seeing the
numbers.

Usage:
    python test_cliff_sniper_corrected.py --rows 6 --cols 7 --repeats 5

Requires the real `psf_zero_core` Rust extension to be built
(`maturin develop --release`) -- this script does not run against, and must
never be silently pointed at, `psf_zero_core_stub` (see psf_compile.py's own
import-time check, which already refuses that substitution).
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

# The real, current PSF-Zero API (psf_compile.py, VERSION 2026-09-16),
# imported directly and by name. If this import fails, the script fails
# loudly right here -- it does NOT fall back to guessing at a different
# API with hasattr() and quietly measuring nothing, which is exactly how
# the previous version of this test went wrong.
from psf_compile import compile as psf_compile_only
from psf_compile import compile_for_hardware

BASIS_GATES = ["rz", "sx", "x", "cx"]
GATES_PER_PAIR = 20
SEED_TRANSPILER = 42


def build_dense_pair_blocks_circuit(num_qubits: int, gates_per_pair: int = GATES_PER_PAIR,
                                     seed: int = 0) -> QuantumCircuit:
    """Adjacent pairs (0,1), (2,3), ... each carry a deep chain of random
    2-qubit unitaries, decomposed to elementary gates so Collect2qBlocks
    sees a long same-pair run rather than a single opaque UnitaryGate.

    Copied from this project's own qiskit-issue-draft.md /
    phase3_v6_workload_control.py (`build_dense_pair_blocks_circuit`) -- the
    generator already used to establish the spare-qubit-cliff finding --
    rather than invented fresh for this test. See module docstring for why
    that matters.
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
    """Every 2-qubit gate in a routed circuit must land on a coupled edge.
    A cheap, independent sanity check -- it does not trust anything Qiskit
    or psf_compile reported about their own output."""
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
    """The only full unitary-equivalence check this script performs -- see
    module docstring for why 38-42 qubits cannot get the same treatment."""
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
            "Small-scale correctness pre-check FAILED -- refusing to run the "
            "large-n timing sweep against a build that fails its own "
            "equivalence check. Fix psf_compile / the environment first "
            "rather than trusting timings from it."
        )
    print()


def timed_median(fn, repeats: int):
    """One discarded warm-up call, then `repeats` timed calls; returns
    (median_ms, None) or (None, exception) -- never a substituted placeholder
    value on failure."""
    try:
        fn()  # warm-up, outside the timer
    except Exception as exc:  # noqa: BLE001 - deliberately broad, but reported, never hidden
        return None, exc
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            return None, exc
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--spare-range", type=int, default=4,
                     help="sweep n from grid_size - this to grid_size")
    ap.add_argument("--routing-optimization-level", type=int, default=1,
                     help="passed to psf_compile.compile_for_hardware's "
                          "routing_optimization_level. Deliberately NOT "
                          "forced to match Qiskit's optimization_level=3 -- "
                          "see module docstring for why level 2/3 there "
                          "would just re-run Qiskit's own passes on top of "
                          "PSF-Zero's, per this project's own README.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    grid_size = args.rows * args.cols
    cm = CouplingMap.from_grid(args.rows, args.cols)

    small_scale_correctness_precheck(seed=args.seed)

    print("=" * 88)
    print(f"CLIFF SNIPER TEST (corrected): Grid = {args.rows}x{args.cols} ({grid_size} qubits)")
    print(f"Qiskit L3 (optimization_level=3) vs PSF-Zero "
          f"compile_for_hardware(routing_optimization_level="
          f"{args.routing_optimization_level})")
    print("Circuit: dense adjacent-pair blocks (the family verified to "
          "reproduce this project's spare-qubit-cliff finding), not an "
          "untested ad hoc shape.")
    print(f"Each point: 1 discarded warm-up call + {args.repeats} timed "
          f"calls, median reported. seed_transpiler={SEED_TRANSPILER} "
          f"(pinned).")
    print("=" * 88)

    rows_out = []
    for n in range(max(1, grid_size - args.spare_range), grid_size + 1):
        spare = grid_size - n
        qc = build_dense_pair_blocks_circuit(n, seed=args.seed)

        def run_qiskit(qc=qc):
            out = transpile(qc, coupling_map=cm, basis_gates=BASIS_GATES,
                             optimization_level=3, seed_transpiler=SEED_TRANSPILER)
            ok, violations = check_routing_validity(out, cm)
            if not ok:
                raise RuntimeError(f"Qiskit L3 output has {violations} coupling-map violation(s)")

        fallback_warnings = []

        def run_psf(qc=qc):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = compile_for_hardware(
                    qc, coupling_map=cm, basis_gates=BASIS_GATES,
                    routing_optimization_level=args.routing_optimization_level,
                    entangling_basis="cx",  # basis_gates above is CX-based; match it
                    seed_transpiler=SEED_TRANSPILER,
                )
            fallback_warnings.extend(caught)
            ok, violations = check_routing_validity(out, cm)
            if not ok:
                raise RuntimeError(f"PSF-Zero output has {violations} coupling-map violation(s)")

        q_ms, q_err = timed_median(run_qiskit, args.repeats)
        p_ms, p_err = timed_median(run_psf, args.repeats)

        if q_err is not None:
            print(f"  n={n:3d} spare={spare:2d} | Qiskit L3 FAILED -- {type(q_err).__name__}: {q_err}")
        if p_err is not None:
            print(f"  n={n:3d} spare={spare:2d} | PSF-Zero  FAILED -- {type(p_err).__name__}: {p_err}")
        for w in fallback_warnings:
            print(f"  n={n:3d} spare={spare:2d} | [PSF-Zero fallback] {w.message}")

        speedup = (q_ms / p_ms) if (q_ms is not None and p_ms not in (None, 0)) else None
        q_str = f"{q_ms:10.3f} ms" if q_ms is not None else "     FAILED"
        p_str = f"{p_ms:10.3f} ms" if p_ms is not None else "     FAILED"
        s_str = f"{speedup:8.2f}x" if speedup is not None else "     N/A"
        print(f"n={n:3d} | spare={spare:2d} | Qiskit L3 {q_str} | PSF-Zero {p_str} | speedup {s_str}")

        rows_out.append(dict(
            n=n, spare=spare, grid_rows=args.rows, grid_cols=args.cols,
            qiskit_l3_ms=q_ms, qiskit_l3_error=str(q_err) if q_err else "",
            psf_zero_ms=p_ms, psf_zero_error=str(p_err) if p_err else "",
            psf_zero_fallback_count=len(fallback_warnings),
            speedup=speedup, repeats=args.repeats,
            routing_optimization_level=args.routing_optimization_level,
            seed=args.seed, seed_transpiler=SEED_TRANSPILER,
            cpu=platform.processor(), python_version=platform.python_version(),
            qiskit_version=qiskit.__version__,
        ))

    print("=" * 88)

    # routing_optimization_level is baked into the filename (not just the run2/
    # run3 collision-avoidance suffix below) so that running this script twice
    # in one day at rl=1 and rl=2/3 -- exactly the follow-up this project's
    # Addendum 24 called for -- produces two self-describing filenames rather
    # than two files distinguishable only by opening them and reading a column.
    cpu_tag = (platform.processor() or "unknown_cpu").replace(" ", "_").replace(",", "")
    base_name = (
        f"cliff_sniper_corrected_{args.rows}x{args.cols}_"
        f"rl{args.routing_optimization_level}_{cpu_tag}_{date.today().isoformat()}"
    )
    out_path = os.path.join(args.out_dir, base_name + ".csv")
    if os.path.exists(out_path):
        i = 2
        while os.path.exists(os.path.join(args.out_dir, f"{base_name}_run{i}.csv")):
            i += 1
        out_path = os.path.join(args.out_dir, f"{base_name}_run{i}.csv")

    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(
        "Before pasting this file's contents anywhere outside this machine: "
        "check it for a local file path (e.g. C:\\Users\\...) or any other "
        "machine-identifying string beyond the CPU signature recorded above, "
        "per this project's standing record-keeping rules."
    )


if __name__ == "__main__":
    main()
