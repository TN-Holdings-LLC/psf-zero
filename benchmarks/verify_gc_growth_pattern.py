"""verify_gc_growth_pattern.py -- Addendum 106's own follow-up.

Addendum 106's P3 finding (real net growth in 287 generic object types
even after a full `gc.collect()`) was reproduced byte-for-byte on a
second, independent process invocation -- confirming the phenomenon is
deterministic and repeatable, not noise, but NOT distinguishing between
two very different explanations that both predict identical results
across separate process runs:

  - ONE-TIME WARM-UP: various lazily-initialized Cython/Rust-wrapped
    code paths inside Qiskit's own dependencies create long-lived
    objects (interned functions, method wrappers, format caches) the
    first few times they are touched, then stop growing. A fresh
    process re-pays this cost every time it starts, so two separate
    process runs would look identical regardless.
  - GENUINE PER-ITERATION ACCUMULATION: something in the compile path
    keeps creating new, uncollected objects proportional to call count.
    A fresh process would ALSO look identical to another fresh process
    doing the same number of iterations, for the same reason.

**Re-running the same script cannot distinguish these two. Only a
SINGLE process doing MORE iterations, with growth measured at multiple
checkpoints along the way, can**: if growth per checkpoint shrinks
toward zero (warm-up), the explanation is benign; if it stays roughly
constant checkpoint to checkpoint, growth is genuinely proportional to
call count.

Usage:
    python verify_gc_growth_pattern.py
"""
from __future__ import annotations

import contextlib
import csv
import gc
import io
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary

import psf_compile

N_QUBITS = 15
GATES_PER_PAIR = 20
CHECKPOINTS = [200, 400, 800, 1600]  # cumulative iteration counts


def build_circuit(seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(N_QUBITS)
    pairs = [(i, i + 1) for i in range(0, N_QUBITS - 1, 2)]
    for (a, b) in pairs:
        for _ in range(GATES_PER_PAIR):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def type_census():
    return Counter(type(o).__name__ for o in gc.get_objects())


def run_compiles(n, seed_offset):
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n):
            qc = build_circuit(seed=seed_offset + i)
            psf_compile.compile(qc, verify=False)


def main():
    gc.enable()
    gc.collect()
    baseline = type_census()

    rows = []
    prior_checkpoint = 0
    prior_growth = {}

    print("=" * 100)
    print("Growth pattern across checkpoints, ONE process, GC enabled -- "
          "does per-block growth shrink (warm-up) or stay constant (real "
          "accumulation)?")
    print("=" * 100)

    for checkpoint in CHECKPOINTS:
        n_new = checkpoint - prior_checkpoint
        run_compiles(n_new, seed_offset=prior_checkpoint)
        gc.collect()
        current = type_census()

        cumulative_growth = {t: current[t] - baseline.get(t, 0)
                             for t in current if current[t] > baseline.get(t, 0)}
        # Growth accrued JUST in this block (not cumulative from the very
        # start) -- this is the number that should shrink toward zero if
        # the cause is warm-up, or stay roughly proportional to n_new if
        # it is genuine per-iteration accumulation.
        block_growth = {t: cumulative_growth.get(t, 0) - prior_growth.get(t, 0)
                        for t in set(cumulative_growth) | set(prior_growth)}
        block_growth = {t: v for t, v in block_growth.items() if v != 0}

        top = sorted(block_growth.items(), key=lambda kv: -kv[1])[:10]
        print(f"\n  Checkpoint at {checkpoint} iterations "
              f"(+{n_new} since last checkpoint):")
        print(f"    Top types by NEW growth in this block alone:")
        for t, v in top:
            per_iter = v / n_new
            print(f"      {t:>28}: +{v:>6}  ({per_iter:.2f}/iteration this block)")
            rows.append(dict(checkpoint=checkpoint, block_size=n_new, type=t,
                             growth_this_block=v, per_iteration_this_block=per_iter,
                             cumulative_growth=cumulative_growth.get(t, 0)))

        prior_growth = cumulative_growth
        prior_checkpoint = checkpoint

    print()
    print("=" * 100)
    print("Reading this: for each type, compare 'per iteration this block' across "
          "checkpoints.")
    print("Shrinking toward 0 across successive checkpoints => one-time warm-up.")
    print("Staying roughly constant across checkpoints => genuine per-iteration "
          "accumulation.")
    print("=" * 100)

    out_path = "gc_growth_pattern_2026-09-20.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "block_size", "type",
                                         "growth_this_block",
                                         "per_iteration_this_block",
                                         "cumulative_growth"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
