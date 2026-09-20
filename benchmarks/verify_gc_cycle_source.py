"""verify_gc_cycle_source.py -- Addendum 105.

Addendum 94 found disabling GC during 10,000 `psf_compile()` calls
grows RSS by 2,273.4MB, confirming reference cycles are created at
meaningful volume, but did not identify what object type is involved.
This locates it via `gc.get_objects()` type-census (baseline: does full
collection reclaim everything, or is something genuinely leaking?),
`gc.collect()`'s own steady-state reclaim rate (cycle census: is the
rate bounded, matching Addendum 94's own +56.1MB-over-10,000-iterations
finding for normal GC-enabled operation?), and `gc.DEBUG_SAVEALL` (cycle
capture: what IS the garbage, concretely?).

Uses a much smaller loop (200 iterations) than Addendum 94's own
10,000 -- enough to see a clear signal without an unwieldy object
census.

Usage:
    python verify_gc_cycle_source.py
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
N_ITER = 200


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
    """Counter of live object counts by type name."""
    return Counter(type(o).__name__ for o in gc.get_objects())


def run_compiles(n, seed_offset=0):
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n):
            qc = build_circuit(seed=seed_offset + i)
            psf_compile.compile(qc, verify=False)


def section_p3_baseline():
    """P3: does a full collection reclaim everything a non-growing
    workload creates, or does something genuinely leak past even
    gc.collect()?"""
    print("=" * 100)
    print("Baseline (P3): does gc.collect() reclaim everything after a "
          "fixed-size, non-growing workload?")
    print("=" * 100)
    gc.enable()
    gc.collect()
    before = type_census()

    run_compiles(N_ITER, seed_offset=0)
    gc.collect()
    after = type_census()

    growth = Counter()
    for t, c_after in after.items():
        delta = c_after - before.get(t, 0)
        if delta > 0:
            growth[t] = delta
    top = growth.most_common(15)
    print(f"  Object types with net growth after full collection "
          f"(top 15 of {len(growth)} with growth > 0):")
    for t, d in top:
        print(f"    {t:>30}: +{d}")
    if not growth:
        print("    (none -- no net growth in any type; matches Addendum "
              "94's 'not a true leak' reading)")
    print()
    return growth


def section_p2_cycle_census():
    """P2: with GC enabled (normal operation), how many unreachable
    objects does each gc.collect() call reclaim, in batches, and does
    the rate stay bounded?"""
    print("=" * 100)
    print("Cycle census (P2): gc.collect()'s own reclaimed-object count, "
          "per batch, GC enabled throughout")
    print("=" * 100)
    gc.enable()
    gc.collect()
    batch = 20
    for start in range(0, N_ITER, batch):
        run_compiles(batch, seed_offset=start)
        n_collected = gc.collect()
        print(f"  iterations {start:>4}-{start+batch-1:>4}: "
              f"gc.collect() reclaimed {n_collected} objects")
    print()


def section_p1_capture():
    """P1: what IS the garbage? Disable GC, run the workload, collect
    once with DEBUG_SAVEALL so gc.garbage retains what would otherwise
    be freed, and inspect it by type."""
    print("=" * 100)
    print("Cycle capture (P1): what object types make up the "
          "uncollected garbage?")
    print("=" * 100)
    gc.disable()
    gc.set_debug(gc.DEBUG_SAVEALL)
    del gc.garbage[:]

    run_compiles(N_ITER, seed_offset=10_000)  # disjoint seeds from the other sections

    gc.collect()
    garbage_types = Counter(type(o).__name__ for o in gc.garbage)
    top = garbage_types.most_common(20)
    print(f"  gc.garbage contains {len(gc.garbage)} objects after one "
          f"collection with DEBUG_SAVEALL.")
    print(f"  Top 20 types by count:")
    for t, c in top:
        print(f"    {t:>30}: {c}")

    # For the single most common type, trace one example's own referrers
    # to identify what holds the cycle -- best-effort, since referrer
    # objects can themselves be uninformative reprs.
    if top:
        target_type = top[0][0]
        example = next(o for o in gc.garbage if type(o).__name__ == target_type)
        referrers = gc.get_referrers(example)
        referrer_types = Counter(type(r).__name__ for r in referrers)
        print(f"\n  Referrers of one '{target_type}' instance "
              f"(what holds a reference to it):")
        for t, c in referrer_types.most_common(10):
            print(f"    {t:>30}: {c}")

    gc.set_debug(0)
    del gc.garbage[:]
    gc.enable()
    print()
    return garbage_types


def main():
    rows = []

    growth = section_p3_baseline()
    for t, d in growth.items():
        rows.append(dict(section="P3_baseline_net_growth", type=t, count=d))

    section_p2_cycle_census()

    garbage_types = section_p1_capture()
    for t, c in garbage_types.items():
        rows.append(dict(section="P1_capture_garbage_by_type", type=t, count=c))

    if not rows:
        rows.append(dict(section="none", type="none", count=0))

    out_path = "gc_cycle_source_2026-09-20.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "type", "count"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
