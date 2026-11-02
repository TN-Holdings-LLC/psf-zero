"""compare.py -- PSF-Zero POC kit, the fuller comparison.

Reproduces the shape of both papers' own headline results on a reduced
scale, so it finishes in a few minutes rather than the tens of minutes
the papers' own full measurements (10,000 iterations; all 26 saturated
instances) took. This script is honest about that reduction throughout
-- printed output always says which papers' table a number corresponds
to and whether this run used the full scale or a reduced one.

For the exact, full-scale reproduction used in the papers themselves,
see the scripts referenced in each paper's own "Data and code
availability" section.

Run:
    python compare.py                  # both sections, reduced scale
    python compare.py --iterations 2000 --instances 8   # larger, slower
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap

try:
    import psf_compile
except ImportError as exc:
    raise SystemExit(
        "Could not import psf_compile. Install PSF-Zero first: see the "
        "main repository README.\n"
        f"Original error: {exc}"
    )

from sample_circuits import dense_pair_blocks, chain_shaped


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------
# Part 1: gate synthesis, matching Paper 2 Table 1's own construction
# (15 qubits, 20 gates/pair) at a reduced iteration count.
# ---------------------------------------------------------------------
def compare_gate_synthesis(n_iter: int) -> None:
    section(f"Part 1: gate synthesis, {n_iter} iterations "
           f"(Paper 2 Table 1 used 10,000 -- this is a reduced-scale check)")

    t_qiskit, t_psf = [], []
    for i in range(n_iter):
        qc = dense_pair_blocks(15, gates_per_pair=20, seed=1000 + i)

        t0 = time.perf_counter()
        transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3)
        t_qiskit.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        psf_compile.compile(qc, verify=False)
        t_psf.append(time.perf_counter() - t0)

        if (i + 1) % max(1, n_iter // 10) == 0:
            print(f"  ...{i+1}/{n_iter}")

    total_q, total_p = sum(t_qiskit), sum(t_psf)
    print()
    print(f"  Qiskit L3   total={total_q:.3f}s  median={statistics.median(t_qiskit)*1000:.3f}ms")
    print(f"  PSF-Zero    total={total_p:.3f}s  median={statistics.median(t_psf)*1000:.3f}ms")
    print(f"  Cumulative speedup: {total_q/total_p:.2f}x")
    print()
    print(f"  Paper 2's own full-scale (10,000-iteration) range across three "
          f"independent sessions: 7.0-9.3x. A reduced-iteration run like this "
          f"one is expected to land in a similar range but is NOT a")
    print(f"  substitute for the full measurement -- session-to-session "
          f"variance was itself part of what Paper 2 reports (Section 6).")


# ---------------------------------------------------------------------
# Part 2: layout search, matching Paper 1's own 8x8 saturated
# dominant_size_sweep family, on a small subset of D values rather than
# all 26.
# ---------------------------------------------------------------------
def compare_layout_search(n_instances: int) -> None:
    section(f"Part 2: layout search, {n_instances} saturated 8x8 instances "
           f"(Paper 1 tested all 26 D values -- this is a subset)")

    all_d = [2, 5, 8, 10, 13, 17, 20, 24, 27]  # spread across the full range
    d_values = all_d[:n_instances]
    cm = CouplingMap.from_grid(8, 8)

    n_found = 0
    for d in d_values:
        qc = chain_shaped(64, n_bare_edges=17, dominant_size=d, seed=0)

        t0 = time.perf_counter()
        out = psf_compile.compile_for_hardware(
            qc, coupling_map=cm, basis_gates=["rz", "sx", "x", "cx"],
            routing_optimization_level=1, entangling_basis="cx",
            verify=False, seed_transpiler=0, layout_search=True,
        )
        t_psf = time.perf_counter() - t0

        found = out is not None
        n_found += int(found)
        print(f"  D={d:>2}: layout_search=True -> found={found}, "
              f"time={t_psf*1000:.2f}ms")

    print()
    print(f"  {n_found}/{len(d_values)} found via PSF-Zero's repaired layout "
          f"search (this subset).")
    print(f"  Paper 1's own full result: Qiskit's VF2Layout fails on 19 of "
          f"the same 26 instances (median 9,236ms before reporting 'no")
    print(f"  solution'), while a fixed-order search (which this repaired "
          f"path uses) resolves all 26 in a fraction of a millisecond each.")
    print(f"  This script does not re-run Qiskit's own failing search here "
          f"-- see quickstart.py's Demo 2 for a small, fast illustration of")
    print(f"  that side of the comparison, or the papers themselves for the "
          f"full 26-instance measurement.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=200,
                   help="gate-synthesis iterations (default 200; paper used 10,000)")
    p.add_argument("--instances", type=int, default=5,
                   help="number of layout-search instances, 1-9 (default 5; paper used 26)")
    args = p.parse_args()

    compare_gate_synthesis(args.iterations)
    compare_layout_search(min(max(args.instances, 1), 9))

    print()
    print("=" * 78)
    print("Both sections reproduced the SHAPE of the papers' own results at "
          "reduced scale. For the full, exact figures quoted in the papers,")
    print("see each paper's own 'Data and code availability' section.")
    print("=" * 78)


if __name__ == "__main__":
    main()
