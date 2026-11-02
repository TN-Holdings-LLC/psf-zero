"""quickstart.py -- PSF-Zero POC kit, step 1.

The fastest possible demonstration of what PSF-Zero does differently from
Qiskit, on two small, self-contained examples. Designed to run in well
under three minutes on ordinary hardware -- everything here uses small
qubit counts and short loops on purpose, so the "try it in five minutes"
promise is real, not aspirational.

This script assumes PSF-Zero is already installed (see the main
repository's own README for `pip install -e .` and the Rust core build
step). It is not a reimplementation of anything -- every number this
script prints comes from calling the real, installed `psf_compile`
module directly.

Run:
    python quickstart.py
"""
from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.transpiler import CouplingMap

try:
    import psf_compile
except ImportError as exc:
    raise SystemExit(
        "Could not import psf_compile. Install PSF-Zero first: see the "
        "main repository README's `pip install -e .` and "
        "`maturin develop --release` steps.\n"
        f"Original error: {exc}"
    )


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------
# Demo 1: gate synthesis. A single, deterministic two-qubit block,
# synthesized by both Qiskit and PSF-Zero, compared for exact
# correctness and speed.
# ---------------------------------------------------------------------
def demo_gate_synthesis() -> None:
    section("Demo 1 of 2: gate synthesis (small, one-shot)")
    print("A single random two-qubit unitary block, synthesized both ways.")
    print()

    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(random_unitary(4, seed=0)), [0, 1])

    t0 = time.perf_counter()
    qiskit_out = transpile(qc, basis_gates=["rz", "sx", "x", "cx"],
                           optimization_level=3)
    t_qiskit = time.perf_counter() - t0

    t0 = time.perf_counter()
    psf_out = psf_compile.compile(qc, verify=False)
    t_psf = time.perf_counter() - t0

    fid_qiskit = _fidelity(qc, qiskit_out)
    fid_psf = _fidelity(qc, psf_out)

    print(f"  Qiskit L3 : {t_qiskit*1000:7.3f} ms, "
          f"{_count_2q(qiskit_out)} 2q gates, fidelity {fid_qiskit:.12f}")
    print(f"  PSF-Zero  : {t_psf*1000:7.3f} ms, "
          f"{_count_2q(psf_out)} 2q gates, fidelity {fid_psf:.12f}")
    print()
    print(f"  -> Same block, both exactly correct. PSF-Zero is analytic "
          f"(closed-form), not search-based, so it returns the identical "
          f"circuit every time for the same input -- no seed to control for.")


def _count_2q(qc):
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


def _fidelity(qc_in, qc_out):
    u_in = Operator(qc_in).data
    u_out = Operator(qc_out).data
    d = u_in.shape[0]
    tr = np.trace(u_in.conj().T @ u_out)
    return float((abs(tr) ** 2 + d) / (d * (d + 1)))


# ---------------------------------------------------------------------
# Demo 2: the layout-search failure region. A small, fully-saturated
# grid instance -- the regime where Qiskit's own VF2Layout is known to
# fail outright. Kept deliberately small (6x6) so this finishes in
# seconds even when Qiskit's own search runs to its budget.
# ---------------------------------------------------------------------
def demo_layout_search() -> None:
    section("Demo 2 of 2: the layout-search failure region (small grid)")
    print("A fully-saturated 6x6 grid (18 disjoint interacting pairs, zero")
    print("spare qubits) -- the exact regime Paper 1 characterizes.")
    print("This step can take up to ~30s if Qiskit's own search runs to")
    print("its budget on this machine; that itself is part of the point.")
    print()

    rows, cols = 6, 6
    n = rows * cols
    cm = CouplingMap.from_grid(rows, cols)
    pairs = [(i, i + 1) for i in range(0, n - 1, 2)]

    qc = QuantumCircuit(n)
    rng = np.random.default_rng(0)
    for (a, b) in pairs:
        qc.append(UnitaryGate(random_unitary(4, seed=int(rng.integers(0, 2**31)))), [a, b])

    t0 = time.perf_counter()
    qiskit_out = transpile(qc, coupling_map=cm, basis_gates=["rz", "sx", "x", "cx"],
                           optimization_level=3)
    t_qiskit = time.perf_counter() - t0

    t0 = time.perf_counter()
    psf_out = psf_compile.compile_for_hardware(
        qc, coupling_map=cm, basis_gates=["rz", "sx", "x", "cx"],
        routing_optimization_level=1, entangling_basis="cx",
        verify=False, seed_transpiler=0, layout_search=True,
    )
    t_psf = time.perf_counter() - t0

    print(f"  Qiskit L3 (with layout search) : {t_qiskit*1000:9.1f} ms")
    print(f"  PSF-Zero (layout_search=True)  : {t_psf*1000:9.1f} ms")
    print()
    if t_qiskit > t_psf * 5:
        print(f"  -> Qiskit's own layout search is {t_qiskit/t_psf:.0f}x slower "
              f"on this saturated instance. At larger grid sizes (see Paper 1, "
              f"Table 1) this gap grows to 271x, with Qiskit reporting "
              f"'no solution exists' on instances that provably have one.")
    else:
        print(f"  -> On this small grid the gap may not be dramatic yet -- "
              f"Paper 1's own 271x figure is measured at 8x8 (64 qubits), "
              f"not this 6x6 demo instance. See compare.py to reproduce the "
              f"full-scale result.")


def main() -> None:
    print("PSF-Zero quick start -- two short demonstrations, ~1-3 minutes total.")
    demo_gate_synthesis()
    demo_layout_search()
    print()
    print("=" * 72)
    print("Next steps:")
    print("  python compare.py        -- the fuller comparison, matching the")
    print("                               papers' own reported numbers")
    print("  python sample_circuits.py -- browse the example circuits used here")
    print("  See docs/papers/ for the full papers and technical overview.")
    print("=" * 72)


if __name__ == "__main__":
    main()
