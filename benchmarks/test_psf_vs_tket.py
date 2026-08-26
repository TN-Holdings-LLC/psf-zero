# test_psf_vs_tket.py -- Corrected Version

#

# NOTE: psf_compile.py was not included in what was pasted, so the actual

# PSF-Zero numbers could not be re-run/verified here. Everything below was

# checked against the parts of this file that don't depend on psf_compile's

# internals -- once you drop your real psf_compile.py next to this file,

# it will run exactly as intended. See test_correctness_gap.py for the

# standalone demonstrations backing bugs #1 and #2 below (run with a stand-in

# for psf_compile since the real one wasn't available).

#

# BUG #1 (critical -- fabricated data): `time_hybrid` was never a

# measurement of the hybrid pipeline's actual cost.

#     qc_psf_pure = psf_compile(qc_orig)          # never timed at all

#     t1 = time.perf_counter()

#     qc_hybrid = run_tket_compile(qc_psf_pure)

#     time_hybrid = (time.perf_counter() - t1) + 0.012   # made-up constant

#   psf_compile's own runtime is never measured anywhere, and a flat

#   constant (0.012s, sourced from an unstated "average") is added in its

#   place for every single sample, regardless of that sample's actual

#   circuit size. Demonstrated directly (test_correctness_gap.py): a real

#   compile step's cost scales with circuit size -- e.g. TKET's own

#   optimizer went from 77ms to 755ms as circuit depth grew from 40 to 800

#   in this test's own circuit family. A single flat constant cannot stand

#   in for that, in either direction, for most samples -- and it silently

#   replaces the one number ("Hybrid mean time") the whole benchmark's

#   "Ultimate Weapon" conclusion rests on. Fixed by actually timing

#   psf_compile() and summing it with the measured TKET-on-PSF time, so

#   Hybrid_Time is a real end-to-end measurement, and by timing Qiskit L3

#   and PSF-Zero Native too (the original left both as "-" in the summary,

#   an incomplete/asymmetric comparison across the 4 engines).

#

# BUG #2 (critical -- no correctness checking at all): the "test" never

# asserts anything about whether the optimized circuits still compute the

# same thing as the original.

#   test_final_ultimate_300_samples_war() contained zero `assert`

#   statements; pytest reports it PASSED purely because nothing raised.

#   A compiler that wins on depth by deleting or mis-substituting gates

#   would sail through 300 samples of this "test" completely undetected --

#   the whole point of a correctness-and-performance benchmark is

#   undermined if only performance is ever checked. Demonstrated directly:

#   a real equivalence check (unitary equality up to global phase) built

#   from qiskit.quantum_info.Operator correctly returns True for TKET's

#   real optimization output and correctly returns False for a

#   deliberately broken "compiler" that just drops the last gate --

#   proving the check itself isn't a rubber stamp. Added

#   `assert_equivalent(...)` calls after every compilation step (Qiskit,

#   TKET, PSF, Hybrid), so a functional regression in any engine now

#   actually fails the test instead of quietly inflating a "smaller depth"

#   statistic.

#

# Also added (minor, not a correctness bug): a `seed` parameter so a

# specific run's 300 circuits can be reproduced when investigating a

# failure -- the pasted version used `np.random.default_rng()` with no

# seed at all, so a failing sample could never be looked at again.

import pytest

import numpy as np

import pandas as pd

import time

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile

from qiskit.quantum_info import Operator

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

from pytket.passes import FullPeepholeOptimise

# Our finalized master engine

from psf_compile import compile as psf_compile

def generate_random_mud_circuit(depth_steps=50, rng=None):

    """Deliberately generate a dense, deep 2-qubit random circuit (depth = 4*depth_steps)"""

    qc = QuantumCircuit(2)

    if rng is None:

        rng = np.random.default_rng()

    for _ in range(depth_steps):

        qc.rx(rng.uniform(0.1, 3.0), 0)

        qc.ry(rng.uniform(0.1, 3.0), 1)

        qc.cx(0, 1)

        qc.rz(rng.uniform(0.1, 3.0), 0)

        qc.cx(1, 0)

    return qc

def run_tket_compile(qiskit_circ: QuantumCircuit) -> QuantumCircuit:

    """Native optimization using TKET alone"""

    tket_circ = qiskit_to_tk(qiskit_circ)

    FullPeepholeOptimise().apply(tket_circ)

    return tk_to_qiskit(tket_circ)

def assert_equivalent(qc_orig: QuantumCircuit, qc_new: QuantumCircuit, label: str, atol=1e-6):

    """Fix (bug #2): verify qc_new implements the same unitary as qc_orig,

    up to an unobservable global phase, so a depth win can never come from

    a functional regression. Raises with a clear message if they diverge."""

    Ua = Operator(qc_orig).data

    Ub = Operator(qc_new).data

    prod = Ua.conj().T @ Ub

    ref = prod[0, 0]

    if abs(ref) < 1e-12:

        raise AssertionError(f"[{label}] compiled circuit is not even proportional to the original "

                              f"(top-left overlap ~0) -- almost certainly a wrong circuit")

    phase = ref / abs(ref)

    max_dev = np.max(np.abs(prod / phase - np.eye(prod.shape[0])))

    assert max_dev < atol, (f"[{label}] compiled circuit does NOT implement the same unitary as the "

                             f"original (max deviation {max_dev:.3e} > {atol}) -- functional regression, "

                             f"not just a depth/time difference")

def test_final_ultimate_300_samples_war(seed=1234):

    results = []

    num_samples = 300

    rng = np.random.default_rng(seed)

    print(f"\n🚀 [N={num_samples}] Starting the 4-way ultimate showdown. Challenging the fortress of TKET...")

    for sample_id in range(num_samples):

        qc_orig = generate_random_mud_circuit(depth_steps=50, rng=rng)

        # 1. Baseline: Qiskit Level 3

        t0 = time.perf_counter()

        qc_qiskit = transpile(qc_orig, basis_gates=['rx', 'ry', 'rz', 'cx'], optimization_level=3)

        time_qiskit = time.perf_counter() - t0

        assert_equivalent(qc_orig, qc_qiskit, "Qiskit L3")

        # 2. Competitor: TKET Native

        t0 = time.perf_counter()

        qc_tket_pure = run_tket_compile(qc_orig)

        time_tket_pure = time.perf_counter() - t0

        assert_equivalent(qc_orig, qc_tket_pure, "TKET Native")

        # 3. Ours: PSF-Zero Native

        # Fix (bug #1): actually time this instead of never measuring it.

        t0 = time.perf_counter()

        qc_psf_pure = psf_compile(qc_orig)

        time_psf_pure = time.perf_counter() - t0

        assert_equivalent(qc_orig, qc_psf_pure, "PSF-Zero Native")

        # 4. Ultimate Weapon: Hybrid (PSF -> TKET)

        # Fix (bug #1): Hybrid_Time is now psf_compile's real measured time

        # plus TKET's real measured time on the PSF output -- no invented

        # constant standing in for an unmeasured quantity.

        t1 = time.perf_counter()

        qc_hybrid = run_tket_compile(qc_psf_pure)

        time_hybrid_tket_stage = time.perf_counter() - t1

        time_hybrid = time_psf_pure + time_hybrid_tket_stage

        assert_equivalent(qc_orig, qc_hybrid, "Hybrid (PSF->TKET)")

        results.append({

            "Sample_ID": sample_id,

            "Qiskit_Depth": qc_qiskit.depth(),

            "TKET_Depth": qc_tket_pure.depth(),

            "PSF_Depth": qc_psf_pure.depth(),

            "Hybrid_Depth": qc_hybrid.depth(),

            "Qiskit_Time": time_qiskit,

            "TKET_Time": time_tket_pure,

            "PSF_Time": time_psf_pure,

            "Hybrid_Time": time_hybrid,

        })

        if (sample_id + 1) % 50 == 0:

            print(f"▓ [{sample_id + 1}/{num_samples}] Samples completely decomposed, hybrid-recompiled, "

                  f"and verified functionally equivalent to the original...")

    df = pd.DataFrame(results)

    df.to_csv("psf_vs_tket_300_ultimate.csv", index=False)

    print("\n📁 Saved raw data to 'psf_vs_tket_300_ultimate.csv'.")

    # Display statistical summary

    summary = {

        "Engine": ["Qiskit L3", "TKET Native", "PSF-Zero Native", "Hybrid (PSF➔TKET)"],

        "Mean Depth": [df["Qiskit_Depth"].mean(), df["TKET_Depth"].mean(), df["PSF_Depth"].mean(), df["Hybrid_Depth"].mean()],

        "Max Depth": [df["Qiskit_Depth"].max(), df["TKET_Depth"].max(), df["PSF_Depth"].max(), df["Hybrid_Depth"].max()],

        "Mean Time": [f"{df['Qiskit_Time'].mean():.6f}s", f"{df['TKET_Time'].mean():.6f}s",

                      f"{df['PSF_Time'].mean():.6f}s", f"{df['Hybrid_Time'].mean():.6f}s"],

    }

    print("\n" + "="*65)

    print("📊 [N=300 Final Showdown Results] 4-Way Joint Summary")

    print("="*65)

    print(pd.DataFrame(summary).to_string(index=False))

    print("="*65)

    print("✅ All 300 samples' Qiskit/TKET/PSF/Hybrid outputs verified functionally")

    print("   equivalent to their original circuits (unitary match up to global phase).")

    # Automatically generate a beautiful comparison plot

    plt.figure(figsize=(10, 5))

    # Boxplot for circuit depths

    plt.boxplot(

        [df["Qiskit_Depth"], df["TKET_Depth"], df["PSF_Depth"], df["Hybrid_Depth"]],

        tick_labels=["Qiskit L3", "TKET Pure", "PSF-Zero", "Hybrid (PSF->TKET)"]

    )

    plt.title("Ultimate Circuit Depth Comparison (N=300)")

    plt.ylabel("Depth")

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    plt.savefig("psf_vs_tket_300_boxplot.png", dpi=300)

    print("🎨 Output the definitive empirical evidence graph to 'psf_vs_tket_300_boxplot.png'.")

