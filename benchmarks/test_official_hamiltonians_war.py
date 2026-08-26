# test_official_hamiltonians_war.py -- Corrected Version

#

# NOTE: psf_compile.py was not included in what was pasted (same as the

# earlier test_psf_vs_tket.py round), so the actual PSF-Zero numbers could

# not be re-run here. This file has the exact same two harness-level bugs

# as test_psf_vs_tket.py, verified there with real measurements (see that

# round's test_correctness_gap.py) and re-checked structurally here.

#

# BUG #1 (critical -- fabricated data): `time_hybrid` is not a

# measurement.

#     qc_psf = psf_compile(qc_orig)               # never timed at all

#     t1 = time.perf_counter()

#     qc_hybrid = run_tket_compile(qc_psf)

#     time_hybrid = (time.perf_counter() - t1) + 0.012  # "PSF's constant

#                                                        # geometric overhead"

#   The comment asserts PSF's overhead is a *constant* 0.012s, but nothing

#   in this file (or the earlier one) ever measures it -- it's the same

#   unmeasured flat number carried over from test_psf_vs_tket.py. In the

#   companion round, TKET's own compile time on this project's circuit

#   family was shown to scale from 77ms to 755ms as circuit depth grew from

#   40 to 800 -- there is no reason to expect PSF's real compile cost is

#   depth-independent, and no measurement here to back up that it is. Fixed

#   by actually timing psf_compile() and using that real number, plus timing

#   Qiskit L3 too (previously left unmeasured, same as before).

#

# BUG #2 (critical -- no correctness checking at all): zero `assert`

# statements anywhere in test_official_hamiltonian_demolition(). Pytest

# reports this PASSED purely because nothing raised, regardless of whether

# TKET's / PSF's / the hybrid pipeline's output circuit still implements

# the same unitary as the Hamiltonian circuit it's supposed to be compiling.

# Fixed the same way as test_psf_vs_tket.py: added a real

# `assert_equivalent(...)` (Operator equivalence up to global phase) after

# every compilation step.

#

# ALSO NOTED (not a code bug, but a documentation/claim issue worth fixing):

# the docstring claimed this "[Zero-Dependency Framework] Perfectly

# replicates the official Benchpress Hamiltonian simulation circuits". The

# generator below is a hand-written 4-step local-rotation + interaction

# circuit with arbitrary hardcoded coefficients (0.4, 0.25, 0.6, 2.0, 1.8,

# ...) -- nothing here was cross-checked against the actual Benchpress

# circuits, so "perfectly replicates" is an unverifiable/overclaiming

# statement as written. Reworded to describe what the function actually is:

# a hand-built stand-in circuit family with similar structural features

# (Trotter-like local rotations + two-qubit interaction terms), not a

# verified replica of any specific official benchmark's circuits.

#

# Also noted (verified NOT to matter empirically, so left as a comment

# rather than "fixed"): `evolution_time=1.5+idx` changes together with

# `interaction` in the main loop, confounding the two. Checked directly:

# holding interaction type fixed and sweeping evolution_time from 1.5 to

# 5.5 did not change either the original or the TKET-compiled depth for

# this circuit family, so the confound happens not to bias the depth

# results here -- but it's still worth knowing this loop is not isolating

# "interaction type" as its only varying factor.

import pytest

import time

import pandas as pd

import numpy as np

from qiskit import QuantumCircuit, transpile

from qiskit.quantum_info import Operator

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

from pytket.passes import FullPeepholeOptimise

# Our finalized master engine

from psf_compile import compile as psf_compile

def generate_official_equivalent_hamiltonian(evolution_time: float, interaction: str) -> QuantumCircuit:

    """

    Hand-built stand-in Hamiltonian-simulation circuit (4 Trotter-like

    steps of local rotations + a two-qubit interaction term), used here as

    a black-box test circuit family. This is NOT a verified replica of any

    specific official benchmark suite's actual circuits -- see the bug

    writeup at the top of this file.

    """

    qc = QuantumCircuit(2)

    for step in range(4):

        # Non-trivial local rotations (Physics site potentials)

        qc.rx(0.4 * evolution_time * (step + 1), 0)

        qc.ry(0.25 * evolution_time, 1)

        qc.rz(0.6 * evolution_time, 0)

        qc.rx(0.15 * evolution_time, 1)

        # Two-qubit interaction term

        if interaction == "xx":

            qc.rxx(2.0 * evolution_time, 0, 1)

        elif interaction == "yy":

            qc.ryy(1.8 * evolution_time, 0, 1)

        elif interaction == "zz":

            qc.rzz(2.2 * evolution_time, 0, 1)

        elif interaction == "exchange":

            qc.rxx(1.0 * evolution_time, 0, 1)

            qc.ryy(1.0 * evolution_time, 0, 1)

        elif interaction == "full":

            qc.rxx(1.2 * evolution_time, 0, 1)

            qc.ryy(0.9 * evolution_time, 0, 1)

            qc.rzz(1.5 * evolution_time, 0, 1)

        qc.rz(0.3 * evolution_time, 1)

    return qc

def run_tket_compile(qiskit_circ: QuantumCircuit) -> QuantumCircuit:

    tket_circ = qiskit_to_tk(qiskit_circ)

    FullPeepholeOptimise().apply(tket_circ)

    return tk_to_qiskit(tket_circ)

def assert_equivalent(qc_orig: QuantumCircuit, qc_new: QuantumCircuit, label: str, atol=1e-6):

    """Fix (bug #2): verify qc_new implements the same unitary as qc_orig,

    up to an unobservable global phase."""

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

def test_official_hamiltonian_demolition():

    """

    Compares Qiskit L3 / TKET native / PSF-Zero native / Hybrid (PSF->TKET)

    on a family of hand-built Hamiltonian-simulation-style circuits, one

    per interaction topology.

    """

    results = []

    print("\n⚔️  [OFFICIAL BENCHMARK] Executing self-hosted Hamiltonian compilation...")

    interactions = ["xx", "yy", "zz", "exchange", "full"]

    for idx, interaction in enumerate(interactions):

        qc_orig = generate_official_equivalent_hamiltonian(evolution_time=1.5 + idx, interaction=interaction)

        # 1. Baseline: Qiskit L3

        # Fix (bug #1): actually time this instead of leaving it unmeasured.

        t0 = time.perf_counter()

        qc_qiskit = transpile(qc_orig, basis_gates=['rx', 'ry', 'rz', 'cx'], optimization_level=3)

        time_qiskit = time.perf_counter() - t0

        assert_equivalent(qc_orig, qc_qiskit, f"Qiskit L3 ({interaction})")

        # 2. Competitor: TKET Native

        t0 = time.perf_counter()

        qc_tket = run_tket_compile(qc_orig)

        time_tket = time.perf_counter() - t0

        assert_equivalent(qc_orig, qc_tket, f"TKET Native ({interaction})")

        # 3. Ours: PSF-Zero Native

        # Fix (bug #1): actually time this instead of never measuring it.

        t0 = time.perf_counter()

        qc_psf = psf_compile(qc_orig)

        time_psf = time.perf_counter() - t0

        assert_equivalent(qc_orig, qc_psf, f"PSF-Zero Native ({interaction})")

        # 4. Ultimate Weapon: Hybrid (PSF -> TKET)

        # Fix (bug #1): Hybrid_Time is psf_compile's real measured time plus

        # TKET's real measured time on the PSF output -- no invented

        # constant standing in for an unmeasured "PSF overhead".

        t1 = time.perf_counter()

        qc_hybrid = run_tket_compile(qc_psf)

        time_hybrid_tket_stage = time.perf_counter() - t1

        time_hybrid = time_psf + time_hybrid_tket_stage

        assert_equivalent(qc_orig, qc_hybrid, f"Hybrid PSF->TKET ({interaction})")

        results.append({

            "Interaction": interaction,

            "Original_Depth": qc_orig.depth(),

            "Qiskit_Depth": qc_qiskit.depth(),

            "TKET_Depth": qc_tket.depth(),

            "PSF_Depth": qc_psf.depth(),

            "Hybrid_Depth": qc_hybrid.depth(),

            "Qiskit_Time": time_qiskit,

            "TKET_Time": time_tket,

            "PSF_Time": time_psf,

            "Hybrid_Time": time_hybrid,

        })

    df = pd.DataFrame(results)

    print("\n" + "="*75)

    print("🏆  [THE FINAL EVIDENCE] Hamiltonian Simulation Showdown Results")

    print("="*75)

    print(df.to_string(index=False))

    print("="*75)

    print("✅ All 5 interaction circuits' Qiskit/TKET/PSF/Hybrid outputs verified")

    print("   functionally equivalent to their original circuits.")

    df.to_csv("official_hamiltonian_compressed_victory.csv", index=False)

    print("📁 Target metrics safely secured in 'official_hamiltonian_compressed_victory.csv'.")
