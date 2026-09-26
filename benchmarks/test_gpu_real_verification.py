"""test_gpu_real_verification.py -- exercises psf_pennylane_gpu_real.py's
REAL lightning.gpu verification step, on actual synthesized 2-qubit blocks.

This is NOT a mock test. If lightning.gpu is not available in the
environment this runs in, every test here is expected to FAIL LOUDLY (via
ConnectionContractError from verify_on_gpu itself), not skip silently --
matching this project's own "no silent fallback" convention. If you need to
run this without a GPU present, that is a reason to not run this file, not
a reason for it to pretend to pass.

Run with:  pytest -v test_gpu_real_verification.py
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import random_unitary

from psf_pennylane_gpu_prototype import reference_cpu_synthesize
from psf_pennylane_gpu_real import verify_block_gpu_and_cpu, verify_on_gpu


class TestRealGPUVerification:
    def test_cnot_synthesized_block_matches_on_gpu(self):
        """A trivial, exactly-representable case: CNOT itself. If this
        fails, something is wrong with the GPU verification plumbing
        itself, not with synthesis (CNOT needs no synthesis at all -- the
        decomposer should return it essentially unchanged)."""
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        circ = reference_cpu_synthesize(cnot)
        result = verify_block_gpu_and_cpu(cnot, circ)
        assert result["gpu_pass"], (
            f"CNOT round-trip failed on real GPU: gpu_expval_diff="
            f"{result['gpu_expval_diff']:.3e}, cpu_matrix_infidelity="
            f"{result['cpu_matrix_infidelity']:.3e}"
        )

    def test_random_unitaries_match_on_gpu(self):
        """Several independent random 2-qubit unitaries, synthesized on CPU
        (reference_cpu_synthesize, the same stand-in the mock prototype
        uses), then verified via REAL execution on lightning.gpu against
        the SAME target unitary run directly on default.qubit (CPU)."""
        worst = 0.0
        for seed in range(5):
            u = random_unitary(4, seed=seed).data
            circ = reference_cpu_synthesize(u)
            result = verify_block_gpu_and_cpu(u, circ)
            worst = max(worst, result["gpu_expval_diff"])
            assert result["gpu_pass"], (
                f"seed {seed}: gpu_expval_diff={result['gpu_expval_diff']:.3e} "
                f"exceeds tolerance"
            )
        print(f"\nworst GPU expectation-value diff over 5 random unitaries: {worst:.3e}")

    def test_wrong_circuit_is_caught_by_gpu_check_too(self):
        """Sanity check on the check itself: an intentionally WRONG circuit
        (identity, standing in for a buggy synthesizer) must fail the GPU
        verification, not silently pass -- the GPU check should be at least
        as sensitive as the existing CPU matrix check, not a weaker
        substitute for it."""
        from qiskit import QuantumCircuit

        u = random_unitary(4, seed=42).data
        wrong_circ = QuantumCircuit(2)  # identity, not the target unitary

        result = verify_block_gpu_and_cpu(u, wrong_circ)
        assert not result["gpu_pass"], (
            "an intentionally wrong (identity) circuit was reported as "
            "matching the target unitary by the GPU check -- the check "
            "itself is not sensitive enough."
        )
