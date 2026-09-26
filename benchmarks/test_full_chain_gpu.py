"""test_full_chain_gpu.py -- exercises the COMPLETE connection
(PennyLane -> CPU synthesis -> REAL lightning.gpu verification -> routing
-> mocked IBM submission) as one chain, not the GPU verification step in
isolation (that is test_gpu_real_verification.py's own job).

This is NOT a mock test for the GPU leg -- it runs on real RTX 4070
hardware. Only the IBM leg (mock_ibm_submit) remains a stand-in, unchanged
from the rest of tonight's work; see psf_pennylane_gpu_ibm_prototype.py's
own docstring for what that stands in for and why.

Run with:  pytest -v test_full_chain_gpu.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pennylane as qml
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector, random_unitary

from psf_pennylane_gpu_full_chain import full_chain_with_gpu_verification
from psf_pennylane_gpu_ibm_prototype import ConnectionContractError, mock_ibm_submit
from psf_pennylane_gpu_prototype import tape_to_qiskit

BLOCK_GATE_FLOOR = 12
RUNS_PER_BLOCK = BLOCK_GATE_FLOOR + 3


def _random_unitary_ops(wires, n, seed):
    rng = np.random.default_rng(seed)
    return [
        qml.QubitUnitary(random_unitary(4, seed=int(rng.integers(0, 2**31))).data, wires=wires)
        for _ in range(n)
    ]


def make_two_block_tape():
    ops = _random_unitary_ops([0, 1], RUNS_PER_BLOCK, seed=1)
    ops += _random_unitary_ops([2, 3], RUNS_PER_BLOCK, seed=2)
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


@pytest.fixture
def backend():
    return GenericBackendV2(num_qubits=6, seed=7)


class TestFullChainWithRealGPU:
    def test_full_chain_completes_with_real_gpu_verification(self, backend):
        """The complete chain, with every block genuinely verified on RTX
        4070 before the (still-mocked) IBM submission stage. If this
        passes, every synthesized block in this tape was actually run on
        GPU hardware and matched its target unitary's own physics -- not
        merely accepted because a call returned without raising."""
        tape = make_two_block_tape()
        mock_ibm = MagicMock(side_effect=mock_ibm_submit)

        counts, qc_routed = full_chain_with_gpu_verification(
            tape, mock_ibm, backend, shots=2000, block_gate_floor=BLOCK_GATE_FLOOR
        )

        assert mock_ibm.call_count == 1
        assert sum(counts.values()) == 2000

    def test_physics_survives_the_full_chain(self, backend):
        """The exact (noiseless) probability distribution after the FULL
        chain (CPU synthesis + real-GPU verification + routing) must match
        the original tape's own distribution -- the same check
        test_pennylane_gpu_ibm_pipeline_mock.py's own
        test_physics_survives_gpu_mock_and_routing ran on the mocked chain,
        now run on the chain with real GPU verification in it."""
        tape = make_two_block_tape()
        qc_before, _ = tape_to_qiskit(tape)
        probs_before = Statevector(qc_before).probabilities_dict()

        _counts, qc_routed = full_chain_with_gpu_verification(
            tape, mock_ibm_submit, backend, shots=100, block_gate_floor=BLOCK_GATE_FLOOR
        )

        n_used = qc_before.num_qubits
        qc_routed_unitary_only = qc_routed.remove_final_measurements(inplace=False)
        probs_after_full = Statevector(qc_routed_unitary_only).probabilities_dict()
        probs_after: dict[str, float] = {}
        for bitstring, p in probs_after_full.items():
            suffix = bitstring[-n_used:]
            probs_after[suffix] = probs_after.get(suffix, 0.0) + p

        all_keys = set(probs_before) | set(probs_after)
        diff = sum(abs(probs_before.get(k, 0.0) - probs_after.get(k, 0.0)) for k in all_keys)
        assert diff < 1e-6, f"total variation distance too high: {diff:.3e}"

    def test_ibm_submission_failure_still_propagates_through_full_chain(self, backend):
        """Curveball, re-run through the full (real-GPU-verified) chain:
        the mocked IBM stage raising must still propagate to the caller,
        not be swallowed -- confirms adding the real-GPU verification step
        did not accidentally introduce a try/except that hides downstream
        failures."""
        tape = make_two_block_tape()
        failing_ibm = MagicMock(side_effect=RuntimeError("simulated IBM job failure"))

        with pytest.raises(RuntimeError, match="simulated IBM job failure"):
            full_chain_with_gpu_verification(
                tape, failing_ibm, backend, shots=1000, block_gate_floor=BLOCK_GATE_FLOOR
            )
