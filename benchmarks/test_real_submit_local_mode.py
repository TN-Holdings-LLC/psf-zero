"""test_real_submit_local_mode.py -- Addendum 153.

Runs the PennyLane -> (real-GPU-verified) synthesis -> routing -> IBM
connection with the REAL SamplerV2 submission function
(psf_ibm_real_submit.make_sampler_submit_fn) in IBM's own local testing
mode. No credentials, no network. Requires qiskit-ibm-runtime and
qiskit-aer, plus lightning.gpu (the synthesis step still verifies each
block on real GPU hardware via psf_pennylane_gpu_full_chain).

Run with:  pytest -v -s test_real_submit_local_mode.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pennylane as qml
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_unitary
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from psf_ibm_real_submit import make_sampler_submit_fn
from psf_pennylane_gpu_full_chain import full_chain_with_gpu_verification

BLOCK_GATE_FLOOR = 12
RUNS_PER_BLOCK = BLOCK_GATE_FLOOR + 3
SHOTS = 4000
SEED = 42


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


def tvd(counts: dict[str, int], qc_routed: QuantumCircuit) -> float:
    """Total variation distance between sampled counts and the routed
    circuit's own exact distribution, over the full backend width (both
    sides use Qiskit's convention: qubit/clbit 0 is the rightmost bit)."""
    exact = Statevector(qc_routed.remove_final_measurements(inplace=False)).probabilities_dict()
    total = sum(counts.values())
    keys = set(exact) | set(counts)
    return 0.5 * sum(abs(exact.get(k, 0.0) - counts.get(k, 0) / total) for k in keys)


@pytest.fixture(scope="module")
def backend():
    return FakeManilaV2()


class TestRealSubmitLocalMode:
    # ---- P1: contracts survive the swap from mock to real SamplerV2 ----
    def test_called_once_and_shots_accounted(self, backend):
        submit = MagicMock(side_effect=make_sampler_submit_fn(backend, seed_simulator=SEED))
        counts, _ = full_chain_with_gpu_verification(
            make_two_block_tape(), submit, backend, shots=SHOTS, block_gate_floor=BLOCK_GATE_FLOOR
        )
        assert submit.call_count == 1
        assert sum(counts.values()) == SHOTS

    def test_invalid_shots_rejected_before_submission(self, backend):
        submit = MagicMock(side_effect=make_sampler_submit_fn(backend, seed_simulator=SEED))
        for bad in (0, -5, 100.7, "100", None):
            with pytest.raises(ValueError):
                full_chain_with_gpu_verification(
                    make_two_block_tape(), submit, backend, shots=bad, block_gate_floor=BLOCK_GATE_FLOOR
                )
        assert submit.call_count == 0

    def test_unmeasured_circuit_rejected(self, backend):
        submit = make_sampler_submit_fn(backend, seed_simulator=SEED)
        qc = QuantumCircuit(2)
        qc.h(0)
        with pytest.raises(ValueError, match="no measurements"):
            submit(qc, 100)

    def test_submission_failure_propagates(self, backend):
        failing = MagicMock(side_effect=RuntimeError("simulated runtime failure"))
        with pytest.raises(RuntimeError, match="simulated runtime failure"):
            full_chain_with_gpu_verification(
                make_two_block_tape(), failing, backend, shots=SHOTS, block_gate_floor=BLOCK_GATE_FLOOR
            )

    # ---- P2: noiseless target reproduces the exact distribution ----
    def test_noiseless_target_matches_exact(self, backend):
        submit = make_sampler_submit_fn(AerSimulator(), seed_simulator=SEED)
        counts, qc_routed = full_chain_with_gpu_verification(
            make_two_block_tape(), submit, backend, shots=SHOTS, block_gate_floor=BLOCK_GATE_FLOOR
        )
        d = tvd(counts, qc_routed)
        print(f"\nP2 noiseless TVD = {d:.4f}")
        assert d < 0.1

    # ---- P3: device-snapshot noise visible but not destructive ----
    def test_fake_device_noise_visible_not_destructive(self, backend):
        submit = make_sampler_submit_fn(backend, seed_simulator=SEED)
        counts, qc_routed = full_chain_with_gpu_verification(
            make_two_block_tape(), submit, backend, shots=SHOTS, block_gate_floor=BLOCK_GATE_FLOOR
        )
        d = tvd(counts, qc_routed)
        print(f"\nP3 FakeManilaV2 TVD = {d:.4f}")
        assert 0.01 < d < 0.3

    # ---- P4: known issue, documented (measure_all over the full backend) ----
    def test_bitstring_width_is_full_backend_known_issue(self, backend):
        submit = make_sampler_submit_fn(backend, seed_simulator=SEED)
        counts, _ = full_chain_with_gpu_verification(
            make_two_block_tape(), submit, backend, shots=200, block_gate_floor=BLOCK_GATE_FLOOR
        )
        widths = {len(k) for k in counts}
        print(f"\nP4 bitstring widths = {widths} (backend has {backend.num_qubits} qubits, circuit uses 4)")
        assert widths == {backend.num_qubits}
