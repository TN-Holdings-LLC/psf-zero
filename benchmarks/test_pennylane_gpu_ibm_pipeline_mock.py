"""Mock-based tests for the PennyLane <-> GPU <-> IBM-backend connection.

Extends test_pennylane_gpu_pipeline_mock.py (PennyLane<->GPU only) with a
third mocked leg standing in for IBM Quantum submission. No GPU, no IBM
credentials and no network call are used anywhere in this file -- see
psf_pennylane_gpu_ibm_prototype.py's module docstring for exactly what is
real (Qiskit's own GenericBackendV2 backend + transpile + exact-statevector
sampling) and what is an explicit stand-in (mock_ibm_submit).

This file specifically probes "curveball" inputs the user asked about, not
just the happy path: a circuit too big for the chosen backend, an invalid
shot count, and a submission failure -- in addition to correctness
(ISA-compliance, and that the physics survives the whole chain) and the
batching/call-count contracts already established for the GPU leg.

Run with:  pytest -v test_pennylane_gpu_ibm_pipeline_mock.py
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pennylane as qml
import pytest
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector, random_unitary

from psf_pennylane_gpu_ibm_prototype import (
    ConnectionContractError,
    is_isa_compliant,
    mock_ibm_submit,
    psf_pennylane_gpu_ibm_transform,
    route_for_backend,
)
from psf_pennylane_gpu_prototype import reference_cpu_synthesize, tape_to_qiskit

BLOCK_GATE_FLOOR = 12
RUNS_PER_BLOCK = BLOCK_GATE_FLOOR + 3


def _random_unitary_ops(wires, n, seed):
    rng = np.random.default_rng(seed)
    return [
        qml.QubitUnitary(random_unitary(4, seed=int(rng.integers(0, 2**31))).data, wires=wires)
        for _ in range(n)
    ]


def make_two_block_tape(n_wires_pairs=2):
    ops = []
    for i in range(n_wires_pairs):
        ops += _random_unitary_ops([2 * i, 2 * i + 1], RUNS_PER_BLOCK, seed=100 + i)
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


def gpu_batch_synthesize(matrices):
    return [reference_cpu_synthesize(m) for m in matrices]


@pytest.fixture
def backend():
    # A real Qiskit object (GenericBackendV2): genuine coupling map and basis
    # gates, but NOT actual IBM hardware -- see module docstring for why this
    # stands in for "some IBM-like backend" rather than the real thing.
    return GenericBackendV2(num_qubits=6, seed=7)


class TestIBMConnection:
    def test_full_chain_submits_isa_compliant_circuit_once(self, backend):
        """Happy path across all three mocked/stand-in legs: GPU batch is
        still called exactly once (the contract from the GPU-only test
        suite must survive adding a third stage), IBM submission is called
        exactly once, and the circuit handed to it is genuinely
        ISA-compliant for the backend -- not just assumed to be because
        transpile() ran."""
        tape = make_two_block_tape()
        mock_gpu = MagicMock(side_effect=gpu_batch_synthesize)
        mock_ibm = MagicMock(side_effect=mock_ibm_submit)

        counts, qc_routed = psf_pennylane_gpu_ibm_transform(
            tape, mock_gpu, mock_ibm, backend, shots=2000, block_gate_floor=BLOCK_GATE_FLOOR
        )

        assert mock_gpu.call_count == 1
        assert mock_ibm.call_count == 1
        submitted_qc, submitted_shots = mock_ibm.call_args[0]
        assert submitted_shots == 2000
        ok, reason = is_isa_compliant(submitted_qc, backend)
        assert ok, reason
        assert sum(counts.values()) == 2000

    def test_physics_survives_gpu_mock_and_routing(self, backend):
        """The exact (noiseless) probability distribution after GPU-mock
        synthesis AND hardware routing must match the original tape's own
        distribution -- chains the earlier PennyLane<->GPU fidelity result
        through the new routing/ISA-translation stage. Uses the exact
        statevector, not sampled counts, so this assertion carries no
        sampling noise and can use a tight tolerance."""
        tape = make_two_block_tape()
        qc_before, _ = tape_to_qiskit(tape)
        probs_before = Statevector(qc_before).probabilities_dict()

        mock_gpu = MagicMock(side_effect=gpu_batch_synthesize)
        _counts, qc_routed = psf_pennylane_gpu_ibm_transform(
            tape, mock_gpu, mock_ibm_submit, backend, shots=100, block_gate_floor=BLOCK_GATE_FLOOR
        )
        # route_for_backend() pins the used circuit's qubits to physical
        # qubits [0..n-1] of the (larger) backend register, so the routed
        # circuit comes back sized to the FULL backend (6 qubits here) with
        # the extra physical qubits idle at |0>. Qiskit prints bitstrings
        # MSB-first (highest qubit index leftmost), so those idle qubits are
        # a fixed "00" prefix -- marginalize them out by keeping only the
        # trailing `qc_before.num_qubits` characters before comparing.
        n_used = qc_before.num_qubits
        qc_routed_unitary_only = qc_routed.remove_final_measurements(inplace=False)
        probs_after_full = Statevector(qc_routed_unitary_only).probabilities_dict()
        probs_after: dict[str, float] = {}
        for bitstring, p in probs_after_full.items():
            suffix = bitstring[-n_used:]
            probs_after[suffix] = probs_after.get(suffix, 0.0) + p

        all_keys = set(probs_before) | set(probs_after)
        diff = sum(abs(probs_before.get(k, 0.0) - probs_after.get(k, 0.0)) for k in all_keys)
        assert diff < 1e-6, f"total variation distance too high after routing: {diff:.3e}"

    def test_sampled_counts_favor_the_dominant_outcome(self, backend):
        """A weaker, sampling-based sanity check on top of the exact check
        above: for a tape whose dominant basis state has most of the
        probability mass, the MOCKED 'IBM' sampler's most common count
        should be that same bitstring. Not a statistics-heavy test (the
        exact-probability test above is the real correctness check) -- just
        confirms the mocked sampler isn't returning garbage."""
        tape = qml.tape.QuantumTape(
            [qml.RY(0.05, wires=0), qml.RY(0.05, wires=1)], measurements=[], shots=None
        )
        mock_gpu = MagicMock(side_effect=gpu_batch_synthesize)
        counts, _qc_routed = psf_pennylane_gpu_ibm_transform(
            tape, mock_gpu, mock_ibm_submit, backend, shots=2000, block_gate_floor=BLOCK_GATE_FLOOR
        )
        dominant = max(counts, key=counts.get)
        assert dominant.replace(" ", "").lstrip("0") == "" or set(dominant.replace(" ", "")) == {"0"}
        assert mock_gpu.call_count == 0  # no 2-qubit interaction here, nothing to batch

    def test_circuit_too_big_for_backend_is_rejected_before_submission(self, backend):
        """Curveball: a circuit needing more qubits than the chosen backend
        has. Must fail clearly, BEFORE reaching the IBM-mock stage -- not
        silently truncate qubits and submit something else."""
        big_tape = make_two_block_tape(n_wires_pairs=4)  # 8 wires > backend's 6
        mock_gpu = MagicMock(side_effect=gpu_batch_synthesize)
        mock_ibm = MagicMock(side_effect=mock_ibm_submit)

        with pytest.raises(ConnectionContractError, match="only has"):
            psf_pennylane_gpu_ibm_transform(
                big_tape, mock_gpu, mock_ibm, backend, shots=1000, block_gate_floor=BLOCK_GATE_FLOOR
            )
        assert mock_ibm.call_count == 0

    def test_invalid_shots_rejected(self, backend):
        """Curveball: shots=0 (or negative) must raise, not silently submit
        a zero-shot or nonsensical job."""
        tape = make_two_block_tape()
        mock_gpu = MagicMock(side_effect=gpu_batch_synthesize)
        mock_ibm = MagicMock(side_effect=mock_ibm_submit)

        for bad_shots in (0, -5):
            with pytest.raises(ValueError):
                psf_pennylane_gpu_ibm_transform(
                    tape, mock_gpu, mock_ibm, backend, shots=bad_shots, block_gate_floor=BLOCK_GATE_FLOOR
                )
        assert mock_ibm.call_count == 0

    def test_ibm_submission_failure_propagates_not_swallowed(self, backend):
        """Curveball: the mocked IBM stage raises (job rejected, network
        error, whatever). Must propagate to the caller, matching this
        project's 'no silent fallback' convention -- not be absorbed into a
        fallback that pretends the job succeeded."""
        tape = make_two_block_tape()
        mock_gpu = MagicMock(side_effect=gpu_batch_synthesize)
        mock_ibm = MagicMock(side_effect=RuntimeError("simulated IBM job failure"))

        with pytest.raises(RuntimeError, match="simulated IBM job failure"):
            psf_pennylane_gpu_ibm_transform(
                tape, mock_gpu, mock_ibm, backend, shots=1000, block_gate_floor=BLOCK_GATE_FLOOR
            )

    def test_backend_reporting_isa_noncompliant_circuit_is_caught(self, backend):
        """Curveball: simulate route_for_backend somehow producing a
        non-ISA-compliant circuit (e.g. a future bug, or a backend whose
        Target changed between routing and submission). The independent
        is_isa_compliant check must catch it and refuse to submit, rather
        than trusting transpile() blindly."""
        tape = make_two_block_tape()
        mock_gpu = MagicMock(side_effect=gpu_batch_synthesize)
        mock_ibm = MagicMock(side_effect=mock_ibm_submit)

        # Directly craft a mismatch: route for `backend` as usual, but then
        # validate the routed circuit against a DIFFERENT, minimal backend
        # stand-in whose coupling map has no edges at all -- forces an ISA
        # mismatch that is_isa_compliant must catch. A small plain object is
        # used instead of mutating GenericBackendV2's internals, which are
        # not meant to be poked at from outside.
        from qiskit.transpiler import CouplingMap

        class _BrokenBackend:
            name = "broken-backend-no-edges"
            num_qubits = backend.num_qubits
            operation_names = backend.operation_names
            coupling_map = CouplingMap()  # zero edges

        broken_backend = _BrokenBackend()

        from psf_pennylane_gpu_prototype import psf_pennylane_gpu_transform as _pglt

        tape_synth = _pglt(tape, mock_gpu, block_gate_floor=BLOCK_GATE_FLOOR)
        qc_synth, _ = tape_to_qiskit(tape_synth)
        qc_routed = route_for_backend(qc_synth, backend)
        ok, reason = is_isa_compliant(qc_routed, broken_backend)
        assert not ok
        assert "coupled edge" in reason or "not in backend.operation_names" in reason
        assert mock_ibm.call_count == 0  # this test never calls the transform end-to-end
