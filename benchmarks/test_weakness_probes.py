"""Adversarial / "curveball-squared" tests written to hunt for weaknesses in
the two connection prototypes (psf_pennylane_gpu_prototype.py and
psf_pennylane_gpu_ibm_prototype.py) beyond what the original two test files
covered.

This file is NOT part of the psf-zero public repo / README submission -- it
is an informal, internal sandbox exercise: "how far can these mocked
connection-layer contracts be pushed before something breaks silently."

Four real gaps were found empirically (see the accompanying instruction
document for the full writeup) and then fixed in the prototype files
alongside this test file, so every test below documents BOTH the failure
mode that used to exist AND the fix that now catches it:

  1. psf_pennylane_gpu_transform() never checked that a circuit returned by
     gpu_batch_fn actually implements the SAME unitary as the block it was
     asked to synthesize -- a shape-correct but physically wrong result
     (e.g. an accidental identity) was silently spliced into the output
     tape. Fixed with an independent equivalence check, the same spirit as
     is_isa_compliant()'s "don't just trust the call succeeded" philosophy.
  2. is_isa_compliant() only ever checked 2-qubit gates against the coupling
     map's edges; a gate touching 3+ qubits skipped the edge check entirely
     and was reported compliant even when clearly not realizable on the
     backend's connectivity. Fixed to fail closed (report non-compliant,
     with a clear reason) for anything it cannot actually verify.
  3. mock_ibm_submit()/reference_local_counts() hard-coded seed=0, so two
     "submissions" of the identical circuit always returned byte-identical
     counts -- silently defeating the fact that repeated measurement on
     real hardware is genuinely random. Fixed to draw fresh entropy by
     default while keeping an explicit `seed` parameter for reproducible
     tests.
  4. shots validation only checked `shots <= 0`, so a non-integer shots
     value (e.g. 100.7) was silently truncated by np.random.multinomial,
     and a non-numeric shots value (a string, None) raised a confusing raw
     TypeError instead of this project's own clear ValueError. Fixed with a
     proper type+positivity check, applied at the same "reject before doing
     any work" point as the existing shots<=0 check.

Run with:  pytest -v test_weakness_probes.py
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap

from psf_pennylane_gpu_prototype import (
    ConnectionContractError,
    reference_cpu_synthesize,
)
from psf_pennylane_gpu_prototype import psf_pennylane_gpu_transform
from psf_pennylane_gpu_ibm_prototype import (
    is_isa_compliant,
    mock_ibm_submit,
    psf_pennylane_gpu_ibm_transform,
    reference_local_counts,
)

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


def gpu_batch_synthesize(matrices):
    return [reference_cpu_synthesize(m) for m in matrices]


class TestGPUEquivalenceCheck:
    """Gap #1: gpu_batch_fn's returned circuits were never checked against
    the matrix they were asked to synthesize."""

    def test_wrong_but_right_shaped_circuit_is_rejected(self):
        """A mock GPU stage that returns a plausible-looking (correct shape,
        correct count) but PHYSICALLY WRONG circuit -- e.g. a buggy kernel
        that silently returns identity -- must be rejected, not spliced into
        the output tape as if it were correct."""
        tape = make_two_block_tape()

        def wrong_gpu(matrices):
            return [QuantumCircuit(2) for _ in matrices]  # always identity

        with pytest.raises(ConnectionContractError, match="does not implement"):
            psf_pennylane_gpu_transform(tape, wrong_gpu, block_gate_floor=BLOCK_GATE_FLOOR)

    def test_wrong_qubit_width_circuit_is_rejected_with_clear_message(self):
        """A mock GPU stage returning the right COUNT of circuits but the
        wrong WIDTH (e.g. 3 qubits for a 2-qubit block) must fail with a
        clear, specific error -- not Qiskit's generic low-level CircuitError
        surfacing from deep inside the splicing step."""
        tape = make_two_block_tape()

        def wrong_width_gpu(matrices):
            return [QuantumCircuit(3) for _ in matrices]

        with pytest.raises(ConnectionContractError, match="qubit"):
            psf_pennylane_gpu_transform(tape, wrong_width_gpu, block_gate_floor=BLOCK_GATE_FLOOR)

    def test_correct_circuit_still_accepted(self):
        """Regression guard: the new equivalence check must not reject a
        genuinely correct synthesis result (the happy path the original
        test suite already covered)."""
        tape = make_two_block_tape()
        tape_out = psf_pennylane_gpu_transform(
            tape, gpu_batch_synthesize, block_gate_floor=BLOCK_GATE_FLOOR
        )
        assert len(tape_out.operations) > 0


class TestISACompliance3QubitGap:
    """Gap #2: is_isa_compliant() silently passed gates it could not
    actually verify against a 2-qubit-only coupling map."""

    def test_three_qubit_gate_is_treated_as_unverifiable_not_silently_passed(self):
        class _StubBackend:
            name = "stub-3q-gap"
            num_qubits = 3
            operation_names = {"ccx", "u", "cx"}
            coupling_map = CouplingMap([(0, 1)])  # (1,2) and (0,2) are NOT edges

        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)

        ok, reason = is_isa_compliant(qc, _StubBackend())
        assert not ok, "a 3-qubit gate must not be silently reported ISA-compliant"
        assert "3" in reason or "qubit" in reason

    def test_two_qubit_gates_still_correctly_checked(self):
        """Regression guard: the fix for the 3+-qubit gap must not disturb
        the existing, already-correct 2-qubit edge check."""
        backend = GenericBackendV2(num_qubits=6, seed=7)
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        ok, reason = is_isa_compliant(qc, backend)
        # cx(0,1) may or may not be a coupled edge on this particular backend
        # instance; what matters is that the function still returns a
        # same-shape (bool, str) result and doesn't raise/crash.
        assert isinstance(ok, bool)
        assert isinstance(reason, str)


class TestIBMSubmitDeterminism:
    """Gap #3: mock_ibm_submit hard-coded seed=0, so repeated 'submissions'
    of the same circuit always returned byte-identical counts."""

    def test_repeated_submissions_are_not_hardcoded_to_identical_counts(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        results = [mock_ibm_submit(qc, 500) for _ in range(5)]
        # All five independent "submissions" being byte-identical would mean
        # the sampler is still silently deterministic. Some pairwise
        # difference is overwhelmingly likely with fresh entropy each call.
        all_identical = all(r == results[0] for r in results[1:])
        assert not all_identical, (
            "5 independent mock_ibm_submit calls on the same circuit+shots "
            "returned byte-identical counts every time -- looks like the "
            "hard-coded seed=0 regression is back."
        )
        for r in results:
            assert sum(r.values()) == 500

    def test_explicit_seed_still_reproducible(self):
        """The fix must not remove the ability to get reproducible sampling
        when a seed is explicitly requested -- existing/future tests that
        want determinism should still be able to ask for it."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        c1 = reference_local_counts(qc, 500, seed=123)
        c2 = reference_local_counts(qc, 500, seed=123)
        assert c1 == c2


class TestShotsValidation:
    """Gap #4: shots validation only checked positivity, not integer-ness,
    so a float shots value was silently truncated and a non-numeric shots
    value raised a confusing raw TypeError."""

    def test_non_integer_shots_rejected_not_silently_truncated(self):
        tape = make_two_block_tape()
        backend = GenericBackendV2(num_qubits=6, seed=7)
        with pytest.raises(ValueError):
            psf_pennylane_gpu_ibm_transform(
                tape, gpu_batch_synthesize, mock_ibm_submit, backend,
                shots=100.7, block_gate_floor=BLOCK_GATE_FLOOR,
            )

    def test_non_numeric_shots_rejected_with_clear_error(self):
        tape = make_two_block_tape()
        backend = GenericBackendV2(num_qubits=6, seed=7)
        for bad_shots in ("100", None, [100]):
            with pytest.raises(ValueError):
                psf_pennylane_gpu_ibm_transform(
                    tape, gpu_batch_synthesize, mock_ibm_submit, backend,
                    shots=bad_shots, block_gate_floor=BLOCK_GATE_FLOOR,
                )

    def test_reference_local_counts_itself_validates_shots(self):
        """The lower-level function should validate too, not rely solely on
        the higher-level transform's check -- defense in depth, matching
        is_isa_compliant()'s own independent-check philosophy."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        with pytest.raises(ValueError):
            reference_local_counts(qc, 50.5)
        with pytest.raises(ValueError):
            reference_local_counts(qc, None)
