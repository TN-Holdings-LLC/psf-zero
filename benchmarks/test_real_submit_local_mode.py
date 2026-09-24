"""test_real_submit_local_mode.py -- Addendum 153.

Runs the PennyLane -> (real-GPU-verified) synthesis -> routing -> IBM
connection with the REAL SamplerV2 submission function
(psf_ibm_real_submit.make_sampler_submit_fn) in IBM's own local testing
mode. No credentials, no network. Requires qiskit-ibm-runtime and
qiskit-aer, plus lightning.gpu (the synthesis step still verifies each
block on real GPU hardware via psf_pennylane_gpu_full_chain).

Updated for Addendum 160: TVD is measured against the logical circuit
(before routing), the full-backend-width known-issue test is replaced by
one asserting the fix, and a SWAP-requiring tape checks the
logical-to-physical measurement mapping.

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
from psf_pennylane_gpu_prototype import tape_to_qiskit

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


def make_swap_tape():
    """Two blocks on NON-adjacent qubits of FakeManilaV2's linear chain
    (0-1-2-3-4): (0,2) and (1,3). Single-qubit ops on every wire come first
    so tape_to_qiskit keeps wire i as Qiskit qubit i (it numbers wires by
    first appearance). Routing must insert SWAPs, so logical qubits can end
    up on different physical qubits than they started on -- the case
    logical_measurement() exists to handle."""
    ops = [qml.RX(0.1 * (w + 1), wires=w) for w in range(4)]
    ops += _random_unitary_ops([0, 2], RUNS_PER_BLOCK, seed=11)
    ops += _random_unitary_ops([1, 3], RUNS_PER_BLOCK, seed=12)
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


def make_triangle_tape():
    """Three blocks forming a triangle, (0,1) then (1,2) then (0,2). A
    triangle cannot be embedded in FakeManilaV2's linear chain, so routing
    MUST insert at least one SWAP, whatever wire numbering tape_to_qiskit
    chooses -- unlike make_swap_tape, whose blocks ended up adjacent after
    consolidation and renumbering (Addendum 161)."""
    ops = _random_unitary_ops([0, 1], RUNS_PER_BLOCK, seed=21)
    ops += _random_unitary_ops([1, 2], RUNS_PER_BLOCK, seed=22)
    ops += _random_unitary_ops([0, 2], RUNS_PER_BLOCK, seed=23)
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


def tvd(counts: dict[str, int], tape) -> float:
    """TVD between sampled counts and the LOGICAL circuit's own exact
    distribution -- the tape converted to Qiskit BEFORE any routing (Addendum
    160). Classical bit i = logical qubit i on both sides (Qiskit convention:
    bit 0 rightmost). A wrong logical-to-physical measurement mapping shows
    up here as a large TVD; the previous version compared against the routed
    full-width circuit and could not detect that."""
    qc_logical, _ = tape_to_qiskit(tape)
    exact = Statevector(qc_logical).probabilities_dict()
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
        tape = make_two_block_tape()
        counts, _ = full_chain_with_gpu_verification(
            tape, submit, backend, shots=SHOTS, block_gate_floor=BLOCK_GATE_FLOOR
        )
        d = tvd(counts, tape)
        print(f"\nP2 noiseless TVD = {d:.4f}")
        assert d < 0.1

    # ---- P3: device-snapshot noise visible but not destructive ----
    def test_fake_device_noise_visible_not_destructive(self, backend):
        submit = make_sampler_submit_fn(backend, seed_simulator=SEED)
        tape = make_two_block_tape()
        counts, _ = full_chain_with_gpu_verification(
            tape, submit, backend, shots=SHOTS, block_gate_floor=BLOCK_GATE_FLOOR
        )
        d = tvd(counts, tape)
        print(f"\nP3 FakeManilaV2 TVD = {d:.4f}")
        assert 0.01 < d < 0.3

    # ---- Addendum 160: the P4 fix -- logical qubits only, at final positions ----
    def test_bitstring_width_equals_logical_qubits(self, backend):
        submit = make_sampler_submit_fn(backend, seed_simulator=SEED)
        counts, _ = full_chain_with_gpu_verification(
            make_two_block_tape(), submit, backend, shots=200, block_gate_floor=BLOCK_GATE_FLOOR
        )
        widths = {len(k) for k in counts}
        print(f"\nA160 P1 bitstring widths = {widths} (backend has {backend.num_qubits} qubits, circuit uses 4)")
        assert widths == {4}

    def test_wire_renumbering_keeps_logical_bit_order(self, backend):
        """Addendum 161: blocks on (0,2) and (1,3) after single-qubit ops on
        every wire. Consolidation absorbs the single-qubit ops, the
        synthesized tape's wires then first appear as 0,2,1,3, and
        tape_to_qiskit renumbers them. Before the fix, classical bits came
        back in that renumbered order (noiseless TVD 0.1598 with no SWAPs
        inserted). Bit i must mean the ORIGINAL tape's wire i."""
        submit = make_sampler_submit_fn(AerSimulator(), seed_simulator=SEED)
        tape = make_swap_tape()
        counts, qc_routed = full_chain_with_gpu_verification(
            tape, submit, backend, shots=SHOTS, block_gate_floor=BLOCK_GATE_FLOOR
        )
        d = tvd(counts, tape)
        print(f"\nA161 renumbering tape: routed 2q gates "
              f"{sum(1 for i in qc_routed.data if len(i.qubits) == 2)}, noiseless TVD = {d:.4f}")
        assert {len(k) for k in counts} == {4}
        assert d < 0.1

    def test_swap_routed_circuit_maps_logical_qubits_correctly(self, backend):
        """A triangle of blocks cannot be embedded in a linear chain, so
        routing must insert SWAPs (checked: more than the 9 CX three generic
        blocks need). Noiseless counts must still match the logical
        circuit's exact distribution -- which fails if any logical qubit is
        measured at its initial rather than final physical position."""
        submit = make_sampler_submit_fn(AerSimulator(), seed_simulator=SEED)
        tape = make_triangle_tape()
        counts, qc_routed = full_chain_with_gpu_verification(
            tape, submit, backend, shots=SHOTS, block_gate_floor=BLOCK_GATE_FLOOR
        )
        final_pos = list(qc_routed.layout.final_index_layout(filter_ancillas=True))
        twoq = sum(1 for inst in qc_routed.data if len(inst.qubits) == 2)
        d = tvd(counts, tape)
        print(f"\nA161 triangle tape: final positions {final_pos}, routed 2q gates {twoq}, noiseless TVD = {d:.4f}")
        assert twoq > 9, "no SWAP was inserted -- this test would not exercise final-position mapping"
        assert {len(k) for k in counts} == {3}
        assert d < 0.1
