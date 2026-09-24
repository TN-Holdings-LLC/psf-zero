"""test_tape_conversion_fidelity.py -- Addendum 165.

Checks the prototype's tape <-> Qiskit conversion against PennyLane's OWN
meaning of a tape (qml.matrix), never against another tape_to_qiskit
output. Every earlier connection test compared the conversion with itself,
which is why a qubit-order bug in it went undetected (Addendum 164, P5).

CPU only; no GPU or IBM backend needed.

Run with:  pytest -v test_tape_conversion_fidelity.py
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from qiskit.quantum_info import Operator, random_unitary

from psf_pennylane_gpu_prototype import (
    psf_pennylane_gpu_transform,
    qiskit_to_tape,
    reference_cpu_synthesize,
    tape_to_qiskit,
)

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
BLOCK_GATE_FLOOR = 12


def infidelity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - |tr(a^dagger b)| / d: zero iff equal up to global phase."""
    d = a.shape[0]
    return 1.0 - abs(np.trace(a.conj().T @ b)) / d


def pennylane_in_qiskit_order(tape, wire_order):
    """PennyLane's own matrix for `tape`, written in Qiskit's bit order:
    Qiskit qubit i is wire_order[i], and Qiskit's qubit 0 is the LEAST
    significant index, so PennyLane's wire_order must be reversed."""
    return qml.matrix(tape, wire_order=list(wire_order)[::-1])


def random_u(seed):
    return random_unitary(4, seed=seed).data


def mixed_tape():
    ops = [qml.Hadamard(wires=0), qml.RX(0.3, wires=1), qml.RY(0.7, wires=2), qml.RZ(0.2, wires=3)]
    ops += [
        qml.QubitUnitary(random_u(1), wires=[2, 0]),
        qml.QubitUnitary(random_u(2), wires=[3, 1]),
        qml.QubitUnitary(random_u(3), wires=[1, 2]),
        qml.PauliX(wires=3),
    ]
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


def test_t1_cnot_matches_pennylane_meaning():
    """Exactly Addendum 164's P5 case, which returned False before the fix."""
    tape = qml.tape.QuantumTape([qml.QubitUnitary(CNOT, wires=[0, 1])], measurements=[], shots=None)
    qc, wo = tape_to_qiskit(tape)
    assert np.allclose(Operator(qc).data, pennylane_in_qiskit_order(tape, wo))


def test_t2_mixed_tape_matches_pennylane_meaning():
    tape = mixed_tape()
    qc, wo = tape_to_qiskit(tape)
    inf = infidelity(Operator(qc).data, pennylane_in_qiskit_order(tape, wo))
    print(f"\nT2 infidelity = {inf:.2e}")
    assert inf < 1e-9


def test_t3_round_trip_preserves_pennylane_meaning():
    tape = mixed_tape()
    qc, wo = tape_to_qiskit(tape)
    back = qiskit_to_tape(qc, wo)
    inf = infidelity(qml.matrix(back, wire_order=wo), qml.matrix(tape, wire_order=wo))
    print(f"\nT3 infidelity = {inf:.2e}")
    assert inf < 1e-9


def test_t4_transform_preserves_pennylane_meaning():
    """The full synthesis transform (CPU stand-in synthesizer) must return a
    tape with the same PennyLane meaning as its input, on blocks acting on
    reversed and non-adjacent wire pairs."""
    runs = BLOCK_GATE_FLOOR + 3
    ops = [qml.RX(0.1 * (w + 1), wires=w) for w in range(4)]
    ops += [qml.QubitUnitary(random_u(100 + k), wires=[2, 0]) for k in range(runs)]
    ops += [qml.QubitUnitary(random_u(200 + k), wires=[3, 1]) for k in range(runs)]
    tape = qml.tape.QuantumTape(ops, measurements=[], shots=None)

    def gpu_batch(matrices):
        return [reference_cpu_synthesize(m) for m in matrices]

    out = psf_pennylane_gpu_transform(tape, gpu_batch, block_gate_floor=BLOCK_GATE_FLOOR)
    wires = [0, 1, 2, 3]
    inf = infidelity(qml.matrix(out, wire_order=wires), qml.matrix(tape, wire_order=wires))
    print(f"\nT4 infidelity = {inf:.2e}")
    assert inf < 1e-9
