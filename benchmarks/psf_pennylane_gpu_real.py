"""psf_pennylane_gpu_real.py -- adds a REAL lightning.gpu verification step
to psf_pennylane_gpu_prototype.py's connection.

Status, spelled out (per this project's own "no silent fallback" rule)
------------------------------------------------------------------------
This file changes exactly ONE thing from the mock prototype: the
CORRECTNESS CHECK on each synthesized block now runs on real hardware
(lightning.gpu, RTX 4070), not the CPU-only matrix-trace check
(_matrix_infidelity) the prototype used.

Still NOT changed, and still a stand-in:
  - The SYNTHESIS itself (turning a 4x4 unitary into a gate sequence) is
    still `reference_cpu_synthesize()` -- Qiskit's own CPU-based
    `TwoQubitBasisDecomposer`, not psf_zero_core, and not GPU-accelerated.
    This project's own measurements (Addendum 137-146, this same session)
    found lightning.gpu only wins over CPU above roughly n=20 qubits
    (noiseless) or n=8-10 qubits (noisy) -- a single 2-qubit block, the unit
    this connection works on, is far below either crossover, so running
    SYNTHESIS on GPU would not be expected to help and was not attempted.
  - The IBM leg (psf_pennylane_gpu_ibm_prototype.py) is untouched by this
    file; mock_ibm_submit() is still a stand-in, not a real submission.

What IS now real: `verify_on_gpu()` below builds an ACTUAL PennyLane
QNode on the ACTUAL `lightning.gpu` device (the same device confirmed
genuinely GPU-backed earlier this session via the CUDA_VISIBLE_DEVICES=""
check -- disabling the GPU there produced a CUDA-level RuntimeError, not a
silent CPU fallback) and computes a real expectation value from it, compared
against the same quantity computed on `default.qubit` (CPU, exact). This
replaces (for the correctness check only) the CPU-only matrix-trace
computation the prototype used, with an end-to-end check that the
synthesized circuit, RUN on lightning.gpu, reproduces the SAME PHYSICS as
the original block -- not merely that the matrices are algebraically close.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from psf_pennylane_gpu_prototype import (
    ConnectionContractError,
    _matrix_infidelity,
)

# A fixed observable and a fixed pair of test states (both computational
# basis and superposition inputs) so the GPU check exercises more than one
# point of the block's own action -- a single expectation value on |00>
# alone would not catch every kind of wrong-unitary error (e.g. a circuit
# that is correct only on |00> but wrong elsewhere).
_TEST_STATE_PREP = (
    [],  # |00>
    [qml.Hadamard(wires=0)],
    [qml.Hadamard(wires=1)],
    [qml.Hadamard(wires=0), qml.Hadamard(wires=1)],  # equal superposition
)
# Asymmetric observables (Z on one wire only) are included deliberately: the
# first version of this file used only Z0 Z1, which is symmetric under
# swapping the two qubits and therefore cannot on its own detect a
# qubit-ORDER error -- exactly the class of bug this file turned out to have
# (see verify_on_gpu's own docstring).
_OBSERVABLES = (
    lambda: qml.PauliZ(0),
    lambda: qml.PauliZ(1),
    lambda: qml.PauliZ(0) @ qml.PauliZ(1),
)

_GPU_INFIDELITY_TOLERANCE = 1e-6  # looser than the CPU matrix check's 1e-7,
# to leave headroom for lightning.gpu's own floating-point path being a
# different numerical route than the CPU matrix-trace computation, not
# because the physics is expected to differ.


def _qiskit_circuit_to_qnode_body(qc: QuantumCircuit, prep_ops):
    """Builds a PennyLane circuit body: apply `prep_ops`, then `qc`'s own
    gates (reusing psf_pennylane_gpu_prototype's own qiskit_to_tape
    conversion logic indirectly, by delegating single-qubit gate mapping
    inline here rather than importing a private helper -- this stays
    self-contained and does not reach into that module's internals)."""
    def body():
        for op in prep_ops:
            qml.apply(op)
        for inst in qc.data:
            op = inst.operation
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            name = op.name
            if name == "cx":
                qml.CNOT(wires=qubits)
            elif len(qubits) == 1:
                mat = np.asarray(Operator(op).data, dtype=complex)
                qml.QubitUnitary(mat, wires=qubits)
            else:
                raise ConnectionContractError(
                    f"verify_on_gpu: unexpected multi-qubit gate {name!r} in "
                    "a synthesized 2-qubit block -- reference_cpu_synthesize "
                    "is expected to emit only single-qubit rotations and cx."
                )
        return [qml.expval(obs()) for obs in _OBSERVABLES]
    return body


def verify_on_gpu(target_matrix: np.ndarray, synthesized_circuit: QuantumCircuit) -> float:
    """Runs `synthesized_circuit` on lightning.gpu and on default.qubit
    (both driven by the SAME target unitary applied directly, for the
    default.qubit side), across the three test-state preparations above,
    and returns the worst (largest) expectation-value difference found.

    This is a REAL execution on REAL GPU hardware -- not a mock, not a
    matrix-only check. Raises ConnectionContractError (this project's own
    "no silent fallback" convention) if lightning.gpu is not actually
    available, rather than silently falling back to CPU and reporting a
    result that did not actually test the GPU.
    """
    try:
        dev_gpu = qml.device("lightning.gpu", wires=2)
    except Exception as exc:
        raise ConnectionContractError(
            "verify_on_gpu: lightning.gpu device could not be created "
            f"({type(exc).__name__}: {exc}) -- refusing to silently fall "
            "back to CPU for what is supposed to be a GPU verification "
            "step. Install pennylane-lightning[gpu] and confirm CUDA is "
            "available (see this session's own CUDA_VISIBLE_DEVICES=\"\" "
            "check for how to confirm the device is genuinely GPU-backed, "
            "not silently CPU)."
        )
    dev_cpu = qml.device("default.qubit", wires=2)

    worst_diff = 0.0
    for prep_ops in _TEST_STATE_PREP:
        qnode_gpu = qml.QNode(
            _qiskit_circuit_to_qnode_body(synthesized_circuit, prep_ops), dev_gpu
        )
        vals_gpu = [float(v) for v in qnode_gpu()]

        def cpu_body(prep_ops=prep_ops):
            for op in prep_ops:
                qml.apply(op)
            # wires=[1, 0], NOT [0, 1]. `target_matrix` comes from Qiskit
            # (qubit 0 is the LEAST significant index); qml.QubitUnitary
            # reads its matrix with the FIRST listed wire as the MOST
            # significant. The GPU side above places gates gate-by-gate on
            # wire == Qiskit qubit index, which is already correct, so only
            # this CPU reference needs the reversal. The first version used
            # wires=[0, 1] and compared the GPU result against the
            # qubit-order-REVERSED operation; found from a real failing run
            # (gpu_expval_diff=4.623e-01 alongside
            # cpu_matrix_infidelity=1.110e-15 -- the synthesis itself was
            # correct, the reference was not). Confirmed by a pure-numpy
            # check of the two index conventions before this change.
            qml.QubitUnitary(target_matrix, wires=[1, 0])
            return [qml.expval(obs()) for obs in _OBSERVABLES]

        qnode_cpu = qml.QNode(cpu_body, dev_cpu)
        vals_cpu = [float(v) for v in qnode_cpu()]

        worst_diff = max(worst_diff, max(abs(a - b) for a, b in zip(vals_gpu, vals_cpu)))

    return worst_diff


def verify_block_gpu_and_cpu(target_matrix: np.ndarray, synthesized_circuit: QuantumCircuit) -> dict:
    """Runs BOTH the original CPU-only matrix check (from the prototype)
    AND the new real-GPU execution check, and returns both results -- so a
    caller (or a test) can compare them directly rather than trusting either
    one alone. Matches this project's own repeated practice of an
    independent second check, not a replacement that silently drops the
    first."""
    cpu_matrix_infidelity = _matrix_infidelity(target_matrix, Operator(synthesized_circuit).data)
    gpu_expval_diff = verify_on_gpu(target_matrix, synthesized_circuit)
    return dict(
        cpu_matrix_infidelity=cpu_matrix_infidelity,
        gpu_expval_diff=gpu_expval_diff,
        gpu_pass=gpu_expval_diff < _GPU_INFIDELITY_TOLERANCE,
    )
