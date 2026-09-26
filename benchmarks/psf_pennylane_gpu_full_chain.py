"""psf_pennylane_gpu_full_chain.py -- wires psf_pennylane_gpu_real.py's REAL
lightning.gpu verification into the full PennyLane -> GPU -> IBM connection
(psf_pennylane_gpu_ibm_transform), not just as a standalone check.

Status, spelled out
--------------------
Still NOT changed: `reference_cpu_synthesize` remains the synthesis
stand-in (CPU, Qiskit's TwoQubitBasisDecomposer) -- see
psf_pennylane_gpu_real.py's own docstring for why synthesis itself was not
moved to GPU (this project's own measurements put a single 2-qubit block
far below either the noiseless (~n=20) or noisy (n=8-10) GPU/CPU crossover).

What changes here: `full_chain_with_gpu_verification()` runs the existing
mocked PennyLane -> GPU-batch -> IBM connection
(psf_pennylane_gpu_ibm_transform), but the "gpu_batch_fn" it hands in wraps
`reference_cpu_synthesize` with a REAL lightning.gpu check
(verify_block_gpu_and_cpu) on every returned block, on real RTX 4070
hardware, BEFORE any block is accepted and spliced into the circuit that
eventually goes to `route_for_backend` / `mock_ibm_submit`. A block that
fails the real-GPU check raises ConnectionContractError, exactly the same
"no silent fallback" convention as the rest of this project's mocked
connection layer -- a block is never silently passed through unverified.

This is still NOT a claim that GPU makes anything here FASTER -- no timing
is measured or reported by this file (see psf_pennylane_gpu_real.py's own
docstring on why, and this project's own publication-policy.md). It IS a
claim that, going into the IBM submission stage, every block was verified
on genuine GPU hardware, not merely trusted because a mock call returned.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2

from psf_pennylane_gpu_prototype import ConnectionContractError, reference_cpu_synthesize
from psf_pennylane_gpu_real import verify_block_gpu_and_cpu
from psf_pennylane_gpu_ibm_prototype import psf_pennylane_gpu_ibm_transform


def gpu_verified_batch_synthesize(matrices: list[np.ndarray]) -> list[QuantumCircuit]:
    """The gpu_batch_fn used by the full chain below. For each 4x4 target
    matrix: synthesize on CPU (the stand-in synthesizer, unchanged), then
    verify the result on REAL lightning.gpu hardware before returning it.
    Raises ConnectionContractError (not a silent skip) if any block fails
    the real-GPU check, so a bad block can never reach route_for_backend /
    mock_ibm_submit undetected."""
    results = []
    for i, matrix in enumerate(matrices):
        circ = reference_cpu_synthesize(matrix)
        check = verify_block_gpu_and_cpu(matrix, circ)
        if not check["gpu_pass"]:
            raise ConnectionContractError(
                f"block #{i}: real lightning.gpu verification failed "
                f"(gpu_expval_diff={check['gpu_expval_diff']:.3e}, "
                f"cpu_matrix_infidelity={check['cpu_matrix_infidelity']:.3e}) "
                "-- refusing to pass this block on to routing/submission."
            )
        results.append(circ)
    return results


def full_chain_with_gpu_verification(
    tape: "object",
    ibm_submit_fn,
    backend: BackendV2,
    shots: int = 4000,
    block_gate_floor: int = 12,
):
    """The complete connection, with every synthesized block verified on
    real GPU hardware before submission: PennyLane tape -> CPU synthesis
    (per block) -> REAL lightning.gpu verification (per block) -> hardware
    routing -> ISA compliance check -> ibm_submit_fn.

    `ibm_submit_fn` stays injected (e.g. mock_ibm_submit, or a real one once
    written) -- this function only changes what happens BEFORE submission,
    not the submission stage itself.
    """
    return psf_pennylane_gpu_ibm_transform(
        tape,
        gpu_verified_batch_synthesize,
        ibm_submit_fn,
        backend,
        shots=shots,
        block_gate_floor=block_gate_floor,
    )
