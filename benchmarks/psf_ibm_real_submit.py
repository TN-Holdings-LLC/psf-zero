"""psf_ibm_real_submit.py -- a REAL IBM submission function for the
PennyLane -> GPU -> IBM connection, replacing mock_ibm_submit.

Uses qiskit_ibm_runtime.SamplerV2, the same primitive real-hardware
submission uses. Verified (Addendum 153) in IBM's own documented "local
testing mode": passing a fake backend from qiskit_ibm_runtime.fake_provider
(or a Qiskit Aer simulator) as `mode` runs the job locally, with no
credentials and no network call. Moving to real hardware changes only the
backend object passed as `mode`.

Credentials
-----------
Nothing in this file accepts, stores, or prints an API key. For real
hardware, save an account ONCE, typed directly into a terminal (never into
code, a repository, or a chat):

    python -c "from qiskit_ibm_runtime import QiskitRuntimeService; \\
        QiskitRuntimeService.save_account(channel='ibm_quantum_platform', \\
        token='<paste in terminal only>', instance='<instance CRN>')"

then call get_saved_account_backend(name) below, which reads that saved
account and takes no token argument by design.

Measurement width (Addendum 153 P4, fixed in Addendum 160)
---------------------------------------------------------
psf_pennylane_gpu_ibm_transform() used to call measure_all() on the circuit
routed to the FULL backend, measuring every physical qubit. It now measures
only the logical qubits, at their final routed positions
(logical_measurement()), so results are as wide as the circuit, not the
device.
"""
from __future__ import annotations

from typing import Callable

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2

from psf_pennylane_gpu_ibm_prototype import _validate_shots


def make_sampler_submit_fn(mode, seed_simulator: int | None = None) -> Callable[[QuantumCircuit, int], dict[str, int]]:
    """Returns submit(circuit, shots) -> counts, backed by a real
    SamplerV2(mode=mode) call. `mode` is a fake backend or Aer simulator
    for local testing, or a real backend from get_saved_account_backend().

    `seed_simulator` is a simulator option: IBM's own local-testing-mode
    example uses it to get fixed results, and it has no meaning on real
    hardware. None means fresh randomness each call, matching the mock's
    own fixed behaviour (Addendum 152, weakness #3)."""
    options = {}
    if seed_simulator is not None:
        options["simulator"] = {"seed_simulator": seed_simulator}
    sampler = SamplerV2(mode=mode, options=options) if options else SamplerV2(mode=mode)

    def submit(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
        _validate_shots(shots)
        if not any(inst.operation.name == "measure" for inst in circuit.data):
            raise ValueError(
                "circuit has no measurements -- SamplerV2 requires measured "
                "circuits; refusing to submit rather than return an empty result."
            )
        result = sampler.run([circuit], shots=int(shots)).result()
        # join_data() merges every classical register into one bit array,
        # so this does not depend on the register being named "meas".
        return dict(result[0].join_data().get_counts())

    return submit


def get_saved_account_backend(name: str):
    """For 2026-09-28 real-hardware use only; not called by any test.
    Reads an account saved beforehand via QiskitRuntimeService.save_account
    (typed in a terminal). Deliberately takes no token argument."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    return service.backend(name)
