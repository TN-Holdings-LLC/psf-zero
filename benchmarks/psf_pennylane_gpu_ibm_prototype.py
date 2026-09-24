"""psf_pennylane_gpu_ibm_prototype.py -- extends the PennyLane<->GPU connection
prototype with a third mocked leg: submission to an IBM-Quantum-style backend.

PROTOTYPE, NOT the real integration -- same status as
psf_pennylane_gpu_prototype.py, which this file builds on (imports its
tape_to_qiskit / qiskit_to_tape / collect_and_consolidate /
psf_pennylane_gpu_transform / ConnectionContractError rather than
duplicating them).

Why this exists
----------------
The user asked, after the PennyLane<->GPU connection test, for the same
treatment extended to include IBM: reproduce PennyLane, GPU AND IBM as mocks
in this sandbox (which has no GPU and no IBM Quantum credentials) and check
that the CONNECTIONS between all three hold up, including under "curveball"
inputs (a backend too small for the circuit, an invalid shot count, a
submission failure) -- not just the happy path.

This project's own reference script for real-hardware execution
(psf-zero/docs/log/05-fidelity.md's `real_device_15q_fidelity_v2.py`) uses
`qiskit_ibm_runtime.QiskitRuntimeService` + `SamplerV2` against a real IBM
backend. Neither package nor any IBM credentials are available here (and
none should ever be typed into a chat or committed to a repo -- see this
project's psf-zero-repo-publish skill). So, exactly as with the GPU leg, the
"IBM" side here is an explicit, labeled stand-in behind an injected callable,
never a real network call.

REAL vs STAND-IN, spelled out again (this project's "no silent fallback"
rule: a stand-in must say so):

  REAL:
    - `GenericBackendV2` (Qiskit's own class, `qiskit.providers.fake_provider`)
      supplies a genuine coupling map, basis gate set and `Target` -- not
      IBM's actual device, but a real, structurally valid backend object of
      the same kind `compile_for_hardware()` expects, standing in for
      "some IBM-like backend" the way `GenericBackendV2` already is meant to.
    - Routing/translation is Qiskit's own `transpile()`, the same call
      `compile_for_hardware()` makes.
    - `is_isa_compliant()` independently re-checks the routed circuit against
      the backend's own coupling map and operation names -- the same concern
      `compile_for_hardware()`'s own docstring raises about "ISA-submittable".
    - `reference_local_counts()` samples from the EXACT statevector
      (noiseless), so it is mathematically correct sampling, just not on real
      hardware and not through IBM's cloud.

  STAND-IN / MOCK (explicitly, not silently):
    - `mock_ibm_submit()` wraps `reference_local_counts()` and stands in for
      an actual `QiskitRuntimeService`/`SamplerV2` job submission. No network
      call, no credentials, no real device. It exists only so the
      *submission contract* (called once per circuit, with the shots asked
      for, returning a counts dict that sums to shots) can be asserted, and
      so a submission failure can be injected and checked for correct
      propagation.
    - This prototype always measures every LOGICAL qubit in the
      computational basis, at its final routed position
      (`logical_measurement()`; before Addendum 160 it was `measure_all()`
      over every physical qubit of the backend). Converting PennyLane's own
      richer measurement
      process types (`qml.expval`, `qml.probs`, `qml.sample`, `qml.counts`,
      ...) into the corresponding hardware execution and postprocessing is a
      separate, real integration problem this connection test does not
      attempt.

  No timing numbers are produced or reported by this file, and none should be
  quoted from it as a PSF-Zero benchmark result (see publication-policy.md).

Connection under test
----------------------
    PennyLane tape
        --(psf_pennylane_gpu_transform, from the other prototype)-->
    PennyLane tape (GPU-mock-synthesized blocks)
        --(tape_to_qiskit, real conversion)-->
    Qiskit QuantumCircuit
        --(transpile against backend, real Qiskit, pinned identity layout)-->
    ISA-compliant Qiskit QuantumCircuit for `backend`
        --(logical_measurement: logical qubits only, at final routed positions; then hand to ibm_submit_fn)-->
    ibm_submit_fn(circuit, shots) -> counts: dict[str, int]   [injected, mocked in tests]
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.providers import BackendV2
from qiskit.quantum_info import Statevector

from psf_pennylane_gpu_prototype import (  # noqa: F401 -- re-exported for convenience
    ConnectionContractError,
    collect_and_consolidate,
    psf_pennylane_gpu_transform,
    qiskit_to_tape,
    tape_to_qiskit,
)

DEFAULT_SHOTS = 4000


def is_isa_compliant(qc: QuantumCircuit, backend: BackendV2) -> tuple[bool, str]:
    """Independently check that `qc` is submittable to `backend`: every gate
    name is one the backend reports, and every 2+-qubit gate acts on a
    coupled edge. Returns (ok, reason) rather than a bare bool so a caller
    (or a test) gets a specific, actionable message instead of a bare False
    -- this project's convention (see psf_compile.py's fallback reporting)
    is to say WHY something failed, not just that it did."""
    allowed = set(backend.operation_names)
    edges = set(map(tuple, backend.coupling_map.get_edges())) if qc.num_qubits > 1 else set()
    for inst in qc.data:
        op = inst.operation
        if op.name in ("barrier", "measure", "reset", "delay"):
            continue
        if op.name not in allowed:
            return False, f"gate {op.name!r} is not in backend.operation_names ({sorted(allowed)})"
        if len(inst.qubits) >= 2:
            idxs = tuple(qc.find_bit(q).index for q in inst.qubits)
            if len(idxs) == 2:
                if idxs not in edges:
                    return False, f"gate {op.name!r} on qubits {idxs} is not a coupled edge on {backend.name}"
            else:
                # A coupling map only ever encodes PAIRWISE (2-qubit)
                # connectivity, so a gate touching 3+ qubits cannot actually
                # be verified against it here -- there is no notion of
                # "3-qubit adjacency" to check. The previous version of this
                # function silently skipped the edge check for this case and
                # returned True, which defeats the entire point of an
                # independent compliance check (a real IBM-style backend's
                # basis is 1- and 2-qubit gates only, so a 3+-qubit gate
                # reaching this point is already a sign something upstream
                # is wrong). Fail closed instead of silently passing
                # something this function cannot actually confirm.
                return False, (
                    f"gate {op.name!r} acts on {len(idxs)} qubits {idxs}; "
                    "is_isa_compliant() can only verify 1- and 2-qubit gates "
                    "against backend.coupling_map (a coupling map has no "
                    "notion of 3+-qubit adjacency) -- treating an "
                    "unverifiable multi-qubit gate as non-compliant rather "
                    "than silently reporting it OK."
                )
    return True, ""


def route_for_backend(qc: QuantumCircuit, backend: BackendV2, seed_transpiler: int = 0) -> QuantumCircuit:
    """Route + translate `qc` for `backend`. Pinned to the identity layout
    (`initial_layout=list(range(qc.num_qubits))`) deliberately: this
    prototype's fidelity check compares statevectors before and after
    routing directly, and pinning the layout sidesteps having to also track
    an arbitrary logical<->physical permutation to do that comparison. A real
    integration would not pin this -- see psf_compile.py's own
    `compile_for_hardware()` for why leaving layout free usually wins on
    hardware with real error rates (VF2PostLayout etc.)."""
    if qc.num_qubits > backend.num_qubits:
        raise ConnectionContractError(
            f"circuit needs {qc.num_qubits} qubits but backend {backend.name!r} "
            f"only has {backend.num_qubits} -- refusing to silently truncate "
            "or drop qubits. Pick a bigger backend or a smaller circuit."
        )
    return transpile(
        qc,
        backend=backend,
        initial_layout=list(range(qc.num_qubits)),
        seed_transpiler=seed_transpiler,
        optimization_level=1,
    )


def _validate_shots(shots: object) -> None:
    """Shared shots validation: must be a real (non-bool) integer and
    positive. Previously only `shots <= 0` was checked, which meant a
    non-integer value like 100.7 was silently truncated by
    np.random.multinomial (a silent-fallback bug this project's own
    convention says not to allow), and a non-numeric value (a string, None)
    raised a confusing raw TypeError from the `<=` comparison instead of a
    clear, actionable ValueError."""
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)):
        raise ValueError(f"shots must be a positive integer, got {shots!r} ({type(shots).__name__}).")
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots!r}.")


def _measured_qargs(qc: QuantumCircuit) -> list[int]:
    """Qubit index measured into each classical bit, in classical-bit order.
    Raises if the circuit has no measurements or a classical bit is measured
    twice / left unmeasured -- ambiguous results are rejected, not guessed."""
    by_clbit: dict[int, int] = {}
    for inst in qc.data:
        if inst.operation.name == "measure":
            c = qc.find_bit(inst.clbits[0]).index
            if c in by_clbit:
                raise ValueError(f"classical bit {c} is measured more than once")
            by_clbit[c] = qc.find_bit(inst.qubits[0]).index
    if not by_clbit:
        raise ValueError("circuit has no measurements")
    if sorted(by_clbit) != list(range(len(by_clbit))):
        raise ValueError(f"classical bits measured are not contiguous from 0: {sorted(by_clbit)}")
    return [by_clbit[c] for c in range(len(by_clbit))]


def reference_local_counts(qc: QuantumCircuit, shots: int, seed: int | None = None) -> dict[str, int]:
    """Real (exact-statevector) sampling, standing in for hardware execution.
    NOT IBM hardware, NOT noisy -- see module docstring.

    Returns counts over the MEASURED qubits only, keyed in classical-bit
    order (classical bit 0 is the rightmost character, Qiskit's own
    convention) -- the same meaning a real SamplerV2 result has. A previous
    version ignored which qubits were measured and always returned every
    qubit of the circuit; that went unnoticed only because every caller used
    measure_all(), and it would have silently disagreed with the real
    submission path once measurement was restricted to the logical qubits
    (Addendum 160). For measure_all() circuits the result is unchanged.

    `seed=None` (the default) draws fresh entropy every call, so repeated
    "submissions" of the identical circuit behave like repeated measurements
    on real hardware (independently sampled, not byte-identical). Pass an
    explicit `seed` for reproducible tests -- a previous version of this
    function hard-coded seed=0, which made every call deterministic and
    silently defeated the point of sampling more than once."""
    _validate_shots(shots)
    qargs = _measured_qargs(qc)
    qc_unitary_part = qc.remove_final_measurements(inplace=False)
    probs = Statevector(qc_unitary_part).probabilities_dict(qargs=qargs)
    bitstrings = list(probs.keys())
    p = np.array([probs[b] for b in bitstrings])
    p = p / p.sum()  # guard against float drift
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(shots, p)
    return {b: int(c) for b, c in zip(bitstrings, draws) if c > 0}


def mock_ibm_submit(qc: QuantumCircuit, shots: int, seed: int | None = None) -> dict[str, int]:
    """The mocked 'IBM' stage: real sampling math (see
    reference_local_counts), explicitly NOT a real IBM Runtime submission.
    Wrapped in a MagicMock in tests so the submission contract (called once,
    with the right circuit and shot count) can be asserted. `seed` is
    optional and forwarded as-is (None -> fresh entropy each call, see
    reference_local_counts)."""
    return reference_local_counts(qc, shots, seed=seed)


def logical_measurement(qc_routed: QuantumCircuit, logical_to_circuit: list[int] | None = None) -> QuantumCircuit:
    """Measure ONLY the logical qubits, each at its FINAL routed physical
    position, into classical bits 0..n-1 in logical order.

    Replaces measure_all() on the routed circuit (Addendum 153 P4 / Addendum
    155): that measured every physical qubit of the backend -- 127-bit
    results on a 127-qubit device and an infeasible local simulation.
    Routing can insert SWAPs that move a logical qubit to a different
    physical qubit than it started on, so the position comes from the routed
    circuit's own final layout, not from the initial layout. Raises rather
    than guessing if the routed circuit carries no layout.

    `logical_to_circuit[i]` is the pre-routing circuit qubit that holds
    logical (tape) wire i. It exists because tape_to_qiskit numbers wires by
    FIRST APPEARANCE, which can differ between the original tape and the
    synthesized tape (single-qubit ops get absorbed into consolidated
    blocks, changing which wire appears first). Found by Addendum 160's own
    SWAP test, which reported a wrong distribution with NO swaps inserted:
    classical bits were coming back in the synthesized tape's wire order,
    not the original's. None means identity (circuit qubit i is logical i)."""
    if qc_routed.layout is None:
        raise ConnectionContractError(
            "routed circuit has no layout -- cannot tell where each logical "
            "qubit ended up, so refusing to guess which qubits to measure."
        )
    final_pos = list(qc_routed.layout.final_index_layout(filter_ancillas=True))
    if logical_to_circuit is None:
        logical_to_circuit = list(range(len(final_pos)))
    if sorted(logical_to_circuit) != list(range(len(final_pos))):
        raise ConnectionContractError(
            f"logical_to_circuit {logical_to_circuit} is not a permutation of "
            f"the routed circuit's {len(final_pos)} logical qubits."
        )
    measured = qc_routed.copy()
    creg = ClassicalRegister(len(final_pos), "meas")
    measured.add_register(creg)
    for logical, circ_q in enumerate(logical_to_circuit):
        measured.measure(final_pos[circ_q], creg[logical])
    return measured


def psf_pennylane_gpu_ibm_transform(
    tape: "object",
    gpu_batch_fn: Callable[[list[np.ndarray]], list[QuantumCircuit]],
    ibm_submit_fn: Callable[[QuantumCircuit, int], dict[str, int]],
    backend: BackendV2,
    shots: int = DEFAULT_SHOTS,
    block_gate_floor: int = 12,
) -> tuple[dict[str, int], QuantumCircuit]:
    """The full three-stage connection: PennyLane -> GPU-batch mock synthesis
    -> hardware routing -> IBM-mock submission. Returns (counts, routed
    circuit) so tests can inspect either.

    Contracts asserted here (raise ConnectionContractError / ValueError
    rather than silently doing something else -- this project's "no silent
    fallback" convention):
      - the circuit handed to `ibm_submit_fn` is ISA-compliant for `backend`
        (checked independently via is_isa_compliant, not merely assumed
        because transpile() ran);
      - a circuit needing more qubits than the backend has is rejected
        before ever reaching `ibm_submit_fn`, not silently truncated;
      - `shots` must be a positive integer (not just ">0" -- a non-integer
        like 100.7 or a non-numeric value is rejected too, see
        _validate_shots).
    """
    _validate_shots(shots)

    tape_synth = psf_pennylane_gpu_transform(tape, gpu_batch_fn, block_gate_floor=block_gate_floor)
    # Build the circuit in the ORIGINAL tape's wire order, so routed qubit i
    # is always the input tape's wire i (Addendum 166). tape_to_qiskit's
    # default first-appearance numbering can differ between `tape` and
    # `tape_synth`. The explicit logical_to_circuit mapping below (Addendum
    # 161) is kept as a second guard; with this order it is the identity.
    qc_synth, synth_wire_order = tape_to_qiskit(tape_synth, wire_order=list(tape.wires))
    synth_index = {w: i for i, w in enumerate(synth_wire_order)}
    missing = [w for w in tape.wires if w not in synth_index]
    if missing:
        raise ConnectionContractError(
            f"wire(s) {missing} of the input tape are absent from the "
            "synthesized tape -- refusing to guess what to measure for them."
        )
    logical_to_circuit = [synth_index[w] for w in tape.wires]

    qc_routed = route_for_backend(qc_synth, backend)

    ok, reason = is_isa_compliant(qc_routed, backend)
    if not ok:
        raise ConnectionContractError(
            f"routed circuit is not ISA-compliant for backend {backend.name!r}: {reason}"
        )

    qc_measured = logical_measurement(qc_routed, logical_to_circuit)

    counts = ibm_submit_fn(qc_measured, shots)
    return counts, qc_routed
