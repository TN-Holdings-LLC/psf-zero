"""psf_pennylane_gpu_prototype.py -- PROTOTYPE, NOT the real integration.

Purpose
-------
This project's roadmap (psf-zero/docs/log/06-open-questions-and-roadmap.md)
lists two items as "planned, not yet built":

    - PennyLane integration (`qml.transforms`)
    - Parallel (multi-core / GPU) execution of independent block synthesis

Neither exists in the real `psf_compile.py` yet, and neither this sandbox nor
the workplace WSL2 environment has a GPU or the real `psf_zero_core` Rust
extension available to test against. This file is therefore NOT a submission
for either roadmap item. It is a small, explicit prototype of the *plumbing*
between them -- built so the shape of the connection (what data crosses the
boundary, in what order, batched how) can be checked with both sides mocked,
before either side is actually implemented.

What is real and what is a stand-in, spelled out (per this project's own
"no silent fallback" rule -- a stand-in must say so, not quietly pass for the
real thing):

  REAL:
    - The PennyLane tape -> Qiskit QuantumCircuit conversion below performs
      genuine gate-by-gate translation for the small op set it supports:
      single-qubit gates (named) and explicit 2-qubit unitary blocks
      (`qml.QubitUnitary` <-> Qiskit's generic `unitary`/`UnitaryGate`).
      Named multi-qubit gates (e.g. CNOT/"cx") are deliberately NOT
      supported: Qiskit's own multi-qubit gate matrices and PennyLane's use
      different local basis-ordering conventions once a circuit has more
      than 2 qubits total (confirmed empirically while building this
      prototype -- see the comments in `qiskit_to_tape`), and solving that
      generally is a real, separate integration problem this connection
      test does not attempt. Every multi-qubit instruction this module
      converts is an explicit matrix, never a named gate whose "obvious"
      cross-library meaning turned out not to match.
    - `collect_and_consolidate()` uses Qiskit's own `Collect2qBlocks` /
      `ConsolidateBlocks` passes -- the identical mechanism the real
      `psf_compile.compile()` uses to find 2-qubit blocks worth
      re-synthesizing.
    - The round-trip fidelity check in the test file uses PennyLane's own
      `qml.matrix()`, independent of anything in this file.

  STAND-IN / MOCK (explicitly, not silently):
    - `reference_cpu_synthesize()` is Qiskit's own exact
      `TwoQubitBasisDecomposer(CXGate())` -- a real and correct two-qubit
      synthesizer, but it is NOT PSF-Zero's Rust core (`psf_zero_core`), and
      it runs on one CPU core, not a GPU. It stands in for "some correct
      per-block synthesizer" so the pipeline has something genuine to call.
    - `gpu_batch_synthesize()` in the test file wraps that reference
      synthesizer in a loop and calls it a "GPU batch" step. No GPU is
      involved. It exists only so the *batching contract* (one call for all
      of a tape's blocks, not one call per block) can be asserted.

  Any timing measured against this file must not be reported as a PSF-Zero
  benchmark number -- see this project's publication-policy.md and the
  psf-zero-repo-publish skill's numeric rules. This file produces no timing
  numbers on purpose.

Connection under test
----------------------
    PennyLane tape
        --(tape_to_qiskit, real conversion)-->
    Qiskit QuantumCircuit
        --(collect_and_consolidate, real Qiskit passes)-->
    Qiskit QuantumCircuit with consolidated 2-qubit `unitary` blocks
        --(extract 4x4 matrices, all of them, in one batch)-->
    gpu_batch_fn(list[np.ndarray]) -> list[QuantumCircuit]   [injected, mocked in tests]
        --(splice each returned circuit back into the consolidated circuit)-->
    Qiskit QuantumCircuit
        --(qiskit_to_tape, real conversion)-->
    PennyLane tape
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.circuit.library import CXGate
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.quantum_info import Operator

DEFAULT_BLOCK_GATE_FLOOR = 12  # matches psf_compile.py's own default

# The "correct per-block synthesizer" stand-in described above. Real and
# exact, but explicitly not psf_zero_core.
_CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())


class ConnectionContractError(RuntimeError):
    """Raised when a mocked stage does not honor the interface contract this
    prototype is testing (e.g. wrong batch size, wrong count of results).
    This project's convention is to raise loudly rather than silently
    substitute or drop data -- see psf_compile.py's own `on_unsupported`
    handling and its docstring's "no silent fallback" rationale."""


_BLOCK_EQUIVALENCE_TOLERANCE = 1e-7


def _matrix_infidelity(u_expected: np.ndarray, u_actual: np.ndarray) -> float:
    """Average-gate-fidelity-based infidelity between two same-size unitaries,
    ignoring global phase -- the identical metric this prototype's own test
    suite already uses for its end-to-end round-trip check. 0.0 means
    identical up to global phase; 1.0 means completely unrelated. Raises
    ValueError (caught by the caller and turned into a ConnectionContractError
    with context) if the shapes don't even match."""
    if u_expected.shape != u_actual.shape:
        raise ValueError(f"shape mismatch: expected {u_expected.shape}, got {u_actual.shape}")
    d = u_expected.shape[0]
    tr = np.trace(u_expected.conj().T @ u_actual)
    fidelity = (np.abs(tr) ** 2 + d) / (d * (d + 1))
    return 1.0 - float(fidelity)


# --------------------------------------------------------------------------
# 1. PennyLane <-> Qiskit conversion (real, but intentionally small: only the
#    op set this prototype's tests actually exercise).
# --------------------------------------------------------------------------

def tape_to_qiskit(tape: "qml.tape.QuantumTape", wire_order: Sequence | None = None) -> tuple[QuantumCircuit, list]:
    """Convert a PennyLane tape to a Qiskit QuantumCircuit.

    Returns (circuit, wire_order) where wire_order is the PennyLane wire
    label for each Qiskit qubit index, so the reverse conversion and the
    fidelity check can agree on qubit ordering.
    """
    # `wire_order` pins Qiskit qubit i to wire_order[i]. Without it, wires are
    # numbered by FIRST APPEARANCE in the tape, which is not stable across
    # transformations of the same circuit: a synthesized tape can list the
    # same wires in a different first-appearance order than its input, so
    # qubit i of the two resulting circuits would be different wires
    # (Addenda 161 and 166). Callers comparing or combining circuits built
    # from related tapes should pass the input tape's own wire order.
    if wire_order is None:
        wire_order = list(tape.wires)
    else:
        wire_order = list(wire_order)
        missing = [w for w in tape.wires if w not in wire_order]
        if missing:
            raise ValueError(f"tape_to_qiskit: tape wire(s) {missing} are not in wire_order {wire_order}")
    wire_index = {w: i for i, w in enumerate(wire_order)}
    qc = QuantumCircuit(len(wire_order))

    for op in tape.operations:
        qubits = [wire_index[w] for w in op.wires]
        name = op.name
        if name == "Hadamard":
            qc.h(qubits[0])
        elif name == "PauliX":
            qc.x(qubits[0])
        elif name == "PauliY":
            qc.y(qubits[0])
        elif name == "PauliZ":
            qc.z(qubits[0])
        elif name == "RX":
            qc.rx(float(op.parameters[0]), qubits[0])
        elif name == "RY":
            qc.ry(float(op.parameters[0]), qubits[0])
        elif name == "RZ":
            qc.rz(float(op.parameters[0]), qubits[0])
        elif name == "QubitUnitary":
            mat = np.asarray(op.parameters[0], dtype=complex)
            # Reversed on purpose. PennyLane reads `mat` with the FIRST listed
            # wire as the most significant index; Qiskit reads it with the
            # first listed qubit as the LEAST significant. Passing `qubits`
            # unreversed applied the qubit-reversed operation; every earlier
            # test compared this conversion against itself and could not
            # see it (Addenda 164-165).
            qc.unitary(mat, qubits[::-1])
        else:
            raise NotImplementedError(
                f"tape_to_qiskit: no mapping for PennyLane op {name!r}. "
                "This prototype's op set is deliberately small -- extend it "
                "before using this on a tape with other gates, rather than "
                "letting an unsupported op silently vanish."
            )
    return qc, wire_order


def qiskit_to_tape(qc: QuantumCircuit, wire_order: Sequence, measurements=()) -> "qml.tape.QuantumTape":
    """Convert a Qiskit QuantumCircuit back to a PennyLane tape, using
    `wire_order` to map Qiskit qubit index -> PennyLane wire label."""
    ops = []
    for inst in qc.data:
        op = inst.operation
        qubits = [wire_order[qc.find_bit(q).index] for q in inst.qubits]
        name = op.name
        if name == "h":
            ops.append(qml.Hadamard(wires=qubits[0]))
        elif name == "x":
            ops.append(qml.PauliX(wires=qubits[0]))
        elif name == "y":
            ops.append(qml.PauliY(wires=qubits[0]))
        elif name == "z":
            ops.append(qml.PauliZ(wires=qubits[0]))
        elif name == "rx":
            ops.append(qml.RX(float(op.params[0]), wires=qubits[0]))
        elif name == "ry":
            ops.append(qml.RY(float(op.params[0]), wires=qubits[0]))
        elif name == "rz":
            ops.append(qml.RZ(float(op.params[0]), wires=qubits[0]))
        elif name in ("unitary", "Unitary"):
            mat = np.asarray(Operator(op).data, dtype=complex)
            # Reversed for the same reason as in tape_to_qiskit (Addenda
            # 164-165): Qiskit's matrix has its first listed qubit least
            # significant, PennyLane expects its first listed wire most
            # significant.
            ops.append(qml.QubitUnitary(mat, wires=qubits[::-1]))
        elif len(qubits) == 1:
            # Generic fallback, SINGLE-QUBIT ONLY (e.g. Qiskit's "u", which is
            # what TwoQubitBasisDecomposer emits for local rotations). Safe:
            # a 1-qubit gate has no relative qubit ordering to get wrong, so
            # its matrix means the same thing to both libraries.
            mat = np.asarray(Operator(op).data, dtype=complex)
            ops.append(qml.QubitUnitary(mat, wires=qubits))
        else:
            # Deliberately NOT a generic fallback for multi-qubit named gates
            # (e.g. "cx"). Confirmed empirically (see this prototype's test
            # suite) that Qiskit's own multi-qubit gate matrices and
            # PennyLane's use different local basis-ordering conventions once
            # a circuit has more than 2 qubits total, so naively feeding
            # Operator(op).data into qml.QubitUnitary(..., wires=qubits) for
            # a *named* multi-qubit gate like "cx" silently produces the
            # WRONG physical operation -- this bit us once already while
            # building this prototype (a 0.94 round-trip infidelity that
            # looked like a real bug until traced to exactly this). Only
            # "unitary"/"Unitary" ops (i.e. blocks this module itself
            # produced or that ConsolidateBlocks produced) are trusted for
            # multi-qubit conversion; anything else raises rather than
            # silently miscomputing.
            raise NotImplementedError(
                f"qiskit_to_tape: {name!r} is a named multi-qubit gate this "
                "prototype does not trust for cross-library conversion "
                "(see the comment above). Collapse it to a single 'unitary' "
                "instruction first (Operator(sub_circuit).data), the way "
                "psf_pennylane_gpu_transform does for gpu_batch_fn's "
                "returned circuits, rather than converting it gate-by-gate."
            )
    return qml.tape.QuantumTape(ops, measurements=list(measurements), shots=None)


# --------------------------------------------------------------------------
# 2. Block collection (real Qiskit passes, same mechanism psf_compile.py uses)
# --------------------------------------------------------------------------

def collect_and_consolidate(qc: QuantumCircuit, block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR) -> QuantumCircuit:
    def worth_consolidating(dag, block):
        return len(block) > block_gate_floor

    pm = PassManager([
        Collect2qBlocks(filter_fn=worth_consolidating),
        ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True),
    ])
    return pm.run(qc)


def reference_cpu_synthesize(matrix: np.ndarray) -> QuantumCircuit:
    """Real, correct, single-core stand-in for a per-block synthesizer.
    NOT psf_zero_core. See module docstring."""
    return _CX_DECOMPOSER(matrix)


# --------------------------------------------------------------------------
# 3. The connection under test: PennyLane -> collect -> GPU batch -> splice -> PennyLane
# --------------------------------------------------------------------------

def psf_pennylane_gpu_transform(
    tape: "qml.tape.QuantumTape",
    gpu_batch_fn: Callable[[list[np.ndarray]], list[QuantumCircuit]],
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
) -> "qml.tape.QuantumTape":
    """The prototype connection. `gpu_batch_fn` is injected so tests can
    substitute a mock and inspect exactly how it was called -- this is the
    thing being tested, not an implementation detail to hide.

    Contract asserted here (raises ConnectionContractError if violated,
    rather than silently truncating/padding -- see this project's
    "no silent fallback" convention):
      - `gpu_batch_fn` is called AT MOST ONCE per transform call, with every
        block's matrix batched into a single list. This is the entire point
        of a GPU path: one call carrying many blocks, not one call per
        block re-paying dispatch overhead each time.
      - `gpu_batch_fn` must return exactly as many circuits as it was given
        matrices, in the same order.
      - each returned circuit must actually IMPLEMENT the same unitary as
        the matrix it was asked to synthesize (checked independently via
        _matrix_infidelity, not merely assumed because gpu_batch_fn returned
        without raising). A batch call "succeeding" with the right count is
        not proof its answers are correct -- the same reasoning
        is_isa_compliant() applies on the IBM leg to transpile()'s output.
    """
    qc, wire_order = tape_to_qiskit(tape)
    qc_blocked = collect_and_consolidate(qc, block_gate_floor=block_gate_floor)

    block_indices: list[int] = []
    matrices: list[np.ndarray] = []
    for i, inst in enumerate(qc_blocked.data):
        op = inst.operation
        if len(inst.qubits) == 2 and op.name == "unitary":
            mat = op.to_matrix()
            if mat is not None and mat.shape == (4, 4):
                block_indices.append(i)
                matrices.append(mat)

    if matrices:
        synthesized = gpu_batch_fn(matrices)
        if len(synthesized) != len(matrices):
            raise ConnectionContractError(
                f"gpu_batch_fn returned {len(synthesized)} circuit(s) for "
                f"{len(matrices)} block(s) -- the batch contract requires "
                "one result per input matrix, in order."
            )
        for i, (mat, circ) in enumerate(zip(matrices, synthesized)):
            if circ.num_qubits != 2:
                raise ConnectionContractError(
                    f"gpu_batch_fn's result #{i} has {circ.num_qubits} "
                    "qubit(s), expected 2 -- refusing to splice a "
                    "wrong-width circuit back into the consolidated circuit."
                )
            infidelity = _matrix_infidelity(mat, Operator(circ).data)
            if infidelity > _BLOCK_EQUIVALENCE_TOLERANCE:
                raise ConnectionContractError(
                    f"gpu_batch_fn's result #{i} does not implement the "
                    f"requested block's unitary (infidelity {infidelity:.3e} "
                    f"> {_BLOCK_EQUIVALENCE_TOLERANCE:.0e}) -- refusing to "
                    "silently accept a physically wrong synthesis result "
                    "just because the batch call returned without error."
                )
    else:
        synthesized = []

    synth_by_index = dict(zip(block_indices, synthesized))

    qc_out = qc_blocked.copy_empty_like()
    for i, inst in enumerate(qc_blocked.data):
        if i in synth_by_index:
            # Collapse the returned circuit to its overall 4x4 matrix and
            # splice it back in as ONE 'unitary' instruction, rather than
            # composing its individual gates (e.g. the u/cx gates a KAK-style
            # decomposer emits) directly into qc_out. This is deliberate, not
            # a loss of fidelity: it keeps every multi-qubit instruction that
            # reaches qiskit_to_tape typed as 'unitary', the only multi-qubit
            # shape this prototype trusts for PennyLane conversion (see
            # qiskit_to_tape's docstring/comment on why named multi-qubit
            # gates like 'cx' are not safely convertible here). A real
            # PennyLane integration would need to solve that conversion
            # properly (Qiskit's little-endian vs PennyLane's convention for
            # named multi-qubit gates); this prototype's job is the
            # PennyLane<->GPU-batch connection, not that separate interop
            # problem, so it is sidestepped rather than silently gotten wrong.
            combined = Operator(synth_by_index[i]).data
            qc_out.unitary(combined, inst.qubits)
        else:
            qc_out.append(inst.operation, inst.qubits, inst.clbits)

    return qiskit_to_tape(qc_out, wire_order, measurements=tape.measurements)
