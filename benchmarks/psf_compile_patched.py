"""
psf_compile_patched.py -- an "improved compiler" built from this project's
own findings, with an honest account of what is real code and what is a
verified stand-in.

WHAT THIS FILE ACTUALLY IS
---------------------------
1. Everything above the line marked "PATCH STARTS HERE" below is the real,
   unmodified content of the reference copy of `psf_compile.py` we have in
   full (the file uploaded to this project, whose own header calls it
   "v3" -- it predates the `compile_for_hardware()` function). It is
   reproduced byte-for-byte except for ONE line: the Rust-core import is
   redirected to `psf_zero_core_stub` (see that file for why -- the real
   `.so` we were given is architecturally incompatible with this sandbox)
   instead of the real `psf_zero_core`. Nothing else in this section is
   changed.

2. `compile_for_hardware()` below the marker is the EXACT real function
   from the user's actual production repository (their "v6"), as they
   pasted it into this conversation, with the validated fix applied:
   a `basis_gates` parameter added and threaded through to the internal
   `transpile(...)` call, and `routing_optimization_level` defaulted to 2
   instead of 0. This fix was independently derived, then independently
   validated in `verify_compile_for_hardware_fix.py` (0/50 correctness
   failures at every level, native-gate inflation eliminated at level >=2),
   and independently re-run by the user themselves with matching results.

   `compile_for_hardware_buggy()` is the ORIGINAL, unfixed version of that
   same real function, kept here only so `test_improved_compiler_end_to_end.py`
   can measure old-vs-new on equal footing. It is not meant to be used.

3. A SECOND, independent bug fix is now also applied inside `compile()`
   itself: `ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True)`
   -- see `compile_force_consolidate.patch` for the full writeup. Found
   while investigating why the user's own real-machine run of `phase2.py`
   (their "dead zone" scale test, using the real Rust core) showed
   PSF-Zero's compile time growing linearly with qubit count and becoming
   ~4x SLOWER than Qiskit at 1000 qubits, reversing this project's own
   speed claims. Root cause: `phase2.py` builds its test circuits from
   chains of raw `UnitaryGate` instructions on each pair, which is exactly
   the case `ConsolidateBlocks`'s `force_consolidate=False` default fails
   to merge (see point 3 in `compile_force_consolidate.patch`) -- so
   `compile()` was calling the synthesizer once per ORIGINAL gate (20 times
   per pair, matching `phase2.py`'s `GATES_PER_PAIR=20`) instead of once
   per pair, multiplying both output size and compile time ~20x. This
   fix has NOT yet been validated against the real Rust core or against a
   fairly-configured Qiskit comparison (the user's `phase2.py` also has a
   separate, independent issue: its Qiskit worker calls
   `transpile(circuit, backend=None, optimization_level=3)`, which has no
   target basis and measurably does nothing to the circuit at all -- see
   `phase2_qiskit_worker.patch`) -- both are proposed fixes pending
   real-machine confirmation, not yet-confirmed root causes.

WHAT WE DO NOT CLAIM
---------------------
We do not have the user's full real "v6" `psf_compile.py` file -- only
`compile_for_hardware()` itself was ever shown to us, in the chat, not as
a file. Section 1 above (`compile()`, `SU4GeodesicPSFSynthesizer`, block
filtering, etc.) is real code, but from an older reference copy (v3), not
proven byte-identical to whatever "v6" actually contains around
`compile_for_hardware()`. `compile_for_hardware()` itself calls
`compile(qc, block_gate_floor=...)` with the same signature this v3 file
provides, so wiring them together here is a faithful, but not 100%
confirmed-identical, integration. Apply the ONE-FUNCTION patch below
directly to your real repository file instead of replacing the whole file
with this one -- see `compile_for_hardware.patch` for a minimal, literal
diff meant for exactly that.
"""

from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler import PassManager, CouplingMap

# Real Rust core `.so` given to us is a non-x86 binary and will not load in
# this sandbox -- see psf_zero_core_stub.py for the verified (1-fidelity <
# 1e-15, matching the real core's own claimed order of magnitude) stand-in
# this line substitutes in. This is the ONLY changed line in this section.
from psf_zero_core_stub import geometric_decompose

# v3: the synthesizer's own cheapest possible output is 4 local Rz.Ry.Rz
# triples (12 gates); up to 3 more RXX/RYY/RZZ gates are added only when
# the corresponding canonical angle is non-negligible. A block with this
# many or fewer ORIGINAL gates is not worth replacing.
DEFAULT_BLOCK_GATE_FLOOR = 12


@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-5
    phase_fix: bool = True
    on_unsupported: str = "keep"


def unitary_fidelity(U_target: np.ndarray, qc: QuantumCircuit) -> float:
    U_out = Operator(qc).data
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float((np.abs(tr) ** 2 + d) / (d * (d + 1)))


class SU4GeodesicPSFSynthesizer:
    def __init__(self, hyper: GeodesicPSFHyper):
        self.hyper = hyper
        self.fallback_count = 0

    def _fallback(self, U_target: np.ndarray, msg: str) -> QuantumCircuit:
        self.fallback_count += 1
        if self.hyper.on_unsupported == "raise":
            raise RuntimeError(msg)
        warnings.warn(f"{msg} -> Falling back to CX-basis synthesis.", UserWarning, stacklevel=2)
        decomposer = TwoQubitBasisDecomposer(CXGate())
        return decomposer(U_target)

    def synthesize(self, U_target: np.ndarray) -> QuantumCircuit:
        if U_target.shape != (4, 4):
            raise ValueError("Input must be a 4x4 unitary matrix.")

        u_r = U_target.real.tolist()
        u_i = U_target.imag.tolist()

        try:
            cartan_angles, k1, k2, global_phase = geometric_decompose(u_r, u_i)

            qc = QuantumCircuit(2)
            qc.global_phase = global_phase

            def local(triple, qubit):
                phi, theta, lam = triple
                qc.rz(lam, qubit)
                qc.ry(theta, qubit)
                qc.rz(phi, qubit)

            local(k2[0], 1)
            local(k2[1], 0)

            a, b, c = cartan_angles
            if abs(a) > 1e-10: qc.rxx(-2 * a, 0, 1)
            if abs(b) > 1e-10: qc.ryy(-2 * b, 0, 1)
            if abs(c) > 1e-10: qc.rzz(-2 * c, 0, 1)

            local(k1[0], 1)
            local(k1[1], 0)

        except Exception as e:
            return self._fallback(U_target, f"Decomposition failed or degenerate: {e}")

        fid = unitary_fidelity(U_target, qc)
        if (1.0 - fid) > self.hyper.tol:
            return self._fallback(U_target, f"Fidelity loss exceeded tolerance: {1.0 - fid:.2e}")
        return qc


def compile_buggy(qc: QuantumCircuit, block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR) -> QuantumCircuit:
    """ORIGINAL (unfixed) real code, verbatim -- force_consolidate omitted,
    i.e. left at Qiskit's own default (False). Do not use -- kept only as
    the "before" side of the comparison in
    diagnose_consolidate_blocks_dead_zone.py."""
    def worth_consolidating(dag, block):
        return len(block) > block_gate_floor

    pm_consolidate = PassManager([
        Collect2qBlocks(filter_fn=worth_consolidating),
        ConsolidateBlocks(kak_basis_gate=None),
    ])
    qc_blocked = pm_consolidate.run(qc)

    hyper = GeodesicPSFHyper(tol=1e-5, on_unsupported="keep")
    synth = SU4GeodesicPSFSynthesizer(hyper)

    qc_psf = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    qc_psf.global_phase = qc_blocked.global_phase

    blocks_processed = 0
    blocks_seen = 0

    for inst in qc_blocked.data:
        op = inst.operation
        qargs = inst.qubits
        cargs = inst.clbits

        if len(qargs) == 2 and op.name == "unitary":
            mat = op.to_matrix()
            if mat is not None and mat.shape == (4, 4):
                blocks_seen += 1
                before = synth.fallback_count
                synthesized_block = synth.synthesize(mat)
                if synth.fallback_count == before:
                    blocks_processed += 1
                qc_psf.compose(synthesized_block, qargs, inplace=True)
                continue

        qc_psf.append(op, qargs, cargs)

    print(
        f"      [Debug] PSF-Zero Rust Core executed for {blocks_processed}/{blocks_seen} "
        f"blocks ({synth.fallback_count} fell back to the original gate); "
        f"block_gate_floor={block_gate_floor}."
    )
    return qc_psf


def compile(qc: QuantumCircuit, block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR) -> QuantumCircuit:
    def worth_consolidating(dag, block):
        return len(block) > block_gate_floor

    # PATCHED (see compile_force_consolidate.patch for the full writeup):
    # Qiskit's ConsolidateBlocks defaults to force_consolidate=False, which
    # silently leaves a candidate block UNMERGED whenever it's made entirely
    # of pre-existing 'unitary'-named nodes (e.g. a chain of UnitaryGate
    # instructions already in the circuit, as opposed to elementary gates).
    # Without force_consolidate=True, any circuit built by repeatedly
    # appending UnitaryGate objects to the same pair -- an ordinary way to
    # build a Trotter step or QAOA layer, and exactly how this project's own
    # phase2.py "dead zone" test builds its circuits -- never gets merged at
    # all: compile() ends up calling synth.synthesize() once per ORIGINAL
    # gate instead of once per pair, multiplying both output gate count and
    # compile time by (gates per pair). Measured on phase2.py's own
    # 156/300/500/1000-qubit, 20-gates-per-pair circuits: exactly 20x more
    # output gates and 16-21x more compile time without this flag.
    pm_consolidate = PassManager([
        Collect2qBlocks(filter_fn=worth_consolidating),
        ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True),
    ])
    qc_blocked = pm_consolidate.run(qc)

    hyper = GeodesicPSFHyper(tol=1e-5, on_unsupported="keep")
    synth = SU4GeodesicPSFSynthesizer(hyper)

    qc_psf = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    qc_psf.global_phase = qc_blocked.global_phase

    blocks_processed = 0
    blocks_seen = 0

    for inst in qc_blocked.data:
        op = inst.operation
        qargs = inst.qubits
        cargs = inst.clbits

        if len(qargs) == 2 and op.name == "unitary":
            mat = op.to_matrix()
            if mat is not None and mat.shape == (4, 4):
                blocks_seen += 1
                before = synth.fallback_count
                synthesized_block = synth.synthesize(mat)
                if synth.fallback_count == before:
                    blocks_processed += 1
                qc_psf.compose(synthesized_block, qargs, inplace=True)
                continue

        qc_psf.append(op, qargs, cargs)

    print(
        f"      [Debug] PSF-Zero Rust Core executed for {blocks_processed}/{blocks_seen} "
        f"blocks ({synth.fallback_count} fell back to the original gate); "
        f"block_gate_floor={block_gate_floor}."
    )
    return qc_psf


# ============================= PATCH STARTS HERE =============================
#
# Below is `compile_for_hardware()` exactly as it exists in the user's real
# repository (as pasted into this conversation), with the fix this project
# validated applied. `compile_for_hardware_buggy()` is the original,
# UNFIXED version of the same real function, kept only for the head-to-head
# comparison in test_improved_compiler_end_to_end.py.

def compile_for_hardware_buggy(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: int = 0,
) -> QuantumCircuit:
    """ORIGINAL (unfixed) real code, verbatim. Do not use -- kept only as the
    "before" side of the comparison. `routing_optimization_level` defaults
    to 0 (routing only), and no `basis_gates` is ever passed to the internal
    `transpile(...)` call, so RXX/RYY/RZZ pass straight through unchanged
    (see diagnose_compile_for_hardware.py) -- this is the confirmed bug.
    """
    qc_compressed = compile(qc, block_gate_floor=block_gate_floor)
    return transpile(
        qc_compressed,
        coupling_map=coupling_map,
        optimization_level=routing_optimization_level,
    )


def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    basis_gates: list[str] | None = None,       # new
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: int = 2,        # was 0
) -> QuantumCircuit:
    """PATCHED real code. `basis_gates` is now threaded through to the
    internal `transpile(...)` call, so the target basis is actually known
    and RXX/RYY/RZZ (or anything else non-native) gets resynthesized to it
    instead of passed through unchanged; `routing_optimization_level`
    defaults to 2 so that resynthesis actually happens (transpile() only
    targets a basis at optimization_level >= 2 for this call shape -- see
    diagnose_compile_for_hardware.py). Validated in
    verify_compile_for_hardware_fix.py: 0/50 correctness failures at every
    level, and native `ecr` gate count drops from 12.00/6.00 (levels 0/1) to
    3.00 (levels 2/3) once `basis_gates` is supplied.
    """
    qc_compressed = compile(qc, block_gate_floor=block_gate_floor)
    return transpile(
        qc_compressed,
        coupling_map=coupling_map,
        basis_gates=basis_gates,                # new
        optimization_level=routing_optimization_level,
    )
