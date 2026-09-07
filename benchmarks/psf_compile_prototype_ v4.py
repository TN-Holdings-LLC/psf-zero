# psf_compile.py -- Prototype v4 (Added verify option to v3 "Fully Integrated & Corrected Version")
#
# Based on the "Fully Integrated & Corrected Version" uploaded previously (force_consolidate=True +
# real-device fixes for basis_gates/optimization_level=2 + loading the genuine Rust core).
# To this, we added the 4th clue discovered in our recent investigation as a "safe opt-in":
#
# 4. Added the ability to explicitly skip the Operator()-based self-check that synthesize() 
#    performs on every iteration by setting verify=False [benchmarks/profile_synthesize_fast_vs_verified.py]
#
#    - Defaults to verify=True. Leaving it unspecified preserves identical behavior and speed.
#    - Degeneracy-detection fallbacks (for CNOT/SWAP/iSWAP/Identity, etc.) via except Exception 
#      remain active regardless of the verify value. This is essential because it is the only 
#      mechanism to know if an input genuinely requires a fallback.
#    - Setting verify=False disables only the step of re-proving the decomposition mathematics 
#      on every production call—mathematics already rigorously verified offline 
#      (worst-case 1-fidelity = 1.11e-15 across 1,000 trials in test_geometric_decompose.py, 
#      8.88e-16 across 200 trials in psf_zero_core_stub.py, reproduced on separate machines).
#    - Measurements with the stub core showed that removing this check made synthesize() 
#      itself 8.11x faster (1.2117ms/block -> 0.1494ms/block, N=2000). Accuracy was verified 
#      separately rather than per-call, yielding an identical worst-case (1-fidelity) = 8.88e-16 
#      compared to verify=True.
#    - Not yet validated on the actual psf_zero_core or real hardware. This prototype is 
#      intended strictly as an opt-in for those who wish to test it, and does not alter default behavior.
#
# Recommended Real-Device Validation Steps:
#   1. Use this file directly, or apply compile_optional_verify.patch to an existing psf_compile.py.
#   2. Re-run phase1.py / phase2.py (with basis_gates fixes, force_consolidate=True, and symmetric warm-up applied) 
#      using verify=False to re-measure the speed ratio against Qiskit.
#   3. Confirm via Operator equivalence that output circuits match the original unitaries 
#      (verifying that decomposition integrity remains intact even when verify=False bypasses runtime checking).

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
from psf_zero_core import geometric_decompose

DEFAULT_BLOCK_GATE_FLOOR = 12


@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-5
    phase_fix: bool = True
    on_unsupported: str = "keep"
    # New: Determines whether synthesize() repeatedly verifies non-degenerate results using Operator().
    # Default is True (as before). Setting to False retains degeneracy fallbacks (via except Exception)
    # while eliminating verification overhead.
    verify: bool = True


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
            # Degeneracy fallback: always active regardless of the verify value.
            return self._fallback(U_target, f"Decomposition failed or degenerate: {e}")

        # Verification: Executed only when verify=True (existing default behavior).
        if self.hyper.verify:
            fid = unitary_fidelity(U_target, qc)
            if (1.0 - fid) > self.hyper.tol:
                return self._fallback(U_target, f"Fidelity loss exceeded tolerance: {1.0 - fid:.2e}")
        return qc


def compile(
    qc: QuantumCircuit,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    verify: bool = True,
) -> QuantumCircuit:
    """
    verify (New, default True): When set to False, skips the Operator()-based self-check 
    performed by synthesize() on non-degenerate results. Does not affect fallbacks detecting 
    degeneracies (CNOT/SWAP/iSWAP/Identity, etc.). 
    In profile_synthesize_fast_vs_verified.py measurements (stub core), this made 
    synthesize() itself 8.11x faster. Because confirmation on real hardware with psf_zero_core 
    is still pending, verify on real hardware before changing any production defaults.
    """
    def worth_consolidating(dag, block):
        return len(block) > block_gate_floor

    # Specify force_consolidate=True to prevent unmerged UnitaryGate chains
    pm_consolidate = PassManager([
        Collect2qBlocks(filter_fn=worth_consolidating),
        ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True),
    ])
    qc_blocked = pm_consolidate.run(qc)

    hyper = GeodesicPSFHyper(tol=1e-5, on_unsupported="keep", verify=verify)
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
        f"block_gate_floor={block_gate_floor}; verify={verify}."
    )
    return qc_psf


def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    basis_gates: list[str] | None = None,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: int = 2,
    verify: bool = True,
) -> QuantumCircuit:
    """PSF block-compression followed by topology-aware routing and ISA basis translation.

    verify: Propagates directly to compile(). See compile() docstring for details.
    """
    qc_compressed = compile(qc, block_gate_floor=block_gate_floor, verify=verify)
    return transpile(
        qc_compressed,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=routing_optimization_level,
    )
