# psf_compile.py -- Latest version (equipped with entangling_basis option)
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

# Reuse Qiskit's rigorous optimal CX decomposer
_CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())


@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-5
    phase_fix: bool = True
    on_unsupported: str = "keep"
    entangling_basis: str = "canonical"  # "canonical" (default) | "cx" (direct native CX output)


def unitary_fidelity(U_target: np.ndarray, qc: QuantumCircuit) -> float:
    U_out = Operator(qc).data
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float((np.abs(tr) ** 2 + d) / (d * (d + 1)))


class SU4GeodesicPSFSynthesizer:
    def __init__(self, hyper: GeodesicPSFHyper, verify: bool = True):
        self.hyper = hyper
        self.fallback_count = 0
        self.verify = verify

    def _fallback(self, U_target: np.ndarray, msg: str) -> QuantumCircuit:
        self.fallback_count += 1
        if self.hyper.on_unsupported == "raise":
            raise RuntimeError(msg)
        warnings.warn(f"{msg} -> Falling back to CX-basis synthesis.", UserWarning, stacklevel=2)
        return _CX_DECOMPOSER(U_target)

    def _entangling_core(self, a: float, b: float, c: float) -> QuantumCircuit:
        core = QuantumCircuit(2)
        if abs(a) > 1e-10: core.rxx(-2 * a, 0, 1)
        if abs(b) > 1e-10: core.ryy(-2 * b, 0, 1)
        if abs(c) > 1e-10: core.rzz(-2 * c, 0, 1)
        if self.hyper.entangling_basis == "cx":
            if len(core.data) == 0:
                return core
            core_matrix = Operator(core).data
            return _CX_DECOMPOSER(core_matrix)
        return core

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
            qc.compose(self._entangling_core(a, b, c), [0, 1], inplace=True)

            local(k1[0], 1)
            local(k1[1], 0)

        except Exception as e:
            return self._fallback(U_target, f"Decomposition failed or degenerate: {e}")

        if self.verify:
            fid = unitary_fidelity(U_target, qc)
            if (1.0 - fid) > self.hyper.tol:
                return self._fallback(U_target, f"Fidelity loss exceeded tolerance: {1.0 - fid:.2e}")
        return qc


def compile(
    qc: QuantumCircuit,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    verify: bool = True,
    entangling_basis: str = "canonical",
) -> QuantumCircuit:
    def worth_consolidating(dag, block):
        return len(block) > block_gate_floor

    pm_consolidate = PassManager([
        Collect2qBlocks(filter_fn=worth_consolidating),
        ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True),
    ])
    qc_blocked = pm_consolidate.run(qc)

    hyper = GeodesicPSFHyper(tol=1e-5, on_unsupported="keep", entangling_basis=entangling_basis)
    synth = SU4GeodesicPSFSynthesizer(hyper, verify=verify)

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
        f"blocks ({synth.fallback_count} fell back); "
        f"block_gate_floor={block_gate_floor}; verify={verify}; entangling_basis={entangling_basis}."
    )
    return qc_psf


def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    basis_gates: list[str] | None = None,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: int = 2,
    verify: bool = True,
    entangling_basis: str = "canonical",
) -> QuantumCircuit:
    qc_compressed = compile(
        qc, 
        block_gate_floor=block_gate_floor, 
        verify=verify, 
        entangling_basis=entangling_basis
    )
    return transpile(
        qc_compressed,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=routing_optimization_level,
    )
