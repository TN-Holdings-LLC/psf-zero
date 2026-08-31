from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, fields
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler import PassManager

@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-5
    phase_fix: bool = True
    on_unsupported: str = "raise"

def unitary_fidelity(U_target: np.ndarray, qc: QuantumCircuit) -> float:
    U_out = Operator(qc).data
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float((np.abs(tr)**2 + d) / (d * (d + 1)))

class SU4GeodesicPSFSynthesizer:
    def __init__(self, hyper: GeodesicPSFHyper):
        self.hyper = hyper

    def _fallback(self, U_target: np.ndarray, msg: str) -> QuantumCircuit:
        if self.hyper.on_unsupported == "raise":
            raise RuntimeError(msg)
        else:
            warnings.warn(f"{msg} -> Keeping original gate.", UserWarning)
            qc = QuantumCircuit(2)
            qc.append(UnitaryGate(U_target), [0, 1])
            return qc

    def synthesize(self, U_target: np.ndarray) -> QuantumCircuit:
        if U_target.shape != (4, 4):
            raise ValueError("Input must be a 4x4 unitary matrix.")
        
        u_r = U_target.real.tolist()
        u_i = U_target.imag.tolist()
        
        try:

            from psf_zero_core import geometric_decompose
            cartan_angles, k1, k2, global_phase = geometric_decompose(u_r, u_i)
            
            qc = QuantumCircuit(2)
            qc.global_phase = global_phase
            
            qc.u(k1[0][1], k1[0][0], k1[0][2], 0)
            qc.u(k1[1][1], k1[1][0], k1[1][2], 1)
            
            a, b, c = cartan_angles
            if abs(a) > 1e-10: qc.rxx(2 * a, 0, 1)
            if abs(b) > 1e-10: qc.ryy(2 * b, 0, 1)
            if abs(c) > 1e-10: qc.rzz(2 * c, 0, 1)
            
            qc.u(k2[0][1], k2[0][0], k2[0][2], 0)
            qc.u(k2[1][1], k2[1][0], k2[1][2], 1)
            
        except Exception as e:
            return self._fallback(U_target, f"Decomposition Failed: {e}")

        fid = unitary_fidelity(U_target, qc)
        if (1.0 - fid) > self.hyper.tol:
            return self._fallback(U_target, f"Fidelity loss exceeded tolerance: {1.0 - fid:.2e}")
        return qc

def compile(qc: QuantumCircuit) -> QuantumCircuit:

    
    pm_consolidate = PassManager([Collect2qBlocks(), ConsolidateBlocks(kak_basis_gate=None)])
    qc_blocked = pm_consolidate.run(qc)
    
    hyper = GeodesicPSFHyper(tol=1e-5, on_unsupported="raise")
    synth = SU4GeodesicPSFSynthesizer(hyper)
    
    qc_psf = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    qc_psf.global_phase = qc_blocked.global_phase 
    
    blocks_processed = 0
    
    for inst in qc_blocked.data:
        op = inst.operation
        qargs = inst.qubits
        cargs = inst.clbits
        
        if len(qargs) == 2 and hasattr(op, 'to_matrix'):
            try:
                mat = op.to_matrix()
                if mat.shape == (4, 4):
                    synthesized_block = synth.synthesize(mat)
                    qc_psf.compose(synthesized_block, qargs, inplace=True)
                    blocks_processed += 1
                    continue
            except Exception:
                pass
                

        qc_psf.append(op, qargs, cargs)
            
    print(f"      [Debug] PSF-Zero Rust Core executed for {blocks_processed} blocks.")
    return qc_psf
