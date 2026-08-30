from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, fields
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin
from qiskit.quantum_info import Operator, random_unitary

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
            cartan_qc = QuantumCircuit(2)
            if abs(a) > 1e-10: cartan_qc.rxx(2 * a, 0, 1)
            if abs(b) > 1e-10: cartan_qc.ryy(2 * b, 0, 1)
            if abs(c) > 1e-10: cartan_qc.rzz(2 * c, 0, 1)
            
            cartan_cx = transpile(cartan_qc, basis_gates=['u', 'cx'], optimization_level=3)
            qc.compose(cartan_cx, [0, 1], inplace=True)
            
        
            qc.u(k2[0][1], k2[0][0], k2[0][2], 0)
            qc.u(k2[1][1], k2[1][0], k2[1][2], 1)
            

            qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=3)
            # =================================================================

        except Exception as e:
            return self._fallback(U_target, f"Decomposition Failed: {e}")

        fid = unitary_fidelity(U_target, qc)
        if (1.0 - fid) > self.hyper.tol:
            return self._fallback(U_target, f"Fidelity loss exceeded tolerance: {1.0 - fid:.2e}")
        return qc
    
class SU4GeodesicPSFUnitarySynthesis(UnitarySynthesisPlugin):
    @property
    def max_qubits(self) -> int: return 2
    @property
    def min_qubits(self) -> int: return 2
    @property
    def supported_bases(self) -> list[str]: return ['rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz', 'cx']
    @property
    def supports_coupling_map(self) -> bool: return False
    @property
    def supports_basis_gates(self) -> bool: return False
    @property
    def supports_natural_direction(self) -> bool: return False
    @property
    def supports_gate_errors(self) -> bool: return False
    @property
    def supports_gate_lengths(self) -> bool: return False
    @property
    def supports_pulse_optimize(self) -> bool: return False

    def run(self, unitary: np.ndarray, **options) -> QuantumCircuit:
        valid_fields = {f.name for f in fields(GeodesicPSFHyper)}
        hyper_kwargs = {k: v for k, v in options.items() if k in valid_fields}
        hyper = GeodesicPSFHyper(**hyper_kwargs)
        synth = SU4GeodesicPSFSynthesizer(hyper)
        return synth.synthesize(unitary)

def get_plugin():
    return SU4GeodesicPSFUnitarySynthesis()

if __name__ == "__main__":
    print("🔬 Running standalone SU(4) synthesis test...")
    hyper = GeodesicPSFHyper(tol=1e-5, on_unsupported="raise")
    synth = SU4GeodesicPSFSynthesizer(hyper)
    np.random.seed(42)
    success_count = 0
    total_tests = 10
    for i in range(total_tests):
        U = random_unitary(4).data
        try:
            qc = synth.synthesize(U)
            fid = unitary_fidelity(U, qc)
            print(f"Test {i+1:02d} | SUCCESS | Fidelity: {fid:.12f}")
            success_count += 1
        except Exception as e:
            print(f"Test {i+1:02d} | FAILED  | Reason: {e}")
    print(f"\nCompleted: {success_count}/{total_tests} passed.")

def compile(qc: QuantumCircuit) -> QuantumCircuit:
    from qiskit import transpile
    return transpile(qc, basis_gates=['rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz', 'cx'], optimization_level=3)
