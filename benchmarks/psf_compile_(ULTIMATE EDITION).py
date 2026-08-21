from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, fields

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin
from qiskit.quantum_info import Operator, random_unitary

# Rust Core (psf_zero_core)
from psf_zero_core import geometric_decompose


# =========================================================
# Hyperparameters
# =========================================================
@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-6
    phase_fix: bool = True
    on_unsupported: str = "raise"  # "raise" or "keep"


# =========================================================
# Fidelity Validation
# =========================================================
def unitary_fidelity(U_target: np.ndarray, qc: QuantumCircuit) -> float:
    U_out = Operator(qc).data
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float((np.abs(tr)**2 + d) / (d * (d + 1)))


# =========================================================
# Synthesizer Core
# =========================================================
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
            (c1, c2, c3), k1, k2, global_phase = geometric_decompose(u_r, u_i)
        except Exception as e:
            return self._fallback(U_target, f"Rust Core Decomposition Failed: {e}")

        # ---------------------------------------------------------
        # Build physical circuit
        # ---------------------------------------------------------
        qc = QuantumCircuit(2)

        # K2 local rotations (applied first)
        qc.rz(k2[0][0], 0)
        qc.ry(k2[0][1], 0)
        qc.rz(k2[0][2], 0)

        qc.rz(k2[1][0], 1)
        qc.ry(k2[1][1], 1)
        qc.rz(k2[1][2], 1)

        # Cartan core (non-local)
        qc.rxx(2 * c1, 0, 1)
        qc.ryy(2 * c2, 0, 1)
        qc.rzz(2 * c3, 0, 1)

        # K1 local rotations (applied last)
        qc.rz(k1[0][0], 0)
        qc.ry(k1[0][1], 0)
        qc.rz(k1[0][2], 0)

        qc.rz(k1[1][0], 1)
        qc.ry(k1[1][1], 1)
        qc.rz(k1[1][2], 1)

        if self.hyper.phase_fix:
            qc.global_phase += global_phase

        # ---------------------------------------------------------
        # Verification / Fidelity Check
        # ---------------------------------------------------------
        fid = unitary_fidelity(U_target, qc)
        projection_fidelity_loss = 1.0 - fid

        if projection_fidelity_loss > self.hyper.tol:
            return self._fallback(
                U_target, 
                f"Fidelity loss ({projection_fidelity_loss:.2e}) exceeded tolerance ({self.hyper.tol:.2e})"
            )

        return qc


# =========================================================
# Qiskit Plugin Interface
# =========================================================
class SU4GeodesicPSFUnitarySynthesis(UnitarySynthesisPlugin):
    @property
    def max_qubits(self) -> int:
        return 2

    @property
    def min_qubits(self) -> int:
        return 2

    @property
    def supported_bases(self) -> list[str]:
        return ['rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz']

    def run(self, unitary: np.ndarray, **options) -> QuantumCircuit:
        valid_fields = {f.name for f in fields(GeodesicPSFHyper)}
        hyper_kwargs = {k: v for k, v in options.items() if k in valid_fields}
        hyper = GeodesicPSFHyper(**hyper_kwargs)

        synth = SU4GeodesicPSFSynthesizer(hyper)
        return synth.synthesize(unitary)


def get_plugin():
    return SU4GeodesicPSFUnitarySynthesis()


# =========================================================
# Demo & Sanity Check
# =========================================================
if __name__ == "__main__":
    print("Running random SU(4) synthesis tests...")
    hyper = GeodesicPSFHyper(tol=1e-6, on_unsupported="raise")
    synth = SU4GeodesicPSFSynthesizer(hyper)
    
    np.random.seed(42)
    success_count = 0
    total_tests = 10

    for i in range(total_tests):
        U = random_unitary(4).data
        # Project exactly to SU(4) to avoid trivial global phase det mismatches
        det = np.linalg.det(U)
        U_su4 = U * (det ** (-0.25))

        try:
            qc = synth.synthesize(U_su4)
            fid = unitary_fidelity(U_su4, qc)
            print(f"Test {i+1:02d} | SUCCESS | Fidelity: {fid:.12f}")
            success_count += 1
        except Exception as e:
            print(f"Test {i+1:02d} | FAILED  | Reason: {e}")

    print(f"\nCompleted: {success_count}/{total_tests} passed.")
