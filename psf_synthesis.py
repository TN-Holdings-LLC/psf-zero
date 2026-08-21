"""
SU4 Geodesic PSF Synthesizer — CORRECTED
Qiskit UnitarySynthesisPlugin + Rust Core Integration
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, fields

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

# Rust Core (psf_zero_core)
from psf_zero_core import geometric_decompose


# =========================================================
# Hyperparameters (Minimal & Deterministic)
# =========================================================
@dataclass
class GeodesicPSFHyper:
    """
    Hyperparameters for deterministic geometric synthesis.
    No learning rate, no randomness — pure geometry.
    """
    tol: float = 1e-6
    phase_fix: bool = True
    verify: bool = True  # Enable self-verification


# =========================================================
# Fidelity Validation
# =========================================================
def unitary_fidelity(U_target: np.ndarray, qc: QuantumCircuit) -> float:
    """Simple fidelity proxy for validation. 1.0 = perfect match."""
    U_out = np.array(qc.to_gate().to_matrix(), dtype=complex)
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float((np.abs(tr) ** 2 + d) / (d * (d + 1)))


class SynthesisVerificationError(RuntimeError):
    """Raised when the synthesized circuit does not match the target unitary."""


# =========================================================
# Final Synthesizer
# =========================================================
class SU4GeodesicPSFSynthesizer:
    """
    Core synthesizer.
    Delegates heavy computation to Rust core.
    """
    def __init__(self, hyper: GeodesicPSFHyper):
        self.hyper = hyper

    def synthesize(self, U_target: np.ndarray) -> QuantumCircuit:
        """Main synthesis pipeline."""
        if U_target.shape != (4, 4):
            raise ValueError("Input must be 4x4 unitary matrix")

        # 1. Rust Core: Geometric Decomposition (Cartan + KAK)
        u_r = U_target.real.tolist()
        u_i = U_target.imag.tolist()

        (c1, c2, c3), k1, k2, global_phase = geometric_decompose(u_r, u_i)

        # 2. Build native circuit
        qc = QuantumCircuit(2)

        # K1 local rotations
        qc.rz(k1[0][0], 0)
        qc.ry(k1[0][1], 0)
        qc.rz(k1[0][2], 0)

        qc.rz(k1[1][0], 1)
        qc.ry(k1[1][1], 1)
        qc.rz(k1[1][2], 1)

        # Cartan core (non-local)
        qc.rxx(2 * c1, 0, 1)
        qc.ryy(2 * c2, 0, 1)
        qc.rzz(2 * c3, 0, 1)

        # K2 local rotations
        qc.rz(k2[0][0], 0)
        qc.ry(k2[0][1], 0)
        qc.rz(k2[0][2], 0)

        qc.rz(k2[1][0], 1)
        qc.ry(k2[1][1], 1)
        qc.rz(k2[1][2], 1)

        # Global phase correction
        if self.hyper.phase_fix:
            qc.global_phase += global_phase

        # 3. Self-verification
        if self.hyper.verify:
            fid = unitary_fidelity(U_target, qc)
            if (1.0 - fid) > self.hyper.tol:
                raise SynthesisVerificationError(
                    f"Synthesized circuit fidelity {fid:.10f} is below "
                    f"required threshold (1 - tol = {1.0 - self.hyper.tol:.10f}). "
                    "This likely indicates a sign/convention mismatch between "
                    "the Rust Cartan decomposition and the RXX/RYY/RZZ gate "
                    "convention used here."
                )

        return qc


# =========================================================
# Qiskit Official Plugin
# =========================================================
class SU4GeodesicPSFUnitarySynthesis(UnitarySynthesisPlugin):
    """
    Official Qiskit UnitarySynthesisPlugin.
    Can be registered and used transparently.
    """
    @property
    def max_qubits(self) -> int:
        return 2

    @property
    def min_qubits(self) -> int:
        return 2

    @property
    def supported_bases(self) -> list[str]:
        return ['ry', 'rz', 'rxx', 'ryy', 'rzz']

    def run(self, unitary: np.ndarray, **options) -> QuantumCircuit:
        """Entry point for Qiskit transpiler."""
        valid = {f.name for f in fields(GeodesicPSFHyper)}
        hyper_kwargs = {k: v for k, v in options.items() if k in valid}
        hyper = GeodesicPSFHyper(**hyper_kwargs)

        synth = SU4GeodesicPSFSynthesizer(hyper)
        return synth.synthesize(unitary)


# Helper for easy registration
def get_plugin():
    """Returns the plugin instance for Qiskit ecosystem."""
    return SU4GeodesicPSFUnitarySynthesis()
