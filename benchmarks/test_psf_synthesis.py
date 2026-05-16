# psf_synthesis.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, fields

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

# Rust Core (Frictionless Mathematical Engine)
from psf_zero_core import geometric_decompose

@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-9
    phase_fix: bool = True

class SU4GeodesicPSFSynthesizer:
    def __init__(self, hyper: GeodesicPSFHyper):
        self.hyper = hyper

    def synthesize(self, U_target: np.ndarray) -> QuantumCircuit:
        if U_target.shape != (4, 4):
            raise ValueError("Input must be 4x4 unitary matrix")

        # 1. Rust Core: Geometric Decomposition
        u_r = U_target.real.tolist()
        u_i = U_target.imag.tolist()
        
        (c1, c2, c3), k1, k2, global_phase = geometric_decompose(u_r, u_i)

        # 2. Build native circuit
        qc = QuantumCircuit(2)

        # K1 local
        qc.rz(k1[0][0], 0); qc.ry(k1[0][1], 0); qc.rz(k1[0][2], 0)
        qc.rz(k1[1][0], 1); qc.ry(k1[1][1], 1); qc.rz(k1[1][2], 1)

        # Cartan core
        qc.rxx(2 * c1, 0, 1)
        qc.ryy(2 * c2, 0, 1)
        qc.rzz(2 * c3, 0, 1)

        # K2 local
        qc.rz(k2[0][0], 0); qc.ry(k2[0][1], 0); qc.rz(k2[0][2], 0)
        qc.rz(k2[1][0], 1); qc.ry(k2[1][1], 1); qc.rz(k2[1][2], 1)

        if self.hyper.phase_fix:
            qc.global_phase += global_phase

        return qc

class SU4GeodesicPSFUnitarySynthesis(UnitarySynthesisPlugin):
    @property
    def max_qubits(self) -> int: return 2
    @property
    def min_qubits(self) -> int: return 2
    @property
    def supported_bases(self) -> list[str]:
        return ['rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz']

    # --- Full compliance with the strict Qiskit 1.x API ---
    @property
    def supports_basis_gates(self) -> bool: return False
    @property
    def supports_coupling_map(self) -> bool: return False
    @property
    def supports_natural_direction(self) -> bool: return False
    @property
    def supports_pulse_optimize(self) -> bool: return False
    @property
    def supports_gate_lengths(self) -> bool: return False
    @property
    def supports_gate_errors(self) -> bool: return False
    @property
    def supports_target(self) -> bool: return False

    def run(self, unitary: np.ndarray, **options) -> QuantumCircuit:
        valid = {f.name for f in fields(GeodesicPSFHyper)}
        hyper_kwargs = {k: v for k, v in options.items() if k in valid}
        hyper = GeodesicPSFHyper(**hyper_kwargs)
        synth = SU4GeodesicPSFSynthesizer(hyper)
        return synth.synthesize(unitary)

def get_plugin():
    return SU4GeodesicPSFUnitarySynthesis()
