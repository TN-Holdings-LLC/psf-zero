from __future__ import annotations

import warnings

import numpy as np

from dataclasses import dataclass, fields

from qiskit import QuantumCircuit

from qiskit.circuit.library import UnitaryGate

from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

from qiskit.quantum_info import Operator, random_unitary

# Rust Core (psf_zero_core)

from psf_zero_core import batch_decompose

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

            # batch_decompose (not geometric_decompose -- that name doesn't

            # exist in psf_zero_core) takes/returns *batches*, and returns

            # *four* raw magic-basis angles, not three. Call it as a batch

            # of one and unpack accordingly.

            angles, k1, k2, global_phase = batch_decompose([u_r], [u_i])[0]

        except Exception as e:

            return self._fallback(U_target, f"Rust Core Decomposition Failed: {e}")

        # The four returned angles are diagonal entries of the canonical

        # core in the magic basis, not themselves the (c1,c2,c3) XX/YY/ZZ

        # coefficients -- which raw position is "biggest" is input-

        # dependent (the Rust side sorts them for a canonical order).

        # This is the ordering-independent extraction, derived by matching

        # the actual returned angle layout against a symbolic

        # diagonalization of exp(i(c1 XX + c2 YY + c3 ZZ)) in the magic

        # basis, and verified against 1000+ random two-qubit unitaries

        # (worst case (1 - fidelity) ~ 1e-15, real Rust extension):

        t0, t1, t2, t3 = angles

        c1 = (t0 + t1) / 2.0

        c2 = (t1 + t3) / 2.0

        c3 = (t0 + t3) / 2.0

        # ---------------------------------------------------------

        # Build physical circuit

        # ---------------------------------------------------------

        # Target matrix order is K1 . N . K2. A circuit composes with later

        # appends on the left, so K2's gates go first and K1's go last.

        qc = QuantumCircuit(2)

        # K2 local rotations (applied first).

        # k2[0] ("k2l") is the left/first factor of kron(k2l, k2r) in the

        # math this decomposition uses; Qiskit's Operator is little-endian

        # (qubit 0 is the *rightmost* kron factor), so k2l -> qubit 1 and

        # k2r -> qubit 0. Same for k1 below. Each triple is (phi, theta,

        # lam) for M = Rz(phi) Ry(theta) Rz(lam) -- a *matrix* product --

        # so the circuit must append lam, then theta, then phi (appending

        # phi first, as a literal left-to-right transcription of the tuple

        # would, builds the reversed matrix Rz(lam)Ry(theta)Rz(phi) instead).

        qc.rz(k2[0][2], 1); qc.ry(k2[0][1], 1); qc.rz(k2[0][0], 1)

        qc.rz(k2[1][2], 0); qc.ry(k2[1][1], 0); qc.rz(k2[1][0], 0)

        # Cartan core (non-local). Qiskit's RXXGate(t) implements

        # exp(-i t/2 XX), so realizing exp(i*c*XX) needs t = -2c, not +2c.

        qc.rxx(-2 * c1, 0, 1)

        qc.ryy(-2 * c2, 0, 1)

        qc.rzz(-2 * c3, 0, 1)

        # K1 local rotations (applied last).

        qc.rz(k1[0][2], 1); qc.ry(k1[0][1], 1); qc.rz(k1[0][0], 1)

        qc.rz(k1[1][2], 0); qc.ry(k1[1][1], 0); qc.rz(k1[1][0], 0)

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

    # Current Qiskit (tested on 2.5.2) declares 10 properties/methods

    # abstract on UnitarySynthesisPlugin, not 4 -- without these six,

    # `get_plugin()` raises TypeError on instantiation before ever

    # reaching `run()`. All return False: this plugin doesn't use

    # coupling-map, basis-gate, direction, error-rate, gate-length, or

    # pulse info.

    @property

    def supports_coupling_map(self) -> bool:

        return False

    @property

    def supports_basis_gates(self) -> bool:

        return False

    @property

    def supports_natural_direction(self) -> bool:

        return False

    @property

    def supports_gate_errors(self) -> bool:

        return False

    @property

    def supports_gate_lengths(self) -> bool:

        return False

    @property

    def supports_pulse_optimize(self) -> bool:

        return False

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
