"""diag_psf_drift.py -- locate where PSF-Zero's per-lap drift (Addendum 184)
enters. Exploratory diagnosis, not a pre-registered test.

Rebuilds the exact 60 qubit pairs of Addendum 183/184 (same generator, same
seed) and runs each pair, on its own 2-qubit problem, around a loop for 10
laps with progressively more of the Addendum 184 pipeline switched on:

  S1  PSF-Zero block synthesizer only (the Rust core, entangling_basis="cx")
  S2  psf_compile.compile()  (tol=1e-5, block_gate_floor default, verify=True)
  S3  S2 + Qiskit transpile to FakeNighthawk's native basis (cz, rz, sx, x) at level 1
      (what compile_for_hardware does after compile())
  S4  S3 + the PennyLane round trip (tape_to_qiskit / qiskit_to_tape), i.e.
      one Addendum 184 lap restricted to one pair
  Q4  control: Qiskit transpile(optimization_level=3) + the same round trip

For each stage: the maximum over pairs of the phase-aligned distance from
the pair's original operator, at laps 1..10. The stage at which the growth
jumps from ~1e-13 per lap to ~7e-11 per lap is where the drift enters.
S4 should reproduce Addendum 184's PSF-Zero numbers (1.76e-11 at lap 1,
6.47e-10 at lap 10) if this diagnosis rebuilds the pipeline faithfully.

Usage (WSL, repository root):
    python -u diag_psf_drift.py 2>&1 | tee diag_psf_drift.txt
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import psf_compile  # noqa: E402
from psf_compile import GeodesicPSFHyper, SU4GeodesicPSFSynthesizer  # noqa: E402
from psf_pennylane_gpu_prototype import qiskit_to_tape, tape_to_qiskit  # noqa: E402

N_PAIRS = 60          # FakeNighthawk, spare = 0: 120 qubits
GATES_PER_PAIR = 20
SEED = 0              # same as deadline_compound_chain.py
LAPS = 10
NATIVE = ["cz", "rz", "sx", "x"]


def pair_unitaries():
    """The same random 2-qubit unitaries, in the same order, as
    deadline_compound_chain.initial_tape (pair by pair, 20 each)."""
    rng = np.random.default_rng(SEED)
    return [[random_unitary(4, seed=int(rng.integers(0, 2**31))).data for _ in range(GATES_PER_PAIR)]
            for _ in range(N_PAIRS)]


def dist(a, b):
    t = np.trace(a.conj().T @ b)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(b - ph * a))


def tape_of(mats):
    """PennyLane tape of one pair (wires 0, 1), PennyLane's own convention."""
    return qml.tape.QuantumTape([qml.QubitUnitary(m, wires=[0, 1]) for m in mats], measurements=[], shots=None)


def wrap2q(qc):
    out = QuantumCircuit(2)
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if inst.operation.name in ("barrier", "delay", "measure"):
            continue
        if len(qs) == 2:
            out.append(UnitaryGate(Operator(inst.operation).data), qs)
        else:
            out.append(inst.operation, qs)
    return out


def main():
    cm2 = [[0, 1], [1, 0]]
    psf = SU4GeodesicPSFSynthesizer(GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"), verify=True)
    pairs = pair_unitaries()

    def s1(state):  # state: 4x4 Qiskit-convention operator
        with contextlib.redirect_stdout(io.StringIO()):
            return Operator(psf.synthesize(state)).data

    def compile_step(qc):
        with contextlib.redirect_stdout(io.StringIO()):
            return psf_compile.compile(qc, verify=True, entangling_basis="cx", on_unsupported="raise")

    def s2(qc):
        return compile_step(qc)

    def s3(qc):
        return transpile(compile_step(qc), coupling_map=cm2, basis_gates=NATIVE, optimization_level=1,
                         seed_transpiler=0, initial_layout=[0, 1])

    def roundtrip(qc_out):
        return qiskit_to_tape(wrap2q(qc_out), [0, 1])

    stages = {}
    # S1: operator in, operator out (Qiskit convention)
    worst = np.zeros(LAPS)
    for mats in pairs:
        qc0 = tape_to_qiskit(tape_of(mats), wire_order=[0, 1])[0]
        u0 = Operator(qc0).data
        u = u0
        for k in range(LAPS):
            u = s1(u)
            worst[k] = max(worst[k], dist(u0, u))
    stages["S1 synthesizer only"] = worst.copy()

    # S2, S3: circuit in, circuit out (Qiskit side only)
    for name, fn in (("S2 compile()", s2), ("S3 compile()+level-1 translate", s3)):
        worst = np.zeros(LAPS)
        for mats in pairs:
            qc = tape_to_qiskit(tape_of(mats), wire_order=[0, 1])[0]
            u0 = Operator(qc).data
            for k in range(LAPS):
                qc = fn(qc)
                worst[k] = max(worst[k], dist(u0, Operator(qc).data))
        stages[name] = worst.copy()

    # S4 and Q4: full lap including the PennyLane round trip; distance in
    # PennyLane's own convention, as in Addendum 184
    for name, fn in (("S4 S3+PennyLane round trip", s3),
                     ("Q4 Qiskit opt3+PennyLane round trip",
                      lambda qc: transpile(qc, coupling_map=cm2, basis_gates=NATIVE, optimization_level=3,
                                           seed_transpiler=0, initial_layout=[0, 1]))):
        worst = np.zeros(LAPS)
        for mats in pairs:
            tape = tape_of(mats)
            u0 = qml.matrix(tape, wire_order=[0, 1])
            for k in range(LAPS):
                qc = tape_to_qiskit(tape, wire_order=[0, 1])[0]
                tape = roundtrip(fn(qc))
                worst[k] = max(worst[k], dist(u0, qml.matrix(tape, wire_order=[0, 1])))
        stages[name] = worst.copy()

    print(f"{'stage':38s} " + " ".join(f"lap{k + 1:>2d}   " for k in range(LAPS)) + "  per-lap growth")
    for name, w in stages.items():
        growth = (w[-1] - w[0]) / (LAPS - 1)
        print(f"{name:38s} " + " ".join(f"{x:.2e}" for x in w) + f"  {growth:.1e}", flush=True)


if __name__ == "__main__":
    main()
