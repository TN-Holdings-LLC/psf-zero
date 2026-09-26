"""diag_construction.py -- where in circuit construction does PSF-Zero's
1.8e-11 error enter? Exploratory diagnosis following Addendum 187.

Addendum 187 found the core's parameters accurate: the error appears only
after the circuit is built. For each of the 60 pairs of Addenda 183-187 this
measures, separately:
  d_model      || U - _reconstruct(core parameters) ||          (the model)
  d_circ_cx    || U - Operator(circuit, entangling_basis="cx") || (as used)
  d_core_cx    || canonical(a,b,c) - Operator(_cx_core_cached(a,b,c)) ||
               (the middle part, which Qiskit's decomposer builds in CX mode)
  d_circ_can   || U - Operator(circuit, entangling_basis="canonical") ||
               (middle part as RXX/RYY/RZZ, no Qiskit decomposer)
  d_local      worst || Operator(rz ry rz) - _zyz_matrix(triple) || over
               the four local triples
All phase-aligned Frobenius distances; U is the Qiskit-convention operator
S1 of diag_psf_drift.py fed to the synthesizer.

Usage (WSL, repository root, psf_compile.py VERSION 2026-09-21):
    python -u diag_construction.py 2>&1 | tee diag_construction.txt
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_unitary

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import psf_compile as pc  # noqa: E402
from psf_pennylane_gpu_prototype import tape_to_qiskit  # noqa: E402

N_PAIRS = 60
GATES_PER_PAIR = 20
SEED = 0


def pair_operators():
    rng = np.random.default_rng(SEED)
    ops = []
    for _ in range(N_PAIRS):
        mats = [random_unitary(4, seed=int(rng.integers(0, 2**31))).data for _ in range(GATES_PER_PAIR)]
        tape = qml.tape.QuantumTape([qml.QubitUnitary(m, wires=[0, 1]) for m in mats], measurements=[], shots=None)
        ops.append(Operator(tape_to_qiskit(tape, wire_order=[0, 1])[0]).data)
    return ops


def dist(a, b):
    t = np.trace(a.conj().T @ b)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(b - ph * a))


def canonical(a, b, c):
    return (pc._CORE_BASIS * np.exp(1j * (a * pc._WX + b * pc._WY + c * pc._WZ))) @ pc._CORE_BASIS_H


def local_err(triple):
    phi, theta, lam = triple
    q = QuantumCircuit(1)
    q.rz(lam, 0); q.ry(theta, 0); q.rz(phi, 0)
    return dist(pc._zyz_matrix(triple), Operator(q).data)


def main():
    print(f"psf_compile VERSION {pc.VERSION}")
    s_cx = pc.SU4GeodesicPSFSynthesizer(pc.GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"), verify=True)
    s_can = pc.SU4GeodesicPSFSynthesizer(pc.GeodesicPSFHyper(entangling_basis="canonical", on_unsupported="raise"), verify=True)
    rows = []
    for p, u in enumerate(pair_operators()):
        out = pc.geometric_decompose(u.real.tolist(), u.imag.tolist())
        cartan, k1, k2, phase = out[0], out[1], out[2], out[3]
        a, b, c = cartan
        sub = pc._cx_core_cached(a, b, c)
        rows.append(dict(
            pair=p, c=c,
            d_model=dist(u, pc._reconstruct(cartan, k1, k2, phase)),
            d_circ_cx=dist(u, Operator(s_cx._build_circuit(cartan, k1, k2, phase)).data),
            d_core_cx=dist(canonical(a, b, c), Operator(sub).data) if sub is not None else float("nan"),
            d_circ_can=dist(u, Operator(s_can._build_circuit(cartan, k1, k2, phase)).data),
            d_local=max(local_err(t) for t in (*k1, *k2)),
        ))
    keys = ("d_model", "d_circ_cx", "d_core_cx", "d_circ_can", "d_local")
    print("\nmaximum over 60 pairs:  " + "  ".join(f"{k}={max(r[k] for r in rows):.2e}" for k in keys))
    print("median over 60 pairs:   " + "  ".join(f"{k}={float(np.median([r[k] for r in rows])):.2e}" for k in keys))
    print("\nFive worst pairs by d_circ_cx:")
    print(f"{'pair':>4} {'c':>8} " + " ".join(f"{k:>11}" for k in keys))
    for r in sorted(rows, key=lambda r: -r["d_circ_cx"])[:5]:
        print(f"{r['pair']:>4} {r['c']:8.4f} " + " ".join(f"{r[k]:11.2e}" for k in keys))


if __name__ == "__main__":
    main()
