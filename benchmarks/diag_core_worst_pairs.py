"""diag_core_worst_pairs.py -- which inputs make PSF-Zero's Rust core
imprecise? Exploratory diagnosis following Addendum 185.

For each of the 60 pair operators of Addenda 183-185 (same generator, same
seed, same Qiskit-convention matrix that stage S1 fed to the synthesizer),
one call each to:
  - PSF-Zero's block synthesizer (entangling_basis="cx"), and
  - Qiskit's TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX"),
and the phase-aligned distance of each output from the input. Alongside,
properties of the input that plausibly control the core's precision:
  - Weyl coordinates (a, b, c) from Qiskit's TwoQubitWeylDecomposition;
  - in the magic basis, M = U_B^T U_B (U normalized to det 1): the smallest
    gap between eigenvalues of M, and the smallest gaps among the real parts
    and among the imaginary parts separately (the core resolves degeneracy
    in Re and Im in turn, per its own changelog). A small gap means a
    near-degenerate input.
Reports the ten worst pairs for PSF-Zero, and Spearman rank correlations of
PSF-Zero's error with each property.

Usage (WSL, repository root):
    python -u diag_core_worst_pairs.py 2>&1 | tee diag_core_worst_pairs.txt
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import sys

import numpy as np
import pennylane as qml
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer, TwoQubitWeylDecomposition

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from psf_compile import GeodesicPSFHyper, SU4GeodesicPSFSynthesizer  # noqa: E402
from psf_pennylane_gpu_prototype import tape_to_qiskit  # noqa: E402

N_PAIRS = 60
GATES_PER_PAIR = 20
SEED = 0
OUT_CSV = "diag_core_worst_pairs_2026-09-26.csv"

MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex) / np.sqrt(2)


def pair_operators():
    """Qiskit-convention 4x4 operator of each pair, built exactly as stage S1
    of diag_psf_drift.py built its input."""
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


def min_gap(values):
    v = np.sort_complex(np.asarray(values)) if np.iscomplexobj(values) else np.sort(np.asarray(values))
    return float(min(abs(v[i] - v[j]) for i in range(len(v)) for j in range(i + 1, len(v))))


def spectral_props(u):
    u = u / np.linalg.det(u) ** 0.25
    ub = MAGIC.conj().T @ u @ MAGIC
    m = ub.T @ ub
    ev = np.linalg.eigvals(m)
    return min_gap(ev), min_gap(ev.real), min_gap(ev.imag)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    psf = SU4GeodesicPSFSynthesizer(GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"), verify=True)
    qk = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
    rows = []
    for p, u in enumerate(pair_operators()):
        with contextlib.redirect_stdout(io.StringIO()):
            e_psf = dist(u, Operator(psf.synthesize(u)).data)
        e_qk = dist(u, Operator(qk(u)).data)
        w = TwoQubitWeylDecomposition(u)
        g_all, g_re, g_im = spectral_props(u)
        rows.append(dict(pair=p, err_psf=e_psf, err_qiskit=e_qk, weyl_a=w.a, weyl_b=w.b, weyl_c=w.c,
                         gap_eig=g_all, gap_re=g_re, gap_im=g_im))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    errs = np.array([r["err_psf"] for r in rows])
    print(f"PSF-Zero error: median {np.median(errs):.2e}, max {errs.max():.2e} | "
          f"Qiskit error: median {np.median([r['err_qiskit'] for r in rows]):.2e}, "
          f"max {max(r['err_qiskit'] for r in rows):.2e}\n")
    print("Ten worst pairs for PSF-Zero:")
    print(f"{'pair':>4} {'err_psf':>9} {'err_qk':>9} {'a':>7} {'b':>7} {'c':>8} {'gap_eig':>9} {'gap_re':>9} {'gap_im':>9}")
    for r in sorted(rows, key=lambda r: -r["err_psf"])[:10]:
        print(f"{r['pair']:>4} {r['err_psf']:9.2e} {r['err_qiskit']:9.2e} {r['weyl_a']:7.4f} {r['weyl_b']:7.4f} "
              f"{r['weyl_c']:8.4f} {r['gap_eig']:9.2e} {r['gap_re']:9.2e} {r['gap_im']:9.2e}")
    print("\nSpearman rank correlation of PSF-Zero's error with:")
    for key in ("gap_eig", "gap_re", "gap_im", "weyl_a", "weyl_b", "weyl_c", "err_qiskit"):
        print(f"  {key:10s} {spearman(errs, [r[key] for r in rows]):+.2f}")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
