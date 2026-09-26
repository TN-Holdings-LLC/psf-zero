"""diag_cx_fallback.py -- follow-up to Addendum 194, C1.

C1 found large exact (phase-included) distances on two blocks that do NOT
use the new closed form: SWAP on the old path (4.00, which is exactly a
global phase of -1) and a dressed core with c = 1e-7 on both paths (0.598;
the new code falls back to the old path there). This script separates a
global-phase-only difference from a wrong unitary, and locates the stage
that introduces it: the Rust core's (a, b, c), the cached CX core
(`_cx_core_cached`, i.e. Qiskit's TwoQubitBasisDecomposer applied to the
canonical core), or the full block.

Diagnostic only; no predictions.

Usage (repository root):
    python -u benchmarks/diag_cx_fallback.py 2>&1 | tee diag_cx_fallback.txt
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qiskit.circuit.library import SwapGate
from qiskit.quantum_info import Operator, random_unitary

import psf_compile as pc


def pauli_core(a, b, c):
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)
    h = a * np.kron(X, X) + b * np.kron(Y, Y) + c * np.kron(Z, Z)
    w, v = np.linalg.eigh(h)
    return (v * np.exp(1j * w)) @ v.conj().T


def dressed(core, seed):
    k = [random_unitary(2, seed=seed + i).data for i in range(4)]
    return np.kron(k[0], k[1]) @ core @ np.kron(k[2], k[3])


def dists(u, v):
    """(raw Frobenius distance, phase-aligned distance, phase angle of v relative to u)."""
    raw = float(np.linalg.norm(u - v))
    t = np.trace(v.conj().T @ u)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return raw, float(np.linalg.norm(u - ph * v)), float(np.angle(ph))


def core_params(u):
    u_r, u_i = u.real.tolist(), u.imag.tolist()
    if pc._CORE_CHECKED is not None:
        cartan, k1, k2, phase, _ = pc._CORE_CHECKED(u_r, u_i)
    else:
        cartan, k1, k2, phase = pc.geometric_decompose(u_r, u_i)
    return cartan


def main():
    print("LOADED", pc.__file__, pc.VERSION)
    cases = [("swap", Operator(SwapGate()).data)]
    for c in (0.0, 1e-9, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5):
        cases.append((f"dressed c={c:g}", dressed(pauli_core(0.6, 0.3, c), 13)))
        cases.append((f"bare    c={c:g}", pauli_core(0.6, 0.3, c)))
    print(f"\n{'case':18s} {'a':>8s} {'b':>8s} {'c':>10s} | {'cached core: raw / aligned / phase':36s} | "
          f"{'old block: raw / aligned / cx':32s} | {'new block: raw / aligned / cx / closed':38s} | "
          f"{'Qiskit on U: raw / aligned / cx'}")
    for label, u in cases:
        try:
            a, b, c = core_params(u)
        except Exception as exc:  # degenerate for the core
            a = b = c = float("nan")
            print(f"{label:18s} core raised {type(exc).__name__}: {exc}")
        if np.isfinite(a):
            sub = pc._cx_core_cached(a, b, c)
            cc = dists(pauli_core(a, b, c), Operator(sub).data) if sub is not None else (0.0, 0.0, 0.0)
        else:
            cc = (float("nan"),) * 3
        blocks = {}
        for flag in (False, True):
            pc.USE_CX_CLOSED_FORM = flag
            pc._CX_CORE_CACHE.clear()
            synth = pc.SU4GeodesicPSFSynthesizer(
                pc.GeodesicPSFHyper(tol=1e-5, on_unsupported="keep", entangling_basis="cx"), verify=True)
            with contextlib.redirect_stdout(io.StringIO()):
                qc = synth.synthesize(u)
            r, al, _ = dists(u, Operator(qc).data)
            blocks[flag] = (r, al, qc.count_ops().get("cx", 0), "rx" in qc.count_ops(), synth.fallback_count)
        pc.USE_CX_CLOSED_FORM = True
        qk = pc._CX_DECOMPOSER(u)
        qr, qa, _ = dists(u, Operator(qk).data)
        o, n = blocks[False], blocks[True]
        print(f"{label:18s} {a:8.4f} {b:8.4f} {c:10.3e} | {cc[0]:.2e} / {cc[1]:.2e} / {cc[2]:+.4f}       | "
              f"{o[0]:.2e} / {o[1]:.2e} / {o[2]}         | {n[0]:.2e} / {n[1]:.2e} / {n[2]} / {n[3]!s:5s}"
              f"          | {qr:.2e} / {qa:.2e} / {qk.count_ops().get('cx', 0)}")


if __name__ == "__main__":
    main()
