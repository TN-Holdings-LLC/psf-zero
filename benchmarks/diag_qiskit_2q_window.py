"""diag_qiskit_2q_window.py -- follow-up to diag_cx_fallback.py.

diag_cx_fallback.py found that Qiskit's TwoQubitBasisDecomposer(CXGate(),
euler_basis="ZSX"), applied directly to a 2-qubit unitary whose smallest
canonical coordinate is c = 1e-7 or 3e-7, returns a 3-CX circuit that is
wrong by 0.598 in phase-aligned Frobenius distance (about 7% average gate
infidelity), while at c = 1e-6 .. 1e-5 its error is small but grows as
7.5e-17 / c. PSF-Zero's CX path uses that decomposer for degenerate cores.

This scan maps the window and checks who is exposed:
  - the decomposer as PSF-Zero configures it (euler_basis="ZSX"), and with
    Qiskit's default Euler basis;
  - plain Qiskit transpile() to basis [cx, rz, sx, x] with no error data
    (optimization levels 1, 2, 3);
  - plain Qiskit transpile() to a 2-qubit GenericBackendV2 with CZ and error
    data (levels 1 and 3), i.e. the ordinary hardware-targeted path;
  - PSF-Zero's block synthesizer, old path and closed-form path.
Families of canonical cores exp(i(a XX + b YY + c ZZ)), dressed with random
single-qubit layers (3 seeds): (0.6, 0.3, e), (0.7, 0.1, e), (0.5, e, e),
(pi/4, e, e), with e from 1e-12 to 1e-2.

Diagnostic only; no predictions.

Usage (repository root):
    python -u benchmarks/diag_qiskit_2q_window.py 2>&1 | tee diag_qiskit_2q_window.txt
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate, UnitaryGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer

import psf_compile as pc

EPS = [10.0 ** (-k / 4) for k in range(8, 49)]  # 1e-2 .. 1e-12, 4 per decade
FAMILIES = {
    "(0.6,0.3,e)": lambda e: (0.6, 0.3, e),
    "(0.7,0.1,e)": lambda e: (0.7, 0.1, e),
    "(0.5,e,e)": lambda e: (0.5, e, e),
    "(pi/4,e,e)": lambda e: (np.pi / 4, e, e),
}
SEEDS = (13, 101, 202)
BAD = 1e-6  # phase-aligned Frobenius distance counted as a failure
OUT_CSV = "diag_qiskit_2q_window_2026-09-26.csv"


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


def aligned(u, v):
    t = np.trace(v.conj().T @ u)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(u - ph * v))


def twoq(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2)


def main():
    print(f"qiskit {qiskit.__version__} | psf_compile {pc.VERSION}")
    dec_zsx = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
    dec_def = TwoQubitBasisDecomposer(CXGate())
    backend = GenericBackendV2(num_qubits=2, basis_gates=["cz", "rz", "sx", "x"], seed=0)

    def m_dec_zsx(u):
        q = dec_zsx(u)
        return aligned(u, Operator(q).data), twoq(q)

    def m_dec_def(u):
        q = dec_def(u)
        return aligned(u, Operator(q).data), twoq(q)

    def m_transpile(level):
        def f(u):
            qc = QuantumCircuit(2)
            qc.append(UnitaryGate(u), [0, 1])
            out = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=level, seed_transpiler=0)
            return aligned(u, Operator(out).data), twoq(out)
        return f

    def m_backend(level):
        def f(u):
            qc = QuantumCircuit(2)
            qc.append(UnitaryGate(u), [0, 1])
            out = transpile(qc, backend, optimization_level=level, seed_transpiler=0)
            return aligned(u, Operator.from_circuit(out).data), twoq(out)
        return f

    def m_psf(flag):
        def f(u):
            pc.USE_CX_CLOSED_FORM = flag
            pc._CX_CORE_CACHE.clear()
            synth = pc.SU4GeodesicPSFSynthesizer(
                pc.GeodesicPSFHyper(tol=1e-5, on_unsupported="keep", entangling_basis="cx"), verify=True)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                q = synth.synthesize(u)
            return aligned(u, Operator(q).data), twoq(q)
        return f

    methods = {
        "decomposer ZSX (as PSF-Zero)": m_dec_zsx,
        "decomposer default Euler": m_dec_def,
        "transpile L1 [cx,rz,sx,x]": m_transpile(1),
        "transpile L2 [cx,rz,sx,x]": m_transpile(2),
        "transpile L3 [cx,rz,sx,x]": m_transpile(3),
        "transpile L1 GenericBackendV2 cz": m_backend(1),
        "transpile L3 GenericBackendV2 cz": m_backend(3),
        "PSF-Zero old path": m_psf(False),
        "PSF-Zero closed form": m_psf(True),
    }
    rows = []
    fails = defaultdict(list)
    worst = defaultdict(float)
    for fam, mk in FAMILIES.items():
        for e in EPS:
            for seed in SEEDS:
                u = dressed(pauli_core(*mk(e)), seed)
                for name, fn in methods.items():
                    try:
                        err, n2 = fn(u)
                    except Exception as exc:  # recorded, not fatal
                        err, n2 = float("nan"), None
                        print(f"  {name} raised {type(exc).__name__} at {fam} e={e:.1e} seed={seed}: {exc}")
                    rows.append(dict(family=fam, eps=e, seed=seed, method=name, aligned_err=err, twoq=n2))
                    if not np.isnan(err):
                        worst[name] = max(worst[name], err)
                        if err > BAD:
                            fails[name].append((fam, e, seed, err, n2))
    pc.USE_CX_CLOSED_FORM = True

    total = len(FAMILIES) * len(EPS) * len(SEEDS)
    print(f"\n{total} unitaries per method; failure = phase-aligned Frobenius distance > {BAD:g}\n")
    for name in methods:
        f = fails[name]
        print(f"{name:34s} worst {worst[name]:.2e}  failures {len(f)}/{total}")
        if f:
            eps_seen = sorted({x[1] for x in f})
            fams = sorted({x[0] for x in f})
            print(f"{'':34s} eps range {eps_seen[0]:.1e} .. {eps_seen[-1]:.1e}; families {fams}")
            for fam, e, seed, err, n2 in sorted(f, key=lambda x: -x[3])[:3]:
                print(f"{'':34s}   e.g. {fam} e={e:.1e} seed={seed}: err={err:.3e}, 2q gates={n2}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
