"""benchmarks/verify_core_infidelity.py"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
from qiskit.quantum_info import Operator, random_unitary

from psf_compile import SU4GeodesicPSFSynthesizer, GeodesicPSFHyper
from psf_zero_core import geometric_decompose_checked
from psf_zero_core import PsfDegenerateError, PsfNumericError, PsfSU2SingularError

TOL = 1e-12
FALLBACK = (PsfDegenerateError, PsfNumericError, PsfSU2SingularError)


def as_lists(u: np.ndarray):
    u = np.asarray(u, dtype=complex)
    return u.real.tolist(), u.imag.tolist()


def haar_su4(rng=None):
    u = random_unitary(4, seed=rng).data
    u = u / np.linalg.det(u) ** 0.25
    return u


def check_core(u: np.ndarray) -> float:
    cartan, k1, k2, phase, infid = geometric_decompose_checked(*as_lists(u))
    assert all(np.isfinite(x) for x in cartan)
    return float(infid)


def check_strict(u: np.ndarray) -> float:
    synth = SU4GeodesicPSFSynthesizer(
        GeodesicPSFHyper(tol=1.0, entangling_basis="canonical"),
        verify="strict",
    )
    qc = synth.synthesize(u)
    U = Operator(qc).data
    tr = np.trace(u.conj().T @ U)
    d = 4.0
    return float(1.0 - (np.abs(tr) ** 2 + d) / (d * (d + 1)))


def sweep_haar(n: int = 500):
    worst_core = worst_strict = 0.0
    fallback = 0
    for i in range(n):
        u = haar_su4()
        try:
            worst_core = max(worst_core, check_core(u))
            worst_strict = max(worst_strict, check_strict(u))
        except FALLBACK:
            fallback += 1
    return worst_core, worst_strict, fallback


def near_cnot(n: int = 200, eps: float = 1e-7):
    cnot = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    fail = 0
    worst = 0.0
    for i in range(n):
        pert = haar_su4()
        mixed = (1 - eps) * cnot + eps * pert
        uu, ss, vv = np.linalg.svd(mixed)
        u = uu @ vv
        u = u / np.linalg.det(u) ** 0.25
        try:
            worst = max(worst, check_core(u), check_strict(u))
        except FALLBACK:
            fail += 1
    return worst, fail


def cargo_test() -> None:
    r = subprocess.run(["cargo", "test", "--release"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("cargo test failed")


if __name__ == "__main__":
    cargo_test()
    wc, ws, fb = sweep_haar(500)
    wn, fn = near_cnot(200, 1e-7)
    print(f"haar core worst={wc:.3e}  strict worst={ws:.3e}  fallback={fb}/500")
    print(f"near-CNOT worst={wn:.3e}  rejected={fn}/200")
    assert fb == 0 and fn == 0
    assert wc < TOL and ws < TOL and wn < TOL
    print("STATUS: PASSED")