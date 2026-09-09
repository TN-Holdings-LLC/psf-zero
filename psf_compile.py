"""TEST DOUBLE ONLY -- never ship this, never benchmark against it.

psf_compile.py raises on import if the real compiled core is missing, and it
says why: substituting a Qiskit-based stand-in makes every PSF-vs-Qiskit
measurement compare Qiskit against Qiskit. This file exists solely so the
*plumbing* in psf_compile.py (register mapping, tol, verify modes, cache) can
be exercised without the .so. It deliberately does no PSF maths at all.
"""
import numpy as np
from qiskit.synthesis import TwoQubitWeylDecomposition


class PsfError(ValueError): pass
class PsfNotUnitaryError(PsfError): pass
class PsfDegenerateError(PsfError): pass
class PsfNumericError(PsfError): pass
class PsfSU2SingularError(PsfError): pass


def _zyz(m):
    """Same convention as psf_compile._zyz_matrix / lib.rs su2_to_euler_zyz."""
    m = m / np.sqrt(np.linalg.det(m) + 0j)          # into SU(2)
    a, b = m[0, 0], m[0, 1]
    theta = 2.0 * np.arctan2(abs(b), abs(a))
    if theta < 1e-12:
        return (0.0, 0.0, float(-2.0 * np.angle(a)))
    phi_raw = np.pi - np.angle(a) - np.angle(b)
    lam_raw = np.angle(b) - np.angle(a) - np.pi
    phi = phi_raw % (2 * np.pi)
    lam = (lam_raw + (phi_raw - phi)) % (4 * np.pi)
    return (float(phi), float(theta), float(lam))


def geometric_decompose(u_r, u_i):
    U = np.array(u_r, dtype=float) + 1j * np.array(u_i, dtype=float)
    d = TwoQubitWeylDecomposition(U)
    cartan = (float(d.a), float(d.b), float(d.c))
    k1 = [_zyz(d.K1l), _zyz(d.K1r)]
    k2 = [_zyz(d.K2l), _zyz(d.K2r)]
    return cartan, k1, k2, float(d.global_phase)


def geometric_decompose_checked(u_r, u_i):
    import psf_compile
    U = np.array(u_r, dtype=float) + 1j * np.array(u_i, dtype=float)
    cartan, k1, k2, phase = geometric_decompose(u_r, u_i)
    infid = psf_compile._infidelity(U, psf_compile._reconstruct(cartan, k1, k2, phase))
    return cartan, k1, k2, phase, float(infid)
