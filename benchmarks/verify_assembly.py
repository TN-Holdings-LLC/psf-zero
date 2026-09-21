"""Independent check of both r0_psf_zero_transform versions' gate assembly,
using numpy only (no PennyLane, no psf_zero_core).

Rust contract (lib.rs, geometric_decompose docstring):
    U = e^{i*phase} * (e1l kron e1r) * N(c1,c2,c3) * (e2l kron e2r)
    each local == Rz(phi) Ry(theta) Rz(lam)  (matrix product)
    batch_decompose returns raw angles (t0..t3) with
    c1=(t0+t1)/2, c2=(t1+t3)/2, c3=(t0+t3)/2   (cartan_from_angles)

PennyLane conventions assumed (documented, NOT executed here):
    Rot(a,b,c) = RZ(c) RY(b) RZ(a); IsingPP(t) = exp(-i t/2 P⊗P);
    GlobalPhase(p) = exp(-i p) I; big-endian wires (wires[0] = left kron factor)
"""
import numpy as np
from scipy.linalg import expm

def Rz(t): return np.array([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]])
def Ry(t): return np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])
X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
I2 = np.eye(2)

# ---- Rust's own su2_to_euler_zyz, ported line-for-line from lib.rs ----
def rust_su2_to_euler_zyz(m):
    a, b = m[0, 0], m[0, 1]
    theta = 2.0 * np.arctan2(abs(b), abs(a))
    if theta < 1e-12:
        lam = (-2.0 * np.angle(a)) % (4*np.pi)
        return (0.0, 0.0, lam)
    phi_raw = np.pi - np.angle(a) - np.angle(b)
    lam_raw = np.angle(b) - np.angle(a) - np.pi
    phi = phi_raw % (2*np.pi)
    lam = (lam_raw + (phi_raw - phi)) % (4*np.pi)
    return (phi, theta, lam)

def rust_su2_from_euler_zyz(phi, theta, lam):  # ported from lib.rs
    c, s = np.cos(theta/2), np.sin(theta/2)
    ep, em = np.exp(-1j*phi/2), np.exp(1j*phi/2)
    lp, lm = np.exp(-1j*lam/2), np.exp(1j*lam/2)
    return np.array([[ep*c*lp, -ep*s*lm], [em*s*lp, em*c*lm]])

# ---- PennyLane conventions, in numpy ----
def pl_Rot(a, b, c): return Rz(c) @ Ry(b) @ Rz(a)
def pl_ising(P, t): return expm(-1j * t/2 * np.kron(P, P))
def on_w0(g): return np.kron(g, I2)
def on_w1(g): return np.kron(I2, g)
def pl_globalphase(p): return np.exp(-1j*p) * np.eye(4)

def circuit_unitary(ops):  # ops in time order
    U = np.eye(4, dtype=complex)
    for g in ops:
        U = g @ U
    return U

# ---- v2's own su2_to_euler, ported from the '(v2)' variant ----
def v2_su2_to_euler(U):
    a, b = U[0, 0], U[0, 1]
    theta = 2*np.arctan2(abs(b), abs(a))
    if theta < 1e-12:
        return 0.0, 0.0, (-2*np.angle(a)) % (4*np.pi)
    phi = (np.angle(b) - np.angle(a) - np.pi) % (4*np.pi)
    lam = (np.pi - np.angle(a) - np.angle(b)) % (4*np.pi)
    return phi, theta, lam

def random_su2(rng):
    v = rng.normal(size=4); v /= np.linalg.norm(v)
    w, x, y, z = v
    return np.array([[w+1j*z, y+1j*x], [-y+1j*x, w-1j*z]])

rng = np.random.default_rng(7)
worst = {"v1": 0.0, "v2": 0.0, "v1_wrong_argorder": 0.0, "v1_no_globalphase": 0.0}
N = 2000
for _ in range(N):
    e1l, e1r, e2l, e2r = (random_su2(rng) for _ in range(4))
    c1, c2, c3 = rng.uniform(-np.pi/4, np.pi/4, 3)
    phase = rng.uniform(-np.pi, np.pi)
    Ncore = expm(1j*(c1*np.kron(X, X) + c2*np.kron(Y, Y) + c3*np.kron(Z, Z)))
    U = np.exp(1j*phase) * np.kron(e1l, e1r) @ Ncore @ np.kron(e2l, e2r)

    # what batch_decompose hands back (triples via Rust's own extractor)
    k1 = [rust_su2_to_euler_zyz(e1l), rust_su2_to_euler_zyz(e1r)]
    k2 = [rust_su2_to_euler_zyz(e2l), rust_su2_to_euler_zyz(e2r)]
    t0, t1, t3 = c1 - c2 + c3, c1 + c2 - c3, -c1 + c2 + c3
    angles = (t0, t1, 0.0, t3)

    # both versions' cartan mapping
    tt0, tt1, _, tt3 = angles
    C = ((tt0+tt1)/2, (tt1+tt3)/2, (tt0+tt3)/2)
    ising = [pl_ising(X, -2*C[0]), pl_ising(Y, -2*C[1]), pl_ising(Z, -2*C[2])]

    # v1 ('Corrected Version'): Rot(lam, theta, phi) directly
    ops_v1 = [on_w0(pl_Rot(k2[0][2], k2[0][1], k2[0][0])), on_w1(pl_Rot(k2[1][2], k2[1][1], k2[1][0])),
              *ising,
              on_w0(pl_Rot(k1[0][2], k1[0][1], k1[0][0])), on_w1(pl_Rot(k1[1][2], k1[1][1], k1[1][0])),
              pl_globalphase(-phase)]
    worst["v1"] = max(worst["v1"], np.linalg.norm(circuit_unitary(ops_v1) - U))

    # v2 ('(v2)' variant): Rust triple -> matrix -> own su2_to_euler -> Rot(*a)
    m = lambda tr: rust_su2_from_euler_zyz(*tr)
    a1, a2 = v2_su2_to_euler(m(k2[0])), v2_su2_to_euler(m(k2[1]))
    a3, a4 = v2_su2_to_euler(m(k1[0])), v2_su2_to_euler(m(k1[1]))
    ops_v2 = [on_w0(pl_Rot(*a1)), on_w1(pl_Rot(*a2)), *ising,
              on_w0(pl_Rot(*a3)), on_w1(pl_Rot(*a4)), pl_globalphase(-phase)]
    worst["v2"] = max(worst["v2"], np.linalg.norm(circuit_unitary(ops_v2) - U))

    # negative controls: prove the check can actually catch a bug
    ops_bad = [on_w0(pl_Rot(*k2[0])), on_w1(pl_Rot(*k2[1])), *ising,
               on_w0(pl_Rot(*k1[0])), on_w1(pl_Rot(*k1[1])), pl_globalphase(-phase)]
    worst["v1_wrong_argorder"] = max(worst["v1_wrong_argorder"], np.linalg.norm(circuit_unitary(ops_bad) - U))
    ops_nophase = ops_v1[:-1]
    worst["v1_no_globalphase"] = max(worst["v1_no_globalphase"], np.linalg.norm(circuit_unitary(ops_nophase) - U))

print(f"{N} random instances built from the Rust core's documented contract:")
for k, v in worst.items():
    print(f"  {k:22s} worst ||circuit - U|| = {v:.2e}")
