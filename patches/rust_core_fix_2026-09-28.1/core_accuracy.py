"""Run the installed Rust core directly on a fixed set of 4x4 blocks and
record, per block, the error tag and the phase-aligned Frobenius distance
between the target and the core's raw (unpolished) reconstruction."""
import sys, os, numpy as np
REPO, INP, OUT = sys.argv[1:4]
sys.path[:0] = [REPO]
import psf_compile as pc
import psf_zero_core as core
from qiskit.quantum_info import random_unitary
print("LOADED", pc.__file__, pc.VERSION)
print("CORE", pc.psf_zero_core.__file__, "CORE_VERSION", getattr(pc.psf_zero_core, "CORE_VERSION", "none (built before 2026-09-28.1)"))
d = np.load(INP); U = list(d["U"])
rng = np.random.default_rng(2026)
U += [random_unitary(4, seed=int(s)).data for s in rng.integers(0, 2**31, 5000)]
# structured: local factors with (nearly) traceless SU(2) parts around a random core
paulis = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.diag([1,-1])]
for s in range(2000):
    g = np.random.default_rng(s)
    core_u = random_unitary(4, seed=s).data
    def loc():
        if g.random() < 0.5:
            v = g.normal(size=3); v /= np.linalg.norm(v); eps = 10.0 ** g.uniform(-14, -2) * (g.random() < 0.8)
            h = sum(v[k] * paulis[k] for k in range(3))
            th = np.pi / 2 - eps
            return np.cos(th) * np.eye(2) - 1j * np.sin(th) * h
        return random_unitary(2, seed=int(g.integers(2**31))).data
    U.append(np.kron(loc(), loc()) @ core_u @ np.kron(loc(), loc()))
tags, dist = [], []
for u in U:
    try:
        cartan, k1, k2, ph = core.geometric_decompose(u.real.tolist(), u.imag.tolist())
        v = pc._reconstruct(cartan, k1, k2, ph)
        z = np.vdot(v, u); z = z / abs(z)
        tags.append("ok"); dist.append(np.linalg.norm(u - z * v))
    except Exception as e:
        tags.append(type(e).__name__); dist.append(np.nan)
np.savez(OUT, tags=np.array(tags), dist=np.array(dist), n_captured=len(d["U"]))
print("done", len(U))
