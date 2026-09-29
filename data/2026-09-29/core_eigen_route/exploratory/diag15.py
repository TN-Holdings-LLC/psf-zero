"""diag15.py -- exploratory: run the 15 captured blocks through a debug-instrumented
build of the fixed core (same source plus eprintln lines gated by PSF_CORE_DEBUG) and
print which acceptance check fails for each group_tol candidate; also the magic-basis
singular values of Re(u_m) and their gaps."""
import os, sys
os.environ["PSF_CORE_DEBUG"] = "1"
sys.path.insert(0, sys.argv[1])
import numpy as np
import psf_zero_core as core
print("core from", core.__file__, "CORE_VERSION", getattr(core, "CORE_VERSION", None), flush=True)
Q = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
d = np.load("blocks15.npz")
for u, lap in zip(d["mats"], d["laps"]):
    un = u / np.linalg.det(u) ** 0.25
    um = Q.conj().T @ un @ Q
    s = np.linalg.svd(um.real, compute_uv=False)
    ev = np.linalg.eigvals(um.T @ um)
    print(f"=== lap {lap}: sv(Re u_m) {np.round(s, 6)} gaps {np.round(-np.diff(s), 6)}; "
          f"eig phases/2 {np.round(np.sort(np.angle(ev) / 2), 6)}", flush=True)
    sys.stderr.flush()
    try:
        core.geometric_decompose_checked(u.real.tolist(), u.imag.tolist())
        print("   accepted", flush=True)
    except Exception as e:
        print("   rejected:", type(e).__name__, flush=True)
