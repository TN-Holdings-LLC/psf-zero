"""Capture the 4x4 blocks psf_compile hands to the Rust core during the
Addendum 219/222 E-part compiles (training circuit, floor 8), and record
which ones the core rejects and with what error."""
import sys, os, contextlib, io, warnings
import numpy as np
REPO = sys.argv[1]; N = int(sys.argv[2]); OUT = sys.argv[3]
sys.path[:0] = [REPO, os.path.join(REPO, "benchmarks")]
import psf_compile as pc
import loop_endurance as le
from qiskit.transpiler import CouplingMap
print("LOADED", pc.__file__, pc.VERSION)
print("CORE", pc.psf_zero_core.__file__, "CORE_VERSION", getattr(pc.psf_zero_core, "CORE_VERSION", "none (built before 2026-09-28.1)"))
cap = []
orig_checked, orig_plain = pc._CORE_CHECKED, pc.geometric_decompose
def wrap(f):
    def g(u_r, u_i):
        u = np.array(u_r) + 1j * np.array(u_i)
        try:
            r = f(u_r, u_i); cap.append((u, "ok")); return r
        except Exception as e:
            cap.append((u, type(e).__name__)); raise
    return g
if orig_checked is not None: pc._CORE_CHECKED = wrap(orig_checked)
pc.geometric_decompose = wrap(orig_plain)
cmap = CouplingMap.from_line(le.E_QUBITS)
rng_e = np.random.default_rng(7)
target_theta = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS); target_theta[8::15] = 0.0
rng_t = np.random.default_rng(202)
for lap in range(N):
    qc = le.e_circuit(target_theta + rng_t.normal(0.0, 0.5, le.E_NPARAMS))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        pc._CX_CORE_CACHE.clear()
        pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                                entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                on_unsupported="keep", seed_transpiler=0, block_gate_floor=8)
U = np.array([c[0] for c in cap]); tags = np.array([c[1] for c in cap])
np.savez(OUT, U=U, tags=tags)
from collections import Counter
print("blocks", len(cap), Counter(tags.tolist()))
