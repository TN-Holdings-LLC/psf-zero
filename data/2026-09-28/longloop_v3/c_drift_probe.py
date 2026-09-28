"""c_drift_probe.py -- exploratory (after the scored v3 run): part C of
long_loop_100k_v3.py alone for LAPS laps, with whichever psf_zero_core is
installed, printing the per-pair distance to lap 0 every 100 laps. Used to
tell whether the doubled C drift of v3 comes from the core or the machine."""
import contextlib, io, sys, warnings
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/benchmarks"]
import numpy as np
from qiskit import QuantumCircuit
import loop_endurance as le
import psf_compile as pc
import psf_zero_core as core
LAPS = int(sys.argv[2])
print("CORE_VERSION", getattr(core, "CORE_VERSION", None), "psf_compile", pc.VERSION)
backend, native = le.nighthawk()
n = backend.coupling_map.size()
rng_c = np.random.default_rng(13)
initial = QuantumCircuit(n)
th = rng_c.uniform(-np.pi, np.pi, (n // 2, 24))
for k in range(n // 2):
    le.add_pair24(initial, 2 * k, 2 * k + 1, th[k])
ref = le.pair_matrices(initial, n)
cur = initial
for lap in range(1, LAPS + 1):
    pc._CX_CORE_CACHE.clear()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        out = pc.compile_for_hardware(cur, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True, on_unsupported="keep",
                                      seed_transpiler=0, block_gate_floor=8)
    cur = le.back_to_logical(out, n)
    if lap in (1, 2, 5, 10) or lap % 100 == 0:
        mats = le.pair_matrices(cur, n)
        d = {k: le.aligned(ref[k], mats[k]) for k in ref}
        worst = max(d, key=d.get)
        print(f"lap {lap}: max distance {d[worst]:.4e} (pair {worst})", flush=True)
