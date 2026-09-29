"""capture15.py -- exploratory (workplace, 2026-09-29): capture the 15 blocks that the
fixed core rejects in part E of the v4 run (Addendum 247 A), as 4x4 matrices, by the
input reconstruction of probe_v4_fallbacks.py (target seed 7, rzz angles 0; one
normal(0, 0.5) draw per lap from seed 202). Saves blocks15.npz."""
import contextlib, csv, io, sys, warnings
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/benchmarks"]
import numpy as np
from qiskit.transpiler import CouplingMap
import loop_endurance as le
import psf_compile as pc
print("LOADED", pc.VERSION, "CORE_VERSION", pc.CORE_VERSION)
laps = [int(r["lap"]) for r in csv.DictReader(open(sys.argv[1] + "/data/long_loop_E_2026-09-28_v4.csv"))
        if int(r["fallbacks"] or 0) > 0]
rng_e = np.random.default_rng(7); tt = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS); tt[8::15] = 0.0
rng_t = np.random.default_rng(202); want = set(laps); thetas = {}
for lap in range(1, 30001):
    th = tt + rng_t.normal(0.0, 0.5, le.E_NPARAMS)
    if lap in want:
        thetas[lap] = th
cap = []
orig = pc._CORE_CHECKED
def g(u_r, u_i):
    u = np.array(u_r) + 1j * np.array(u_i)
    try:
        r = orig(u_r, u_i); cap.append((u, None)); return r
    except Exception as e:
        cap.append((u, type(e).__name__)); raise
pc._CORE_CHECKED = g
mats, labels = [], []
for lap in laps:
    cap.clear()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore"); pc._CX_CORE_CACHE.clear()
        pc.compile_for_hardware(le.e_circuit(thetas[lap]), coupling_map=CouplingMap.from_line(le.E_QUBITS),
                                basis_gates=["cx", "rz", "sx", "x"], entangling_basis="cx",
                                initial_layout=list(range(le.E_QUBITS)), on_unsupported="keep",
                                seed_transpiler=0, block_gate_floor=8)
    for pos, (u, err) in enumerate(cap):
        if err:
            mats.append(u); labels.append((lap, pos, err))
            print(lap, pos, err)
np.savez("blocks15.npz", mats=np.array(mats), laps=np.array([l[0] for l in labels]), pos=np.array([l[1] for l in labels]))
print("captured", len(mats))
