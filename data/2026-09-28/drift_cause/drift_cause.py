"""drift_cause.py -- exploratory: (1) the Weyl coordinates of the pair's
block and the spread of the core's raw residual over all 60 blocks at lap 1;
(2) causal check over a few laps: with the polish forced on every block
(REFINE_THRESHOLD set to 0 in memory), does the pair's per-lap change
disappear?"""
import contextlib, io, sys, warnings
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/benchmarks"]
import numpy as np
from qiskit import QuantumCircuit
import loop_endurance as le
import psf_compile as pc
import psf_zero_core as core
LAPS = int(sys.argv[2]); FORCE = sys.argv[3] == "force"; PAIR = 51
print("CORE_VERSION", getattr(core, "CORE_VERSION", None), "force polish", FORCE)
if FORCE:
    pc.REFINE_THRESHOLD = 0.0
    pc._refine_batch.__defaults__ = (0.0, 3)
    pc._refine_decomposition.__defaults__ = (0.0, 3)
backend, native = le.nighthawk()
n = backend.coupling_map.size()
rng = np.random.default_rng(13)
cur = QuantumCircuit(n)
th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
for k in range(n // 2):
    le.add_pair24(cur, 2 * k, 2 * k + 1, th[k])
mats = le.pair_matrices(cur, n)
raws = {}
for k, u in mats.items():
    cartan, k1, k2, ph, _ = pc._CORE_CHECKED(u.real.tolist(), u.imag.tolist())
    raws[k] = (float(np.linalg.norm(pc._reconstruct(cartan, k1, k2, ph) - u)), cartan)
r = np.array([v[0] for v in raws.values()])
print(f"lap-1 raw residual over 60 blocks: median {np.median(r):.2e}, max {r.max():.2e}, "
      f"<= 1e-13 (unpolished): {(r <= 1e-13).sum()}, in (1e-14, 1e-13]: {((r > 1e-14) & (r <= 1e-13)).sum()}")
print(f"pair {PAIR}: raw {raws[PAIR][0]:.3e}, Weyl (a, b, c) = {tuple(round(x, 6) for x in raws[PAIR][1])}")
d = le.pair_matrices(cur, n)
ref = d
for lap in range(1, LAPS + 1):
    pc._CX_CORE_CACHE.clear()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        out = pc.compile_for_hardware(cur, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True, on_unsupported="keep",
                                      seed_transpiler=0, block_gate_floor=8)
    cur = le.back_to_logical(out, n)
    new = le.pair_matrices(cur, n)
    dist = {k: le.aligned(ref[k], new[k]) for k in ref}
    w = max(dist, key=dist.get)
    print(f"lap {lap}: pair {PAIR} distance to lap 0 {dist[PAIR]:.3e}; worst pair {w} {dist[w]:.3e}", flush=True)
