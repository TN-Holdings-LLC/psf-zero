"""drift_stages.py -- exploratory: where does part C's per-lap drift enter?

For LAPS laps of part C (as in long_loop_100k_v3.py), for every pair: the
input block U_k, the core's raw residual ||reconstruct(core params) - U_k||,
whether the polish ran (raw > REFINE_THRESHOLD), the residual after polish,
the synthesizer's own error aligned(U_k, Operator(synthesize(U_k))), and the
actual per-lap change aligned(U_k, U_{k+1}); for pair PAIR also the direction
of successive changes (cosine of the generators log-free first-order form
G_k = (U_{k+1} ph - U_k) U_k^dagger)."""
import contextlib, io, sys, warnings
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/benchmarks"]
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import loop_endurance as le
import psf_compile as pc
import psf_zero_core as core
LAPS = int(sys.argv[2]); PAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 51
print("CORE_VERSION", getattr(core, "CORE_VERSION", None), "REFINE_THRESHOLD", pc.REFINE_THRESHOLD)
backend, native = le.nighthawk()
n = backend.coupling_map.size()
rng = np.random.default_rng(13)
cur = QuantumCircuit(n)
th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
for k in range(n // 2):
    le.add_pair24(cur, 2 * k, 2 * k + 1, th[k])
syn = pc.SU4GeodesicPSFSynthesizer(pc.GeodesicPSFHyper(entangling_basis="cx"), verify=True)

def stages(u):
    cartan, k1, k2, ph, infid = pc._CORE_CHECKED(u.real.tolist(), u.imag.tolist())
    raw = float(np.linalg.norm(pc._reconstruct(cartan, k1, k2, ph) - u))
    (c2, k1b, k2b, ph2), before, after = pc._refine_decomposition(u, cartan, k1, k2, ph)
    polished = before > pc.REFINE_THRESHOLD
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        qc = syn.synthesize(u)
    emitted = le.aligned(u, Operator(qc).data)
    return raw, polished, after, emitted

prev_g = None
mats = le.pair_matrices(cur, n)
for lap in range(1, LAPS + 1):
    st = {k: stages(mats[k]) for k in mats}
    pc._CX_CORE_CACHE.clear()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        out = pc.compile_for_hardware(cur, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True, on_unsupported="keep",
                                      seed_transpiler=0, block_gate_floor=8)
    cur = le.back_to_logical(out, n)
    new = le.pair_matrices(cur, n)
    step = {k: le.aligned(mats[k], new[k]) for k in mats}
    u0, u1 = mats[PAIR], new[PAIR]
    t = np.trace(u0.conj().T @ u1); ph = np.conj(t / abs(t))
    g = (u1 * ph - u0) @ u0.conj().T
    cos = None if prev_g is None else float(abs(np.vdot(prev_g, g)) / (np.linalg.norm(prev_g) * np.linalg.norm(g)))
    prev_g = g
    raw, pol, after, emitted = st[PAIR]
    unpol = [k for k, s in st.items() if not s[1]]
    print(f"lap {lap}: pair {PAIR}: raw {raw:.3e} polished {pol} after {after:.3e} synth-alone {emitted:.3e} "
          f"lap-step {step[PAIR]:.3e} cos(prev) {cos if cos is None else round(cos, 6)} | all pairs: "
          f"unpolished {len(unpol)}, max step {max(step.values()):.3e} (pair {max(step, key=step.get)}), "
          f"max unpolished raw {max((st[k][0] for k in unpol), default=0):.3e}", flush=True)
    mats = new
