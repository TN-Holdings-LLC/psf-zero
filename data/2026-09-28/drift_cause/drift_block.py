"""drift_block.py -- exploratory: inside compile(), compare for one pair the
matrix Qiskit's ConsolidateBlocks hands to the synthesizer (M) with the
pair's exact operator (U), and the batched synthesis (synthesize_many, as
compile() uses it) with the single path (synthesize)."""
import contextlib, io, sys, warnings
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/benchmarks"]
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import loop_endurance as le
import psf_compile as pc
import psf_zero_core as core
LAPS = int(sys.argv[2]); PAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 51
print("CORE_VERSION", getattr(core, "CORE_VERSION", None))
backend, native = le.nighthawk()
n = backend.coupling_map.size()
rng = np.random.default_rng(13)
cur = QuantumCircuit(n)
th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
for k in range(n // 2):
    le.add_pair24(cur, 2 * k, 2 * k + 1, th[k])
SW = np.eye(4)[[0, 2, 1, 3]]
rec = []
_orig = pc.SU4GeodesicPSFSynthesizer.synthesize_many
def _wrap(self, mats):
    out = _orig(self, mats); rec.append((list(mats), out, self)); return out
pc.SU4GeodesicPSFSynthesizer.synthesize_many = _wrap
for lap in range(1, LAPS + 1):
    rec.clear()
    U = le.pair_matrices(cur, n)[PAIR]
    pc._CX_CORE_CACHE.clear()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        out = pc.compile_for_hardware(cur, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True, on_unsupported="keep",
                                      seed_transpiler=0, block_gate_floor=8)
    mats, outs, syn = rec[-1]
    best = min(((min(le.aligned(U, m), le.aligned(U, SW @ m @ SW)), i) for i, m in enumerate(mats)))
    i = best[1]; M = mats[i]
    swapped = le.aligned(U, SW @ M @ SW) < le.aligned(U, M)
    Mu = SW @ M @ SW if swapped else M
    circ_b = outs[i][0]
    Ob = Operator(circ_b).data
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        Os = Operator(syn.synthesize(M)).data
    cartan, k1, k2, ph, _ = pc._CORE_CHECKED(M.real.tolist(), M.imag.tolist())
    raw = float(np.linalg.norm(pc._reconstruct(cartan, k1, k2, ph) - M))
    p1, before, after = pc._refine_batch(M[None], pc._pack(cartan, k1, k2, ph)[None])
    print(f"lap {lap}: consolidation |U-M| {best[0]:.3e} (qubit order swapped: {swapped}) | core raw {raw:.3e} "
          f"batched polish {before[0]:.3e}->{after[0]:.3e} | emitted(batched) vs M {le.aligned(M, Ob):.3e} | "
          f"emitted(single) vs M {le.aligned(M, Os):.3e}", flush=True)
    cur = le.back_to_logical(out, n)
