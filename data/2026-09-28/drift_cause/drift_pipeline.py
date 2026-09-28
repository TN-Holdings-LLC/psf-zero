"""drift_pipeline.py -- exploratory: split part C's per-lap change of one pair
into (a) PSF-Zero's compile() (consolidate + synthesize) and (b) the Qiskit
transpile at routing level 1 (layout, basis translation to the native set,
1q optimization). compile() is wrapped to capture its output."""
import contextlib, io, sys, warnings
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/benchmarks"]
import numpy as np
from qiskit import QuantumCircuit
import loop_endurance as le
import psf_compile as pc
import psf_zero_core as core
LAPS = int(sys.argv[2]); PAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 51
print("CORE_VERSION", getattr(core, "CORE_VERSION", None))
backend, native = le.nighthawk()
print("native basis", native)
n = backend.coupling_map.size()
rng = np.random.default_rng(13)
cur = QuantumCircuit(n)
th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
for k in range(n // 2):
    le.add_pair24(cur, 2 * k, 2 * k + 1, th[k])
cap = {}
_orig = pc.compile
def _wrap(*a, **kw):
    out = _orig(*a, **kw); cap["c"] = out; return out
pc.compile = _wrap

def gen(u0, u1):
    t = np.trace(u0.conj().T @ u1); ph = np.conj(t / abs(t))
    return (u1 * ph - u0) @ u0.conj().T

prev = {}
for lap in range(1, LAPS + 1):
    pc._CX_CORE_CACHE.clear()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        out = pc.compile_for_hardware(cur, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True, on_unsupported="keep",
                                      seed_transpiler=0, block_gate_floor=8)
    A = le.pair_matrices(cur, n)[PAIR]
    B = le.pair_matrices(cap["c"], n)[PAIR]
    nxt = le.back_to_logical(out, n)
    C = le.pair_matrices(nxt, n)[PAIR]
    ga, gb, gt = gen(A, B), gen(B, C), gen(A, C)
    def cos(key, g):
        p = prev.get(key); prev[key] = g
        return None if p is None else round(float(abs(np.vdot(p, g)) / (np.linalg.norm(p) * np.linalg.norm(g))), 4)
    ops = {}
    for inst in cap["c"].data:
        qs = [cap["c"].find_bit(q).index for q in inst.qubits]
        if all(q // 2 == PAIR for q in qs):
            ops[inst.operation.name] = ops.get(inst.operation.name, 0) + 1
    ops2 = {}
    for inst in nxt.data:
        qs = [nxt.find_bit(q).index for q in inst.qubits]
        if all(q // 2 == PAIR for q in qs):
            ops2[inst.operation.name] = ops2.get(inst.operation.name, 0) + 1
    print(f"lap {lap}: compile {le.aligned(A, B):.3e} (cos {cos('a', ga)}) | transpile {le.aligned(B, C):.3e} "
          f"(cos {cos('b', gb)}) | total {le.aligned(A, C):.3e} (cos {cos('t', gt)}) | compiled ops {ops} | out ops {ops2}",
          flush=True)
    cur = nxt
