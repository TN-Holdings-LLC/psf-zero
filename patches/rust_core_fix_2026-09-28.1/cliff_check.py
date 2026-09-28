"""Cliff-regime compiles (FakeNighthawk, pair24 circuits as in Addendum 219 F)
with the installed core: 2q count, pair check, fallbacks, time, output hash."""
import sys, os, contextlib, io, warnings, re, time, hashlib, statistics as st
import numpy as np
REPO = sys.argv[1]
sys.path[:0] = [REPO, os.path.join(REPO, "benchmarks")]
import psf_compile as pc
import loop_endurance as le
from qiskit import QuantumCircuit
print("LOADED", pc.__file__, pc.VERSION)
print("CORE", pc.psf_zero_core.__file__, "CORE_VERSION", getattr(pc.psf_zero_core, "CORE_VERSION", "none (built before 2026-09-28.1)"))
FELL = re.compile(r"(\d+) block\(s\) fell back")
backend, native = le.nighthawk(); n = backend.coupling_map.size()
rng = np.random.default_rng(101)
res = []
for rep in range(6):
    th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
    qc = QuantumCircuit(n)
    for k in range(n // 2): le.add_pair24(qc, 2 * k, 2 * k + 1, th[k])
    with warnings.catch_warnings(record=True) as ws, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always"); pc._CX_CORE_CACHE.clear()
        t0 = time.perf_counter()
        out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True, on_unsupported="keep",
                                      seed_transpiler=0, block_gate_floor=8)
        el = time.perf_counter() - t0
    fb = sum(int(m.group(1)) for w in ws for m in [FELL.search(str(w.message))] if m)
    ok, worst = le.pair_check(qc, out, n)
    ops = out.count_ops(); twoq = sum(v for k, v in ops.items() if k in ("cz", "cx", "ecr"))
    h = hashlib.sha256(repr([(i.operation.name, tuple(out.find_bit(q).index for q in i.qubits),
                               tuple(float(p) for p in i.operation.params)) for i in out.data]).encode()).hexdigest()[:12]
    res.append((rep, twoq, ok, worst, fb, el))
    print(f"rep {rep}: 2q {twoq} | pair check ok {ok} worst {worst:.2e} | fallbacks {fb} | {el*1e3:.1f} ms | out {h}")
print("median ms (reps 1-5):", round(st.median(r[5] for r in res[1:]) * 1e3, 1))
