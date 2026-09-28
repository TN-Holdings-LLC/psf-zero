"""E-part compiles (Addendum 219/222 training circuit, floor 8) with the
installed core: fallbacks, GUARD_STATS, loss error, 2q count, compile time."""
import sys, os, contextlib, io, warnings, re, time, gc, statistics as st
import numpy as np
REPO = sys.argv[1]; N = int(sys.argv[2])
sys.path[:0] = [REPO, os.path.join(REPO, "benchmarks")]
import psf_compile as pc
import loop_endurance as le
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
print("LOADED", pc.__file__, pc.VERSION)
print("CORE", pc.psf_zero_core.__file__, "CORE_VERSION", getattr(pc.psf_zero_core, "CORE_VERSION", "none (built before 2026-09-28.1)"))
FELL = re.compile(r"(\d+) block\(s\) fell back to CX-basis synthesis \((\d+) degenerate/numeric, (\d+) unexpected\)")
cmap = CouplingMap.from_line(le.E_QUBITS)
rng_e = np.random.default_rng(7)
tt = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS); tt[8::15] = 0.0
target = Statevector(le.e_circuit(tt))
rng_t = np.random.default_rng(202)
fb_total, worst, times, cx = 0, 0.0, [], []
def run(qc):
    with warnings.catch_warnings(record=True) as ws, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        pc._CX_CORE_CACHE.clear()
        t0 = time.perf_counter()
        out = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                                      entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                      on_unsupported="keep", seed_transpiler=0, block_gate_floor=8)
        el = time.perf_counter() - t0
    fb = 0
    for w in ws:
        m = FELL.search(str(w.message))
        if m: fb = int(m.group(1))
    return out, el, fb
run(le.e_circuit(tt)); gc.collect(); gc.freeze()
for k in pc.GUARD_STATS: pc.GUARD_STATS[k] = 0 if isinstance(pc.GUARD_STATS[k], int) else 0.0
for lap in range(N):
    qc = le.e_circuit(tt + rng_t.normal(0.0, 0.5, le.E_NPARAMS))
    gc.disable(); out, el, fb = run(qc); gc.enable()
    if lap % 50 == 0: gc.collect()
    times.append(el); fb_total += fb; cx.append(out.count_ops().get("cx", 0))
    err = abs((1 - abs(target.inner(Statevector(out))) ** 2) - (1 - abs(target.inner(Statevector(qc))) ** 2))
    worst = max(worst, err)
print(f"compiles {N} | core fallbacks {fb_total} | worst loss error {worst:.2e} | cx per compile {sorted(set(cx))} "
      f"| median compile {st.median(times)*1e3:.2f} ms (p90 {np.percentile(times,90)*1e3:.2f})")
print("GUARD_STATS", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pc.GUARD_STATS.items()})
