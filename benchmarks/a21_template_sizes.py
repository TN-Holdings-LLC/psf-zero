"""a21: the ring template (item 35) on CZ ring classifiers of 8-20 qubits now that item 40 lifts the ring limit:
build time, median bind time over 200 random angle sets, duration against the formula path, exactness (n <= 16).
    python benchmarks/a21_template_sizes.py   (from the repository root)"""
import sys, os, time, warnings, contextlib, io, statistics, random, math
warnings.simplefilter("ignore")
REPO = os.getcwd()  # run from the repository root
sys.path[:0] = [REPO + "/benchmarks", REPO]
import core_fix_c2_eval as H
H.load_module("benchmarks/psf_smart_layout.py", "psf_smart_layout")
pc = H.load_module("patches/psf_compile_c59_2026-10-15/psf_compile.py", "psf_compile"); pc.WARN_WITHOUT_TARGET = False
a = H.load_module("patches/psf_ai_compile_a21_2026-10-15/psf_ai_compile.py", "psf_ai_a21")
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit import QuantumCircuit
t = FakeFez().target
def clf(n, L=4, seed=1):
    rng = random.Random(seed); qc = QuantumCircuit(n)
    for _ in range(L):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q); qc.rz(rng.uniform(-3, 3), q)
        for q in range(n): qc.cz(q, (q + 1) % n)
    for q in range(n): qc.ry(rng.uniform(-3, 3), q)
    qc.measure_all(); return qc
for n in (8, 12, 16, 20):
    qc = clf(n)
    s = time.perf_counter(); tpl = a.ring_template(qc, t); build = time.perf_counter() - s
    vals = tpl.values_of(qc); rng = random.Random(5); ts = []
    for _ in range(200):
        v = [rng.uniform(-3, 3) for _ in vals]
        s = time.perf_counter(); c = tpl.bind_values(v); ts.append(time.perf_counter() - s)
    with contextlib.redirect_stdout(io.StringIO()):
        full = a.compile_for_model_circuit(qc, t.build_coupling_map(), ["cz", "rz", "sx", "x"], target=t)
    b = tpl.bind(qc)
    ex = pc._exact93(qc, b) if n <= 16 else "skipped"
    print(n, "build ms", round(build * 1000), "bind median ms", round(statistics.median(ts) * 1000, 2),
          "dur tpl/formula", round(pc._duration_s(b, t) / pc._duration_s(full, t), 3), "exact", ex, flush=True)
