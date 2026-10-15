"""a21 / a20: the front end on in-scope chains and lattices (handed to psf_compile c59) and on ring classifiers,
FakeFez T and F: call time, path, the hand-off verdict, duration. AFILE selects the front end (default a20).
    AFILE=patches/psf_ai_compile_a21_2026-10-15/psf_ai_compile.py python benchmarks/a21_handoff_time.py"""
import sys, os, runpy, shutil, time, warnings, contextlib, io, copy, math, random
warnings.simplefilter("ignore")
REPO = os.getcwd()  # run from the repository root
sys.path[:0] = [REPO + "/benchmarks", REPO]
import tempfile
S = tempfile.mkdtemp() + "/c55tmp"
sys.argv = ["x", S]
g = runpy.run_path("benchmarks/c55_val.py", run_name="nm"); shutil.rmtree(S, ignore_errors=True)
import core_fix_c2_eval as H
H.load_module("benchmarks/psf_smart_layout.py", "psf_smart_layout")
pc = H.load_module("patches/psf_compile_c59_2026-10-15/psf_compile.py", "psf_compile"); pc.WARN_WITHOUT_TARGET = False
AFILE = os.environ.get("AFILE", "patches/psf_ai_compile_a20_2026-10-15/psf_ai_compile.py")
a = H.load_module(AFILE, "psf_ai_x")
t, f = g["device"]("FakeFez")
basis = ["cz", "rz", "sx", "x"]
def clf(n, L, seed):
    from qiskit import QuantumCircuit
    rng = random.Random(seed); qc = QuantumCircuit(n)
    for _ in range(L):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q); qc.rz(rng.uniform(-3, 3), q)
        for q in range(n): qc.cz(q, (q + 1) % n)
    for q in range(n): qc.ry(rng.uniform(-3, 3), q)
    qc.measure_all(); return qc
cases = [(nm, g["build"](nm)) for nm in ("tfim_n20_r3", "hhI_n20", "tfim_n40_r2")] + [(f"clf_n{n}", clf(n, 4, 1)) for n in (4, 6, 8)]
for nm, qc in cases:
    for tn, tg in (("T", t), ("F", f)):
        s = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out, info = a.compile_for_model_circuit(qc, tg.build_coupling_map(), basis, target=tg, return_info=True)
            r = (info["path"], info["chosen"], info.get("exact"), round(pc._duration_s(out, tg) * 1e6, 3))
        except Exception as e:
            r = ("ERR", type(e).__name__, str(e)[:60])
        print(nm, tn, round((time.perf_counter() - s) * 1000), "ms", r, flush=True)
print("A18_STATS", a.A18_STATS)
