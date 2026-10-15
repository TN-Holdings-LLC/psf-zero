"""a21 items 40-41: ring classifiers (CZ) and QAOA rings (rzz) for n in NS on FakeFez T and F: path, exactness by
psf_compile's item 93, duration and two-qubit count against Qiskit level 3 (seed 0), compile time; for CZ rings at
n <= 8 the output is compared with a20's."""
import sys, os, runpy, shutil, time, warnings, contextlib, io, math, random, json
warnings.simplefilter("ignore")
REPO = os.getcwd()  # run from the repository root
sys.path[:0] = [REPO + "/benchmarks", REPO]
S = os.environ.get("OUTDIR", "data/2026-10-15/a21/rings"); os.makedirs(S, exist_ok=True); sys.argv = ["x", S + "/c55tmp"]
g = runpy.run_path("benchmarks/c55_val.py", run_name="nm"); shutil.rmtree(S + "/c55tmp", ignore_errors=True)
import core_fix_c2_eval as H
H.load_module("benchmarks/psf_smart_layout.py", "psf_smart_layout")
pc = H.load_module("patches/psf_compile_c59_2026-10-15/psf_compile.py", "psf_compile"); pc.WARN_WITHOUT_TARGET = False
a21 = H.load_module("patches/psf_ai_compile_a21_2026-10-15/psf_ai_compile.py", "psf_ai_a21")
a20 = H.load_module("patches/psf_ai_compile_a20_2026-10-15/psf_ai_compile.py", "psf_ai_a20"); a20.EXACT_CHECK = False
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from c29_identity import value_sig
t, f = g["device"]("FakeFez")
NS = [int(x) for x in os.environ.get("NS", "4,6,8,9,10,11,12,16,20").split(",")]
CHECK = set(int(x) for x in os.environ.get("CHECK", "4,6,9,12,16,20").split(","))
def clf(n, L=4, seed=1):
    rng = random.Random(seed); qc = QuantumCircuit(n)
    for _ in range(L):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q); qc.rz(rng.uniform(-3, 3), q)
        for q in range(n): qc.cz(q, (q + 1) % n)
    for q in range(n): qc.ry(rng.uniform(-3, 3), q)
    qc.measure_all(); return qc
def qaoa(n, p=2, seed=2):
    rng = random.Random(seed); qc = QuantumCircuit(n); qc.h(range(n))
    for _ in range(p):
        gam = rng.uniform(-1.5, 1.5)
        for q in range(n): qc.rzz(gam * rng.choice((1, -1)), q, (q + 1) % n)
        bet = rng.uniform(-1, 1)
        for q in range(n): qc.rx(bet, q)
    qc.measure_all(); return qc
rows = []
for fam, fn in (("clf", clf), ("qaoa", qaoa)):
    for n in NS:
        qc = fn(n)
        for tn, tg in (("T", t), ("F", f)):
            r = dict(fam=fam, n=n, target=tn)
            s = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    out, info = a21.compile_for_model_circuit(qc, tg.build_coupling_map(), ["cz", "rz", "sx", "x"], target=tg, return_info=True)
                r.update(ms=round((time.perf_counter() - s) * 1000), path=info["path"], chosen=info["chosen"],
                         dur=pc._duration_s(out, tg) * 1e6, n2q=sum(1 for i in out.data if len(i.qubits) == 2),
                         isa=bool(pc._isa_ok(out, tg)), rzz_ok=bool(pc._rzz_in_range(out)))
                if n in CHECK:
                    s2 = time.perf_counter(); r["exact"] = pc._exact93(qc, out); r["check_s"] = round(time.perf_counter() - s2, 1)
                if fam == "clf" and n <= 8:
                    with contextlib.redirect_stdout(io.StringIO()):
                        o20 = a20.compile_for_model_circuit(qc, tg.build_coupling_map(), ["cz", "rz", "sx", "x"], target=tg)
                    r["same_as_a20"] = value_sig(o20) == value_sig(out)
            except Exception as e:
                r.update(path="ERR", err=f"{type(e).__name__}: {str(e)[:80]}")
            q3 = generate_preset_pass_manager(3, target=tg, seed_transpiler=0).run(qc)
            if tn == "F" and not pc._rzz_in_range(q3):
                q3 = pc._fold(q3, tg)
            r.update(qk3_dur=pc._duration_s(q3, tg) * 1e6, qk3_n2q=sum(1 for i in q3.data if len(i.qubits) == 2))
            if "dur" in r:
                r["ratio"] = round(r["dur"] / r["qk3_dur"], 3)
            rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(S + "/a21_rings.json", "w"), indent=1)
print("DONE")
