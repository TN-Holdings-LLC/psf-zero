"""Effect of removing search and checks: one arm per process.

    TOQB=<TOQB root> python data/2026-10-16/effect_search_checks/arm.py <arm> <out.json>   (from the PSF-Zero root)
arm: rel | before | now | qk3
rel    = release 2026-10-14.1 (psf_compile.py at the root) + a18 (benchmarks/psf_ai_compile.py)
before = c63a + a22 (this morning)
now    = c65 + a23
qk3    = Qiskit's transpile, optimization level 3, seed 0
"""
import contextlib, importlib.util, io, json, os, runpy, shutil, statistics, sys, tempfile, time, warnings
import numpy as np
warnings.simplefilter("ignore")
ARM, OUT = sys.argv[1], sys.argv[2]
_tmp = tempfile.mkdtemp() + "/imp"; sys.argv = ["x", _tmp]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("benchmarks/c55_val.py", run_name="eff")
shutil.rmtree(_tmp, ignore_errors=True)
sys.path[:0] = ["data/2026-10-15/workplace/workplace_2026-10-15b_W1-W2/w30", "benchmarks"]
import c60_tests as TT
def load(p, nm):
    s = importlib.util.spec_from_file_location(nm, p); m = importlib.util.module_from_spec(s); sys.modules[nm] = m
    s.loader.exec_module(m); return m
C63A = "data/2026-10-15/workplace/workplace_2026-10-15c_W1-W3/w34/w3/c63a/psf_compile.py"
FILES = {"rel": ("psf_compile.py", "benchmarks/psf_ai_compile.py"),
         "before": (C63A, "patches/psf_ai_compile_a22_2026-10-15/psf_ai_compile.py"),
         "now": ("patches/psf_compile_c65_2026-10-16/psf_compile.py", "patches/psf_ai_compile_a23_2026-10-16/psf_ai_compile.py")}
with contextlib.redirect_stdout(io.StringIO()):
    helper = load(C63A, "dist_helper")
    if ARM in FILES:
        pc = load(FILES[ARM][0], "psf_compile"); pc.WARN_WITHOUT_TARGET = False
        ai = load(FILES[ARM][1], "ai_front")
from qiskit import transpile
_IN = {}
def tvd(qc, out):
    k = id(qc)
    if k not in _IN:
        flat = qc
        for _ in range(4):
            if not any(i.operation.name == "PauliEvolution" for i in flat.data):
                break
            flat = flat.decompose(gates_to_decompose=["PauliEvolution"])
        _IN[k] = (qc, helper._dist93(flat, 12))
    p = _IN[k][1]
    if p is None:
        return None
    q = helper._dist93(out, 12)
    if p is None or q is None:
        return None
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))
def med(fn, reps=3):
    ts, out = [], None
    for _ in range(reps):
        s = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()): out = fn()
        ts.append(time.perf_counter() - s)
    return out, statistics.median(ts)
def measure(qc, tg, compile_fn, template_fn=None):
    r = {}
    try:
        out, t = med(compile_fn)
        m = g["metrics"](out, tg)
        r.update(compile_s=t, twoq=m["twoq"], dur_us=m["dur_us"], isa=m["isa"], rzz_ok=m["rzz_in_range"], tvd=tvd(qc, out))
    except Exception as e:
        r.update(compile_error=type(e).__name__ + ": " + str(e)[:200])
    if template_fn is not None:
        try:
            tp, tb = med(template_fn)
            if tp is not None:
                bound, tbind = med(lambda: tp.bind(qc), reps=5)
                r.update(template_s=tb, bind_s=tbind)
        except Exception as e:
            r.update(template_error=type(e).__name__)
    return r
rows = []
FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")
CIRC = {name: (g["build"](name) if name in FORMULA else TT.build(name)) for name in FORMULA + NATIVE}
for d in g["DEVICES"]:
    t, ft = g["device"](d)
    print(d, time.strftime("%H:%M:%S"), flush=True)
    for name in FORMULA + NATIVE:
        qc = CIRC[name]
        for lab, tg in (("T", t), ("F", ft)):
            if ARM == "qk3":
                r = measure(qc, tg, lambda: transpile(qc, target=tg, optimization_level=3, seed_transpiler=0))
            elif ARM == "rel":
                r = measure(qc, tg, lambda: pc.compile_for_hardware(qc, target=tg))
            else:
                r = measure(qc, tg, lambda: pc.compile_for_hardware(qc, target=tg), lambda: pc.compile_template(qc, tg))
            rows.append(dict(src="c63val", dev=d, case=name, tg=lab, **r))
            print("  ", name, lab, time.strftime("%H:%M:%S"), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
sys.path.insert(0, os.environ["TOQB"])
from toqb.task_loop import model_circuit, angle_list, n_params
from qiskit_ibm_runtime import fake_provider
rng = np.random.default_rng(7)
for d in ("FakeFez", "FakeTorino", "FakeMarrakesh", "FakeKingston"):
    print("toqb", d, time.strftime("%H:%M:%S"), flush=True)
    t = getattr(fake_provider, d)().target
    cm = t.build_coupling_map(); basis = [x for x in ("cz", "rz", "sx", "x") if x in t.operation_names]
    for n in range(4, 21):
        for L in (1, 2, 4, 8):
            qc = model_circuit(angle_list(rng.uniform(-3, 3, n_params(n, L)), rng.uniform(-1, 1, n), n, L, np.pi / 2), n, L)
            if ARM == "qk3":
                r = measure(qc, t, lambda: transpile(qc, target=t, optimization_level=3, seed_transpiler=0))
            elif ARM == "rel":
                r = measure(qc, t, lambda: ai.compile_for_model_circuit(qc, cm, basis, target=t))
            else:
                r = measure(qc, t, lambda: ai.compile_for_model_circuit(qc, cm, basis, target=t), lambda: ai.model_template(qc, t))
            rows.append(dict(src="toqb", dev=d, case=f"ring{n}", L=L, tg="T", **r))
json.dump(dict(arm=ARM, rows=rows), open(OUT, "w"), indent=0)
print(ARM, len(rows), sum(1 for r in rows if "compile_error" in r), "errors")
