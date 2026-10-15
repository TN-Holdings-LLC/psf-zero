"""a22 (items 42-43) against a21: ring classifiers (CZ) and QAOA rings (rzz) for n in NS on FakeFez T and F, with
psf_compile c61 under both front ends (so only the front end differs). Per cell: a22's path, exactness by item 93 (n in
CHECK), duration and two-qubit count against a21 and Qiskit level 3 (seed 0), compile time; whether a22's output is
identical to a21's where the ring does not embed. Then item 43: model_template on each CZ ring (n <= 16, T), its kind,
build time, median bind over 50 random angle sets, and whether a bind equals the front end's own compile.
    OUTDIR=<dir> python benchmarks/a22_rings.py   (from the repository root)"""
import contextlib
import io
import json
import os
import random
import runpy
import shutil
import statistics
import sys
import tempfile
import time
import warnings

warnings.simplefilter("ignore")
REPO = os.getcwd()
sys.path[:0] = [REPO + "/benchmarks", REPO]
S = os.environ.get("OUTDIR", "data/2026-10-15/a22")
os.makedirs(S, exist_ok=True)
sys.argv = ["x", tempfile.mkdtemp() + "/c55tmp"]
g = runpy.run_path("benchmarks/c55_val.py", run_name="nm")
import core_fix_c2_eval as H  # noqa: E402
H.load_module("benchmarks/psf_smart_layout.py", "psf_smart_layout")
pc = H.load_module("patches/psf_compile_c61_2026-10-15/psf_compile.py", "psf_compile")
pc.WARN_WITHOUT_TARGET = False
a21 = H.load_module("patches/psf_ai_compile_a21_2026-10-15/psf_ai_compile.py", "psf_ai_a21_x")
a22 = H.load_module("patches/psf_ai_compile_a22_2026-10-15/psf_ai_compile.py", "psf_ai_a22_x")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from c29_identity import value_sig  # noqa: E402
t, f = g["device"]("FakeFez")
NS = [int(x) for x in os.environ.get("NS", "4,6,8,9,10,11,12,16,20").split(",")]
CHECK = set(int(x) for x in os.environ.get("CHECK", "4,6,9,12,16").split(","))
BASIS = ["cz", "rz", "sx", "x"]


def clf(n, L=4, seed=1):
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(L):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q)
            qc.rz(rng.uniform(-3, 3), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(rng.uniform(-3, 3), q)
    qc.measure_all()
    return qc


def qaoa(n, p=2, seed=2):
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        gam = rng.uniform(-1.5, 1.5)
        for q in range(n):
            qc.rzz(gam * rng.choice((1, -1)), q, (q + 1) % n)
        bet = rng.uniform(-1, 1)
        for q in range(n):
            qc.rx(bet, q)
    qc.measure_all()
    return qc


def call(mod, qc, tg):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_model_circuit(qc, tg.build_coupling_map(), BASIS, target=tg, return_info=True)


dur = lambda c, tg: pc._duration_s(c, tg) * 1e6  # noqa: E731
n2q = lambda c: sum(1 for i in c.data if len(i.qubits) == 2)  # noqa: E731
rows = []
for fam, fn in (("clf", clf), ("qaoa", qaoa)):
    for n in NS:
        qc = fn(n)
        for tn, tg in (("T", t), ("F", f)):
            r = dict(fam=fam, n=n, target=tn)
            try:
                s = time.perf_counter()
                o22, i22 = call(a22, qc, tg)
                r["ms"] = round((time.perf_counter() - s) * 1000)
                o21, i21 = call(a21, qc, tg)
                q3 = generate_preset_pass_manager(3, target=tg, seed_transpiler=0).run(qc)
                if tn == "F" and not pc._rzz_in_range(q3):
                    q3 = pc._fold(q3, tg)
                r.update(path22=f"{i22['path']}:{i22['chosen']}", path21=f"{i21['path']}:{i21['chosen']}",
                         dur22=dur(o22, tg), dur21=dur(o21, tg), dur_qk3=dur(q3, tg), q2_22=n2q(o22), q2_21=n2q(o21),
                         q2_qk3=n2q(q3), same_as_a21=value_sig(o22) == value_sig(o21),
                         isa=bool(pc._isa_ok(o22, tg)), rzz_ok=bool(pc._rzz_in_range(o22)))
                r["ratio_a21"] = round(r["dur22"] / r["dur21"], 3)
                r["ratio_qk3"] = round(r["dur22"] / r["dur_qk3"], 3)
                if n in CHECK:
                    r["exact"] = pc._exact93(qc, o22)
            except Exception as exc:  # noqa: BLE001 - recorded
                r["error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
            rows.append(r)
            print(json.dumps(r), flush=True)
tpl_rows = []
for n in [x for x in NS if x <= 16]:
    qc = clf(n)
    s = time.perf_counter()
    tpl = a22.model_template(qc, t)
    build = time.perf_counter() - s
    r = dict(n=n, kind=None if tpl is None else type(tpl).__name__, build_ms=round(build * 1000, 1))
    if tpl is not None:
        rng = random.Random(7)
        ts, same, exact = [], 0, 0
        for k in range(50):
            q = clf(n, seed=100 + k)
            s = time.perf_counter()
            b = tpl.bind(q)
            ts.append(time.perf_counter() - s)
            o, _ = call(a22, q, t)
            same += value_sig(b) == value_sig(o)
            if k < 3 and n <= 16:
                exact += pc._exact93(q, b) is True
        r.update(bind_ms_median=round(statistics.median(ts) * 1000, 2), same_as_compile=same, of=50, exact_of_3=exact)
    tpl_rows.append(r)
    print(json.dumps(r), flush=True)
json.dump(dict(rings=rows, templates=tpl_rows, a22_stats=a22.A22_STATS), open(S + "/a22_rings.json", "w"), indent=1)
print("DONE")
