"""a22: item 93 on a22's outputs for the 20-qubit embedded rings (CZ classifier and rzz ring), FakeFez T and F; it reuses
the builders of benchmarks/a22_rings.py. Run from the repository root:  python benchmarks/a22_n20_exact.py"""
import sys, os, json, contextlib, io, warnings, runpy, tempfile, time
warnings.simplefilter("ignore")
sys.argv = ["x"]; os.environ["NS"] = "4"; os.environ["CHECK"] = "4"
# reuse the builders of benchmarks/a22_rings.py without running it
src = open("benchmarks/a22_rings.py").read().split("dur = lambda")[0]
src = src.replace('os.makedirs(S, exist_ok=True)', '').replace('S = os.environ.get("OUTDIR", "data/2026-10-15/a22")', 'S = None')
exec(compile(src, "a22_rings_head", "exec"))
for fam, fn in (("clf", clf), ("qaoa", qaoa)):
    qc = fn(20)
    for tn, tg in (("T", t), ("F", f)):
        o, i = call(a22, qc, tg)
        s = time.perf_counter(); ex = pc._exact93(qc, o)
        print(json.dumps(dict(fam=fam, n=20, target=tn, path=f"{i['path']}:{i['chosen']}", exact=ex, check_s=round(time.perf_counter() - s, 1))), flush=True)
