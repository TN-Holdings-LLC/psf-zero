"""Development check on C44-VAL's (already seen) tests: c45 with and without item 80, against C44-VAL's records."""
import sys, os, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
REPO = "<repo>"
PY = "python"
C45 = os.environ.get("C45_FILE", "<c45 psf_compile.py>")
if sys.argv[1] == "one":
    import contextlib, io, warnings, math
    warnings.simplefilter("ignore")
    sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
    os.environ["C44_TESTS"] = "<c44_val run dir>/c44_tests.json"
    import c44_val as V
    from qiskit_ibm_runtime import fake_provider
    test, dev = sys.argv[2], sys.argv[3]
    qc = V.build("<benchpress>", test)
    b = getattr(fake_provider, dev)(); t = b.target
    V.load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = V.load(C45, "c45")
    for arm, kw in (("C45", {}), ("C45n", dict(level3_candidate=False))):
        before = dict(m.L3C_STATS)
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = m.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)
            r = dict(test=test, device=dev, arm=arm, t=round(time.perf_counter() - t0, 3))
            r.update(V.breakdown(out, t))
            r["l3c"] = {k: m.L3C_STATS[k] - before[k] for k in before if m.L3C_STATS[k] - before[k]}
        except Exception as e:
            r = dict(test=test, device=dev, arm=arm, error=f"{type(e).__name__}: {e}"[:200])
        print(json.dumps(r), flush=True)
    sys.exit()
out = sys.argv[2]
assert not os.path.exists(out)
rows = [json.loads(x) for x in open("<c44_val run dir>/c44_val.jsonl")][1:]
jobs = sorted({(r["test"], r["device"]) for r in rows})
def go(j):
    p = subprocess.run([PY, __file__, "one", *j], capture_output=True, text=True, timeout=1800, cwd=REPO)
    return [l for l in p.stdout.splitlines() if l.startswith("{")] or [json.dumps(dict(test=j[0], device=j[1], arm="C45", error=p.stderr[-200:]))]
n = 0
with ThreadPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex, open(out, "w") as fh:
    for f in as_completed([ex.submit(go, j) for j in jobs]):
        for l in f.result(): fh.write(l + "\n")
        fh.flush(); n += 1
        if n % 10 == 0: print(n, len(jobs), flush=True)
