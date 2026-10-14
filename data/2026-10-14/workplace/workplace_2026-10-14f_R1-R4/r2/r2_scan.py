"""R2 scan: C52-VAL's 24 tests x 5 devices x {T, F} through c54's default call (as C52-VAL's arms C52 / C52F:
entangling_basis="cx", layout_search=True, seed 0). Largest outcome-probability difference (C52-VAL's exactness, at
most 20 active qubits). Cells above 1e-9 are compiled again with item 93's tolerance at 1e-9: which fallback is taken,
and at what duration."""
import sys, os, json, time, contextlib, io, tempfile, importlib.util
sys.argv = ["c52_val.py", tempfile.mkdtemp() + "/c52"]
spec = importlib.util.spec_from_file_location("cv", "benchmarks/c52_val.py"); CV = importlib.util.module_from_spec(spec)
sys.modules["cv"] = CV; spec.loader.exec_module(CV)
P = CV.load(os.environ.get("PSF_UNDER_TEST", "patches/psf_compile_c54_2026-10-14/psf_compile.py"), "c54_r2"); P.WARN_WITHOUT_TARGET = False
OUT = os.environ.get("R2_OUT", "scan_rows.jsonl")
fh = open(OUT, "w")
def comp(qc, tg, tol):
    P.EXACT_CHECK_TOL = tol
    s0 = {k: v for k, v in P.EXACT93_STATS.items()}
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            c = P.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    except Exception as ex:
        return None, {"error": repr(ex)[:120]}, time.time() - t0
    d = {k: P.EXACT93_STATS[k] - s0[k] for k in ("failed", "fallback_usual", "fallback_level1", "raised", "not_checkable") if P.EXACT93_STATS[k] != s0[k]}
    return c, d, time.time() - t0
for dev in CV.DEVICES:
    T, F = CV.device(dev)
    for name in CV.TESTS:
        qc = CV.build(name)
        for tk, tg in (("T", T), ("F", F)):
            c, st, secs = comp(qc, tg, 1e-6)
            row = dict(device=dev, test=name, target=tk)
            if c is None:
                row.update(st); fh.write(json.dumps(row) + "\n"); fh.flush(); continue
            ex = CV.exactness(qc, c)
            row.update(dur_us=CV.metrics(c, tg)["dur_us"], exact=ex, stats=st, secs=round(secs, 1))
            if ex is not None and ex > 1e-9:
                c9, st9, s9 = comp(qc, tg, 1e-9)
                row["at_1e-9"] = dict(dur_us=CV.metrics(c9, tg)["dur_us"] if c9 is not None else None,
                                      exact=CV.exactness(qc, c9) if c9 is not None else None, stats=st9)
            P.EXACT_CHECK_TOL = 1e-6
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(dev, name, tk, row.get("exact"), row.get("at_1e-9"), flush=True)
