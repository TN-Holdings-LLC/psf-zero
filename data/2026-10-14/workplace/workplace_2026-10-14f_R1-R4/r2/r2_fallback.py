"""R2: what does c54's item 93 give at 1e-9 if its check also covers 20 active qubits? (tfim_n20_r3, c48's _tfim(12)
ring on FakeTorino); and what the exact level-3 pass manager (exact2q.exact_level3) gives as the final transpile."""
import sys, os, contextlib, io, tempfile, importlib.util
sys.argv = ["c52_val.py", tempfile.mkdtemp() + "/c52"]
spec = importlib.util.spec_from_file_location("cv", "benchmarks/c52_val.py"); CV = importlib.util.module_from_spec(spec)
sys.modules["cv"] = CV; spec.loader.exec_module(CV)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact2q as X
P = CV.load("patches/psf_compile_c54_2026-10-14/psf_compile.py", "c54_fb"); P.WARN_WITHOUT_TARGET = False
sys.path.insert(0, "patches/psf_compile_c54_2026-10-14")
from test_c48_on_c49_for_c54 import _tfim
orig_dist = P._dist93
P._dist93 = lambda c, limit=20: orig_dist(c, limit)
cases = [("tfim_n20_r3", CV.build("tfim_n20_r3"), "FakeFez"), ("tfim_n20_r3", CV.build("tfim_n20_r3"), "FakeMarrakesh"), ("c48 _tfim(12,4)", _tfim(12, 4, 8350), "FakeTorino")]
orig_tr = P.transpile
for name, qc, dev in cases:
    T, F = CV.device(dev)
    for mode in ("c54 tol 1e-6", "c54 tol 1e-9", "c54 tol 1e-9 + exact level 3"):
        P.EXACT_CHECK_TOL = 1e-6 if "1e-6" in mode else 1e-9
        if "exact level 3" in mode:
            def tr(circ, *a, **k):
                if k.get("optimization_level") == 3 and k.get("target") is not None and not a:
                    kk = {x: v for x, v in k.items() if x in ("target", "initial_layout", "routing_method", "seed_transpiler")}
                    return X.exact_level3(**kk).run(circ)
                return orig_tr(circ, *a, **k)
            P.transpile = tr
        s0 = dict(P.EXACT93_STATS)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c = P.compile_for_hardware(qc, target=T, entangling_basis="cx", layout_search=True, seed_transpiler=0)
            ex = CV.exactness(qc, c); d = CV.metrics(c, T)["dur_us"]
        except Exception as e:
            ex, d = repr(e)[:80], None
        P.transpile = orig_tr
        st = {k: P.EXACT93_STATS[k] - s0[k] for k in ("failed", "fallback_usual", "fallback_level1", "raised", "not_checkable") if P.EXACT93_STATS[k] != s0[k]}
        print(f"{name:16s} {dev:13s} {mode:30s} dur {d} exact {ex} {st}", flush=True)
