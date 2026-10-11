"""C39-KNOWN (a supplementary check, not part of C39-VAL's scoring): the HamLib circuits item 68 is known to lose on
(tsp ts225 and lin105, unary encoding) and the slow one (1-fullins, 5335 terms): R1 (2026-10-11.1), R2 (2026-10-11.2)
and C39 on six devices; ESP by the device's errors and compile time."""
import contextlib, io, json, math, os, sys, time, warnings, importlib.util
warnings.simplefilter("ignore")
REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
import bp_mock as B
import c39_val as V
from qiskit_ibm_runtime import fake_provider
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
mods = {"R1": load(V.R1, "r1"), "R2": load(os.path.join(REPO, "psf_compile.py"), "r2"), "C39": load(V.C39, "c39")}
bp, test, dev = sys.argv[1], sys.argv[2], sys.argv[3]
arg = [t for t in B.strata(bp)["HamLib, FakeTorino"] if test in t[0]][0][2]
qc, _ = B.build(bp, "ham_device", arg)
backend = getattr(fake_provider, dev)()
rec = dict(test=test, device=dev)
for arm, m in mods.items():
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = m.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    b = V.breakdown(out, backend.target)
    rec[arm] = dict(t=round(time.perf_counter() - t0, 2), n2=b["n2"], log10esp=round(-(b["s2"] + b["s1"] + b["sm"]), 4),
                    failed=b["failed"], stats=dict(getattr(m, "STEINER_STATS", {})))
print(json.dumps(rec), flush=True)
