"""n2_a15.py -- a14 (searched) and a15 (constructions chosen by duration, a14 as last candidate) on W3's classifier
circuits, for comparison with N2's ring formula. python n2_a15.py <out> [FakeFez,FakeTorino] (from the repository root)"""
import contextlib, io, json, os, sys, time, warnings, importlib.util
import numpy as np
warnings.simplefilter("ignore")
REPO = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.getcwd(), "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w2"), os.path.join(HERE, "..", "..", "w17", "w2"), os.path.join(REPO, "benchmarks")]
import w2_tfim as W2
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
pc = load(os.path.join(REPO, "patches", "psf_compile_c49_2026-10-14", "psf_compile.py"), "psf_compile")
pc.WARN_WITHOUT_TARGET = False
A15 = load(os.path.join(REPO, "patches", "psf_ai_compile_a15_2026-10-14", "psf_ai_compile.py"), "a15")
D = load(os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1", "depth_eval.py"), "depth_eval")
from qiskit_ibm_runtime import fake_provider
rng = np.random.default_rng(20261014)
angles = {(n, L): (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L))) for n in (4, 6) for L in range(1, 17)}
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
devs = (sys.argv[2] if len(sys.argv) > 2 else "FakeFez,FakeTorino").split(",")
Ls = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(1, 17))
fh = open(os.path.join(out, "a15_rows.jsonl"), "a")
basis = ["cz", "rz", "sx", "x"]
for dev in devs:
    b = getattr(fake_provider, dev)(); t = b.target; cm = t.build_coupling_map()
    for n in (4, 6):
        for L in Ls:
            x, th = angles[(n, L)]
            qc = D.circuit(x, th, n, L); qc.measure_all()
            ref = W2.probs(qc)
            row = dict(device=dev, n=n, L=L)
            for arm in ("a14", "a15"):
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    if arm == "a14":
                        c = A15.a14.compile_for_model_circuit(qc, cm, basis, target=t)
                    else:
                        s0 = A15.A15_STATS["chosen_search"]
                        c = A15.compile_for_model_circuit(qc, cm, basis, target=t)
                        row["a15_chose_search"] = A15.A15_STATS["chosen_search"] > s0
                d, cz, _ = W2.duration_and_cz(c, t)
                row.update({f"{arm}_cz": cz, f"{arm}_dur": round(d, 4), f"{arm}_err": W2.tvd(W2.probs(c), ref),
                            f"{arm}_s": round(time.perf_counter() - t0, 1)})
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(json.dumps(row), flush=True)
