"""n3_parity.py -- N3 (2026-10-14): why c48 differs between odd and even layer counts on W3's classifier, and what c49
changes. For every (device, n, L) and for c48 and c49 (default call, `compile_for_hardware(qc, backend=b,
seed_transpiler=0)`): duration, CZ, and the change of STEINER_STATS during the call (item 68-73 reading, item 82
mirror tried/kept, item 83 reorder, item 69 chosen/kept_usual), plus whether item 73's reader accepts the circuit and
what it leaves at the end. Run from the repository root."""
import contextlib, io, json, os, sys, time, warnings, importlib.util
import numpy as np
warnings.simplefilter("ignore")
REPO = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w2"),
                os.path.join(REPO, "benchmarks")]
import w2_tfim as W2
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
M = {"c48": load(os.path.join(REPO, "patches", "psf_compile_c48_2026-10-12", "psf_compile.py"), "n3_c48"),
     "c49": load(os.path.join(REPO, "patches", "psf_compile_c49_2026-10-14", "psf_compile.py"), "n3_c49")}
for m in M.values():
    m.WARN_WITHOUT_TARGET = False
D = load(os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1", "depth_eval.py"), "depth_eval")
from qiskit_ibm_runtime import fake_provider
rng = np.random.default_rng(20261014)                     # W3's angles, in W3's order
ang = {(n, L): (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L))) for n in (4, 6) for L in range(1, 17)}
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
fh = open(os.path.join(out, "n3_rows.jsonl"), "w")
for dev in ("FakeFez", "FakeTorino"):
    b = getattr(fake_provider, dev)()
    for n in (4, 6):
        for L in range(1, 17):
            x, th = ang[(n, L)]
            qc = D.circuit(x, th, n, L); qc.measure_all()
            row = dict(device=dev, n=n, L=L)
            ev = M["c49"]._st_parse(qc)
            ev2 = None if ev is not None else M["c49"]._st_parse_gates(qc)
            row["read_as_rotations"] = ev is not None
            row["read_as_gates"] = ev2 is not None
            if ev2 is not None:
                row["terms"] = len(M["c49"]._st_merge(ev2[2]))
                row["suffix_gates"] = len(ev2[4])
                row["perm"] = ev2[5] is not None and ev2[5] != list(range(n))
            for name, P in M.items():
                before = dict(P.STEINER_STATS)
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    c = P.compile_for_hardware(qc, backend=b, seed_transpiler=0)
                d, cz, _ = W2.duration_and_cz(c, b.target)
                st = {k: P.STEINER_STATS[k] - before.get(k, 0) for k in P.STEINER_STATS if P.STEINER_STATS[k] != before.get(k, 0)}
                row[name] = dict(cz=cz, dur_us=round(d, 3), steiner=st, secs=round(time.perf_counter() - t0, 1))
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(json.dumps(row), flush=True)
