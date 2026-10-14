"""n1_e2e.py -- W2's 48 cells through the patched c49's default call (compile_for_hardware(qc, backend=b,
seed_transpiler=0)), against W2's QK3-BW, QK3 and c48 rows. python n1_e2e.py <w2_rows.jsonl> <patched psf_compile.py> <out>"""
import json, os, sys, time, warnings, importlib.util
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.getcwd()
sys.path[:0] = [os.path.join(os.getcwd(), "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w2"), os.path.join(HERE, "..", "..", "w17", "w2"), os.path.join(REPO, "benchmarks")]
import w2_tfim as W2
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
P = load(sys.argv[2], "c49n1"); P.WARN_WITHOUT_TARGET = False
from qiskit_ibm_runtime import fake_provider
w2 = {(r["device"], r["n"], r["steps"], r["arm"]): r for r in map(json.loads, open(sys.argv[1]))}
os.makedirs(sys.argv[3], exist_ok=True)
fh = open(os.path.join(sys.argv[3], "n1_e2e_rows.jsonl"), "w")
for d in ("FakeFez", "FakeMarrakesh"):
    b = getattr(fake_provider, d)()
    for n in (12, 16, 24, 32, 50, 100):
        for s in (2, 4, 8, 10):
            qc = W2.tfim_pauli(n, s)
            u0 = P.CHAIN_STATS["used"]; t0 = time.perf_counter()
            out = P.compile_for_hardware(qc, backend=b, seed_transpiler=0)
            secs = time.perf_counter() - t0
            dur, cz, _ = W2.duration_and_cz(out, b.target)
            ref = W2.probs(W2.unrolled(qc)) if n <= 24 else W2.corr(W2.unrolled(qc), n)
            err = W2.tvd(W2.probs(out), ref) if n <= 24 else float(np.max(np.abs(W2.corr(out, n) - ref)))
            r = dict(device=d, n=n, steps=s, cz=cz, dur_us=round(dur, 4), err=err, correct=bool(err <= 1e-6),
                     isa=W2.isa_ok(out, b.target), item86=P.CHAIN_STATS["used"] - u0, compile_s=round(secs, 2),
                     **{f"{a}_{k}": w2[(d, n, s, a)][k2] for a in ("QK3-BW", "QK3", "PSF") for k, k2 in (("cz", "cz"), ("dur", "dur_us"))})
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(f"{d} n={n:3d} s={s:2d} c49+n1 {cz:5d} {dur:7.2f}us | BW {r['QK3-BW_cz']:5d} {r['QK3-BW_dur']:7.2f} | err {err:.1e} isa {r['isa']} item86 {r['item86']} {secs:.1f}s", flush=True)
