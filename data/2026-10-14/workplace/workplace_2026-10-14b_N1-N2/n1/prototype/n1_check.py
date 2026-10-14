"""n1_check.py -- N1 prototype against W2's cells (QK3-BW from W2's rows). python n1_check.py <w2_rows.jsonl> <out>
[--cells all|quick]; run from the repository root with c49 in patches/."""
import json, os, sys, time, warnings, importlib.util
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.getcwd()
sys.path[:0] = [os.path.join(os.getcwd(), "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w2"), HERE, os.path.join(HERE, "..", "..", "w17", "w2"), os.path.join(REPO, "benchmarks")]
import n1_chain as N1
import w2_tfim as W2
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
C = load(os.path.join(REPO, "patches", "psf_compile_c49_2026-10-14", "psf_compile.py"), "c49")
from qiskit_ibm_runtime import fake_provider
w2 = {(r["device"], r["n"], r["steps"], r["arm"]): r for r in map(json.loads, open(sys.argv[1]))}
quick = "--cells" in sys.argv and sys.argv[sys.argv.index("--cells") + 1] == "quick"
cells = [(d, n, s) for d in ("FakeFez", "FakeMarrakesh") for n in (12, 16, 24, 32, 50, 100) for s in (2, 4, 8, 10)]
if quick:
    cells = [("FakeFez", 12, 10), ("FakeFez", 100, 10), ("FakeMarrakesh", 100, 2), ("FakeMarrakesh", 100, 10), ("FakeMarrakesh", 50, 10)]
os.makedirs(sys.argv[2], exist_ok=True)
fh = open(os.path.join(sys.argv[2], "n1_rows.jsonl"), "w")
backs = {}
for d, n, s in cells:
    b = backs.setdefault(d, getattr(fake_provider, d)())
    qc = W2.tfim_pauli(n, s)
    t0 = time.perf_counter()
    out, how = N1.chain_brickwork(qc, b.target, C)
    secs = time.perf_counter() - t0
    dur, cz, _ = W2.duration_and_cz(out, b.target)
    ref = W2.probs(W2.unrolled(qc)) if n <= 24 else W2.corr(W2.unrolled(qc), n)
    err = W2.tvd(W2.probs(out), ref) if n <= 24 else float(np.max(np.abs(W2.corr(out, n) - ref)))
    bw, qk = w2[(d, n, s, "QK3-BW")], w2[(d, n, s, "QK3")]
    r = dict(device=d, n=n, steps=s, cz=cz, dur_us=round(dur, 4), err=err, correct=bool(err <= 1e-6), isa=W2.isa_ok(out, b.target),
             how=how, secs=round(secs, 2), bw_cz=bw["cz"], bw_dur=bw["dur_us"], qk3_cz=qk["cz"], qk3_dur=qk["dur_us"],
             psf_cz=w2[(d, n, s, "PSF")]["cz"], psf_dur=w2[(d, n, s, "PSF")]["dur_us"])
    fh.write(json.dumps(r) + "\n"); fh.flush()
    print(f"{d} n={n:3d} s={s:2d} N1 {cz:5d} {dur:7.2f}us | BW {bw['cz']:5d} {bw['dur_us']:7.2f} | c48 {r['psf_cz']:5d} {r['psf_dur']:7.2f} | err {err:.1e} isa {r['isa']} {how} {secs:.1f}s", flush=True)
