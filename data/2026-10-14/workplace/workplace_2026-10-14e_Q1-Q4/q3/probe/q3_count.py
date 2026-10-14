"""q3_count.py -- how many of a17's candidates item 30 finds inexact on W3's 64 cells (c53; T and F)."""
import os, sys, contextlib, io, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
P = load(REPO + "/patches/psf_compile_c53_2026-10-14/psf_compile.py", "psf_compile"); P.WARN_WITHOUT_TARGET = False
A = load(REPO + "/patches/psf_ai_compile_a17_2026-10-14/psf_ai_compile.py", "a17")
D = load(REPO + "/data/2026-10-05/workplace/depth1/depth_eval.py", "depth_eval")
from qiskit_ibm_runtime import fake_provider
rng = np.random.default_rng(20261014)
angles = {(n, L): (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L))) for n in (4, 6) for L in range(1, 17)}
orig = A._exact
log = []
def spy(qc, c, ref=None):
    r = orig(qc, c, ref)
    if r is False:
        p0 = A._outcomes(qc) if ref is None else ref; p1 = A._outcomes(c)
        log.append(0.5 * float(abs(p0 - p1).sum()))
    return r
A._exact = spy
for dev in ("FakeFez", "FakeTorino"):
    T0 = getattr(fake_provider, dev)().target
    for tk, tg in (("T", T0), ("F", fractional(T0))):
        for n in (4, 6):
            for L in range(1, 17):
                x, th = angles[(n, L)]
                qc = D.circuit(x, th, n, L); qc.measure_all()
                k = len(log)
                with contextlib.redirect_stdout(io.StringIO()):
                    A.compile_for_model_circuit(qc, tg.build_coupling_map(), ["cz", "rz", "sx", "x"], target=tg)
                if len(log) > k:
                    print(dev, tk, n, L, "dropped", len(log) - k, ["%.1e" % v for v in log[k:]], flush=True)
print("TOTAL", A.A17_STATS, "dropped tvd range", (min(log), max(log)) if log else None)
