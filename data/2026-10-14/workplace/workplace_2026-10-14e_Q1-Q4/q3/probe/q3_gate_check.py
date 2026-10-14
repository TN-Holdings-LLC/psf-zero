import os, sys, contextlib, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
P = load(REPO + "/patches/psf_compile_c51_2026-10-14/psf_compile.py", "psf_compile"); P.WARN_WITHOUT_TARGET = False
A = load(REPO + "/patches/psf_ai_compile_a17_2026-10-14/psf_ai_compile.py", "a17")
D = load(REPO + "/data/2026-10-05/workplace/depth1/depth_eval.py", "depth_eval")
from qiskit_ibm_runtime.fake_provider import FakeFez
rng = np.random.default_rng(20261014)
angles = {(n, L): (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L))) for n in (4, 6) for L in range(1, 17)}
F = fractional(FakeFez().target)
x, th = angles[(6, 12)]
qc = D.circuit(x, th, 6, 12); qc.measure_all()
for ring in (False, True):
    A.RING_FORMULA = ring
    for ex in (False, True):
        A.EXACT_CHECK = ex
        with contextlib.redirect_stdout(io.StringIO()):
            c, info = A.compile_for_model_circuit(qc, F.build_coupling_map(), ["cz", "rz", "sx", "x"], target=F, return_info=True)
        print("ring", ring, "check", ex, "chosen", info["chosen"], "tvd %.1e" % W2.tvd(W2.probs(c), W2.probs(qc)), "dur %.2f" % dur_2q(c, F)[0], A.A17_STATS)
