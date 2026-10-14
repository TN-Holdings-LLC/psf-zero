import os, sys, contextlib, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation
P = load(REPO + "/patches/psf_compile_c51_2026-10-14/psf_compile.py", "psf_compile"); P.WARN_WITHOUT_TARGET = False
D = load(REPO + "/data/2026-10-05/workplace/depth1/depth_eval.py", "depth_eval")
from qiskit_ibm_runtime.fake_provider import FakeFez
rng = np.random.default_rng(20261014)
angles = {(n, L): (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L))) for n in (4, 6) for L in range(1, 17)}
T = FakeFez().target; F = fractional(T)
x, th = angles[(6, 12)]
qc = D.circuit(x, th, 6, 12); qc.measure_all()
ref = W2.probs(qc)
cc = PassManager([CommutativeCancellation()]).run(qc)
print("commuted start itself tvd", W2.tvd(W2.probs(cc), ref))
for frac in ("auto", False):
    P.FRACTIONAL = frac
    for k in list(P.FRAC_STATS): P.FRAC_STATS[k] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        o = P.compile_for_hardware(cc, target=F, basis_gates=["cz", "rz", "sx", "x"], entangling_basis="cx", layout_search=True, seed_transpiler=0)
    print("FRACTIONAL", frac, "tvd", W2.tvd(W2.probs(o), ref), dur_2q(o, F), {k: v for k, v in P.FRAC_STATS.items() if v})
print("EXACT_TOL", P.EXACT_TOL)
with contextlib.redirect_stdout(io.StringIO()):
    raw = P._compile_for_hardware_c49(cc, target=F, basis_gates=["cz", "rz", "sx", "x"], entangling_basis="cx", layout_search=True, seed_transpiler=0)
print("c49 part (before fold) tvd", W2.tvd(W2.probs(raw), ref), "rzz", sum(1 for i in raw.data if i.operation.name=="rzz"), "in range", rzz_ok(raw))
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
f1 = PassManager([FoldRzzAngle()]).run(raw)
f1 = P._strip_global_phase_gates(f1)
print("after FoldRzzAngle only tvd", W2.tvd(W2.probs(f1), ref))
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
f2 = PassManager([Optimize1qGatesDecomposition(target=P._errorless(F))]).run(f1)
print("after 1q merge tvd", W2.tvd(W2.probs(f2), ref))
# where in c49: which stage? check STEINER / level3 / resynth stats
for nm in ("STEINER_STATS", "EXACT_STATS", "SELECT_STATS", "L3_STATS", "RESYNTH_STATS"):
    if hasattr(P, nm): print(nm, getattr(P, nm))
