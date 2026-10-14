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
cc = PassManager([CommutativeCancellation()]).run(qc)
orig = P.transpile
def wrapped(circ, *a, **k):
    out = orig(circ, *a, **k)
    try:
        if any(i.operation.name == "measure" for i in circ.data) and not isinstance(circ, list):
            e = W2.tvd(W2.probs(out), W2.probs(circ))
            if e > 1e-9:
                import pickle; pickle.dump((circ, k), open("bad_in.pkl","wb"))
            print(f"transpile opt={k.get('optimization_level')} approx={k.get('approximation_degree')} tvd {e:.2e} rzz_out {sum(1 for i in out.data if i.operation.name=='rzz')}")
    except Exception as ex:
        print("check failed", repr(ex)[:80])
    return out
P.transpile = wrapped
with contextlib.redirect_stdout(sys.stderr):
    pass
o = P._compile_for_hardware_c49(cc, target=F, basis_gates=["cz", "rz", "sx", "x"], entangling_basis="cx", layout_search=True, seed_transpiler=0)
print("final", W2.tvd(W2.probs(o), W2.probs(qc)))
