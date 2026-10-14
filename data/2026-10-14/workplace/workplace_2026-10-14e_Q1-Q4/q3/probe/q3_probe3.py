import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
from qiskit import transpile
P = load(REPO + "/patches/psf_compile_c51_2026-10-14/psf_compile.py", "psf_compile")
from qiskit_ibm_runtime.fake_provider import FakeFez
T = FakeFez().target; F = fractional(T)
circ, k = pickle.load(open("bad_in.pkl", "rb"))
ref = W2.probs(circ)
def run(tg, **kw):
    kk = dict(k); kk.update(kw); kk.pop("target", None)
    o = transpile(circ, target=tg, **kk)
    return W2.tvd(W2.probs(o), ref), sum(1 for i in o.data if i.operation.name == "rzz"), dur_2q(o, tg)[0]
print("as called", run(F))
print("approx 1.0 explicit", run(F, approximation_degree=1.0))
print("errorless F", run(P._errorless(F)))
print("plain T (no rzz)", run(T))
for opt in (1, 2):
    print("opt", opt, run(F, optimization_level=opt))
from qiskit.transpiler import generate_preset_pass_manager
