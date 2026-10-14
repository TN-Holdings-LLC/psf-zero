import os, sys, pickle
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from p1_common import *  # noqa
import exact2q as X
from qiskit import transpile
from qiskit_ibm_runtime import fake_provider
for path, dev in (("bad_in_FakeTorino.pkl", "FakeTorino"), ("bad_in_FakeFez.pkl", "FakeFez"), ("bad_in.pkl", "FakeFez")):
    circ, k = pickle.load(open(path, "rb"))
    T = getattr(fake_provider, dev)().target
    kk = dict(k); kk.pop("target", None); lvl = kk.pop("optimization_level", 3)
    ref = W2.probs(circ)
    a = transpile(circ, target=T, optimization_level=lvl, **kk)
    pm = X.exact_level3(target=T, **{x: v for x, v in kk.items() if x in ("initial_layout", "routing_method", "seed_transpiler")})
    for s in X.EXACT_STATS: X.EXACT_STATS[s] = 0
    b = pm.run(circ)
    print(path.split("/")[-1], dev, "| Qiskit L3: max|dp| %.1e dur %.3f 2q %d | exact L3: max|dp| %.1e dur %.3f 2q %d |" % (
        float(abs(W2.probs(a) - ref).max()), dur_2q(a, T)[0], dur_2q(a, T)[1], float(abs(W2.probs(b) - ref).max()), dur_2q(b, T)[0], dur_2q(b, T)[1]), X.EXACT_STATS, flush=True)
