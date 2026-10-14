import os, sys, pickle
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from p1_common import *  # noqa
import exact2q as X
from qiskit import transpile
from qiskit_ibm_runtime import fake_provider
for path, dev in (("bad_in_FakeTorino.pkl", "FakeTorino"), ("bad_in_FakeFez.pkl", "FakeFez"), ("bad_in.pkl", "FakeFez")):
    circ, k = pickle.load(open(path, "rb"))
    T = getattr(fake_provider, dev)().target
    kk = {x: v for x, v in k.items() if x in ("initial_layout", "routing_method", "seed_transpiler")}
    ref = W2.probs(circ)
    a = transpile(circ, target=T, optimization_level=3, **kk)
    row = [f"Qiskit L3 {float(abs(W2.probs(a) - ref).max()):.1e} {dur_2q(a, T)[0]:.3f}"]
    for ncc in (False, True):
        b = X.exact_level3(no_commutative_cancellation=ncc, target=T, **kk).run(circ)
        row.append(f"exact{' -CC' if ncc else ''} {float(abs(W2.probs(b) - ref).max()):.1e} {dur_2q(b, T)[0]:.3f}")
    print(path.split('/')[-1], dev, " | ".join(row), flush=True)
