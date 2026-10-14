import os, sys, pickle, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.fake_provider import FakeFez
T = FakeFez().target
circ, k = pickle.load(open("bad_in.pkl", "rb"))
ref = W2.probs(circ)
small = []
for i in circ.data:
    for p in i.operation.params:
        try:
            v = float(p); r = abs(math.remainder(v, 2 * math.pi))
            if r < 0.05: small.append((i.operation.name, v))
        except Exception: pass
print("small-angle gates in input:", small[:10], len(small))
from qiskit.transpiler.passes import RemoveIdentityEquivalent
o = PassManager([RemoveIdentityEquivalent(target=T)]).run(circ)
print("RemoveIdentityEquivalent(target) alone tvd", W2.tvd(W2.probs(o), ref), len(circ.data) - len(o.data), "gates removed")
o = PassManager([RemoveIdentityEquivalent(approximation_degree=1.0, target=T)]).run(circ)
print("RemoveIdentityEquivalent(1.0, target) tvd", W2.tvd(W2.probs(o), ref), len(circ.data) - len(o.data), "gates removed")
import inspect
print(inspect.signature(RemoveIdentityEquivalent))
