import os, sys, pickle
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from p1_common import *  # noqa
import exact2q as X
from qiskit.converters import dag_to_circuit
from qiskit_ibm_runtime import fake_provider
circ, k = pickle.load(open("bad_in_FakeTorino.pkl", "rb"))
T = fake_provider.FakeTorino().target
ref = W2.probs(circ)
st = {"last": 0.0, "n": 0}
def cb(**kw):
    st["n"] += 1
    c = dag_to_circuit(kw["dag"])
    if not any(i.operation.name == "measure" for i in c.data): return
    try: e = float(abs(W2.probs(c) - ref).max())
    except Exception as ex: return
    if abs(e - st["last"]) > 1e-11:
        print(f"  pass #{st['n']} {type(kw['pass_']).__name__}: {st['last']:.2e} -> {e:.2e}"); st["last"] = e
pm = X.exact_level3(target=T, initial_layout=k.get("initial_layout"), routing_method="none", seed_transpiler=k.get("seed_transpiler"))
pm.run(circ, callback=cb)
snap = {}
def cb2(**kw):
    st2["n"] += 1
    if st2["n"] in (27, 28):
        snap[st2["n"]] = dag_to_circuit(kw["dag"])
st2 = {"n": 0}
pm = X.exact_level3(target=T, initial_layout=k.get("initial_layout"), routing_method="none", seed_transpiler=k.get("seed_transpiler"))
pm.run(circ, callback=cb2)
from collections import Counter
a, b = snap[27], snap[28]
print("before", Counter(i.operation.name for i in a.data), "\nafter ", Counter(i.operation.name for i in b.data))
def ops(c):
    return [(i.operation.name, tuple(c.find_bit(q).index for q in i.qubits), tuple(round(float(p), 9) for p in i.operation.params)) for i in c.data]
A, B = ops(a), ops(b)
import difflib
d = [l for l in difflib.unified_diff([str(x) for x in A], [str(x) for x in B], lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---"))]
print(len(d)); print("\n".join(d[:30]))
