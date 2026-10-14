import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
from qiskit import transpile
from qiskit.converters import dag_to_circuit
from qiskit_ibm_runtime.fake_provider import FakeFez
T = FakeFez().target
circ, k = pickle.load(open("bad_in.pkl", "rb"))
ref = W2.probs(circ)
state = {"last": 0.0, "n": 0}
def cb(**kw):
    state["n"] += 1
    p = kw["pass_"]; dag = kw["dag"]
    try:
        c = dag_to_circuit(dag)
        if not any(i.operation.name == "measure" for i in c.data): return
        e = W2.tvd(W2.probs(c), ref)
    except Exception as ex:
        return
    if abs(e - state["last"]) > 1e-9:
        print(f"pass #{state['n']} {type(p).__name__}: tvd {state['last']:.2e} -> {e:.2e}  {getattr(p, '__dict__', {}).get('approximation_degree', '')}")
        state["last"] = e
kk = dict(k); kk.pop("target", None)
transpile(circ, target=T, callback=cb, **kk)
