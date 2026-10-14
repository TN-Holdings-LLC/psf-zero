import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
from qiskit import transpile, QuantumCircuit
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, TwoQubitPeepholeOptimization
from qiskit.synthesis import TwoQubitWeylDecomposition
from qiskit_ibm_runtime import fake_provider
dev = sys.argv[1]
T = getattr(fake_provider, dev)().target
circ, k = pickle.load(open(f"bad_in_{dev}.pkl", "rb"))
ref = W2.probs(circ)
st = {"last": 0.0, "n": 0}
def cb(**kw):
    st["n"] += 1
    c = dag_to_circuit(kw["dag"])
    if not any(i.operation.name == "measure" for i in c.data): return
    try: e = float(abs(W2.probs(c) - ref).max())
    except Exception: return
    if abs(e - st["last"]) > 1e-11:
        print(f"  pass #{st['n']} {type(kw['pass_']).__name__}: max|dp| {st['last']:.2e} -> {e:.2e}"); st["last"] = e
kk = dict(k); kk.pop("target", None)
transpile(circ, target=T, callback=cb, **kk)
# block analysis for the peephole pass
c2 = circ.remove_final_measurements(inplace=False)
dag = circuit_to_dag(c2); col = Collect2qBlocks(); col.run(dag)
worst = []
for blk in col.property_set["block_list"]:
    qs = sorted({dag.find_bit(q).index for nd in blk for q in nd.qargs})
    if len(qs) != 2: continue
    sub = QuantumCircuit(2)
    for nd in blk: sub.append(nd.op, [qs.index(dag.find_bit(q).index) for q in nd.qargs])
    big = QuantumCircuit(c2.num_qubits); big.compose(sub, qubits=qs, inplace=True)
    out = PassManager([TwoQubitPeepholeOptimization(T, approximation_degree=1.0)]).run(big)
    r = QuantumCircuit(2, global_phase=out.global_phase)
    for i in out.data: r.append(i.operation, [qs.index(out.find_bit(q).index) for q in i.qubits])
    f = 1 - average_gate_fidelity(Operator(sub), Operator(r))
    w = TwoQubitWeylDecomposition(Operator(sub).data)
    worst.append((f, (round(w.a, 5), round(w.b, 5), round(w.c, 5)), qs))
worst.sort(reverse=True)
print(dev, "blocks", len(worst), "infidelity > 1e-13:", sum(1 for w in worst if w[0] > 1e-13))
for w in worst[:6]: print("  infid %.2e weyl %s qubits %s" % w)
