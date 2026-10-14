import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, TwoQubitPeepholeOptimization, ConsolidateBlocks, UnitarySynthesis
from qiskit_ibm_runtime.fake_provider import FakeFez
T = FakeFez().target
circ, k = pickle.load(open("bad_in.pkl", "rb"))
c2 = circ.remove_final_measurements(inplace=False)
dag = circuit_to_dag(c2)
pm = PassManager([Collect2qBlocks()]); pm.run(c2)
from qiskit.transpiler.passes import Collect2qBlocks as C
cb = C(); cb.run(dag)
blocks = cb.property_set["block_list"]
print(len(blocks), "blocks")
worst = []
for bi, blk in enumerate(blocks):
    qs = sorted({dag.find_bit(q).index for node in blk for q in node.qargs})
    if len(qs) != 2: continue
    sub = QuantumCircuit(c2.num_qubits)
    for node in blk:
        sub.append(node.op, [dag.find_bit(q).index for q in node.qargs])
    out = PassManager([TwoQubitPeepholeOptimization(T, approximation_degree=1.0)]).run(sub)
    def red(c):
        r = QuantumCircuit(2)
        for i in c.data:
            r.append(i.operation, [qs.index(c.find_bit(q).index) for q in i.qubits])
        r.global_phase = c.global_phase
        return Operator(r)
    f = average_gate_fidelity(red(sub), red(out))
    ncz_in = sum(1 for i in sub.data if len(i.qubits) == 2); ncz_out = sum(1 for i in out.data if len(i.qubits) == 2)
    worst.append((1 - f, bi, qs, ncz_in, ncz_out))
worst.sort(reverse=True)
for w in worst[:5]: print("infidelity %.2e block %d qubits %s cz %d -> %d" % w)
bi = worst[0][1]
blk = blocks[bi]
sub = QuantumCircuit(2)
qs = worst[0][2]
for node in blk:
    sub.append(node.op, [qs.index(dag.find_bit(q).index) for q in node.qargs])
pickle.dump((sub, qs), open("bad_block.pkl", "wb"))
from qiskit.synthesis import TwoQubitWeylDecomposition
print("weyl", TwoQubitWeylDecomposition(Operator(sub).data).a, TwoQubitWeylDecomposition(Operator(sub).data).b, TwoQubitWeylDecomposition(Operator(sub).data).c)
print(sub)
