import os
import pickle, numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.synthesis import TwoQubitBasisDecomposer, TwoQubitWeylDecomposition
from qiskit.circuit.library import CZGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ConsolidateBlocks, UnitarySynthesis, TwoQubitPeepholeOptimization
from qiskit_ibm_runtime.fake_provider import FakeFez
sub, qs = pickle.load(open("bad_block.pkl", "rb"))
U = Operator(sub)
dec = TwoQubitBasisDecomposer(CZGate(), euler_basis="ZSX")
for bf in (1.0, None):
    c = dec(U.data, basis_fidelity=bf) if bf is not None else dec(U.data)
    print("TwoQubitBasisDecomposer basis_fidelity", bf, "infid %.2e" % (1 - average_gate_fidelity(U, Operator(c))), "cz", c.count_ops().get("cz", 0))
c = dec(U.data, approximate=False)
print("approximate=False infid %.2e" % (1 - average_gate_fidelity(U, Operator(c))), c.count_ops().get("cz", 0))
T = FakeFez().target
big = QuantumCircuit(T.num_qubits); big.compose(sub, qubits=qs, inplace=True)
def red(c):
    r = QuantumCircuit(2, global_phase=c.global_phase)
    for i in c.data: r.append(i.operation, [qs.index(c.find_bit(q).index) for q in i.qubits])
    return Operator(r)
o1 = PassManager([ConsolidateBlocks(target=T, approximation_degree=1.0), UnitarySynthesis(target=T, approximation_degree=1.0)]).run(big)
print("ConsolidateBlocks+UnitarySynthesis infid %.2e" % (1 - average_gate_fidelity(U, red(o1))))
o2 = PassManager([TwoQubitPeepholeOptimization(T, approximation_degree=1.0)]).run(big)
print("TwoQubitPeepholeOptimization infid %.2e" % (1 - average_gate_fidelity(U, red(o2))))
w = TwoQubitWeylDecomposition(U.data)
print("weyl a,b,c", repr(w.a), repr(w.b), repr(w.c))
