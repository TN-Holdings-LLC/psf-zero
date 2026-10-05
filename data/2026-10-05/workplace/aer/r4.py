import warnings; warnings.simplefilter("ignore")
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
def ev(c, q, trunc=True):
    c = c.copy(); c.save_expectation_value(SparsePauliOp("Z"), [q], label="e")
    return round(float(AerSimulator(method="statevector", enable_truncation=trunc).run(c).result().data()["e"]), 6)
print("width ctl tgt | Z(tgt) trunc / exact | Z(ctl) trunc / exact")
for n, a, b in ((2, 1, 0), (2, 0, 1), (3, 2, 1), (3, 1, 2), (3, 2, 0), (5, 4, 3), (5, 3, 4), (15, 14, 13), (27, 14, 13), (27, 13, 14), (27, 1, 0), (27, 0, 1)):
    c = QuantumCircuit(n); c.x(a); c.cx(a, b)
    print(n, a, b, "|", ev(c, b), "/", ev(c, b, False), "|", ev(c, a), "/", ev(c, a, False))
# single-qubit only, non-zero index
for n, q in ((27, 13), (27, 0), (3, 2)):
    c = QuantumCircuit(n); c.x(q)
    print("x only", n, q, ev(c, q), "/", ev(c, q, False))
# two separate x
c = QuantumCircuit(27); c.x(14); c.h(13)
print("x14 h13: Z14", ev(c, 14), "/", ev(c, 14, False), " Z13", ev(c, 13), "/", ev(c, 13, False))
c = QuantumCircuit(27); c.x(14); c.id(13)
print("x14 id13: Z14", ev(c, 14), "/", ev(c, 14, False), " Z13", ev(c, 13), "/", ev(c, 13, False))
