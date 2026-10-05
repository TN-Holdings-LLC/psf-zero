import warnings; warnings.simplefilter("ignore")
from qiskit import qpy, QuantumCircuit
from qiskit_aer import AerSimulator
import qiskit_aer, qiskit
from qiskit.quantum_info import SparsePauliOp, Statevector
print("qiskit", qiskit.__version__, "aer", qiskit_aer.__version__)
m = qpy.load(open("minimal.qpy", "rb"))[0]
def ev(c, q, trunc, label="e"):
    c = c.copy(); c.save_expectation_value(SparsePauliOp("Z"), [q], label=label)
    return round(float(AerSimulator(method="statevector", enable_truncation=trunc).run(c).result().data()[label]), 6)
for q in (13, 14, 16):
    print("Z on", q, "trunc", ev(m, q, True), "no trunc", ev(m, q, False))
# relabel onto 3 qubits
def relabel(c, mp, n):
    r = QuantumCircuit(n)
    for i in c.data: r.append(i.operation, [mp[c.find_bit(b).index] for b in i.qubits])
    return r
for mp, n, tag in (({13: 0, 14: 1, 16: 2}, 3, "3q 13,14,16->0,1,2"), ({13: 13, 14: 14, 16: 16}, 17, "17q same"),
                   ({13: 2, 14: 1, 16: 0}, 3, "3q reversed"), ({13: 0, 14: 1, 16: 3}, 4, "4q gap at 2")):
    r = relabel(m, mp, n)
    print(tag, "trunc", ev(r, mp[13], True), "no trunc", ev(r, mp[13], False))
r = relabel(m, {13: 0, 14: 1, 16: 2}, 3)
print("Statevector Z0:", Statevector(r).expectation_value(SparsePauliOp("IIZ")).real)
# simpler gates: try to replace sx/rz block
for desc, build in (
    ("x16; cx14,16; cx14,13", lambda c: (c.x(16), c.cx(14, 16), c.cx(14, 13))),
    ("h16; cx14,16; cx14,13", lambda c: (c.h(16), c.cx(14, 16), c.cx(14, 13))),
    ("x14; cx14,16; cx14,13", lambda c: (c.x(14), c.cx(14, 16), c.cx(14, 13))),
    ("x14; cx14,13 (no 16)", lambda c: (c.x(14), c.cx(14, 13))),
    ("x14; cx14,16; cx14,13; only 13,14,16", lambda c: (c.x(14), c.cx(14, 16), c.cx(14, 13))),
):
    c = QuantumCircuit(27); build(c)
    print(desc, ": trunc", ev(c, 13, True), "no trunc", ev(c, 13, False))
