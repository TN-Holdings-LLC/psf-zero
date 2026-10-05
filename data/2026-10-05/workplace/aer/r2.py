import warnings; warnings.simplefilter("ignore")
import numpy as np
from qiskit import qpy, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
out = qpy.load(open("case.qpy", "rb"))[0]
Q = 13
def ev(c, trunc):
    c = c.copy(); c.save_expectation_value(SparsePauliOp("Z"), [Q], label="e")
    return AerSimulator(method="statevector", enable_truncation=trunc).run(c).result().data()["e"]
def bad(c):
    return abs(ev(c, True) - ev(c, False)) > 1e-6
base = QuantumCircuit(out.num_qubits)
ins = [(i.operation, [out.find_bit(q).index for q in i.qubits]) for i in out.data if i.operation.name not in ("barrier", "measure")]
def build(lst):
    c = QuantumCircuit(out.num_qubits)
    for op, qs in lst: c.append(op, qs)
    return c
assert bad(build(ins))
cur = ins[:]
changed = True
while changed:
    changed = False
    i = 0
    while i < len(cur):
        trial = cur[:i] + cur[i+1:]
        if trial and bad(build(trial)):
            cur = trial; changed = True
        else:
            i += 1
c = build(cur)
print("minimal:", len(cur), "instructions")
for op, qs in cur: print("  ", op.name, qs, [round(float(p), 4) for p in op.params])
print("ev trunc", ev(c, True), "no trunc", ev(c, False))
with open("minimal.qpy", "wb") as f: qpy.dump(c, f)
