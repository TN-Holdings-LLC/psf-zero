import numpy as np, warnings; warnings.simplefilter("ignore")
import pilot_depth as P
from qiskit import transpile, QuantumCircuit, qpy
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_ibm_runtime import fake_provider as fp
be = fp.FakeAuckland()
Xtr,ytr,Xte,yte = P.data(6); th = np.random.default_rng(3).normal(size=P.n_params(6,1))
qc = P.circuit(Xte[0], th, 6, 1); zi = P.forward(th, Xte[:1], 6, 1)[0]
out = transpile(qc, target=be.target, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)
fin = out.layout.final_index_layout(filter_ancillas=True)
print("ideal", zi, "fin0", fin[0])
with open("case.qpy", "wb") as f: qpy.dump(out, f)
def ev(c, **o):
    c = c.copy(); c.save_expectation_value(SparsePauliOp("Z"), [fin[0]], label="e")
    return AerSimulator(method="statevector", **o).run(c).result().data()["e"]
def dm(c, **o):
    c = c.copy(); c.save_density_matrix(qubits=[fin[0]], label="d")
    r = np.asarray(AerSimulator(method="statevector", **o).run(c).result().data()["d"]); return (r[0,0]-r[1,1]).real
for o in ({}, {"enable_truncation": False}, {"fusion_enable": False}, {"enable_truncation": False, "fusion_enable": False}):
    print(o, "ev %.4f" % ev(out, **o), "dm %.4f" % dm(out, **o))
c = out.copy(); c.save_expectation_value(Pauli("Z"), [fin[0]], label="e")
print("Pauli object ev %.4f" % AerSimulator(method="statevector").run(c).result().data()["e"])
