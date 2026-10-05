import numpy as np, warnings; warnings.simplefilter("ignore")
import pilot_depth as P
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import fake_provider as fp
be = fp.FakeAuckland(); C = P.compilers(be)
Xtr,ytr,Xte,yte = P.data(6); th = np.random.default_rng(3).normal(size=P.n_params(6,1))
qc = P.circuit(Xte[0], th, 6, 1); zi = P.forward(th, Xte[:1], 6, 1)[0]
out = C["L3T"](qc); fin = out.layout.final_index_layout(filter_ancillas=True)
c = out.copy(); c.save_density_matrix(qubits=[fin[0]])
rho = np.asarray(AerSimulator(method="density_matrix").run(c).result().data()["density_matrix"])
print("ideal %.4f  dm-save %.4f" % (zi, (rho[0,0]-rho[1,1]).real))
c = out.copy(); c.save_expectation_value(SparsePauliOp("Z"), [fin[0]])
print("ev-save dm method %.4f" % AerSimulator(method="density_matrix").run(c).result().data()["expectation_value"])
print("ev-save sv method %.4f" % AerSimulator(method="statevector").run(c).result().data()["expectation_value"])
print("ev-save sv method fusion off %.4f" % AerSimulator(method="statevector", fusion_enable=False).run(c).result().data()["expectation_value"])
