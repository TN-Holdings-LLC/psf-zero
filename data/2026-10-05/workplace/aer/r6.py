import warnings; warnings.simplefilter("ignore")
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
import qiskit_aer; print("aer", qiskit_aer.__version__)
c = QuantumCircuit(3); c.x(2); c.cx(2, 1)
for trunc in (True, False):
    s = c.copy(); s.save_expectation_value(SparsePauliOp("Z"), [1], label="e")
    try:
        print("trunc", trunc, float(AerSimulator(method="statevector", enable_truncation=trunc).run(s).result().data()["e"]))
    except Exception as e:
        print("trunc", trunc, "ERROR", str(e)[:120])
