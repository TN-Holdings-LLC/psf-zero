"""Minimal reproducer: AerSimulator.save_expectation_value with an operator on a SUBSET of the qubits returns a wrong
value when qubit truncation is on (the default) and the circuit's active qubits are not 0..k-1.
Expected Z on qubit 1 of X(2) CX(2,1)|000> is -1."""
import warnings; warnings.simplefilter("ignore")
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
import qiskit, qiskit_aer
from qiskit_aer import AerSimulator

c = QuantumCircuit(3)
c.x(2)
c.cx(2, 1)
print("qiskit", qiskit.__version__, "| qiskit-aer", qiskit_aer.__version__)
print("exact (Statevector):", Statevector(c).expectation_value(SparsePauliOp("IZI")).real)
for method in ("statevector", "density_matrix"):
    for trunc in (True, False):
        s = c.copy()
        s.save_expectation_value(SparsePauliOp("Z"), [1], label="e")
        try:
            v = float(AerSimulator(method=method, enable_truncation=trunc).run(s).result().data()["e"])
        except Exception as e:
            v = f"ERROR {str(e)[:80]}"
        print(f"{method:15s} enable_truncation={trunc!s:5s} save_expectation_value(Z, [1]) = {v}")
s = c.copy(); s.save_expectation_value(SparsePauliOp("IZI"), [0, 1, 2], label="e")
print("full-width operator IZI on [0,1,2], truncation on:", float(AerSimulator().run(s).result().data()["e"]))
