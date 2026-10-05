import warnings; warnings.simplefilter("ignore")
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
import qiskit, qiskit_aer
from qiskit_aer import AerSimulator
print("qiskit", qiskit.__version__, "aer", qiskit_aer.__version__)
c = QuantumCircuit(3); c.x(2); c.cx(2, 1)
exact = Statevector(c).expectation_value(SparsePauliOp("IZI")).real
s = c.copy(); s.save_expectation_value(SparsePauliOp("Z"), [1], label="e")
s2 = c.copy(); s2.save_expectation_value(SparsePauliOp("IZI"), [0, 1, 2], label="e")
r = lambda x, **o: float(AerSimulator(method="statevector", **o).run(x).result().data()["e"])
print("exact", exact, "| save_expval Z on [1]:", r(s), "| no trunc:", r(s, enable_truncation=False), "| full-width IZI:", r(s2))
try:
    from qiskit_aer.primitives import EstimatorV2
    print("EstimatorV2 IZI:", float(EstimatorV2().run([(c, SparsePauliOp("IZI"))]).result()[0].data.evs))
except Exception as e:
    print("EstimatorV2 n/a", e)
try:
    from qiskit_aer.primitives import Estimator
    print("Estimator(V1) IZI:", Estimator().run(c, SparsePauliOp("IZI")).result().values[0])
except Exception as e:
    print("Estimator V1 n/a", type(e).__name__)
