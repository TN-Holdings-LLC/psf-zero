"""Does Qiskit's CX-basis synthesis fail on the #17057 inputs in this environment?"""
import platform

import numpy as np
import qiskit
import scipy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.synthesis import TwoQubitBasisDecomposer

print("qiskit", qiskit.__version__, "| numpy", np.__version__, "| scipy", scipy.__version__,
      "|", platform.platform())
np.show_config()
dec = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
for c in (1e-8, 3e-8, 1e-7, 2e-7, 3e-7, 1e-6):
    core = QuantumCircuit(2)
    core.rxx(-1.2, 0, 1)
    core.ryy(-0.6, 0, 1)
    core.rzz(-2 * c, 0, 1)
    u = Operator(core)
    e1 = 1 - average_gate_fidelity(Operator(dec(u.data)), u)
    t = transpile(core, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
    e2 = 1 - average_gate_fidelity(Operator(t), u)
    print(f"c={c:.0e}  decomposer {e1:.3e}   transpile L1 {e2:.3e}")
