"""confirm.py -- exploratory: (1) the snapped block synthesized exactly
(Qiskit's Weyl decomposition with fidelity=None, via psf_compile._exact_rebuild)
has no error; (2) passing approximation_degree=1.0 explicitly changes nothing;
(3) the same circuit at L3 with the peephole pass's result compared per
setting."""
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
import routing_pressure_arms as rp
import psf_compile as pc

def dist(u, v):
    t = np.trace(u.conj().T @ v); ph = t / abs(t)
    return float(np.linalg.norm(v - ph * u))

def mirror(qc, out, n=12):
    pos = rp.final_positions(out, n)
    full = QuantumCircuit(out.num_qubits); full.compose(out, range(out.num_qubits), inplace=True)
    full.compose(qc.inverse(), pos, inplace=True)
    a = Statevector(full).data
    return float(np.sqrt(np.sum(np.abs(a[1:]) ** 2)))

qc = rp.build("F2_brick_grid", 12, 240100)
cmap = CouplingMap.from_grid(3, 4)
snap = {}
def cb(**kw):
    if kw["count"] == 24:
        snap["pre"] = dag_to_circuit(kw["dag"])
transpile(qc, coupling_map=cmap, basis_gates=rp.BASIS, optimization_level=3, seed_transpiler=0, callback=cb)
cons = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(snap["pre"])
u = next(Operator(i.operation).data for i in cons.data if i.operation.name == "unitary"
         and [cons.find_bit(x).index for x in i.qubits] == [5, 6])
print(f"block [5,6]: TwoQubitBasisDecomposer(CX) error {dist(u, Operator(TwoQubitBasisDecomposer(CXGate())(u)).data):.3e}")
print(f"block [5,6]: exact rebuild (Weyl fidelity=None + polish) error {dist(u, Operator(pc._exact_rebuild(u)).data):.3e}")
for ad in (None, 1.0):
    kw = {} if ad is None else {"approximation_degree": ad}
    out = transpile(qc, coupling_map=cmap, basis_gates=rp.BASIS, optimization_level=3, seed_transpiler=0, **kw)
    print(f"L3 approximation_degree={'default' if ad is None else ad}: mirror {mirror(qc, out):.3e}")
