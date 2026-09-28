"""block.py -- exploratory: capture the circuit just before the first
TwoQubitPeepholeOptimization of the L3 run, collect its 2-qubit blocks, and
for each block compare (a) TwoQubitWeylDecomposition with the default
fidelity (1 - 1e-9) and with fidelity=None, (b) TwoQubitBasisDecomposer(CX)
output against the block."""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer, TwoQubitWeylDecomposition
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
import routing_pressure_arms as rp

qc = rp.build("F2_brick_grid", 12, 240100)
snap = {}
def cb(**kw):
    if type(kw["pass_"]).__name__ == "TwoQubitPeepholeOptimization" and "before" not in snap:
        snap["before"] = dag_to_circuit(kw["dag"])  # dag AFTER this pass
    if kw["count"] == 24:
        snap["pre"] = dag_to_circuit(kw["dag"])
transpile(qc, coupling_map=CouplingMap.from_grid(3, 4), basis_gates=rp.BASIS, optimization_level=3,
          seed_transpiler=0, callback=cb)
pre = snap["pre"]
cons = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(pre)
dec = TwoQubitBasisDecomposer(CXGate())

def dist(u, v):
    t = np.trace(u.conj().T @ v); ph = t / abs(t)
    return float(np.linalg.norm(v - ph * u))

rows = []
for inst in cons.data:
    if inst.operation.name != "unitary" or len(inst.qubits) != 2:
        continue
    u = Operator(inst.operation).data
    wd = TwoQubitWeylDecomposition(u)
    we = TwoQubitWeylDecomposition(u, fidelity=None)
    exact_spec = (float(we.a), float(we.b), float(we.c))
    syn = dec(u)
    e = dist(u, Operator(syn).data)
    q = [cons.find_bit(x).index for x in inst.qubits]
    rows.append((e, q, f"calc fid {wd.calculated_fidelity:.12f}, 1-fid {1 - wd.calculated_fidelity:.2e}", exact_spec, (float(wd.a), float(wd.b), float(wd.c)), syn.count_ops().get("cx", 0)))
rows.sort(reverse=True)
print(f"{len(rows)} blocks; worst synthesis errors:")
for e, q, s1, s2, abc, ncx in rows[:5]:
    print(f"  qubits {q}: synthesis error {e:.3e}; default Weyl {s1}; (a,b,c) default {tuple(round(x, 9) for x in abc)} "
          f"vs fidelity=None {tuple(round(x, 9) for x in s2)}; CX {ncx}")
