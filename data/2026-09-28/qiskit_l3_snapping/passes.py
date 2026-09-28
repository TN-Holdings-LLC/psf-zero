"""passes.py -- exploratory: which transpiler pass introduces the 1.17e-5?
Runs the same L3 transpile with a callback; after every pass that changes
the circuit before the layout stage, compares its operator with the
original (12 qubits, 4096 x 4096 via Statevector on a random input, cheap)."""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from qiskit import transpile
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info import Statevector, random_statevector
from qiskit.transpiler import CouplingMap
import routing_pressure_arms as rp

n = 12
cmap = CouplingMap.from_grid(3, 4)
qc = rp.build("F2_brick_grid", n, 240100)
psi = random_statevector(2 ** n, seed=1)
ref = psi.evolve(qc).data
log = []

def cb(**kw):
    p = kw["pass_"]; name = type(p).__name__
    dag = kw["dag"]
    circ = dag_to_circuit(dag)
    lay = kw["property_set"].get("layout")
    if lay is not None and circ.num_qubits == n and kw["count"] >= 16:
        # after ApplyLayout: mirror test with the initial layout (VF2Layout found a perfect
        # layout, so there is no routing and the final positions equal the initial ones)
        from qiskit import QuantumCircuit
        pos = [lay[qc.qubits[i]] for i in range(n)]
        full = QuantumCircuit(n); full.compose(circ, range(n), inplace=True); full.compose(qc.inverse(), pos, inplace=True)
        a = Statevector(full).data
        log.append((kw["count"], name, float(np.sqrt(np.sum(np.abs(a[1:]) ** 2))))); return
    if circ.num_qubits != n or lay is not None:
        log.append((kw["count"], name, None)); return
    try:
        v = psi.evolve(circ).data
    except Exception as e:
        log.append((kw["count"], name, "err " + type(e).__name__)); return
    t = np.vdot(ref, v); ph = t / abs(t)
    log.append((kw["count"], name, float(np.linalg.norm(v - ph * ref))))

transpile(qc, coupling_map=cmap, basis_gates=rp.BASIS, optimization_level=3, seed_transpiler=0, callback=cb)
prev = 0.0
for c, name, e in log:
    mark = ""
    if isinstance(e, float):
        if e > 1e-10 and prev <= 1e-10: mark = "  <-- first above 1e-10"
        prev = e
        print(f"{c:3d} {name:40s} {e:.3e}{mark}")
    else:
        print(f"{c:3d} {name:40s} (after layout: {e})")
