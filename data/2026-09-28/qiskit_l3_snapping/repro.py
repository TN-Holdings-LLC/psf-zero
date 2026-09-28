"""repro.py -- exploratory: reproduce the Qiskit L3 output with mirror error
1.17e-5 seen in the routing-arms dry run (F2 brick grid, n = 12, instance 0,
seed 240100, 3x4 grid, basis cx/rz/sx/x, optimization_level 3, seed 0).
Error measure: the mirror test of routing_pressure_arms.py, computed with
Qiskit's Statevector (output, then the inverse of the original circuit on
the final positions; norm of all amplitudes other than |0...0>)."""
import sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
import routing_pressure_arms as rp

def mirror(qc, out, n):
    pos = rp.final_positions(out, n)
    m = out.num_qubits
    full = QuantumCircuit(m)
    full.compose(out.remove_final_measurements(inplace=False) if False else out, range(m), inplace=True)
    full.compose(qc.inverse(), pos, inplace=True)
    a = Statevector(full).data
    return float(np.sqrt(np.sum(np.abs(a[1:]) ** 2)))

print("qiskit", qiskit.__version__, flush=True)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
cmap = rp.GRIDS and CouplingMap.from_grid(*rp.GRIDS[n])
qc = rp.build("F2_brick_grid", n, 20000 * n + 100 * 1 + 0)
for lvl, reps in ((3, 3), (2, 1), (1, 1)):
    for rep in range(reps):
        t0 = time.time()
        out = transpile(qc, coupling_map=cmap, basis_gates=rp.BASIS, optimization_level=lvl, seed_transpiler=0)
        t1 = time.time()
        e = mirror(qc, out, n)
        print(f"L{lvl} rep {rep}: cx {out.count_ops().get('cx', 0)} ops {sum(out.count_ops().values())} "
              f"mirror {e:.3e} (transpile {t1 - t0:.1f} s, check {time.time() - t1:.1f} s)", flush=True)
