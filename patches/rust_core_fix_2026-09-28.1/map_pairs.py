"""Map captured block positions to E-circuit pairs by comparing each
captured block's Weyl coordinates with each pair's (rxx, ryy, rzz) core."""
import sys, os, numpy as np
REPO = sys.argv[1]
sys.path[:0] = [REPO, os.path.join(REPO, "benchmarks")]
import loop_endurance as le
from qiskit.synthesis import TwoQubitWeylDecomposition
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit
U = np.load("blocks_base3000.npz")["U"]
rng_e = np.random.default_rng(7)
tt = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS); tt[8::15] = 0.0
rng_t = np.random.default_rng(202)
thetas = [tt + rng_t.normal(0.0, 0.5, le.E_NPARAMS) for _ in range(3000)]
def weyl(u):
    d = TwoQubitWeylDecomposition(u); return np.array([d.a, d.b, d.c])
def pair_weyl(th, k):
    qc = QuantumCircuit(2); p = th[15*k:15*k+15]
    qc.rxx(p[6], 0, 1); qc.ryy(p[7], 0, 1); qc.rzz(p[8], 0, 1)
    return weyl(Operator(qc).data)
for lap in [int(x) for x in sys.argv[2].split(",")]:
    th = thetas[lap - 1]
    pw = [pair_weyl(th, k) for k in range(len(le.E_BLOCKS))]
    m = []
    for pos in range(11):
        w = weyl(U[(lap - 1) * 11 + pos]); dist = [np.linalg.norm(w - q) for q in pw]; k = int(np.argmin(dist))
        m.append(f"{pos}->{le.E_BLOCKS[k]}" + ("" if dist[k] < 1e-6 else f"(?{dist[k]:.0e})"))
    print(lap, " ".join(m))
