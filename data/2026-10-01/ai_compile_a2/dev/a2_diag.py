import sys, io, contextlib, warnings, random
sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, core_fix_c2_eval as H
comp = H.load_module(sys.argv[1], "psf_compile"); lay = H.load_module(sys.argv[2], "psf_smart_layout")
a2 = H.load_module("psf_ai_compile.py", "ai_a2")
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider
b = fake_provider.FakeAuckland(); tgt = b.target; cm = tgt.build_coupling_map(); nat = ["cx","rz","sx","x"]
noisy = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(b))
def inf(qc, out):
    n = qc.num_qubits; fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n]); c = out.copy(); c.save_density_matrix(qubits=fin)
    rho = np.asarray(noisy.run(c).result().data()["density_matrix"]); psi = Statevector(qc).data; return 1 - float(np.real(psi.conj() @ rho @ psi))
rows = []
for seed in range(7001, 7041):
    rng = random.Random(seed); n = rng.choice([3,4,5]); qc = H.rand_dense(n, rng.randint(6,20), rng)
    with contextlib.redirect_stdout(io.StringIO()):
        o2 = a2.compile_for_model_circuit(qc, cm, nat, target=tgt)
    t = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0)
    rows.append((seed, H.two_q(o2), H.two_q(t), a2.estimated_cost(o2, tgt), a2.estimated_cost(t, tgt), inf(qc, o2), inf(qc, t)))
est_worse = sum(r[3] > r[4] + 1e-9 for r in rows); act_worse = sum(r[5] > r[6] + 1e-9 for r in rows)
both = sum((r[3] > r[4] + 1e-9) and (r[5] > r[6]) for r in rows)
print("A2 est cost worse than L3T:", est_worse, "| actual worse:", act_worse, "| both:", both)
import math
pred = [1 - math.exp(-r[3]) for r in rows] + [1 - math.exp(-r[4]) for r in rows]; act = [r[5] for r in rows] + [r[6] for r in rows]
print("corr(est, actual) =", round(float(np.corrcoef(pred, act)[0,1]), 3), "mean est", round(sum(pred)/len(pred),4), "mean act", round(sum(act)/len(act),4))
for r in sorted(rows, key=lambda r: r[5]-r[6], reverse=True)[:6]: print("seed %d 2q %d/%d est %.4f/%.4f act %.4f/%.4f" % r)
