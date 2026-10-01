import sys, io, contextlib, warnings, random, math
sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, core_fix_c2_eval as H
comp = H.load_module(sys.argv[1], "psf_compile"); lay = H.load_module(sys.argv[2], "psf_smart_layout")
a2 = H.load_module(sys.argv[3], "ai_a2"); a3 = H.load_module("psf_ai_compile.py", "ai_a3")
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider
b = fake_provider.FakeAuckland(); tgt = b.target; cm = tgt.build_coupling_map(); nat = ["cx","rz","sx","x"]
noisy = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(b))
def inf(qc, out):
    n = qc.num_qubits; fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n]); c = out.copy(); c.save_density_matrix(qubits=fin)
    rho = np.asarray(noisy.run(c).result().data()["density_matrix"]); psi = Statevector(qc).data; return 1 - float(np.real(psi.conj() @ rho @ psi))
est2, est3, act = [], [], []; better = worse = 0; rows = []
for seed in range(8001, 8041):
    rng = random.Random(seed); n = rng.choice([3,4,5]); qc = H.rand_dense(n, rng.randint(6,20), rng)
    with contextlib.redirect_stdout(io.StringIO()):
        o2 = a2.compile_for_model_circuit(qc, cm, nat, target=tgt); o3 = a3.compile_for_model_circuit(qc, cm, nat, target=tgt)
    for o in (o2, o3):
        est2.append(a2.estimated_cost(o, tgt)); est3.append(a3.estimated_cost(o, tgt)); act.append(inf(qc, o))
    i2, i3 = act[-2], act[-1]; better += i3 < i2 - 1e-12; worse += i3 > i2 + 1e-12
    rows.append((seed, H.two_q(o2), H.two_q(o3), est3[-2], est3[-1], i2, i3))
conv = lambda e: [1 - math.exp(-x) for x in e]
print("A3 better/worse than A2:", better, worse)
print("corr(a2 cost, actual) %.3f | corr(a3 cost, actual) %.3f" % (np.corrcoef(conv(est2), act)[0,1], np.corrcoef(conv(est3), act)[0,1]))
print("a3 cost says A3 <= A2 in", sum(r[4] <= r[3] + 1e-12 for r in rows), "of", len(rows))
for r in sorted(rows, key=lambda r: r[6]-r[5], reverse=True)[:6]: print("seed %d 2q %d/%d a3cost %.4f/%.4f actual %.4f/%.4f" % r)
