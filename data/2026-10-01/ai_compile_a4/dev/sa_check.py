import sys, io, contextlib, warnings, random, math
sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, core_fix_c2_eval as H
comp = H.load_module(sys.argv[1], "psf_compile"); lay = H.load_module(sys.argv[2], "psf_smart_layout")
a2 = H.load_module(sys.argv[3], "ai_a2"); a3 = H.load_module("psf_ai_compile.py", "ai_a3")
from stateaware import state_aware_cost
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider
b = getattr(fake_provider, sys.argv[4])(); tgt = b.target; cm = tgt.build_coupling_map(); nat = [g for g in ("cx","cz","rz","sx","x") if g in tgt.operation_names]
noisy = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(b))
def inf(qc, out):
    n = qc.num_qubits; fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n]); c = out.copy(); c.save_density_matrix(qubits=fin)
    rho = np.asarray(noisy.run(c).result().data()["density_matrix"]); psi = Statevector(qc).data; return 1 - float(np.real(psi.conj() @ rho @ psi))
E_avg, E_sa, A = [], [], []
for seed in range(8001, 8021):
    rng = random.Random(seed); n = rng.choice([3,4,5]); qc = H.rand_dense(n, rng.randint(6,20), rng)
    with contextlib.redirect_stdout(io.StringIO()):
        outs = [a2.compile_for_model_circuit(qc, cm, nat, target=tgt), a3.compile_for_model_circuit(qc, cm, nat, target=tgt),
                transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0),
                transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0)]
    for o in outs:
        E_avg.append(1 - math.exp(-a3.estimated_cost(o, tgt))); E_sa.append(state_aware_cost(o, tgt)); A.append(inf(qc, o))
print(sys.argv[4], "corr(avg-infidelity cost, actual) %.3f | corr(state-aware, actual) %.3f | mean actual %.4f mean sa %.4f mean avg %.4f" % (
    np.corrcoef(E_avg, A)[0,1], np.corrcoef(E_sa, A)[0,1], np.mean(A), np.mean(E_sa), np.mean(E_avg)))
# within-circuit ranking accuracy (pairs)
ok_avg = ok_sa = tot = 0
for c in range(len(A)//4):
    for i in range(4):
        for j in range(i+1, 4):
            a, b_ = 4*c+i, 4*c+j
            if abs(A[a]-A[b_]) < 1e-6: continue
            tot += 1; ok_avg += (E_avg[a] < E_avg[b_]) == (A[a] < A[b_]); ok_sa += (E_sa[a] < E_sa[b_]) == (A[a] < A[b_])
print("within-circuit pair ranking correct: avg-cost %d/%d  state-aware %d/%d" % (ok_avg, tot, ok_sa, tot))
