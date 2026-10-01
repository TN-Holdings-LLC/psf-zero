"""Exploratory: A2 vs A3 vs L3 vs L3T, noisy simulation, development inputs (replay set; a2 seeds 8001-8040)."""
import sys, os, io, contextlib, warnings, random, time, statistics
sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, core_fix_c2_eval as H
comp = H.load_module(sys.argv[1], "psf_compile"); lay = H.load_module(sys.argv[2], "psf_smart_layout")
a2 = H.load_module(sys.argv[3], "ai_a2"); a3 = H.load_module("psf_ai_compile.py", "ai_a3")
import os; sys.path.insert(0, os.environ.get('V10_DIR', '.'))
import e2e_vllm_psf_v10 as E, rp_eval
from psf_pennylane_gpu_prototype import tape_to_qiskit
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider
dev, which = sys.argv[4], sys.argv[5]
b = getattr(fake_provider, dev)(); tgt = b.target; cm = tgt.build_coupling_map(); nat = [g for g in ("cx","cz","rz","sx","x") if g in tgt.operation_names]
noisy = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(b))
def inf(qc, out):
    n = qc.num_qubits; fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n]); c = out.copy(); c.save_density_matrix(qubits=fin)
    rho = np.asarray(noisy.run(c).result().data()["density_matrix"]); psi = Statevector(qc).data; return 1 - float(np.real(psi.conj() @ rho @ psi))
circs = []
if which == "replay":
    for task, spec, src in rp_eval.collect(sys.argv[6], E):
        n = E.TASKS[task][0]; tape, _, _, _ = E.to_tape(spec, n); qc, _ = tape_to_qiskit(tape, wire_order=list(range(n))); circs.append(qc)
else:
    for seed in range(8001, 8041):
        rng = random.Random(seed); n = rng.choice([3,4,5]); circs.append(H.rand_dense(n, rng.randint(6,20), rng))
acc = {k: [] for k in ("A2", "A4", "L3", "L3T")}; tq = {k: 0 for k in acc}; ts = []
for qc in circs:
    with contextlib.redirect_stdout(io.StringIO()):
        o2 = a2.compile_for_model_circuit(qc, cm, nat, target=tgt)
        t = time.perf_counter(); o3 = a3.compile_for_model_circuit(qc, cm, nat, target=tgt); ts.append(time.perf_counter() - t)
    outs = dict(A2=o2, A4=o3, L3=transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0),
                L3T=transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0))
    for k, o in outs.items(): acc[k].append(inf(qc, o)); tq[k] += H.two_q(o)
m = {k: sum(v) / len(v) for k, v in acc.items()}
print(f"{dev} {which} n={len(circs)}: mean infidelity " + " ".join(f"{k} {v:.4f}" for k, v in m.items()) + " | 2q " + " ".join(f"{k} {v}" for k, v in tq.items())
      + f" | A4 better/worse than L3 {sum(a<b for a,b in zip(acc['A4'],acc['L3']))}/{sum(a>b for a,b in zip(acc['A4'],acc['L3']))} | A4 median {statistics.median(ts)*1000:.0f} ms", flush=True)
