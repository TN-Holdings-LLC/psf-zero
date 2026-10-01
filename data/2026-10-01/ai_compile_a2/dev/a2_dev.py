"""Exploratory: A1 vs A2 (error-aware) vs L3 vs L3 with target, noisy simulation, development seeds 7001-7040."""
import sys, os, io, contextlib, warnings, random, time, statistics, json
sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np
import core_fix_c2_eval as H
comp = H.load_module(sys.argv[1], "psf_compile"); lay = H.load_module(sys.argv[2], "psf_smart_layout")
a1 = H.load_module(sys.argv[3], "ai_a1"); a2 = H.load_module("psf_ai_compile.py", "ai_a2")
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider
devs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["FakeAuckland", "FakeTorino"]
seeds = range(7001, 7041)
def fid(qc, out, sim):
    n = qc.num_qubits; fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
    c = out.copy(); c.save_density_matrix(qubits=fin)
    rho = np.asarray(sim.run(c).result().data()["density_matrix"]); psi = Statevector(qc).data
    return float(np.real(psi.conj() @ rho @ psi))
for d in devs:
    b = getattr(fake_provider, d)(); tgt = b.target; cm = tgt.build_coupling_map(); nat = [g for g in ("cx","cz","rz","sx","x") if g in tgt.operation_names]
    noisy = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(b)); ideal = AerSimulator(method="density_matrix")
    acc = {k: [] for k in ("A1", "A2", "L3", "L3T")}; tq = {k: 0 for k in acc}; bad = 0; tA2 = []
    for seed in seeds:
        rng = random.Random(seed); n = rng.choice([3,4,5]); qc = H.rand_dense(n, rng.randint(6,20), rng)
        outs = {}
        with contextlib.redirect_stdout(io.StringIO()):
            outs["A1"] = a1.compile_for_model_circuit(qc, cm, nat)
            t = time.perf_counter(); outs["A2"] = a2.compile_for_model_circuit(qc, cm, nat, target=tgt); tA2.append(time.perf_counter()-t)
        outs["L3"] = transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0)
        outs["L3T"] = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0)
        for k, o in outs.items():
            if fid(qc, o, ideal) < 1 - 1e-9: bad += 1; print("NOT EXACT", d, seed, k)
            acc[k].append(1 - fid(qc, o, noisy)); tq[k] += H.two_q(o)
    m = {k: sum(v)/len(v) for k, v in acc.items()}
    w = sum(a < b for a, b in zip(acc["A2"], acc["L3T"]))
    print(f"{d}: mean infidelity " + " ".join(f"{k} {v:.4f}" for k, v in m.items()) + f" | 2q " + " ".join(f"{k} {v}" for k, v in tq.items())
          + f" | A2 better than L3T in {w}/{len(acc['A2'])} | not exact {bad} | A2 median {statistics.median(tA2)*1000:.0f} ms", flush=True)
