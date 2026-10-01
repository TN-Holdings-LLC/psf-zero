"""Was the scored a4 run (a4_eval.py, seeds 9001-9040) affected by the id(target) cache defect? The run built the
FakeAuckland, FakeTorino and FakeKingston Targets one after another in one process, so a freed Target's id could be
reused. Recompute the A4 arm with a5 (value-keyed caches) in a fresh process per device and compare the noisy
infidelity circuit by circuit with the recorded A4 values."""
import sys, io, contextlib, warnings, random, json
sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, core_fix_c2_eval as H
S = sys.argv[1]; dev = sys.argv[2]; raw = json.load(open(sys.argv[3]))
H.load_module(S + '/release_c2/psf_compile.py', 'psf_compile'); H.load_module(S + '/release_c2/psf_smart_layout.py', 'psf_smart_layout')
a5 = H.load_module('psf_ai_compile.py', 'a5')
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider
b = getattr(fake_provider, dev)(); tgt = b.target; cm = tgt.build_coupling_map(); nat = [g for g in ("cx","cz","rz","sx","x") if g in tgt.operation_names]
noisy = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(b))
rec = {r["seed"]: r for r in raw["R"] if r["device"] == dev}
same = diff = 0; worst = 0.0; d_mean = []
for seed in range(9001, 9041):
    rng = random.Random(seed); n = rng.choice([3,4,5]); qc = H.rand_dense(n, rng.randint(6,20), rng)
    with contextlib.redirect_stdout(io.StringIO()):
        o = a5.compile_for_model_circuit(qc, cm, nat, target=tgt)
    fin = list(o.layout.final_index_layout(filter_ancillas=True)[:n]); c = o.copy(); c.save_density_matrix(qubits=fin)
    rho = np.asarray(noisy.run(c).result().data()["density_matrix"]); psi = Statevector(qc).data
    f = float(np.real(psi.conj() @ rho @ psi)); r = rec[seed]["A4"]
    if abs(f - r["noisy"]) < 1e-9 and H.two_q(o) == r["two_q"]: same += 1
    else: diff += 1; worst = max(worst, abs(f - r["noisy"]))
    d_mean.append((1 - f) - (1 - r["noisy"]))
print(f"{dev}: a5 reproduces recorded A4 exactly on {same}/40 circuits; differs on {diff} (max |dF| {worst:.2e}); mean infidelity change {np.mean(d_mean):+.5f}")
