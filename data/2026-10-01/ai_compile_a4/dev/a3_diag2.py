import sys, io, contextlib, warnings, random, math
sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import numpy as np, core_fix_c2_eval as H
comp = H.load_module(sys.argv[1], "psf_compile"); lay = H.load_module(sys.argv[2], "psf_smart_layout")
a2 = H.load_module(sys.argv[3], "ai_a2"); a3 = H.load_module("psf_ai_compile.py", "ai_a3")
from qiskit.quantum_info import average_gate_fidelity
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider
b = fake_provider.FakeAuckland(); tgt = b.target; cm = tgt.build_coupling_map(); nat = ["cx","rz","sx","x"]
nm = NoiseModel.from_backend(b); errs = nm._local_quantum_errors
cache = {}
def chan_cost(name, qs):
    k = (name, tuple(qs))
    if k not in cache:
        qe = errs.get(name, {}).get(tuple(qs))
        cache[k] = -math.log(average_gate_fidelity(qe.to_quantumchannel())) if qe is not None else 0.0
    return cache[k]
for seed in (8031, 8037, 8012):
    rng = random.Random(seed); n = rng.choice([3,4,5]); qc = H.rand_dense(n, rng.randint(6,20), rng)
    with contextlib.redirect_stdout(io.StringIO()):
        o2 = a2.compile_for_model_circuit(qc, cm, nat, target=tgt); o3 = a3.compile_for_model_circuit(qc, cm, nat, target=tgt)
    for nm_, o in (("A2", o2), ("A3", o3)):
        used = sorted({o.find_bit(q).index for i in o.data for q in i.qubits})
        ch = sum(chan_cost(i.operation.name, [o.find_bit(q).index for q in i.qubits]) for i in o.data)
        dur = {}
        print(seed, nm_, "qubits", used, "a3cost %.4f channel-sum %.4f" % (a3.estimated_cost(o, tgt), ch),
              "T2(us)", [round(tgt.qubit_properties[q].t2*1e6) for q in used])
