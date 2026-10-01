import sys, io, contextlib, warnings, random, time, statistics
sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import psf_ai_compile as ai
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import fake_provider
src = open("eqtest.py").read(); exec("def rand_dense" + src.split("def rand_dense")[1].split("rng = random.Random")[0])
tq = lambda c: sum(1 for i in c.data if len(i.qubits) == 2)
cfgs = {"c2 only (1 seed)": dict(seeds=(0,), commutation_start=False, do_polish=False),
        "+ 4 seeds": dict(seeds=(0,1,2,3), commutation_start=False, do_polish=False),
        "+ commuted start": dict(seeds=(0,), commutation_start=True, do_polish=False),
        "+ polish": dict(seeds=(0,), commutation_start=False, do_polish=True),
        "all (a0)": dict()}
for bname, s0 in (("FakeAuckland", 7), ("FakeKingston", 9)):
    t = getattr(fake_provider, bname)().target; cm = t.build_coupling_map(); nat = [g for g in ("cx","cz","rz","sx","x") if g in t.operation_names]
    rng = random.Random(s0); circs = []
    for k in range(60):
        n = rng.choice([3,4,5]); circs.append(rand_dense(n, rng.randint(6,20), rng))
    for name, kw in cfgs.items():
        ts = []; s = 0
        for qc in circs:
            t0 = time.perf_counter(); o = ai.compile_for_model_circuit(qc, cm, nat, **kw); ts.append(time.perf_counter()-t0); s += tq(o)
        print(f"{bname} seeds{s0} {name:18s} sum2q {s} median ms {statistics.median(ts)*1000:.0f}", flush=True)
