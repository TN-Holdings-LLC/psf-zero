"""Exploratory: c2 compile_for_hardware vs psf_ai_compile prototype vs Qiskit L3 on development inputs."""
import sys, io, contextlib, warnings, random, time, json, glob, statistics
sys.path.insert(0, '.'); import os; sys.path.insert(0, os.environ.get('V10_DIR', '.')); warnings.simplefilter("ignore")
import psf_compile as pc, psf_ai_compile as ai
from physeq import phys_equiv
from qiskit import transpile, QuantumCircuit
from qiskit_ibm_runtime import fake_provider
src = open("eqtest.py").read(); exec("def rand_dense" + src.split("def rand_dense")[1].split("rng = random.Random")[0])
tq = lambda c: sum(1 for i in c.data if len(i.qubits) == 2)
def be(name):
    t = getattr(fake_provider, name)().target; return t.build_coupling_map(), [g for g in ("cx","cz","rz","sx","x") if g in t.operation_names]
def c2(qc, cm, nat):
    with contextlib.redirect_stdout(io.StringIO()):
        return pc.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx", layout_search=True, seed_transpiler=0)
part = sys.argv[1]
if part == "random":
    for bname, seed0 in (("FakeAuckland", 7), ("FakeAuckland", 8), ("FakeKingston", 9)):
        cm, nat = be(bname); rng = random.Random(seed0); S = [0, 0, 0]; worse = [0, 0]; bad = 0; T = [[], []]
        for k in range(60):
            n = rng.choice([3, 4, 5]); qc = rand_dense(n, rng.randint(6, 20), rng)
            t = time.perf_counter(); a = c2(qc, cm, nat); T[0].append(time.perf_counter() - t)
            t = time.perf_counter(); b = ai.compile_for_model_circuit(qc, cm, nat); T[1].append(time.perf_counter() - t)
            l3 = tq(transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0))
            for o in (a, b):
                ok, f = phys_equiv(qc, o); bad += not ok
            S[0] += tq(a); S[1] += tq(b); S[2] += l3; worse[0] += tq(b) > tq(a); worse[1] += tq(b) > l3
        print(f"{bname} seeds{seed0}: sum2q c2 {S[0]} ai {S[1]} L3 {S[2]} | ai>c2 {worse[0]} ai>L3 {worse[1]} ai<L3 ? | non-equiv {bad} | median ms c2 {statistics.median(T[0])*1000:.0f} ai {statistics.median(T[1])*1000:.0f}", flush=True)
else:
    import e2e_vllm_psf_v10 as E
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    cm, nat = be("FakeAuckland"); seen = set(); agg = {}
    for f in sorted(glob.glob("../gh_latest/data/2026-09-30/**/best_circuit.json", recursive=True)):
        task = f.split("/")[-2]
        if task not in E.TASKS: continue
        spec = json.load(open(f)); key = task + json.dumps(spec.get("gates"), sort_keys=True)
        if key in seen: continue
        seen.add(key); n, _, groups = E.TASKS[task]
        tape, _, _, _ = E.to_tape(spec, n); qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
        a = c2(qc, cm, nat); t = time.perf_counter(); b = ai.compile_for_model_circuit(qc, cm, nat); dt = time.perf_counter() - t
        l3 = tq(transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0))
        fa, _, _ = E.compiled_check("lightning.qubit", a, n, groups); fb, _, _ = E.compiled_check("lightning.qubit", b, n, groups)
        g = agg.setdefault(task, [0, 0, 0, 0, 0, 0, 0, []]); g[0] += 1; g[1] += tq(a); g[2] += tq(b); g[3] += l3
        g[4] += tq(b) > l3; g[5] += tq(b) > tq(a); g[6] += abs(fa - fb) > 1e-9; g[7].append(dt)
    for t, g in sorted(agg.items()):
        print(f"{t:9s} unique {g[0]:2d} | sum2q c2 {g[1]:4d} ai {g[2]:4d} L3 {g[3]:4d} | ai>L3 {g[4]} ai>c2 {g[5]} | F changed {g[6]} | ai median ms {statistics.median(g[7])*1000:.0f}")
