"""Exploratory: candidate stack vs L3 per model circuit (same file, same process)."""
import json, os, sys, warnings, io, contextlib, glob, collections
sys.path.insert(0, os.environ.get('V10_DIR', '.'))  # folder holding e2e_vllm_psf_v10.py
import e2e_vllm_psf_v10 as E
from psf_pennylane_gpu_prototype import tape_to_qiskit
from qiskit import transpile
from qiskit_ibm_runtime import fake_provider
import psf_compile as pc
with warnings.catch_warnings():
    warnings.simplefilter("ignore"); tgt = fake_provider.FakeAuckland().target
cm = tgt.build_coupling_map(); nat = E.native_of(tgt)
tq = lambda c: sum(1 for i in c.data if len(i.qubits) == 2)
seen = {}; agg = collections.defaultdict(lambda: [0, 0, 0, 0])
for f in sorted(glob.glob(sys.argv[1] + "/**/best_circuit.json", recursive=True)):
    task = f.split("/")[-2]
    if task not in E.TASKS: continue
    spec = json.load(open(f)); key = task + json.dumps(spec.get("gates"), sort_keys=True)
    if key in seen: continue
    n, _, groups = E.TASKS[task]
    tape, _, _, _ = E.to_tape(spec, n); qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    l3 = tq(transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0))
    p = tq(out); seen[key] = (p, l3, f)
    g = agg[task]; g[0] += 1; g[1] += p > l3; g[2] += p; g[3] += l3
    if p > l3: print(task, "psf", p, "L3", l3, f.split("2026-09-30/")[-1])
for t, g in sorted(agg.items()): print("SUM", t, "unique", g[0], "psf>L3", g[1], "psf", g[2], "L3", g[3])
