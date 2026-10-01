"""Exploratory: release stack vs candidate stack (compile c1 + layout c2) on model circuits of 2026-09-30."""
import json, os, time, warnings, sys, io, contextlib, importlib, glob
sys.path.insert(0, os.environ.get('V10_DIR', '.'))  # folder holding e2e_vllm_psf_v10.py
import e2e_vllm_psf_v10 as E
from psf_pennylane_gpu_prototype import tape_to_qiskit
from qiskit import transpile
from qiskit_ibm_runtime import fake_provider
with warnings.catch_warnings():
    warnings.simplefilter("ignore"); tgt = fake_provider.FakeAuckland().target
cm = tgt.build_coupling_map(); nat = E.native_of(tgt)
tq = lambda c: sum(1 for i in c.data if len(i.qubits) == 2)
def run(pc, qc):
    with contextlib.redirect_stdout(io.StringIO()):
        pc._CX_CORE_CACHE.clear(); t = time.perf_counter()
        out = pc.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    return out, time.perf_counter() - t
mode = sys.argv[1]
import psf_compile as pc
print("psf_compile", pc.VERSION, "| layout", importlib.import_module("psf_smart_layout").LAYOUT_VERSION)
files = sorted(glob.glob(sys.argv[2] + "/**/best_circuit.json", recursive=True))
rows = []
for f in files:
    task = f.split("/")[-2]
    if task not in E.TASKS: continue
    n, _, groups = E.TASKS[task]
    spec = json.load(open(f)); tape, l2q, _, _ = E.to_tape(spec, n); qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
    out, dt = run(pc, qc); fc, _, _ = E.compiled_check("lightning.qubit", out, n, groups)
    fl, _ = E.logical_check("lightning.qubit", tape, n, groups)
    rows.append(dict(file=f.split("pod_outputs/")[-1], task=task, psf_2q=tq(out), psf_ms=round(dt*1000,1), F_log=fl, F_cmp=fc))
json.dump(rows, open(f"cmp_{mode}.json", "w"), indent=0)
print(len(rows), "circuits")
