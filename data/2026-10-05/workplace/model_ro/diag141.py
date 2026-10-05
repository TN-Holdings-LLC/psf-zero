import sys, os, warnings, contextlib, io
warnings.simplefilter("ignore"); sys.path.insert(0, ".")
import psf_compile as pc, ai10_eval as E, depth_eval as DE
import numpy as np
from qiskit import qpy, QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, process_fidelity
from qiskit_ibm_runtime import fake_provider as fp
A10 = E.load("psf_ai_compile.py", "d_a10"); A9 = E.load("psf_ai_compile_a9.py", "d_a9")
cs = list(qpy.load(open("model_circuits.qpy", "rb")))
qc0 = cs[141]; qc = qc0.copy(); qc.measure_all()
be = fp.FakeKingston(); t = be.target; cm = t.build_coupling_map()
basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
print("circuit 141:", qc0.num_qubits, dict(qc0.count_ops()))
for A in (A9, A10):
    with contextlib.redirect_stdout(io.StringIO()):
        out, info = A.compile_for_model_circuit(qc, cm, basis, target=t, return_info=True)
    sim = DE.Noisy(be)
    p0, mq, cl = E.reduced_probs(out, sim, False)
    ideal = Statevector(qc0).probabilities()
    tv = 0.5 * np.abs(p0 - ideal).sum()
    print(A.AI_COMPILE_VERSION, "TV %.2e" % tv, {k: info[k] for k in info if k in ("chosen", "start", "seed", "polished", "candidate", "name")} if isinstance(info, dict) else info)
    print("  keys", list(info.keys())[:15] if isinstance(info, dict) else None)
for A in (A9, A10):
    with contextlib.redirect_stdout(io.StringIO()):
        out, info = A.compile_for_model_circuit(qc, cm, basis, target=t, return_info=True)
    print(A.AI_COMPILE_VERSION, "best", info["best"], "path", info["path"])
    # state infidelity of the unmeasured circuit implied

