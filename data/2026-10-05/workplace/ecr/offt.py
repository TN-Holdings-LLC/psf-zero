import sys, warnings, contextlib, io, collections
warnings.simplefilter("ignore"); sys.path.insert(0, ".")
import psf_compile as pc, ai10_eval2 as E2, ecr_explore as X
from qiskit_ibm_runtime import fake_provider as fp
A10 = E2.load("psf_ai_compile.py", "o_a10"); A9 = E2.load("psf_ai_compile_a9.py", "o_a9")
be = fp.FakeBrussels(); t = be.target; cm = t.build_coupling_map()
basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
print("basis", basis)
cnt = collections.Counter(); info = collections.Counter()
for name, qc0 in X.circuits()[:12]:
    qc = qc0.copy(); qc.measure_all()
    for A in (A9, A10):
        with contextlib.redirect_stdout(io.StringIO()):
            out, inf = A.compile_for_model_circuit(qc, cm, basis, target=t, return_info=True)
        for ins in out.data:
            nm = ins.operation.name
            if nm in ("barrier", "measure"): continue
            q = tuple(out.find_bit(b).index for b in ins.qubits)
            if nm not in t.operation_names or q not in t[nm]:
                cnt[(A.AI_COMPILE_VERSION, nm, "rev" if (nm in t.operation_names and q[::-1] in t[nm]) else "other")] += 1
        info[(A.AI_COMPILE_VERSION, inf.get("chosen"), inf.get("path"))] += 1
print(cnt); print(info)
