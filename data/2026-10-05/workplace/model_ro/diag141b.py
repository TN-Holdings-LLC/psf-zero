import sys, warnings, contextlib, io
warnings.simplefilter("ignore"); sys.path.insert(0, ".")
import psf_compile as pc, ai10_eval as E
import readout_eval as RE
from qiskit import qpy
from qiskit_ibm_runtime import fake_provider as fp
A10 = E.load("psf_ai_compile.py", "d_a10"); A9 = E.load("psf_ai_compile_a9.py", "d_a9")
cs = list(qpy.load(open("model_circuits.qpy", "rb")))
be = fp.FakeKingston(); t = be.target; cm = t.build_coupling_map()
basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
qc0 = cs[141]; qc = qc0.copy(); qc.measure_all()
for A in (A9, A10):
    with contextlib.redirect_stdout(io.StringIO()):
        out = A.compile_for_model_circuit(qc, cm, basis, target=t)
    print(A.AI_COMPILE_VERSION, "state infidelity %.2e" % RE.state_infid(qc0, RE.strip_measure(out)))
