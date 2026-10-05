import sys, warnings, contextlib, io, collections
warnings.simplefilter("ignore"); sys.path.insert(0, ".")
import psf_compile as pc, ai10_eval2 as E2, ecr_explore as X, readout_eval as RE
from qiskit_ibm_runtime import fake_provider as fp
A10 = E2.load("psf_ai_compile_a10.py", "q_a10"); A11 = E2.load("psf_ai_compile.py", "q_a11")
import psf_ai_compile  # noqa
for dev in ("FakeBrussels", "FakeTorino", "FakeAuckland"):
    be = getattr(fp, dev)(); t = be.target; cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    off10 = off11 = same = 0; maxinf = 0; n = 0
    for name, qc0 in X.circuits()[::4]:
        qc = qc0.copy(); qc.measure_all()
        with contextlib.redirect_stdout(io.StringIO()):
            o10 = A10.compile_for_model_circuit(qc, cm, basis, target=t)
            o11 = A11.compile_for_model_circuit(qc, cm, basis, target=t)
        off10 += A11._off_target_2q(o10, t); off11 += A11._off_target_2q(o11, t)
        same += RE.sig(o10) == RE.sig(o11); n += 1
        maxinf = max(maxinf, RE.state_infid(qc0, RE.strip_measure(o11)))
    print(dev, "circuits", n, "off-target 2q a10", off10, "a11", off11, "identical", same, "max state infid a11 %.1e" % maxinf, A11.DIRECTION_STATS)
