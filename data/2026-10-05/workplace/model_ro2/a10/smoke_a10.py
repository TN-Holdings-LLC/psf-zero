import sys, os, warnings, contextlib, io, time
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
import psf_compile as pc
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
A9 = load("psf_ai_compile_a9.py", "a9"); A10 = load("psf_ai_compile.py", "a10")
from qiskit import qpy
from qiskit_ibm_runtime import fake_provider as fp
with open("<repo>/data/2026-10-02/ai6/outputs/model_circuits.qpy", "rb") as f: cs = qpy.load(f)
for dev in ("FakeTorino", "FakeKingston"):
    be = getattr(fp, dev)(); t = be.target; cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    for k in (0, 40, 120):
        qc = cs[k].copy(); qc.measure_all()
        res = []
        for A in (A9, A10):
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                out = A.compile_for_model_circuit(qc, cm, basis, target=t)
            mq = [out.find_bit(i.qubits[0]).index for i in out.data if i.operation.name == "measure"]
            res.append((round(sum(t["measure"][(q,)].error for q in mq), 4), sum(len(i.qubits) == 2 for i in out.data), round(time.perf_counter() - t0, 2)))
        print(dev, k, qc.num_qubits, "a9", res[0], "a10", res[1])
