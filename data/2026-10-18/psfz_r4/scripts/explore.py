"""Exploratory: psfz against the release on chosen devices/tests; prints both placements (the release's from its
two-qubit gates) and durations."""
import json, os, runpy, subprocess, sys, tempfile, io, contextlib
dev_tests = [a.split(":") for a in sys.argv[1:]]
tmp = tempfile.mkdtemp()
sys.argv = ["psfz_val", os.path.join(tmp, "none.jsonl")]
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path("data/2026-10-17/psfz_val/psfz_val.py", run_name="x")
from qiskit import qasm3, transpile
from qiskit_ibm_runtime import fake_provider
BIN = os.environ.get("PSFZ_BIN", G["BIN"])
for d, name in dev_tests:
    t = getattr(fake_provider, d)().target
    dj = os.path.join(tmp, d + ".json"); json.dump(G["device_json"](t, d), open(dj, "w"))
    qc = G["build"](name)
    src = os.path.join(tmp, "in.qasm"); open(src, "w").write(qasm3.dumps(transpile(qc, basis_gates=G["BASIS"], optimization_level=0)))
    subprocess.run([BIN, "compile", dj, src, os.path.join(tmp, "o.qasm"), os.path.join(tmp, "r.json")], capture_output=True)
    rep = json.load(open(os.path.join(tmp, "r.json")))
    out = G["from_report"](rep, t.num_qubits) if rep["ok"] else None
    with contextlib.redirect_stdout(io.StringIO()):
        o2 = G["REL"].compile_for_hardware(qc, target=t)
    edges = sorted({tuple(sorted(o2.find_bit(q).index for q in i.qubits)) for i in o2.data if len(i.qubits) == 2})
    print(d, name, "psfz", G["numbers"](out, t) if out else rep, rep.get("layout"))
    print("   rel", G["numbers"](o2, t), "edges", edges)
