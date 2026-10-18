"""Exploratory check: two psfz binaries give the same output (report ops) on every PSFZ-VAL input of the devices."""
import json, os, random, runpy, subprocess, sys, tempfile, io, contextlib
b1, b2, devs = sys.argv[1], sys.argv[2], sys.argv[3:]
tmp = tempfile.mkdtemp(); sys.argv = ["x", os.path.join(tmp, "n.jsonl")]
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path("data/2026-10-17/psfz_val/psfz_val.py", run_name="x")
from qiskit import qasm3, transpile
from qiskit_ibm_runtime import fake_provider
n = same = 0
for d in devs:
    t = getattr(fake_provider, d)().target
    dj = os.path.join(tmp, "d.json"); json.dump(G["device_json"](t, d), open(dj, "w"))
    for name in G["CHAINS"] + G["RINGS"] + G["NEW"]:
        qc0 = G["build"](name); rng = random.Random(f"PSFZ-VAL|{d}|{name}")
        for k in range(-1, G["SAME"] + G["FREE"]):
            qc = qc0 if k < 0 else G["randomize"](qc0, rng, same=k < G["SAME"])
            src = os.path.join(tmp, "in.qasm"); open(src, "w").write(qasm3.dumps(transpile(qc, basis_gates=G["BASIS"], optimization_level=0)))
            outs = []
            for b in (b1, b2):
                r = os.path.join(tmp, "r.json")
                if os.path.exists(r): os.remove(r)
                subprocess.run([b, "compile", dj, src, os.path.join(tmp, "o.qasm"), r], capture_output=True)
                rep = json.load(open(r)); rep.pop("compile_s", None); rep.pop("device_s", None); outs.append(json.dumps(rep, sort_keys=True))
            n += 1; same += outs[0] == outs[1]
print("inputs", n, "same output", same)
