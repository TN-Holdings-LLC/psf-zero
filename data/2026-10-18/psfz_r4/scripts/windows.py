"""Exploratory: psfz's duration for every window (both orientations) of the longest modal snake piece."""
import json, os, runpy, subprocess, sys, tempfile, io, contextlib, collections
d, name = sys.argv[1], sys.argv[2]
tmp = tempfile.mkdtemp()
sys.argv = ["psfz_val", os.path.join(tmp, "none.jsonl")]
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path("data/2026-10-17/psfz_val/psfz_val.py", run_name="x")
from qiskit import qasm3, transpile
from qiskit_ibm_runtime import fake_provider
BIN = os.environ["PSFZ_BIN"]  # a psfz build that reads PSFZ_WALK (an exploratory patch, not committed)
t = getattr(fake_provider, d)().target
op = "cz" if "cz" in t.operation_names else "ecr"
dirs = {}; dur = {}
for k, p in t[op].items():
    if (p.error or 0) < 0.5: dirs[k] = 1; dur[tuple(sorted(k))] = round(p.duration * 1e12)
modal = collections.Counter(dur.values()).most_common(1)[0][0]
full = {q: set() for q in range(t.num_qubits)}
for a, b in t[op].keys(): full[a].add(b); full[b].add(a)
fe, fq = G["REL"]._failed_elements(t, 0.5)
K = set(range(t.num_qubits)) - set(fq); kadj = {q: {x for x in full[q] if x in K and (q, x) not in fe and (x, q) not in fe} for q in K}
snake = G["REL"]._chain_snake(t.num_qubits, full)
pieces, cur = [], []
for q in snake:
    if q in K and (not cur or (q in kadj[cur[-1]] and dur.get(tuple(sorted((cur[-1], q)))) == modal)): cur.append(q)
    else:
        if cur: pieces.append(cur)
        cur = [q] if q in K else []
pieces.append(cur)
best = max(pieces, key=len)
qc = G["build"](name); n = qc.num_qubits
dj = os.path.join(tmp, "d.json"); json.dump(G["device_json"](t, d), open(dj, "w"))
src = os.path.join(tmp, "in.qasm"); open(src, "w").write(qasm3.dumps(transpile(qc, basis_gates=G["BASIS"], optimization_level=0)))
with contextlib.redirect_stdout(io.StringIO()):
    rel = G["numbers"](G["REL"].compile_for_hardware(qc, target=t), t)["dur_us"]
res = []
for s in range(len(best) - n + 1):
    for rev in (False, True):
        w = best[s:s + n][::-1] if rev else best[s:s + n]
        env = dict(os.environ, PSFZ_WALK=",".join(map(str, w)))
        subprocess.run([BIN, "compile", dj, src, os.path.join(tmp, "o.qasm"), os.path.join(tmp, "r.json")], env=env, capture_output=True)
        rep = json.load(open(os.path.join(tmp, "r.json")))
        du = G["numbers"](G["from_report"](rep, t.num_qubits), t)["dur_us"] if rep["ok"] else None
        fwd = "".join("F" if (a, b) in dirs else "R" for a, b in zip(w, w[1:]))
        res.append((s, rev, du, fwd))
cen = (len(best) - n) // 2
print(d, name, "piece", len(best), "release", round(rel, 4), "centred", [r for r in res if r[0] == cen])
print(" min", min(r[2] for r in res), "max", max(r[2] for r in res), "windows <= release:", sum(r[2] <= rel + 1e-9 for r in res), "of", len(res))
for r in sorted(res, key=lambda r: r[2]): print("  ", r[0], r[1], round(r[2], 4), r[3])
