"""psfz compile / template / bind: the binary at r/old vs the variants named in VARS (comma list, builds under r/), bit for bit, on many inputs and devices (incl. damaged)."""
import json, os, subprocess, sys, tempfile, random, copy
import numpy as np
P = sys.argv[1]; R = f"{P}/r"; REPO = "/tmp/claude-0/work/psf-zero"
VARS = os.environ["VARS"].split(",")
BIN = {v: f"{R}/{v}/target/release/psfz" for v in ["old"] + VARS}
sys.path.insert(0, f"{REPO}/rust/psfz/harness"); os.chdir(REPO)
import twin_check as tc
from qiskit import QuantumCircuit, qasm3
from qiskit_ibm_runtime import fake_provider as fp
pc = tc.load_pc()
tmp = tempfile.mkdtemp(dir=P)
devs = {}
for name in ("FakeFez", "FakeMarrakesh", "FakeTorino", "FakeSherbrooke", "FakeBrisbane"):
    try:
        txt = pc._psfz105_device(getattr(fp, name)().target)
    except Exception as e:
        print("skip", name, e); continue
    devs[name] = txt
rr = random.Random(5)
for base in ("FakeFez", "FakeSherbrooke"):
    for k in range(3):
        d = json.loads(devs[base])
        for q in rr.sample(range(d["num_qubits"]), 6 + 4 * k): d["qubits"][q]["meas_error"] = 0.9
        for c in rr.sample(range(len(d["couplers"])), 8 + 6 * k): d["couplers"][c][2] = 0.9
        devs[f"{base}_dmg{k}"] = json.dumps(d)
dfile = {}
for n, t in devs.items():
    dfile[n] = f"{tmp}/{n}.json"; open(dfile[n], "w").write(t)
rng = np.random.default_rng(1113)
def circ(kind, n):
    qc = QuantumCircuit(n, n)
    ring = kind.startswith("ring")
    edges = [(q, q + 1) for q in range(n - 1)] + ([(n - 1, 0)] if ring else [])
    for q in range(n): qc.h(q)
    for s in range(3):
        for (a, b) in edges[0::2] + edges[1::2]:
            g = rng.integers(0, 6)
            if g == 0: qc.rzz(float(rng.uniform(-3, 3)), a, b)
            elif g == 1: qc.rzz(float(rng.choice([0, np.pi / 2, -np.pi / 2, np.pi])), a, b)
            elif g == 2: qc.cx(a, b)
            elif g == 3: qc.rxx(float(rng.uniform(-3, 3)), a, b); qc.ryy(float(rng.uniform(-3, 3)), a, b)
            elif g == 4: qc.cz(a, b)
            else: qc.cp(float(rng.uniform(-3, 3)), a, b)
        for q in range(n):
            qc.u(*[float(x) for x in rng.uniform(-np.pi, np.pi, 3)], q)
            if rng.random() < 0.3: qc.rz(float(rng.uniform(-3, 3)), q)
    if kind.endswith("full"): qc.measure(range(n), range(n))
    elif kind.endswith("part"):
        m = sorted(rng.choice(n, size=max(1, n // 4), replace=False)); qc.measure(m, m)
    elif kind.endswith("one"): qc.measure(0, 0)
    return qc
inputs = []
for kind in ("chain_full", "chain_part", "chain_one", "chain_none", "ring_full", "ring_part", "ring_one"):
    for n in ((4, 8, 13, 20, 33, 50) if kind.startswith("chain") else (12, 16, 20, 24, 30, 40)):
        inputs.append((f"{kind}_{n}", circ(kind, n)))
inputs.append(("empty0", QuantumCircuit(0))); inputs.append(("one1", QuantumCircuit(1, 1)))
inputs[-1][1].h(0); inputs[-1][1].measure(0, 0)
def run(b, args, outs):
    for o in outs:
        if os.path.exists(o): os.remove(o)
    r = subprocess.run([b] + args, capture_output=True, text=True)
    return r.returncode, [open(o, "rb").read() if os.path.exists(o) else None for o in outs], ("panicked" in r.stderr)
def norm(b, key):
    if b is None: return None
    try:
        j = json.loads(b)
    except Exception:
        return b
    for k in list(j) if isinstance(j, dict) else []:
        if k.endswith("_s"): j.pop(k)
    return json.dumps(j, sort_keys=True)
diffs = {v: [] for v in VARS}; panics = {v: 0 for v in ["old"] + VARS}; total = 0
for name, qc in inputs:
    fin = f"{tmp}/in.qasm"; open(fin, "w").write(qasm3.dumps(qc))
    for dn, df in dfile.items():
        res = {}
        for v, b in BIN.items():
            oq, cr, tj, bq = f"{tmp}/o.qasm", f"{tmp}/c.json", f"{tmp}/t.json", f"{tmp}/b.qasm"
            c = run(b, ["compile", df, fin, oq, cr], [oq, cr])
            t = run(b, ["template", df, fin, tj], [tj])
            bnd = (None, [None], False)
            if t[1][0]:
                tjj = json.loads(t[1][0]); vals = [[float(x) for x in np.random.default_rng(len(name)).uniform(-3, 3, s["nparams"])] for s in tjj["slots"]]
                vf = f"{tmp}/v.json"; json.dump(vals, open(vf, "w"))
                bnd = run(b, ["bind", tj, vf, bq], [bq])
            panics[v] += c[2] + t[2] + bnd[2]
            res[v] = (c[0], c[1][0], norm(c[1][1], 0), t[0], norm(t[1][0], 0), bnd[0], bnd[1][0])
        total += 1
        for v in VARS:
            if res[v] != res["old"]:
                which = [k for k, (a, b) in enumerate(zip(res["old"], res[v])) if a != b]
                diffs[v].append((name, dn, which, res["old"][0], res[v][0]))
print("cases", total, "panics", panics)
FIELDS = ["compile_rc", "qasm", "report", "template_rc", "template", "bind_rc", "bind_qasm"]
for v in VARS:
    print(v, "differ", len(diffs[v]))
    for d in diffs[v][:6]: print("   ", d[0], d[1], [FIELDS[k] for k in d[2]], "rc", d[3], d[4])
json.dump({v: diffs[v] for v in VARS}, open(f"{P}/ab3_diffs.json", "w"))
