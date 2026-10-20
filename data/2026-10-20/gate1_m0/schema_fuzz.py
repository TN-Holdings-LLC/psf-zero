"""Cross-check of gate 1's hand-written schema (psfz gate, checks 0-3) against Python jsonschema on the r2 schema file.

    python data/2026-10-20/gate1_m0/schema_fuzz.py SCRATCH_DIR     (from the repository root; psfz built with cargo)

Mutations of N2's controls (seed 20261020): random field changes, then boundary values that are often valid. Strict
JSON on the Python side too (duplicate keys, NaN and Infinity refused). Prints the counts and every disagreement."""
import json, math, random, subprocess, sys, os, copy
import jsonschema
REPO = os.getcwd()
S = sys.argv[1]
os.makedirs(f"{S}/rec", exist_ok=True)
schema = json.load(open(f"{REPO}/docs/design/psf-intent-0.r2.schema.json"))
V = jsonschema.Draft202012Validator(schema)
n2 = json.load(open(f"{REPO}/data/2026-10-20/n2/n2.json"))
json.dump({"device_dir": f"{REPO}/data/2026-10-20/n2/fixtures", "CZ_max": 500, "SX_max": 2000, "chi_max": 256}, open(f"{S}/cfg.json", "w"))
bases = [json.loads(r["text"]) for r in n2["controls"] if r["id"] != "K6"] + [json.loads(n2["refusals"][i]["text"]) for i in (14, 18)]
rng = random.Random(20261020)
vals = [None, True, False, 0, -1, 1, 1.5, 2, 12, 13, 64, 65, 100, 101, 1e-9, -1e-9, 3.2, 6.29, -6.29, 10, 10.0001, -10.5, "x", "slots", "ry", "rx",
        "z0", "z12", "z123", "zz", [], [0.1], {}, {"requested": True}, {"requested": 1}, "file:F1.json", "file:/abs.json",
        "file:a/../F1.json", "file:F1.json ", "file:", "0"*64, "A"*64, "chain", "ring", "field", "none", 1.0, 2.0, 3]
def mutate(r):
    r = copy.deepcopy(r)
    tgt = r if rng.random() < 0.4 else r.setdefault("params", {})
    if not isinstance(tgt, dict): return r
    op = rng.random()
    keys = list(tgt.keys()) + ["extra", "twin", "device_sha256", "slots", "decline", "angles", "graph"]
    k = rng.choice(keys)
    if op < 0.2 and tgt: tgt.pop(rng.choice(list(tgt.keys())))
    else: tgt[k] = rng.choice(vals)
    return r
def py_verdict(text):
    def nodup(pairs):
        d = {}
        for k, v in pairs:
            if k in d: raise ValueError("dup")
            d[k] = v
        return d
    try: r = json.loads(text, object_pairs_hook=nodup, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except Exception: return "SCHEMA_INVALID"
    if not isinstance(r, dict) or not isinstance(r.get("ir"), str): return "SCHEMA_INVALID"
    if r["ir"] != "psf-intent/0": return "IR_VERSION"
    def fin(x):
        if isinstance(x, float): return math.isfinite(x)
        if isinstance(x, dict): return all(fin(v) for v in x.values())
        if isinstance(x, list): return all(fin(v) for v in x)
        return True
    if not fin(r): return "NUMBER_NOT_FINITE"
    return "past" if V.is_valid(r) else "SCHEMA_INVALID"
texts = []
for b in bases:
    for _ in range(80):
        r = b
        for _ in range(rng.randint(1, 3)): r = mutate(r)
        texts.append(json.dumps(r))
texts += ['{"ir":"psf-intent/0","decline":"out_of_family"}', '{"ir":"psf-intent/0","decline":"x"}', '{"ir":"psf-intent/0","decline":"out_of_family","twin":{"requested":true}}',
          '{"ir": "psf-intent/0", "family": "tfim_chain", "family": "tfim_chain"}', '{"ir":"psf-intent/0","x":NaN}', '{"ir":"psf-intent/0","x":+1}']
bin_ = f"{REPO}/rust/psfz/target/release/psfz"
mism, counts = [], {}
for i, t in enumerate(texts):
    rf, of = f"{S}/rec/r{i}.json", f"{S}/rec/o{i}.json"
    open(rf, "w").write(t)
    subprocess.run([bin_, "gate", rf, f"{S}/cfg.json", of], capture_output=True)
    o = json.load(open(of))
    rust = o.get("code") if o.get("refused") and o.get("check") in ("0", "1", "2", "3") else "past"
    py = py_verdict(t)
    counts[py] = counts.get(py, 0) + 1
    if rust != py: mism.append((t, py, rust, o.get("field"), o.get("reason")))
print(len(texts), "records;", counts, "; mismatches:", len(mism))
for m in mism[:15]: print(m)
import math as _m
valid = {"n": [2, 12, 64, 100, 64.0, 65], "layers": [1, 64, 65, 0], "encoding_scale": [1e-12, _m.pi, 0, 3.1416],
         "readout": ["z0", "z99", "z1"], "J": [-10, 10, 10.000001], "h": [-10, 0, 10], "dt": [1, 1e-300, 0, 1.0000001],
         "steps": [1, 64, 65], "order": [1, 2, 2.0, 1.5], "slots": ["field", "none"], "angles": [[6.283185307179586], [-6.283185307179586], [6.3], "slots"],
         "graph": ["chain", "ring"]}
texts2 = []
for b in bases:
    for _ in range(60):
        r = copy.deepcopy(b)
        for _ in range(rng.randint(1, 2)):
            k = rng.choice([k for k in valid if k in r["params"]] or ["n"])
            r["params"][k] = rng.choice(valid[k])
        if rng.random() < 0.3: r["device_sha256"] = rng.choice(["0"*64, "f"*64, "F"*64])
        if rng.random() < 0.2: r["twin"] = rng.choice([{"requested": False}, {"requested": True}, {"requested": True, "x": 1}])
        if rng.random() < 0.2: r["device"] = rng.choice(["file:F1.json", "file:sub/F1.json", "file:.F1", "file:/F1.json", "file:x..y"])
        texts2.append(json.dumps(r))
mism2, counts2 = [], {}
for i, t in enumerate(texts2):
    rf, of = f"{S}/rec/v{i}.json", f"{S}/rec/w{i}.json"
    open(rf, "w").write(t)
    subprocess.run([bin_, "gate", rf, f"{S}/cfg.json", of], capture_output=True)
    o = json.load(open(of))
    rust = o.get("code") if o.get("refused") and o.get("check") in ("0", "1", "2", "3") else "past"
    py = py_verdict(t)
    counts2[py] = counts2.get(py, 0) + 1
    if rust != py: mism2.append((t, py, rust, o.get("field"), o.get("reason")))
print(len(texts2), "boundary records;", counts2, "; mismatches:", len(mism2))
for m in mism2[:15]: print(m)
