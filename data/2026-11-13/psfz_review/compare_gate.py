"""psfz gate: r/old vs r/new on N2 records and 600 mutated records; usage: python compare_gate.py WORKDIR"""
import json, os, subprocess, sys, random, tempfile
P = sys.argv[1]; OLD = f"{P}/r/old/target/release/psfz"; NEW = f"{P}/r/new/target/release/psfz"
N2 = f"{P}/data/2026-10-20/n2"; j = json.load(open(f"{N2}/n2.json"))
tmp = tempfile.mkdtemp(dir=P); c = dict(j["config_test_only"]); c["device_dir"] = f"{N2}/fixtures"
cfg = f"{tmp}/cfg.json"; json.dump(c, open(cfg, "w"))
texts = []
for sec in ("refusals", "controls", "requests", "fault_injection"):
    for r in j[sec]:
        if isinstance(r, dict) and isinstance(r.get("text"), str): texts.append((f"{sec}:{r.get('id')}", r["text"]))
rnd = random.Random(1113); base = [t for _, t in texts]
for i in range(600):  # mutations: truncation, byte flips, field deletions
    t = rnd.choice(base); k = rnd.random()
    if k < 0.3: t = t[: rnd.randrange(len(t))]
    elif k < 0.6:
        b = list(t)
        for _ in range(rnd.randint(1, 4)): b[rnd.randrange(len(b))] = rnd.choice('{}[]",:0123456789-eE.ax \n')
        t = "".join(b)
    else:
        try:
            d = json.loads(t)
            def walk(x):
                if isinstance(x, dict) and x:
                    kk = rnd.choice(list(x))
                    if rnd.random() < 0.5: del x[kk]
                    else: x[kk] = rnd.choice([None, -1, 1e308, "x", [], {}, 2**63, 0.5])
                    if x and rnd.random() < 0.5: walk(x[rnd.choice(list(x))]) if x else None
                elif isinstance(x, list) and x: walk(rnd.choice(x))
            walk(d); t = json.dumps(d)
        except Exception: t = t[::-1]
    texts.append((f"mut{i}", t))
diff = 0; panics = {"old": 0, "new": 0}; ex = []
for name, t in texts:
    rec = f"{tmp}/rec.json"; open(rec, "w").write(t); res = {}
    for tag, b in (("old", OLD), ("new", NEW)):
        o = f"{tmp}/out_{tag}.json"
        if os.path.exists(o): os.remove(o)
        r = subprocess.run([b, "gate", rec, cfg, o], capture_output=True, text=True)
        if "panicked" in r.stderr or r.returncode not in (0, 1, 2): panics[tag] += 1
        res[tag] = (r.returncode, open(o, "rb").read() if os.path.exists(o) else None)
    if res["old"] != res["new"]:
        diff += 1
        if len(ex) < 8: ex.append((name, res["old"][0], res["new"][0], (res["old"][1] or b"")[:300], (res["new"][1] or b"")[:300]))
print("records", len(texts), "differ", diff, "panics", panics)
for e in ex: print(e)
