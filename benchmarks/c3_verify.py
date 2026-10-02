"""c3_verify.py -- independent re-computation of the c3 run (Addendum 303), written after the run. Reads the raw
json only: provenance, counts, P0, H1-H5, plus the rows behind P0's noiseless maximum and the per-device recompile
counts.   python c3_verify.py <run dir>"""
import glob, json, os, sys
import numpy as np
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston"); ARMS = ("C2", "C3", "L3T"); FAM = ("F1", "F2", "F3", "F4", "F5")
WANT = dict(F1=216, F2=120, F3=150, F4=135, F5=72)
D, flags = {}, []
for p in glob.glob(os.path.join(sys.argv[1], "c3_*.json")):
    r = json.load(open(p)); m = r["meta"]
    if m["git_head"] != "158967c" or not m["sha"]["script"].startswith("a56aa676") or \
            not m["sha"]["c3"].startswith("8fd5e500") or m["smoke"] or m["c3"] != "2026-10-02.c3":
        flags.append(p)
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(f"{p} count")
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
print(f"files {len(D)}, flags {len(flags)}")
rows = [(k, x) for k, v in D.items() for x in v]
hi = sorted(((x["infid_ideal"], k, x["params"]) for k, x in rows if x.get("infid_ideal", 0) > 1e-9), key=lambda t: -t[0])
print(f"noiseless > 1e-9: {len(hi)} rows, arms/families {sorted({(k[1], k[2]) for _, k, _ in hi})}, max {hi[0][0]:.2e}" if hi else "none > 1e-9")
p0 = len(D) == 45 and not flags and max(x.get("infid_ideal", 0) for _, x in rows) <= 1e-6
def pairs(d, fb, a, b):
    out = []
    for f, bc in fb:
        for x, y in zip(D[(d, a, f)], D[(d, b, f)]):
            assert x["params"] == y["params"]
            if bc is None or x["params"]["bc"] == bc:
                out.append((x, y))
    return out
R = lambda pr: np.mean([x["infid"] for x, _ in pr]) / np.mean([y["infid"] for _, y in pr])
V = lambda ok, bad: "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
fail = sum(x["failed_uses"] + x["failed_q_uses"] for (d, a, f), v in D.items() if a == "C3" for x in v)
tor = {f: R(pairs("FakeTorino", [(f, None)], "C3", "L3T")) for f in ("F1", "F2", "F4")}
allp = [p for d in DEV for f in FAM for p in pairs(d, [(f, None)], "C3", "C2")]
clean = [(x, y) for x, y in allp if y["failed_uses"] + y["failed_q_uses"] == 0]
same = sum(x["two_q"] == y["two_q"] and abs(x["infid"] - y["infid"]) <= 1e-12 for x, y in clean)
hit = [(x, y) for x, y in allp if y["failed_uses"] + y["failed_q_uses"] > 0]
better = sum(x["infid"] < y["infid"] for x, y in hit)
chain = {d: R(pairs(d, [("F3", "o"), ("F5", None)], "C3", "L3T")) for d in DEV}
print("P0", "PASS" if p0 else "FAIL")
print("H1", V(fail == 0, fail > 0), fail)
print("H2", V(all(v <= 1.3 for v in tor.values()), any(v >= 1.8 for v in tor.values())), {k: round(v, 3) for k, v in tor.items()})
print("H3", V(same == len(clean), same < len(clean)), same, "of", len(clean))
print("H4", V(better / len(hit) >= 0.9, better / len(hit) < 0.7), better, "of", len(hit))
print("H5", V(all(v >= 1.1 for v in chain.values()), any(v <= 1.0 for v in chain.values())), {k: round(v, 3) for k, v in chain.items()})
print("recompiles by device:", {d: sum(x["recompiled"] > 0 for f in FAM for x in D[(d, "C3", f)]) for d in DEV})
print("C2 failed-qubit uses by device:", {d: sum(x["failed_q_uses"] for f in FAM for x in D[(d, "C2", f)]) for d in DEV})
l3 = [z for d in DEV for f in FAM for (x, y), (_, z) in zip(pairs(d, [(f, None)], "C3", "C2"), pairs(d, [(f, None)], "C3", "L3T"))
      if y["failed_uses"] + y["failed_q_uses"] > 0]
print("recompiled circuits: mean infidelity C2 %.4f, C3 %.4f, L3T %.4f" % (np.mean([y["infid"] for _, y in hit]),
      np.mean([x["infid"] for x, _ in hit]), np.mean([z["infid"] for z in l3])))
print("flags:", flags or "none")
