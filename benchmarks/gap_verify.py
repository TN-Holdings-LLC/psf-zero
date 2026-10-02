"""gap_verify.py -- independent re-computation of the GAP run (Addendum 300), written after the run. Reads the raw
json files only. Checks provenance and counts, lists the circuits that failed P0's noiseless check, recomputes the
table and the H1-H6 quantities. The H1-H6 lines are exploratory: the locked P0 failed, so nothing is scored.
    python gap_verify.py <run dir>"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston"); ARMS = ("C2", "A5", "L3T")
WANT = dict(F1=216, F2=120, F3=150, F4=135, F5=72)
out = sys.argv[1]; D = {}; flags = []
for p in glob.glob(os.path.join(out, "gap_*.json")):
    r = json.load(open(p)); m = r["meta"]
    if m["git_head"] != "4acca8c" or not m["sha"]["script"].startswith("6659bd0b") or m["smoke"]:
        flags.append(p)
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(f"{p}: {len(r['rows'])}")
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
print(f"files {len(D)}, provenance/count flags {len(flags)}")
p0 = [(k, x["params"], x["infid_ideal"]) for k, v in D.items() for x in v if x.get("infid_ideal", 1) > 1e-9]
print(f"P0 noiseless > 1e-9: {len(p0)} rows; arms/families {sorted({(k[1], k[2]) for k, _, _ in p0})}; "
      f"max {max(v for *_, v in p0):.2e}; all <= 1e-6: {all(v <= 1e-6 for *_, v in p0)}")
print("  circuits:", sorted({(k[0], json.dumps(pp)) for k, pp, _ in p0}))
def sub(name):
    return {"cyc": [("F1", None), ("F3", "p")], "chain": [("F3", "o"), ("F5", None)], "F4": [("F4", None)],
            "all": [(f, None) for f in ("F1", "F2", "F3", "F4", "F5")]}[name]
def pairs(d, name, a="C2", b="L3T"):
    out = []
    for f, bc in sub(name):
        for x, y in zip(D[(d, a, f)], D[(d, b, f)]):
            if bc is None or x["params"]["bc"] == bc:
                assert x["params"] == y["params"]
                out.append((x, y))
    return out
ratio = lambda pr: np.mean([x["infid"] for x, _ in pr]) / np.mean([y["infid"] for _, y in pr])
heron = DEV[1:]
q = dict(H1={d: np.mean([x["two_q"] > y["two_q"] for x, y in pairs(d, "cyc")]) for d in heron},
         H2={d: np.mean([x["two_q"] <= y["two_q"] for x, y in pairs(d, "chain")]) for d in DEV},
         H3={d: ratio(pairs(d, "cyc")) for d in heron}, H4={d: ratio(pairs(d, "F4")) for d in DEV},
         H5={d: ratio(pairs(d, "all", "A5", "L3T")) for d in DEV}, H6={d: ratio(pairs(d, "chain")) for d in DEV})
for k, v in q.items():
    print(k, {d: round(float(x), 3) for d, x in v.items()})
win = {d: np.mean([x["infid"] < y["infid"] for x, y in pairs(d, "all")]) for d in DEV}
print("share of all circuits where C2 beats L3T:", {d: round(float(v), 3) for d, v in win.items()})
print("flags:", flags or "none")
