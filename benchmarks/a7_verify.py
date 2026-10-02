"""a7_verify.py -- independent re-computation of the a7 run (Addendum 315), written after the locked score was seen and
before the raw files were read. Reads the raw json only: provenance, counts, P0, H1-H6 with unrounded ratios, how often
A7 chose L3T's output and how those choices turned out, and A5 against the ai6 run's A5 (Addendum 313).
    python a7_verify.py <run dir> [<ai6 outputs dir>]"""
import glob, json, os, sys
import numpy as np
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston"); ARMS = ("A5", "A7", "L3T")
FAM = ("F1", "F2", "F3", "F4", "F5"); SETS = FAM + ("MODEL",)
WANT = dict(F1=216, F2=120, F3=150, F4=135, F5=72)
out = sys.argv[1]
idx = json.load(open(os.path.join(out, "model_index.json")))
D, flags = {}, []
for p in glob.glob(os.path.join(out, "a7_*.json")):
    r = json.load(open(p)); m = r["meta"]
    if (m["git_head"] != "3130e3e" or not m["sha"]["script"].startswith("4d5359fe") or
            not m["sha"]["a7"].startswith("e2132e3c") or m["smoke"] or m["a7"] != "2026-10-02.a7" or
            m["a5"] != "2026-10-01.a5" or m["release"] != "2026-10-02.2"):
        flags.append(p)
    if len(r["rows"]) != (idx["converted"] if m["set"] == "MODEL" else WANT[m["set"]]):
        flags.append(p + " count")
    D[(m["device"], m["arm"], m["set"])] = r["rows"]
rows = [x for v in D.values() for x in v]
p0i = max(x.get("infid_ideal", 0) for x in rows)
wide = sum(1 for x in rows if x.get("too_wide"))
print("files %d, flags %d; model circuits %d" % (len(D), len(flags), idx["converted"]))
print("P0", "PASS" if len(D) == 54 and not flags and p0i <= 1e-6 and wide <= 0.05 * len(rows) and idx["converted"] >= 150
      else "FAIL", "noiseless max %.2e, too wide %d of %d" % (p0i, wide, len(rows)))
sel = lambda d, a, s, bc=None: [x for x in D[(d, a, s)] if bc is None or x["params"].get("bc") == bc]


def R(d, sets, a, b, bc=None):
    xs = [x["infid"] for s in sets for x in sel(d, a, s, bc)]
    ys = [y["infid"] for s in sets for y in sel(d, b, s, bc)]
    return np.mean(xs) / np.mean(ys)


V = lambda ok, bad: "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
G = {"GAP": FAM, "MODEL": ("MODEL",)}
f3 = {bc: R("FakeAuckland", ["F3"], "A7", "L3T", bc) for bc in "op"}
print("H1", V(all(v <= 1.02 for v in f3.values()), any(v >= 1.10 for v in f3.values())), {k: round(v, 5) for k, v in f3.items()})
r75 = {(g, d): R(d, s, "A7", "A5") for g, s in G.items() for d in DEV}
print("H2", V(all(v <= 1.0 for v in r75.values()), any(v > 1.02 for v in r75.values())),
      {g + "/" + d: round(v, 5) for (g, d), v in r75.items()})
ra = R("FakeAuckland", FAM, "A7", "L3T")
print("H3", V(ra <= 1.0, ra > 1.02), round(ra, 5))
frac = {}
for d in DEV:
    pr = [(x, y) for s in SETS for x, y in zip(D[(d, "A7", s)], D[(d, "A5", s)])]
    assert all(x["params"] == y["params"] for x, y in pr)
    frac[d] = np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in pr])
print("H4", V(all(v >= 0.95 for v in frac.values()), any(v < 0.90 for v in frac.values())), {k: round(v, 4) for k, v in frac.items()})
med = {a: float(np.median([x["compile_s"] for (d, aa, s), v in D.items() if aa == a for x in v])) for a in ARMS}
print("H5", V(med["A7"] <= 1.1 * med["A5"], med["A7"] > 1.5 * med["A5"]), {k: round(v, 4) for k, v in med.items()})
nf = sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, s), v in D.items() if aa == "A7" for x in v)
print("H6", V(nf == 0, nf > 0), nf)
print()
print("A7 chose L3T's output, and how those choices compare with A5 and with L3T (per device):")
for d in DEV:
    ch = [(x, y, z) for s in SETS for x, y, z in zip(D[(d, "A7", s)], D[(d, "A5", s)], D[(d, "L3T", s)]) if x["chosen"] == "L3T"]
    eq = sum(abs(x["infid"] - z["infid"]) <= 1e-12 for x, _, z in ch)
    better = sum(x["infid"] < y["infid"] - 1e-12 for x, y, _ in ch)
    worse = sum(x["infid"] > y["infid"] + 1e-12 for x, y, _ in ch)
    gain = np.mean([y["infid"] - x["infid"] for x, y, _ in ch]) if ch else 0.0
    print("  %s: %d chosen (output identical to L3T arm in %d); vs A5 better %d, worse %d; mean infidelity gain %.4f"
          % (d, len(ch), eq, better, worse, gain))
    wr = [x["infid"] / y["infid"] for s in SETS for x, y in zip(D[(d, "A7", s)], D[(d, "A5", s)]) if x["infid"] > y["infid"] + 1e-12]
    print("    circuits where A7 is worse than A5: %d, worst ratio %s, median ratio %s" % (
        len(wr), ("%.3f" % max(wr)) if wr else "-", ("%.3f" % np.median(wr)) if wr else "-"))
if len(sys.argv) > 2:
    s = n = 0
    for d in DEV:
        for st in SETS:
            g = json.load(open(os.path.join(sys.argv[2], "ai6_%s_A5_%s.json" % (d, st))))["rows"]
            for x, y in zip(D[(d, "A5", st)], g):
                n += 1
                s += x["params"] == y["params"] and x["two_q"] == y["two_q"] and abs(x["infid"] - y["infid"]) <= 1e-12
    print("\nA5 against the ai6 run's A5 (Addendum 313): identical %d of %d" % (s, n))
print("off-target:", {a: sum(x["off_target"] > 0 for (d, aa, ss), v in D.items() if aa == a for x in v) for a in ARMS})
print("flags:", flags or "none")
