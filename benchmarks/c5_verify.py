"""c5_verify.py -- independent re-computation of the c5 run (Addendum 309), written after the lock and after the
locked score was seen, before the raw files were read. Reads the raw json only: provenance, counts, P0, H1-H6, and
descriptive per-circuit comparisons (C5 against C3 and L3T), including the circuits item 31 recompiled for C3.
    python c5_verify.py <run dir>"""
import glob, json, os, sys
import numpy as np
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston"); ARMS = ("C3", "C5", "L3T"); FAM = ("F1", "F2", "F3", "F4", "F5")
WANT = dict(F1=216, F2=120, F3=150, F4=135, F5=72)
D, flags = {}, []
for p in glob.glob(os.path.join(sys.argv[1], "c5_*.json")):
    r = json.load(open(p)); m = r["meta"]
    if m["git_head"] != "ce574b0" or not m["sha"]["script"].startswith("44fe82de") or \
            not m["sha"]["c5"].startswith("daa3a44d") or m["smoke"] or m["c5"] != "2026-10-02.c5" or \
            m["c3"] != "2026-10-02.1":
        flags.append(p)
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(p + " count")
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
print("files %d, flags %d" % (len(D), len(flags)))
rows = [x for v in D.values() for x in v]
p0i = max(x.get("infid_ideal", 0) for x in rows)
wide = sum(1 for x in rows if x.get("too_wide"))
print("P0", "PASS" if len(D) == 45 and not flags and p0i <= 1e-6 and wide <= 0.05 * len(rows) else "FAIL",
      "noiseless max %.2e, too wide %d of %d" % (p0i, wide, len(rows)))


def pairs(d, fb, a, b):
    out = []
    for f, bc in fb:
        for x, y in zip(D[(d, a, f)], D[(d, b, f)]):
            assert x["params"] == y["params"]
            if (bc is None or x["params"]["bc"] == bc) and "infid" in x and "infid" in y:
                out.append((x, y))
    return out


R = lambda pr: np.mean([x["infid"] for x, _ in pr]) / np.mean([y["infid"] for _, y in pr])
V = lambda ok, bad: "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
FB = [("F1", None), ("F2", None), ("F3", "o"), ("F3", "p"), ("F4", None), ("F5", None)]
c53 = {(f + (bc or ""), d): R(pairs(d, [(f, bc)], "C5", "C3")) for f, bc in FB for d in DEV}
c5l = {(f + (bc or ""), d): R(pairs(d, [(f, bc)], "C5", "L3T")) for f, bc in FB for d in DEV}
chain = {d: R(pairs(d, [("F3", "o"), ("F5", None)], "C5", "L3T")) for d in DEV}
le = sum(v <= 1.0 for v in c53.values())
frac = {d: np.mean([x["infid"] <= y["infid"] + 1e-12 for f in FAM for x, y in pairs(d, [(f, None)], "C5", "C3")])
        for d in DEV}
same = diff = 0
for d in DEV:
    for f in FAM:
        for x, y in zip(D[(d, "C5", f)], D[(d, "C3", f)]):
            if x["backstop"] or y["backstop"]:
                continue
            same += x["two_q"] == y["two_q"] and x["depth"] == y["depth"]
            diff += not (x["two_q"] == y["two_q"] and x["depth"] == y["depth"])
fail = sum(x["failed_uses"] + x["failed_q_uses"] for (d, a, f), v in D.items() if a == "C5" for x in v)
med = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a for x in v])) for a in ARMS}
r3 = lambda dct: {k: round(float(v), 3) for k, v in dct.items()}
print("H1", V(all(v <= 1.05 for v in chain.values()), any(v >= 1.15 for v in chain.values())), r3(chain))
print("H2", V(le >= 17 and all(v <= 1.02 for v in c53.values()), le < 14 or any(v > 1.10 for v in c53.values())),
      "%d of 18 <= 1.0, range %.3f-%.3f" % (le, min(c53.values()), max(c53.values())))
print("H3", V(all(v >= 0.90 for v in frac.values()), any(v < 0.80 for v in frac.values())), r3(frac))
print("H4", V(diff == 0 and same > 0, diff > 0), same, "same,", diff, "different")
print("H5", V(fail == 0, fail > 0), fail)
print("H6", V(med["C5"] <= 3 * med["C3"], med["C5"] > 10 * med["C3"]), {k: round(v, 4) for k, v in med.items()})
print()
print("| cell | device | C5/C3 | C5/L3T |")
for k in c53:
    print("| %s | %s | %.3f | %.3f |" % (k[0], k[1], c53[k], c5l[k]))
print()
for d in DEV:
    p53 = [p for f in FAM for p in pairs(d, [(f, None)], "C5", "C3")]
    p5l = [p for f in FAM for p in pairs(d, [(f, None)], "C5", "L3T")]
    worse = [(x["infid"] / y["infid"]) for x, y in p53 if x["infid"] > y["infid"] + 1e-12]
    print("%s: pooled C5/C3 %.3f, C5/L3T %.3f; C5 <= L3T in %d of %d; re-placed %d; C5 worse than C3 in %d "
          "(max ratio %s)" % (d, R(p53), R(p5l), sum(x["infid"] <= y["infid"] + 1e-12 for x, y in p5l), len(p5l),
                              sum(x["refined"] > 0 for x, _ in p53), len(worse),
                              ("%.3f" % max(worse)) if worse else "-"))
bs = [(x, y, z) for d in DEV for f in FAM
      for (x, y), (_, z) in zip(pairs(d, [(f, None)], "C5", "C3"), pairs(d, [(f, None)], "C5", "L3T")) if y["backstop"]]
if bs:
    print("circuits item 31 recompiled for C3 (%d): mean infidelity C3 %.4f, C5 %.4f, L3T %.4f; C5 two_q == C3 in %d"
          % (len(bs), np.mean([y["infid"] for _, y, _ in bs]), np.mean([x["infid"] for x, _, _ in bs]),
             np.mean([z["infid"] for _, _, z in bs]), sum(x["two_q"] == y["two_q"] for x, y, _ in bs)))
print("backstop recompiles:", {a: sum(x["backstop"] > 0 for (d, aa, f), v in D.items() if aa == a for x in v)
                               for a in ("C3", "C5")})
print("flags:", flags or "none")
