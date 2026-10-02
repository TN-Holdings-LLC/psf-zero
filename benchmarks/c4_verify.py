"""c4_verify.py -- independent re-computation of the c4 run (Addendum 306), written after the lock and before the
results were seen. Reads the raw json only: provenance, counts, P0, H1-H5, plus descriptive per-circuit comparisons
of C4 against C3.   python c4_verify.py <run dir>"""
import glob, json, os, sys
import numpy as np
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston"); ARMS = ("C3", "C4", "L3T"); FAM = ("F1", "F2", "F3", "F4", "F5")
WANT = dict(F1=216, F2=120, F3=150, F4=135, F5=72)
D, flags = {}, []
for p in glob.glob(os.path.join(sys.argv[1], "c4_*.json")):
    r = json.load(open(p)); m = r["meta"]
    if m["git_head"] != "29dd762" or not m["sha"]["script"].startswith("44978911") or \
            not m["sha"]["c4"].startswith("0a1502f9") or m["smoke"] or m["c4"] != "2026-10-02.c4" or \
            m["c3"] != "2026-10-02.1":
        flags.append(p)
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(p + " count")
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
print("files %d, flags %d" % (len(D), len(flags)))
rows = [(k, x) for k, v in D.items() for x in v]
wide = sum(1 for _, x in rows if x.get("too_wide"))
p0i = max(x.get("infid_ideal", 0) for _, x in rows)
p0 = len(D) == 45 and not flags and p0i <= 1e-6 and wide <= 0.05 * len(rows)
print("P0", "PASS" if p0 else "FAIL", "noiseless max %.2e, too wide %d of %d" % (p0i, wide, len(rows)))


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
cells43 = {(f + (bc or ""), d): R(pairs(d, [(f, bc)], "C4", "C3")) for f, bc in FB for d in DEV}
cells4l = {(f + (bc or ""), d): R(pairs(d, [(f, bc)], "C4", "L3T")) for f, bc in FB for d in DEV}
cells3l = {(f + (bc or ""), d): R(pairs(d, [(f, bc)], "C3", "L3T")) for f, bc in FB for d in DEV}
chain = {d: R(pairs(d, [("F3", "o"), ("F5", None)], "C4", "L3T")) for d in DEV}
le = sum(v <= 1.0 for v in cells43.values())
fail = sum(x["failed_uses"] + x["failed_q_uses"] for (d, a, f), v in D.items() if a == "C4" for x in v)
med = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a for x in v])) for a in ARMS}
print("H1", V(all(v <= 1.05 for v in chain.values()), any(v >= 1.20 for v in chain.values())),
      {k: round(v, 3) for k, v in chain.items()})
print("H2", V(le >= 16 and all(v <= 1.10 for v in cells43.values()), le < 12 or any(v > 1.20 for v in cells43.values())),
      "%d of 18 <= 1.0, range %.3f-%.3f" % (le, min(cells43.values()), max(cells43.values())))
print("H3", V(all(v <= 1.10 for v in cells4l.values()), any(v > 1.30 for v in cells4l.values())),
      "C4/L3T range %.3f-%.3f" % (min(cells4l.values()), max(cells4l.values())))
print("H4", V(fail == 0, fail > 0), fail)
print("H5", V(med["C4"] <= 2 * med["C3"], med["C4"] > 3 * med["C3"]), {k: round(v, 4) for k, v in med.items()})
print()
print("| cell | device | C3/L3T | C4/L3T | C4/C3 |")
for k in cells43:
    print("| %s | %s | %.3f | %.3f | %.3f |" % (k[0], k[1], cells3l[k], cells4l[k], cells43[k]))
print()
# Descriptive, per circuit: C4 against C3.
allp = [p for d in DEV for f in FAM for p in pairs(d, [(f, None)], "C4", "C3")]
same_2q = sum(x["two_q"] == y["two_q"] for x, y in allp)
same_inf = sum(abs(x["infid"] - y["infid"]) <= 1e-12 for x, y in allp)
win = sum(x["infid"] < y["infid"] - 1e-12 for x, y in allp)
lose = sum(x["infid"] > y["infid"] + 1e-12 for x, y in allp)
print("per circuit C4 vs C3: %d pairs; same 2q count %d; same infidelity %d; C4 lower %d; C4 higher %d"
      % (len(allp), same_2q, same_inf, win, lose))
lr = np.log([x["infid"] / y["infid"] for x, y in allp if y["infid"] > 0 and x["infid"] > 0])
print("log ratio C4/C3: median %.3f, geometric mean ratio %.3f" % (np.median(lr), np.exp(np.mean(lr))))
for d in DEV:
    pr = [p for f in FAM for p in pairs(d, [(f, None)], "C4", "C3")]
    print("  %s: pooled C4/C3 %.3f, C4 lower in %d of %d, same 2q %d" % (
        d, R(pr), sum(x["infid"] < y["infid"] - 1e-12 for x, y in pr), len(pr), sum(x["two_q"] == y["two_q"] for x, y in pr)))
print("C4 backstop recompiles:", sum(x.get("backstop", 0) > 0 for (d, a, f), v in D.items() if a == "C4" for x in v))
print("C3/C4 failed uses by arm:", {a: sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items()
                                          if aa == a for x in v) for a in ARMS})
print("flags:", flags or "none")
