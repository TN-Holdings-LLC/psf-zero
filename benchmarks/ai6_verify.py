"""ai6_verify.py -- independent re-computation of the ai6 run (Addendum 312), written after the locked score was seen
and before the raw files were read. Reads the raw json only: provenance, counts, P0, H1-H8 with unrounded ratios,
per-circuit agreement of A6 with A5, per-task MODEL figures, and A5 against the GAP run's A5 arm (Addendum 301).
    python ai6_verify.py <run dir> [<GAP outputs dir>]"""
import collections, glob, json, os, sys
import numpy as np
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston"); HERON = DEV[1:]
ARMS = ("C5", "A5", "A6", "A6F", "L3T"); FAM = ("F1", "F2", "F3", "F4", "F5"); SETS = FAM + ("MODEL",)
WANT = dict(F1=216, F2=120, F3=150, F4=135, F5=72)
out = sys.argv[1]
idx = json.load(open(os.path.join(out, "model_index.json")))
D, flags = {}, []
for p in glob.glob(os.path.join(out, "ai6_*.json")):
    r = json.load(open(p)); m = r["meta"]
    if (m["git_head"] != "c8dc903" or not m["sha"]["script"].startswith("adef4385") or
            not m["sha"]["a6"].startswith("2b8d11fa") or m["smoke"] or m["a6"] != "2026-10-02.a6" or
            m["a5"] != "2026-10-01.a5" or m["release"] != "2026-10-02.2"):
        flags.append(p)
    want = idx["converted"] if m["set"] == "MODEL" else WANT[m["set"]]
    if len(r["rows"]) != want:
        flags.append(p + " count")
    D[(m["device"], m["arm"], m["set"])] = r["rows"]
rows = [x for v in D.values() for x in v]
p0i = max(x.get("infid_ideal", 0) for x in rows)
wide = sum(1 for x in rows if x.get("too_wide"))
print("files %d, flags %d; model circuits collected %d, converted %d, skipped %d" % (
    len(D), len(flags), idx["collected"], idx["converted"], len(idx["skipped"])))
print("P0", "PASS" if len(D) == 90 and not flags and p0i <= 1e-6 and wide <= 0.05 * len(rows) and idx["converted"] >= 150
      else "FAIL", "noiseless max %.2e, too wide %d of %d" % (p0i, wide, len(rows)))


def pairs(d, sets, a, b):
    o = []
    for s in sets:
        for x, y in zip(D[(d, a, s)], D[(d, b, s)]):
            assert x["params"] == y["params"]
            if "infid" in x and "infid" in y:
                o.append((x, y))
    return o


Rt = lambda pr: np.mean([x["infid"] for x, _ in pr]) / np.mean([y["infid"] for _, y in pr])
G = {"GAP": FAM, "MODEL": ("MODEL",)}
R = {(g, d, a, b): Rt(pairs(d, s, a, b)) for g, s in G.items() for d in DEV
     for a, b in (("A6", "A5"), ("A6", "C5"), ("A6F", "A6"), ("A6", "L3T"), ("A5", "L3T"), ("C5", "L3T"), ("A6F", "C5"),
                  ("A6F", "L3T"))}
V = lambda ok, bad: "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
r = lambda k: "%.5f" % R[k]
print("H1", V(all(R[(g, d, "A6", "A5")] <= 1.0 for g in G for d in DEV), any(R[(g, d, "A6", "A5")] > 1.02 for g in G for d in DEV)),
      {g + "/" + d: r((g, d, "A6", "A5")) for g in G for d in DEV})
print("H2", V(all(R[("GAP", d, "A6", "A5")] <= 0.98 for d in HERON), all(R[("GAP", d, "A6", "A5")] >= 1.0 for d in HERON)),
      {d: r(("GAP", d, "A6", "A5")) for d in HERON})
print("H3", V(all(R[("MODEL", d, "A6", "C5")] <= 0.95 for d in DEV), any(R[("MODEL", d, "A6", "C5")] >= 1.0 for d in DEV)),
      {d: r(("MODEL", d, "A6", "C5")) for d in DEV})
print("H4", V(all(R[("GAP", d, "A6", "C5")] <= 1.0 for d in DEV), any(R[("GAP", d, "A6", "C5")] > 1.05 for d in DEV)),
      {d: r(("GAP", d, "A6", "C5")) for d in DEV})
med = {a: float(np.median([x["compile_s"] for (d, aa, s), v in D.items() if aa == a for x in v])) for a in ARMS}
fq = {(g, d): R[(g, d, "A6F", "A6")] for g in G for d in HERON}
print("H5", V(all(v <= 1.05 for v in fq.values()) and med["A6F"] <= 0.5 * med["A6"],
              any(v > 1.15 for v in fq.values()) or med["A6F"] > med["A6"]),
      {g + "/" + d: "%.4f" % v for (g, d), v in fq.items()}, "median A6F %.4f A6 %.4f" % (med["A6F"], med["A6"]))
print("H6", V(all(R[(g, d, "A6", "L3T")] <= 1.0 for g in G for d in DEV), any(R[(g, d, "A6", "L3T")] > 1.05 for g in G for d in DEV)),
      {g + "/" + d: r((g, d, "A6", "L3T")) for g in G for d in DEV})
nf = {a: sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, s), v in D.items() if aa == a for x in v) for a in ARMS}
print("H7", V(nf["C5"] + nf["A6"] + nf["A6F"] == 0, nf["C5"] + nf["A6"] + nf["A6F"] > 0), nf)
print("H8", V(med["A6"] <= 1.3 * med["A5"], med["A6"] > 2 * med["A5"]), {k: round(v, 4) for k, v in med.items()})
print()
print("| set | device | A6/A5 | A6/C5 | A6F/A6 | A6F/C5 | A6/L3T | A6F/L3T | A5/L3T | C5/L3T |")
for g in G:
    for d in DEV:
        print("| %s | %s | " % (g, d) + " | ".join("%.3f" % R[(g, d, a, b)] for a, b in
              (("A6", "A5"), ("A6", "C5"), ("A6F", "A6"), ("A6F", "C5"), ("A6", "L3T"), ("A6F", "L3T"), ("A5", "L3T"), ("C5", "L3T"))) + " |")
print()
for d in DEV:
    for s in SETS:
        pr = pairs(d, [s], "A6", "A5")
        same = sum(abs(x["infid"] - y["infid"]) <= 1e-12 and x["two_q"] == y["two_q"] for x, y in pr)
        lo = sum(x["infid"] < y["infid"] - 1e-12 for x, y in pr); hi = sum(x["infid"] > y["infid"] + 1e-12 for x, y in pr)
        if same != len(pr):
            print("A6 vs A5 %s %s: identical %d of %d, A6 lower %d, higher %d" % (d, s, same, len(pr), lo, hi))
print("A6 vs A5: all other device-set pairs identical circuit by circuit")
print()
print("MODEL per task (mean infidelity C5 / A6 / A6F / L3T; mean 2q C5 / A6 / L3T):")
for d in DEV:
    by = collections.defaultdict(list)
    for i, x in enumerate(D[(d, "A6", "MODEL")]):
        by[x["params"]["task"]].append(i)
    for t, ii in sorted(by.items()):
        mi = lambda a, k: np.mean([D[(d, a, "MODEL")][i][k] for i in ii])
        print("  %s %-9s n=%3d  %.4f / %.4f / %.4f / %.4f   %.1f / %.1f / %.1f" % (
            d, t, len(ii), mi("C5", "infid"), mi("A6", "infid"), mi("A6F", "infid"), mi("L3T", "infid"),
            mi("C5", "two_q"), mi("A6", "two_q"), mi("L3T", "two_q")))
if len(sys.argv) > 2:
    s = n = 0
    for d in DEV:
        for f in FAM:
            p = os.path.join(sys.argv[2], "gap_%s_A5_%s.json" % (d, f))
            if not os.path.exists(p):
                continue
            g = json.load(open(p))["rows"]
            for x, y in zip(D[(d, "A5", f)], g):
                n += 1
                s += x["params"] == y["params"] and x["two_q"] == y.get("two_q") and abs(x["infid"] - y.get("infid", -1)) <= 1e-12
    print("\nA5 against the GAP run's A5 (Addendum 301): identical %d of %d" % (s, n))
print("off-target:", {a: sum(x["off_target"] > 0 for (d, aa, ss), v in D.items() if aa == a for x in v) for a in ARMS})
print("flags:", flags or "none")
