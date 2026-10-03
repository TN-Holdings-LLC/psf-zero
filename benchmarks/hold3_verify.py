"""hold3_verify.py -- independent re-computation of the HOLD3 run (Addendum 323), written after the run finished and
before any of its output (score or raw files) was seen. Reads the raw json only: provenance, counts, P0, H1-H9 with
unrounded ratios, how C8's choices relate to C5 and C7F, per-circuit win rates and the choices by family.
    python hold3_verify.py <run dir>"""
import collections
import glob
import json
import os
import sys

import numpy as np

CX = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva")
CZ = ("FakeTorino", "FakeKingston", "FakeFez", "FakeMarrakesh", "FakeAachen")
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
       "FakeMarrakesh", "FakeAachen")
ARMS = ("C5", "C7F", "C8", "A7", "L3T")
FAM = ("F1", "F2", "F3", "F4", "F5", "F6")
WANT = dict(F1=432, F2=240, F3=300, F4=270, F5=144, F6=120)
SHA = dict(script="95d0fc5fb60f", c8="ae48dd7a9ea5")
CELLS = [("F1", None), ("F2", None), ("F3", "o"), ("F3", "p"), ("F4", None), ("F5", None), ("F6", None)]
CHAIN = [("F3", "o"), ("F5", None)]
F3O = [("F3", "o")]
ALL = [(f, None) for f in FAM]

D, flags, heads, vers = {}, [], set(), set()
for p in glob.glob(os.path.join(sys.argv[1], "hold3_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    vers.add((m["release"], m["c8"], m["a7"], m["sha"]["release"][:12], m["sha"]["a7"][:12]))
    if (m["smoke"] or not m["sha"]["script"].startswith(SHA["script"]) or not m["sha"]["c8"].startswith(SHA["c8"])
            or m["c8"] != "2026-10-03.c8" or m["release"] != "2026-10-02.2" or m["a7"] != "2026-10-02.a7"):
        flags.append(p)
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(p + " count")
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
rows = [x for v in D.values() for x in v]
p0i = max(x.get("infid_ideal", 0.0) for x in rows)
wide = sum(1 for x in rows if x.get("too_wide"))
print("files %d, rows %d, git_head %s, versions %s" % (len(D), len(rows), sorted(heads), sorted(vers)))
print("P0", "PASS" if len(D) == 270 and not flags and len(heads) == 1 and p0i <= 1e-6 and wide <= 0.05 * len(rows)
      else "FAIL", "noiseless max %.2e, too wide %d" % (p0i, wide))


def sel(d, a, f, bc=None):
    return [x for x in D[(d, a, f)] if bc is None or x["params"].get("bc") == bc]


def pairs(d, cells, a, b):
    o = []
    for f, bc in cells:
        xa, xb = sel(d, a, f, bc), sel(d, b, f, bc)
        assert len(xa) == len(xb)
        for x, y in zip(xa, xb):
            assert x["params"] == y["params"]
            o.append((x, y))
    return o


def R(pr):
    return np.mean([x["infid"] for x, _ in pr]) / np.mean([y["infid"] for _, y in pr])


def V(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def r4(dct):
    return {k.replace("Fake", ""): round(float(v), 4) for k, v in dct.items()}


h1 = {d: R(pairs(d, ALL, "C8", "C5")) for d in DEV}
print("H1", V(all(v <= 1 for v in h1.values()), any(v > 1.02 for v in h1.values())), r4(h1))
h2 = {d: h1[d] for d in CX}
print("H2", V(sum(v <= 0.99 for v in h2.values()) >= 3, sum(v > 1.00 for v in h2.values()) >= 2), r4(h2))
h3 = {d: h1[d] for d in CZ}
print("H3", V(all(0.98 <= v <= 1.02 for v in h3.values()), any(v < 0.95 or v > 1.05 for v in h3.values())), r4(h3))
h4 = {d: R(pairs(d, F3O, "C8", "L3T")) for d in CX}
print("H4", V(sum(v <= 1.05 for v in h4.values()) >= 3, sum(v >= 1.10 for v in h4.values()) >= 2), r4(h4))
h5 = {d: R(pairs(d, CHAIN, "C8", "L3T")) for d in CX}
print("H5", V(sum(v <= 1.05 for v in h5.values()) >= 3, all(v >= 1.10 for v in h5.values())), r4(h5))
nfd = sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, a, f), v in D.items() if a == "C8" for x in v)
noff = sum(x["off_target"] > 0 for (d, a, f), v in D.items() if a == "C8" for x in v)
print("H6", V(nfd + noff == 0, nfd + noff > 0), "failed uses %d, off-target circuits %d" % (nfd, noff))
med = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a for x in v])) for a in ARMS}
print("H7", V(med["C8"] <= 3 * med["C5"], med["C8"] > 10 * med["C5"]), {a: round(v, 4) for a, v in med.items()})
cv = {(d, f + (bc or "")): R(pairs(d, [(f, bc)], "C8", "C5")) for d in CX for f, bc in CELLS}
sh = np.mean([v <= 1 for v in cv.values()])
print("H8", V(sh >= 0.8, sh < 0.5), "%.4f of %d; cells > 1: %s" % (
    sh, len(cv), {"%s %s" % (d.replace("Fake", ""), c): round(float(v), 4) for (d, c), v in cv.items() if v > 1}))
right = n = same5 = samef = 0
for d in DEV:
    for f in FAM:
        for x5, xf, x8 in zip(sel(d, "C5", f), sel(d, "C7F", f), sel(d, "C8", f)):
            same5 += abs(x8["infid"] - x5["infid"]) <= 1e-12
            samef += abs(x8["infid"] - xf["infid"]) <= 1e-12
            if abs(x5["infid"] - xf["infid"]) <= 1e-12:
                continue
            n += 1
            right += abs(x8["infid"] - min(x5["infid"], xf["infid"])) <= 1e-12
print("H9", V(right / n >= 0.75, right / n < 0.5), "%d of %d (%.4f)" % (right, n, right / n))
print("  C8 equals C5 in %d circuits, equals C7F in %d (of %d)" % (same5, samef, len(DEV) * 1506))
print()
print("by device: C8/C5, C7F/C5, oracle/C5, C8/L3T, C5/L3T, F3o C8/L3T, chains C8/L3T, A7/C8, A7/L3T")
for d in DEV:
    orc = []
    for f in FAM:
        for x5, xf in zip(sel(d, "C5", f), sel(d, "C7F", f)):
            orc.append(min(x5["infid"], xf["infid"]))
    c5 = np.mean([x["infid"] for f in FAM for x in sel(d, "C5", f)])
    print("  %-14s" % d, " ".join("%.4f" % v for v in (
        h1[d], R(pairs(d, ALL, "C7F", "C5")), np.mean(orc) / c5, R(pairs(d, ALL, "C8", "L3T")),
        R(pairs(d, ALL, "C5", "L3T")), R(pairs(d, F3O, "C8", "L3T")), R(pairs(d, CHAIN, "C8", "L3T")),
        R(pairs(d, ALL, "A7", "C8")), R(pairs(d, ALL, "A7", "L3T")))))
print()
print("per circuit: C8 < C5, C8 > C5, A7 <= L3T, C8 <= L3T")
for d in DEV:
    pr = pairs(d, ALL, "C8", "C5")
    a = pairs(d, ALL, "A7", "L3T")
    c = pairs(d, ALL, "C8", "L3T")
    print("  %-14s %.3f %.3f %.3f %.3f" % (d, np.mean([x["infid"] < y["infid"] - 1e-12 for x, y in pr]),
                                          np.mean([x["infid"] > y["infid"] + 1e-12 for x, y in pr]),
                                          np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in a]),
                                          np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in c])))
print()
ch = collections.Counter()
for (d, a, f), v in D.items():
    if a == "C8":
        for x in v:
            ch[(f, x.get("chosen"))] += 1
print("C8 choices by family:", dict(sorted(ch.items())))
print("off-target by arm:", {a: sum(x["off_target"] > 0 for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS})
print("failed uses (coupler, direction) by arm:", {a: (
    sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v),
    sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v)) for a in ARMS})
print("flags: %d %s" % (len(flags), sorted(flags)[:5]))
