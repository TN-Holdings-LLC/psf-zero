"""hold2_verify.py -- independent re-computation of the HOLD2 run (Addendum 320), written after the run finished and
before any of its output (score or raw files) was seen. Reads the raw json only: provenance, counts, P0, H1-H7 with
unrounded ratios, per-circuit win rates, how often C6 and C5 gave the same result, and the chain cells.
    python hold2_verify.py <run dir>"""
import glob
import json
import os
import sys

import numpy as np

CX = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva")
CZ = ("FakeTorino", "FakeKingston", "FakeFez", "FakeMarrakesh", "FakeAachen")
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
       "FakeMarrakesh", "FakeAachen")
ARMS = ("C5", "C6", "A7", "L3T")
FAM = ("F1", "F2", "F3", "F4", "F5", "F6")
WANT = dict(F1=432, F2=240, F3=300, F4=270, F5=144, F6=120)
SHA = dict(script="2802a86512bf", c6="c40e1bf133e0")
CELLS = [("F1", None), ("F2", None), ("F3", "o"), ("F3", "p"), ("F4", None), ("F5", None), ("F6", None)]
CHAIN = [("F3", "o"), ("F5", None)]
ALL = [(f, None) for f in FAM]

D, flags, heads, rel = {}, [], set(), set()
for p in glob.glob(os.path.join(sys.argv[1], "hold2_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    rel.add((m["release"], m["c6"], m["a7"], m["sha"]["release"][:12], m["sha"]["a7"][:12]))
    if (m["smoke"] or not m["sha"]["script"].startswith(SHA["script"]) or not m["sha"]["c6"].startswith(SHA["c6"])
            or m["c6"] != "2026-10-03.c6" or m["release"] != "2026-10-02.2" or m["a7"] != "2026-10-02.a7"):
        flags.append(p)
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(p + " count")
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
rows = [x for v in D.values() for x in v]
p0i = max(x.get("infid_ideal", 0.0) for x in rows)
wide = sum(1 for x in rows if x.get("too_wide"))
print("files %d, rows %d, git_head %s, versions %s" % (len(D), len(rows), sorted(heads), sorted(rel)))
print("P0", "PASS" if len(D) == 216 and not flags and len(heads) == 1 and p0i <= 1e-6 and wide <= 0.05 * len(rows)
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


h1 = {d: R(pairs(d, ALL, "C6", "C5")) for d in DEV}
print("H1", V(all(v <= 1 for v in h1.values()), any(v > 1.02 for v in h1.values())), r4(h1))
h2 = {d: h1[d] for d in CX}
print("H2", V(sum(v <= 0.98 for v in h2.values()) >= 3, sum(v > 1.00 for v in h2.values()) >= 2), r4(h2))
h3 = {d: h1[d] for d in CZ}
print("H3", V(all(0.98 <= v <= 1.02 for v in h3.values()), any(v < 0.95 or v > 1.05 for v in h3.values())), r4(h3))
h4 = {d: R(pairs(d, CHAIN, "C6", "L3T")) for d in CX}
print("H4", V(sum(v <= 1.05 for v in h4.values()) >= 3, all(v >= 1.10 for v in h4.values())), r4(h4))
nf = {a: (sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v),
          sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v))
      for a in ARMS}
print("H5", V(nf["C6"][1] == 0, nf["C6"][1] > 0), "(by coupler, by direction):", nf)
med = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a for x in v])) for a in ARMS}
print("H6", V(med["C6"] <= 3 * med["C5"], med["C6"] > 10 * med["C5"]), {a: round(v, 4) for a, v in med.items()})
cv = {(d, f + (bc or "")): R(pairs(d, [(f, bc)], "C6", "C5")) for d in CX for f, bc in CELLS}
sh = np.mean([v <= 1 for v in cv.values()])
print("H7", V(sh >= 0.8, sh < 0.5), "%.4f of %d" % (sh, len(cv)))
print("  cx cells with C6/C5 > 1:", {"%s %s" % (d.replace("Fake", ""), c): round(float(v), 4)
                                     for (d, c), v in cv.items() if v > 1})
print()
print("by device: C6/C5, C6/L3T, C5/L3T, chains C6/L3T, chains C5/L3T, A7/C6, A7/L3T")
for d in DEV:
    print("  %-14s" % d, " ".join("%.4f" % R(pairs(d, c, a, b)) for c, a, b in (
        (ALL, "C6", "C5"), (ALL, "C6", "L3T"), (ALL, "C5", "L3T"), (CHAIN, "C6", "L3T"), (CHAIN, "C5", "L3T"),
        (ALL, "A7", "C6"), (ALL, "A7", "L3T"))))
print()
print("per circuit, C6 vs C5: same result / C6 better / C6 worse; and A7 <= L3T")
for d in DEV:
    pr = pairs(d, ALL, "C6", "C5")
    same = np.mean([abs(x["infid"] - y["infid"]) <= 1e-12 for x, y in pr])
    better = np.mean([x["infid"] < y["infid"] - 1e-12 for x, y in pr])
    worse = np.mean([x["infid"] > y["infid"] + 1e-12 for x, y in pr])
    a = pairs(d, ALL, "A7", "L3T")
    print("  %-14s same %.3f  better %.3f  worse %.3f  A7<=L3T %.3f  A7 chose L3T %d" % (
        d, same, better, worse, np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in a]),
        sum(x.get("chosen") == "L3T" for f in FAM for x in sel(d, "A7", f))))
print()
print("chain cells on the cx devices (C6/C5, C6/L3T, C5/L3T):")
for d in CX:
    for f, bc in CHAIN:
        c = [(f, bc)]
        print("  %-14s %s%s %.4f %.4f %.4f" % (d, f, bc or "", R(pairs(d, c, "C6", "C5")), R(pairs(d, c, "C6", "L3T")),
                                             R(pairs(d, c, "C5", "L3T"))))
print("off-target:", {a: sum(x["off_target"] > 0 for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS})
print("flags: %d %s" % (len(flags), sorted(flags)[:5]))
