"""hold4_verify.py -- independent re-computation of the HOLD4 run (Addendum 327), written after the run finished and
before any of its output (score or raw files) was seen. Reads the raw json only: provenance, counts, P0, H1-H9 with
unrounded ratios, the oracle bound (per circuit, the measured better of R3 and L3T), per-circuit win rates and C9's
choices by family.
    python hold4_verify.py <run dir>"""
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
ARMS = ("R3", "C9", "A7", "L3T")
FAM = ("F1", "F2", "F3", "F4", "F5", "F6")
WANT = dict(F1=432, F2=240, F3=300, F4=270, F5=144, F6=120)
SHA = dict(script="a834164831ee", c9="e27d241776e8")
CELLS = [("F1", None), ("F2", None), ("F3", "o"), ("F3", "p"), ("F4", None), ("F5", None), ("F6", None)]
ALL = [(f, None) for f in FAM]

D, flags, heads, vers = {}, [], set(), set()
for p in glob.glob(os.path.join(sys.argv[1], "hold4_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    vers.add((m["release"], m["c9"], m["a7"], m["sha"]["release"][:12], m["sha"]["a7"][:12]))
    if (m["smoke"] or not m["sha"]["script"].startswith(SHA["script"]) or not m["sha"]["c9"].startswith(SHA["c9"])
            or m["c9"] != "2026-10-03.c9" or m["release"] != "2026-10-03.1" or m["a7"] != "2026-10-02.a7"):
        flags.append(p)
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(p + " count")
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
rows = [x for v in D.values() for x in v]
p0i = max(x.get("infid_ideal", 0.0) for x in rows)
wide = sum(1 for x in rows if x.get("too_wide"))
print("files %d, rows %d, git_head %s, versions %s" % (len(D), len(rows), sorted(heads), sorted(vers)))
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


h1 = {d: R(pairs(d, ALL, "C9", "R3")) for d in DEV}
print("H1", V(all(v <= 1 for v in h1.values()), any(v > 1.02 for v in h1.values())), r4(h1))
h2 = {d: R(pairs(d, ALL, "C9", "L3T")) for d in DEV}
print("H2", V(sum(v <= 1.00 for v in h2.values()) >= 7, sum(v > 1.03 for v in h2.values()) >= 3), r4(h2))
cl = {(d, f + (bc or "")): R(pairs(d, [(f, bc)], "C9", "L3T")) for d in DEV for f, bc in CELLS}
shl = np.mean([v <= 1.02 for v in cl.values()])
print("H3", V(shl >= 0.9, shl < 0.7), "%.4f of %d; cells > 1.02: %s" % (
    shl, len(cl), {"%s %s" % (d.replace("Fake", ""), c): round(float(v), 4) for (d, c), v in cl.items() if v > 1.02}))
h4 = {d: R(pairs(d, [("F3", "p")], "C9", "L3T")) for d in CX}
print("H4", V(all(v <= 1.03 for v in h4.values()), any(v >= 1.10 for v in h4.values())), r4(h4))
h5 = {d: R(pairs(d, [("F1", None)], "C9", "L3T")) for d in CZ}
print("H5", V(all(v <= 1.02 for v in h5.values()), any(v >= 1.05 for v in h5.values())), r4(h5))
nfd = sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, a, f), v in D.items() if a == "C9" for x in v)
noff = sum(x["off_target"] > 0 for (d, a, f), v in D.items() if a == "C9" for x in v)
print("H6", V(nfd + noff == 0, nfd + noff > 0), "failed uses %d, off-target circuits %d" % (nfd, noff))
med = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a for x in v])) for a in ARMS}
print("H7", V(med["C9"] <= 4 * med["R3"], med["C9"] > 10 * med["R3"]), {a: round(v, 4) for a, v in med.items()})
h8 = {d: R(pairs(d, ALL, "A7", "C9")) for d in DEV}
print("H8", V(sum(v >= 0.96 for v in h8.values()) >= 7, sum(v < 0.93 for v in h8.values()) >= 3), r4(h8))
right = n = 0
for d in DEV:
    for f in FAM:
        for xr, xl, x9 in zip(sel(d, "R3", f), sel(d, "L3T", f), sel(d, "C9", f)):
            if abs(xr["infid"] - xl["infid"]) <= 1e-12:
                continue
            n += 1
            right += abs(x9["infid"] - min(xr["infid"], xl["infid"])) <= 1e-12
print("H9", V(right / n >= 0.75, right / n < 0.55), "%d of %d (%.4f)" % (right, n, right / n))
print()
print("by device: C9/R3, C9/L3T, oracle/L3T, R3/L3T, A7/C9, A7/L3T, A7/oracle")
for d in DEV:
    orc, l3, a7 = [], [], []
    for f in FAM:
        for xr, xl, xa in zip(sel(d, "R3", f), sel(d, "L3T", f), sel(d, "A7", f)):
            orc.append(min(xr["infid"], xl["infid"]))
            l3.append(xl["infid"])
            a7.append(xa["infid"])
    print("  %-14s" % d, " ".join("%.4f" % v for v in (
        h1[d], h2[d], np.mean(orc) / np.mean(l3), R(pairs(d, ALL, "R3", "L3T")), h8[d], R(pairs(d, ALL, "A7", "L3T")),
        np.mean(a7) / np.mean(orc))))
print()
print("per circuit: C9 < R3, C9 > R3, C9 <= L3T, A7 <= C9")
for d in DEV:
    pr, pl, pa = pairs(d, ALL, "C9", "R3"), pairs(d, ALL, "C9", "L3T"), pairs(d, ALL, "A7", "C9")
    print("  %-14s %.3f %.3f %.3f %.3f" % (d, np.mean([x["infid"] < y["infid"] - 1e-12 for x, y in pr]),
                                          np.mean([x["infid"] > y["infid"] + 1e-12 for x, y in pr]),
                                          np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in pl]),
                                          np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in pa])))
print()
ch = collections.Counter()
for (d, a, f), v in D.items():
    if a == "C9":
        for x in v:
            ch[(f, x.get("chosen"))] += 1
print("C9 choices by family:", dict(sorted(ch.items())))
print("off-target by arm:", {a: sum(x["off_target"] > 0 for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS})
print("failed uses (coupler, direction) by arm:", {a: (
    sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v),
    sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v)) for a in ARMS})
print("flags: %d %s" % (len(flags), sorted(flags)[:5]))
