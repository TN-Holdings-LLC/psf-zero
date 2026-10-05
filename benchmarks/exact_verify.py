"""exact_verify.py -- independent re-computation of the EXACT run (Addendum 342), written after the run started and
before any of its output was seen. Reads the raw json only: provenance, counts, P0, E1-E9, the cell table, and the
check counters.
    python exact_verify.py <run dir>"""
import glob
import json
import os
import sys

import numpy as np

RUN = sys.argv[1]
XD = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeTorino", "FakeKingston")
CX = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva")
YD = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
      "FakeMarrakesh", "FakeAachen")
ARMS = ("RPSF", "R41", "C12", "L3T", "A8", "A9")
CELLS = ("X1", "X2", "X3", "X4", "X5")
SHA = dict(script="f99cb9ee57ee", c12="1dc9b9b1d5b0", a9="f6df9f7a28c8")
WANT_X = dict(X1=48, X2=8, X3=40, X4=16, X5=16)

X, Y, flags, heads = {}, {}, [], set()
for p in glob.glob(os.path.join(RUN, "exact_x_*.json")) + glob.glob(os.path.join(RUN, "exact_y_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    if (m["smoke"] or not m["sha"]["script"].startswith(SHA["script"]) or not m["sha"]["c12"].startswith(SHA["c12"])
            or not m["sha"]["a9"].startswith(SHA["a9"]) or m["release"] != "2026-10-04.1" or m["c12"] != "2026-10-05.c12"
            or m["a9"] != "2026-10-05.a9" or m["a8"] != "2026-10-04.a8"):
        flags.append(os.path.basename(p))
    if m["part"] == "x":
        X[(m["device"], m["arm"])] = r
        for c, n in WANT_X.items():
            if sum(1 for x in r["rows"] if x["cell"] == c) != n:
                flags.append(os.path.basename(p) + " count " + c)
    else:
        Y[m["device"]] = r
        if len(r["rows"]) != 1506:
            flags.append(os.path.basename(p) + " count")
errs = sum(1 for r in X.values() for x in r["rows"] if x["error"] or "infid" not in x)
print("files X %d Y %d, git_head %s, flags %d %s" % (len(X), len(Y), sorted(heads), len(flags), flags[:5]))
print("P0", "PASS" if len(X) == 36 and len(Y) == 9 and not errs and not flags and len(heads) == 1 else "FAIL",
      "compile errors / missing infid %d" % errs)


def wrong(d, a, cells=CELLS):
    rs = [x for x in X[(d, a)]["rows"] if x["cell"] in cells]
    return sum(x["infid"] > 1e-6 for x in rs), len(rs), max(x["infid"] for x in rs)


def V(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


print("\ncell table (wrong/n, max infidelity):")
for d in XD:
    for c in CELLS:
        print("  %-13s %s  " % (d, c) + "  ".join("%s %d/%d %.1e" % ((a,) + wrong(d, a, (c,))) for a in ARMS))
e1 = sum(wrong(d, "C12")[0] for d in XD)
e2 = sum(wrong(d, "A9")[0] for d in XD)
e3v = {d: wrong(d, "R41", ("X3",))[0] for d in CX}
e4 = sum(wrong(d, "R41")[0] for d in XD if d not in CX)
e5 = sum(wrong(d, "RPSF")[0] for d in XD)
same = {d: float(np.mean([x["same"] for x in Y[d]["rows"]])) for d in YD}
ais = {d: float(np.mean([x["ai_same"] for x in Y[d]["rows"] if "ai_same" in x])) for d in YD}
nai = {d: sum(1 for x in Y[d]["rows"] if "ai_same" in x) for d in YD}
e8 = sum(1 for d in YD for x in Y[d]["rows"] if x["c12_infid"] > 1e-6)
mr = float(np.median([x["r41_s"] for d in YD for x in Y[d]["rows"]]))
mc = float(np.median([x["c12_s"] for d in YD for x in Y[d]["rows"]]))
print("\nE1", V(e1 == 0, e1 > 0), e1, "| max C12 infid %.1e" % max(wrong(d, "C12")[2] for d in XD))
print("E2", V(e2 == 0, e2 > 0), e2, "| max A9 infid %.1e" % max(wrong(d, "A9")[2] for d in XD))
print("E3", V(sum(v >= 1 for v in e3v.values()) >= 3, all(v == 0 for v in e3v.values())), e3v)
print("E4", V(e4 == 0, e4 > 0), e4)
print("E5", V(e5 == 0, e5 > 0), e5, "| max RPSF infid %.1e" % max(wrong(d, "RPSF")[2] for d in XD))
print("E6", V(all(v >= 0.995 for v in same.values()), any(v < 0.98 for v in same.values())),
      {k.replace("Fake", ""): round(v, 4) for k, v in same.items()})
print("E7", V(all(v >= 0.99 for v in ais.values()), any(v < 0.95 for v in ais.values())),
      {k.replace("Fake", ""): round(v, 4) for k, v in ais.items()}, "n per device", sorted(set(nai.values())))
print("E8", V(e8 == 0, e8 > 0), e8, "| max C12 infid on F %.1e" % max(x["c12_infid"] for d in YD for x in Y[d]["rows"]))
print("E9", V(mc <= 1.2 * mr, mc > 2 * mr), "C12 %.4f s, R41 %.4f s, ratio %.3f" % (mc, mr, mc / mr))
print("\nwrong by arm (all X):", {a: sum(wrong(d, a)[0] for d in XD) for a in ARMS})
print("X3 wrong by arm and device:", {a: {d.replace("Fake", ""): wrong(d, a, ("X3",))[0] for d in XD} for a in ARMS})
print("X1 wrong by arm and device:", {a: {d.replace("Fake", ""): wrong(d, a, ("X1",))[0] for d in XD} for a in ARMS})
print("C12 EXACT_STATS (X):", {d.replace("Fake", ""): X[(d, "C12")]["stats"]["exact"] for d in XD})
print("A9 L3T_CHECK_STATS (X):", {d.replace("Fake", ""): X[(d, "A9")]["stats"]["l3t_check"] for d in XD})
print("C12 EXACT_STATS (Y):", {d.replace("Fake", ""): Y[d]["stats"]["exact"] for d in YD})
print("A9 wrong on sampled F:", sum(1 for d in YD for x in Y[d]["rows"] if x.get("a9_infid", 0) > 1e-6),
      "max %.1e" % max(x.get("a9_infid", 0) for d in YD for x in Y[d]["rows"]))
print("Y rows differing (C12 vs R41):", {d.replace("Fake", ""): sum(1 for x in Y[d]["rows"] if not x["same"]) for d in YD})
print("median compile s by X arm:", {a: round(float(np.median([x["compile_s"] for d in XD for x in X[(d, a)]["rows"]])), 4)
                                     for a in ARMS})
