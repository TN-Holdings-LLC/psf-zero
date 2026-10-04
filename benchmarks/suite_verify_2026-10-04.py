"""suite_verify_2026-10-04.py -- independent re-computation of the STALE and WIDE runs (Addendum 334), written after
the suite finished and before its output files were read (only the last five lines of the suite log, WIDE's H3-H5
verdicts, had been seen). Reads the raw json only: provenance, counts, P0, H1-H5 with unrounded ratios, per-circuit
win rates and R3's choices by family.
    python suite_verify_2026-10-04.py stale <run dir>
    python suite_verify_2026-10-04.py wide <run dir>"""
import collections
import glob
import json
import os
import sys

import numpy as np

TEST, RUN = sys.argv[1], sys.argv[2]
CX = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva")
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
       "FakeMarrakesh", "FakeAachen")
ARMS = ("R2", "R3", "A7", "L3T")
FAM = ("F1", "F2", "F3", "F4", "F5", "F6")
WANT = dict(stale=dict(F1=216, F2=120, F3=150, F4=135, F5=72, F6=60),
            wide=dict(F1=48, F2=48, F3=48, F4=36, F5=24, F6=24))[TEST]
SHA = dict(stale="f2dd16caf17d", wide="47a32ac09648")[TEST]
WIDE_MAX = dict(stale=0.05, wide=0.10)[TEST]
CELLS = [("F1", None), ("F2", None), ("F3", "o"), ("F3", "p"), ("F4", None), ("F5", None), ("F6", None)]
ALL = [(f, None) for f in FAM]

D, flags, heads, vers, tch = {}, [], set(), set(), {}
for p in glob.glob(os.path.join(RUN, TEST + "_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    vers.add((m["release"], m["a7"], m["sha"]["release"][:12], m["sha"]["a7"][:12]))
    if (m["smoke"] or not m["sha"]["script"].startswith(SHA) or m["release"] != "2026-10-03.3"
            or m["a7"] != "2026-10-02.a7"):
        flags.append(os.path.basename(p))
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(os.path.basename(p) + " count")
    if TEST == "stale":
        tch.setdefault(m["device"], set()).add(m.get("stale_t1_changed"))
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
rows = [x for v in D.values() for x in v]
p0i = max(x.get("infid_ideal", 0.0) for x in rows)
wide = sum(1 for x in rows if x.get("too_wide"))
print("%s: files %d, rows %d, git_head %s, versions %s" % (TEST, len(D), len(rows), sorted(heads), sorted(vers)))
print("P0", "PASS" if len(D) == 216 and not flags and len(heads) == 1 and p0i <= 1e-6 and wide <= WIDE_MAX * len(rows)
      else "FAIL", "noiseless max %.2e, too wide %d of %d" % (p0i, wide, len(rows)))
if TEST == "stale":
    print("stale T1 changed per device (one value per device expected):", {d.replace("Fake", ""): sorted(v) for d, v in tch.items()})

dropped = collections.Counter()


def sel(d, a, f, bc=None):
    return [x for x in D[(d, a, f)] if bc is None or x["params"].get("bc") == bc]


def pairs(d, cells, a, b):
    o = []
    for f, bc in cells:
        xa, xb = sel(d, a, f, bc), sel(d, b, f, bc)
        assert len(xa) == len(xb)
        for x, y in zip(xa, xb):
            assert x["params"] == y["params"] and x["n"] == y["n"]
            if "infid" in x and "infid" in y:
                o.append((x, y))
            else:
                dropped[(d, a, b)] += 1
    return o


def R(pr):
    return np.mean([x["infid"] for x, _ in pr]) / np.mean([y["infid"] for _, y in pr]) if pr else float("nan")


def V(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def r4(dct):
    return {k.replace("Fake", ""): round(float(v), 4) for k, v in dct.items()}


h1 = {d: R(pairs(d, ALL, "R2", "L3T")) for d in DEV}
print("H1", V(sum(v <= 1.00 for v in h1.values()) >= 7, sum(v > 1.03 for v in h1.values()) >= 3), r4(h1))
h2 = {d: R(pairs(d, ALL, "R3", "R2")) for d in CX}
print("H2", V(sum(v <= 1.00 for v in h2.values()) >= 3, sum(v > 1.01 for v in h2.values()) >= 2), r4(h2))
cl = {(d, f + (bc or "")): R(pairs(d, [(f, bc)], "R2", "L3T")) for d in DEV for f, bc in CELLS}
vals = [v for v in cl.values() if v == v]
sh = sum(v <= 1.02 for v in vals) / len(vals)
print("H3", V(sh >= 0.85, sh < 0.65), "%.4f of %d pairs" % (sh, len(vals)))
nfd = sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, a, f), v in D.items() if a in ("R2", "R3") for x in v)
noff = sum(x["off_target"] > 0 for (d, a, f), v in D.items() if a in ("R2", "R3") for x in v)
print("H4", V(nfd + noff == 0, nfd + noff > 0), "failed uses %d, off-target circuits %d" % (nfd, noff))
med = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a for x in v])) for a in ARMS}
if TEST == "wide":
    print("H5", V(med["R3"] <= 1.0, med["R3"] > 5.0), {a: round(v, 4) for a, v in med.items()})
else:
    h5 = {d: R(pairs(d, ALL, "A7", "L3T")) for d in DEV}
    print("H5", V(sum(v <= 1.00 for v in h5.values()) >= 7, sum(v > 1.03 for v in h5.values()) >= 3), r4(h5))
    print("median compile s", {a: round(v, 4) for a, v in med.items()})
print()
print("by device: R2/L3T, R3/L3T, R3/R2, A7/R2, A7/L3T, rec/L3T, F5 R3/R2, F3o R3/R2")
for d in DEV:
    rec = "R3" if d in CX else "R2"
    print("  %-14s" % d, " ".join("%.4f" % v for v in (
        h1[d], R(pairs(d, ALL, "R3", "L3T")), R(pairs(d, ALL, "R3", "R2")), R(pairs(d, ALL, "A7", "R2")),
        R(pairs(d, ALL, "A7", "L3T")), R(pairs(d, ALL, rec, "L3T")), R(pairs(d, [("F5", None)], "R3", "R2")),
        R(pairs(d, [("F3", "o")], "R3", "R2")))))
print()
print("cells R2/L3T > 1.02:", {"%s %s" % (d.replace("Fake", ""), c): round(float(v), 4) for (d, c), v in cl.items() if v > 1.02})
c3 = {(d, f + (bc or "")): R(pairs(d, [(f, bc)], "R3", "R2")) for d in DEV for f, bc in CELLS}
print("cells R3/R2 > 1.01:", {"%s %s" % (d.replace("Fake", ""), c): round(float(v), 4) for (d, c), v in c3.items() if v > 1.01})
print("cells R3/R2 < 0.97:", {"%s %s" % (d.replace("Fake", ""), c): round(float(v), 4) for (d, c), v in c3.items() if v < 0.97})
print("per circuit: R2 <= L3T, R3 < R2, R3 > R2, A7 <= R2")
for d in DEV:
    a, b, c = pairs(d, ALL, "R2", "L3T"), pairs(d, ALL, "R3", "R2"), pairs(d, ALL, "A7", "R2")
    print("  %-14s %.3f %.3f %.3f %.3f" % (d, np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in a]),
                                          np.mean([x["infid"] < y["infid"] - 1e-12 for x, y in b]),
                                          np.mean([x["infid"] > y["infid"] + 1e-12 for x, y in b]),
                                          np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in c])))
print()
ch = collections.Counter()
for (d, a, f), v in D.items():
    if a == "R3":
        for x in v:
            ch[(f, x.get("chosen"))] += 1
print("R3 choices by family:", dict(sorted(ch.items())))
print("off-target by arm:", {a: sum(x["off_target"] > 0 for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS})
print("failed uses (coupler, direction) by arm:", {a: (
    sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v),
    sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v)) for a in ARMS})
print("too wide by arm:", {a: sum(1 for (d, aa, f), v in D.items() if aa == a for x in v if x.get("too_wide")) for a in ARMS})
print("pairs dropped for width:", sum(dropped.values()))
print("flags: %d %s" % (len(flags), sorted(flags)[:5]))
