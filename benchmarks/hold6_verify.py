"""hold6_verify.py -- independent re-computation of the HOLD6 run (Addendum 336), written after the run finished and
before its output files were read (only the last three lines of the run log, the too-wide counts by arm, had been
seen). Reads the raw json only: provenance, counts, P0, H1-H11 with unrounded ratios, per-circuit win rates, the
A8 = R3 identity on cz devices, and the choices.
    python hold6_verify.py <run dir>"""
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
FAM = ("F1", "F2", "F3", "F4", "F5", "F6")
WFAM = ("W1", "W2", "W3", "W4", "W5", "W6")
WANT = dict(F1=432, F2=240, F3=300, F4=270, F5=144, F6=120, W1=12, W2=12, W3=12, W4=12, W5=12, W6=12)
SHA = dict(script="8740a33225f2", c11="726defb75c0e", a8="3f17afbf0aaf")

D, flags, heads, vers = {}, [], set(), set()
for p in glob.glob(os.path.join(sys.argv[1], "hold6_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    vers.add((m["release"], m.get("c11"), m["a7"], m.get("a8"), m["sha"]["release"][:12], m["sha"]["a7"][:12]))
    if (m["smoke"] or not m["sha"]["script"].startswith(SHA["script"]) or not m["sha"].get("c11", "").startswith(SHA["c11"])
            or not m["sha"].get("a8", "").startswith(SHA["a8"]) or m["release"] != "2026-10-03.3"
            or m.get("c11") != "2026-10-04.c11" or m["a7"] != "2026-10-02.a7" or m.get("a8") != "2026-10-04.a8"):
        flags.append(os.path.basename(p))
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(os.path.basename(p) + " count")
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
fr = [x for (d, a, f), v in D.items() if f in FAM for x in v]
wr = [x for (d, a, f), v in D.items() if f in WFAM for x in v]
p0i = max(x.get("infid_ideal", 0.0) for x in fr + wr)
fw, ww = sum(1 for x in fr if x.get("too_wide")), sum(1 for x in wr if x.get("too_wide"))
print("files %d, rows F %d W %d, git_head %s, versions %s" % (len(D), len(fr), len(wr), sorted(heads), sorted(vers)))
print("P0", "PASS" if len(D) == 486 and not flags and len(heads) == 1 and p0i <= 1e-6 and fw <= 0.05 * len(fr)
      and ww <= 0.30 * len(wr) else "FAIL", "noiseless max %.2e, too wide F %d of %d, W %d of %d" % (p0i, fw, len(fr), ww, len(wr)))


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
    return o


def R(pr):
    return np.mean([x["infid"] for x, _ in pr]) / np.mean([y["infid"] for _, y in pr]) if pr else float("nan")


def V(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def r4(dct):
    return {k.replace("Fake", ""): round(float(v), 4) for k, v in dct.items()}


FA = [(f, None) for f in FAM]
WA = [(f, None) for f in WFAM]
h1 = {d: R(pairs(d, FA, "C11", "R3")) for d in DEV}
print("H1", V(sum(v <= 1 for v in h1.values()) >= 8, any(v > 1.01 for v in h1.values())), r4(h1))
h2 = {d: h1[d] for d in CX}
print("H2", V(all(v < 1 for v in h2.values()), sum(v > 1.003 for v in h2.values()) >= 2), r4(h2))
h3 = {d: R(pairs(d, [("F3", "o")], "C11", "R3")) for d in CX}
print("H3", V(sum(v <= 1 for v in h3.values()) >= 3, sum(v > 1.01 for v in h3.values()) >= 2), r4(h3))
h4 = {d: R(pairs(d, [("F5", None)], "C11", "R3")) for d in CX}
print("H4", V(all(v <= 1.01 for v in h4.values()), any(v > 1.03 for v in h4.values())), r4(h4))
h5 = {d: h1[d] for d in CZ}
print("H5", V(sum(v <= 1 for v in h5.values()) >= 4, sum(v > 1.005 for v in h5.values()) >= 2), r4(h5))
nfd = sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, a, f), v in D.items() if a in ("C11", "A8") for x in v)
noff = sum(x["off_target"] > 0 for (d, a, f), v in D.items() if a in ("C11", "A8") for x in v)
print("H6", V(nfd + noff == 0, nfd + noff > 0), "failed uses %d, off-target circuits %d" % (nfd, noff))
med = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a and f in FAM for x in v]))
       for a in ("R3", "C11", "A7", "L3T")}
print("H7", V(med["C11"] <= 3 * med["R3"], med["C11"] > 10 * med["R3"]), {a: round(v, 4) for a, v in med.items()})
h8 = {d: R(pairs(d, FA, "C11", "L3T")) for d in DEV}
print("H8", V(all(v <= 1 for v in h8.values()), any(v > 1.02 for v in h8.values())), r4(h8))
h9 = {d: R(pairs(d, WA, "A8", "A7")) for d in DEV}
print("H9", V(sum(v <= 0.95 for v in h9.values()) >= 7, sum(v > 1 for v in h9.values()) >= 2), r4(h9))
h10 = {d: R(pairs(d, WA, "A8", "L3T")) for d in DEV}
print("H10", V(sum(v <= 1 for v in h10.values()) >= 7, sum(v > 1.03 for v in h10.values()) >= 3), r4(h10))
h11 = {d: R(pairs(d, WA, "C11", "R3")) for d in DEV}
print("H11", V(sum(v <= 1 for v in h11.values()) >= 7, sum(v > 1.02 for v in h11.values()) >= 2), r4(h11))
print()
print("F by device: C11/R3, C11/L3T, R3/L3T, A7/C11, F3o C11/R3, F3p C11/R3, F5 C11/R3, F6 C11/R3")
for d in DEV:
    print("  %-14s" % d, " ".join("%.4f" % v for v in (
        h1[d], h8[d], R(pairs(d, FA, "R3", "L3T")), R(pairs(d, FA, "A7", "C11")), R(pairs(d, [("F3", "o")], "C11", "R3")),
        R(pairs(d, [("F3", "p")], "C11", "R3")), R(pairs(d, [("F5", None)], "C11", "R3")),
        R(pairs(d, [("F6", None)], "C11", "R3")))))
print("W by device: C11/R3, A8/A7, A8/L3T, A7/L3T, R3/L3T, A8/R3, pairs A8-A7")
for d in DEV:
    print("  %-14s" % d, " ".join("%.4f" % v for v in (
        h11[d], h9[d], h10[d], R(pairs(d, WA, "A7", "L3T")), R(pairs(d, WA, "R3", "L3T")), R(pairs(d, WA, "A8", "R3")))),
          len(pairs(d, WA, "A8", "A7")))
same = sum(abs(x["infid"] - y["infid"]) <= 1e-12 and x["two_q"] == y["two_q"] for d in CZ for x, y in pairs(d, WA, "A8", "R3"))
tot = sum(len(pairs(d, WA, "A8", "R3")) for d in CZ)
print("A8 = R3 on cz devices (W, identical call): %d of %d pairs identical" % (same, tot))
print()
print("per circuit (F): C11 < R3, C11 > R3, C11 <= L3T")
for d in DEV:
    pr, pl = pairs(d, FA, "C11", "R3"), pairs(d, FA, "C11", "L3T")
    print("  %-14s %.3f %.3f %.3f" % (d, np.mean([x["infid"] < y["infid"] - 1e-12 for x, y in pr]),
                                      np.mean([x["infid"] > y["infid"] + 1e-12 for x, y in pr]),
                                      np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in pl])))
ch = collections.Counter()
for (d, a, f), v in D.items():
    if a in ("C11", "R3"):
        for x in v:
            ch[(a, d in CX, f[0], x.get("chosen"))] += 1
print("choices (arm, cx device, set, chosen):", dict(sorted(ch.items(), key=str)))
print("failed uses (coupler, direction) by arm:", {a: (
    sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v),
    sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v))
    for a in ("R3", "C11", "A7", "A8", "L3T")})
print("too wide by arm and set:", {(a, s): sum(1 for (d, aa, f), v in D.items() if aa == a and f[0] == s for x in v
                                              if x.get("too_wide")) for a in ("R3", "C11", "A7", "A8", "L3T") for s in "FW"})
print("flags: %d %s" % (len(flags), sorted(flags)[:5]))
