"""hold_verify.py -- independent re-computation of the HOLD run (Addendum 318), written after the locked score was seen and
before the raw files were read in full. Reads the raw json only: provenance, counts, P0, H1-H7 with unrounded ratios,
the seen/new split, per-circuit win rates, and where the C3 arm touched a coupler flagged as failed.
    python hold_verify.py <run dir>"""
import collections, glob, json, os, sys
import numpy as np
SEEN = ("FakeAuckland", "FakeTorino", "FakeKingston")
NEW = ("FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez", "FakeMarrakesh", "FakeAachen")
DEV = SEEN + NEW; CZ = ("FakeTorino", "FakeKingston", "FakeFez", "FakeMarrakesh", "FakeAachen")
ARMS = ("C3", "C5", "A7", "L3T"); FAM = ("F1", "F2", "F3", "F4", "F5", "F6")
WANT = dict(F1=432, F2=240, F3=300, F4=270, F5=144, F6=120)
D, flags = {}, []
for p in glob.glob(os.path.join(sys.argv[1], "hold_*.json")):
    r = json.load(open(p)); m = r["meta"]
    if (m["git_head"] != "35637f9" or not m["sha"]["script"].startswith("bd087fa5") or m["smoke"] or
            m["release"] != "2026-10-02.2" or m["a7"] != "2026-10-02.a7"):
        flags.append(p)
    if len(r["rows"]) != WANT[m["family"]]:
        flags.append(p + " count")
    D[(m["device"], m["arm"], m["family"])] = r["rows"]
rows = [x for v in D.values() for x in v]
p0i = max(x.get("infid_ideal", 0) for x in rows); wide = sum(1 for x in rows if x.get("too_wide"))
print("files %d, flags %d, rows %d" % (len(D), len(flags), len(rows)))
print("P0", "PASS" if len(D) == 216 and not flags and p0i <= 1e-6 and wide <= 0.05 * len(rows) else "FAIL",
      "noiseless max %.2e, too wide %d" % (p0i, wide))


def sel(d, a, f, bc=None):
    return [x for x in D[(d, a, f)] if bc is None or x["params"].get("bc") == bc]


def pairs(d, cells, a, b):
    o = []
    for f, bc in cells:
        for x, y in zip(sel(d, a, f, bc), sel(d, b, f, bc)):
            assert x["params"] == y["params"]
            o.append((x, y))
    return o


R = lambda pr: np.mean([x["infid"] for x, _ in pr]) / np.mean([y["infid"] for _, y in pr])
V = lambda ok, bad: "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
ALL = [(f, None) for f in FAM]
CELLS = [("F1", None), ("F2", None), ("F3", "o"), ("F3", "p"), ("F4", None), ("F5", None), ("F6", None)]
CHAIN = [("F3", "o"), ("F5", None)]
r4 = lambda dct: {k.replace("Fake", ""): round(float(v), 4) for k, v in dct.items()}
h1 = {d: R(pairs(d, ALL, "C5", "C3")) for d in DEV}
print("H1", V(all(v <= 1 for v in h1.values()), any(v > 1.02 for v in h1.values())), r4(h1))
cv = [R(pairs(d, [c], "C5", "C3")) for d in DEV for c in CELLS]
sh = np.mean([v <= 1 for v in cv])
print("H2", V(sh >= 0.9, sh < 0.75), "%.4f of %d; worst cell %.3f" % (sh, len(cv), max(cv)))
h3 = {d: R(pairs(d, CHAIN, "C5", "L3T")) for d in CZ}
print("H3", V(all(v <= 1.05 for v in h3.values()), any(v >= 1.15 for v in h3.values())), r4(h3))
h4 = {d: R(pairs(d, ALL, "A7", "L3T")) for d in DEV}
print("H4", V(all(v <= 1 for v in h4.values()), any(v > 1.05 for v in h4.values())), r4(h4))
h5 = {d: R(pairs(d, ALL, "A7", "C5")) for d in DEV}
print("H5", V(all(v <= 1 for v in h5.values()), any(v > 1.03 for v in h5.values())), r4(h5))
nf = {a: sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS}
print("H6", V(nf["C5"] + nf["A7"] == 0, nf["C5"] + nf["A7"] > 0), nf)
h7 = {d: R(pairs(d, [("F6", None)], "A7", "L3T")) for d in DEV}
print("H7", V(all(v <= 1.05 for v in h7.values()), any(v > 1.20 for v in h7.values())), r4(h7))
print()
print("seen vs new (pooled over devices in the group):")
for name, grp in (("seen", SEEN), ("new", NEW)):
    for a, b in (("C5", "C3"), ("A7", "L3T"), ("A7", "C5"), ("C5", "L3T")):
        pr = [p for d in grp for p in pairs(d, ALL, a, b)]
        print("  %-4s %s/%s %.4f (n=%d)" % (name, a, b, R(pr), len(pr)))
print()
print("per circuit, A7 <= L3T and C5 <= C3, by device:")
for d in DEV:
    a = pairs(d, ALL, "A7", "L3T"); c = pairs(d, ALL, "C5", "C3")
    print("  %-14s A7<=L3T %.3f  C5<=C3 %.3f  A7 chose L3T %d" % (
        d, np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in a]), np.mean([x["infid"] <= y["infid"] + 1e-12 for x, y in c]),
        sum(x.get("chosen") == "L3T" for f in FAM for x in sel(d, "A7", f))))
print()
fl = collections.Counter()
for (d, a, f), v in D.items():
    if a == "C3":
        for x in v:
            if x["failed_uses"] + x["failed_q_uses"]:
                fl[(d, f)] += 1
print("C3 circuits touching a flagged element, by (device, family):", dict(fl))
print("median compile s:", {a: round(float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a for x in v])), 4)
                            for a in ARMS})
print("off-target:", {a: sum(x["off_target"] > 0 for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS})
print("flags:", flags or "none")
