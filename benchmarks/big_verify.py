"""big_verify.py -- independent re-computation of the BIG run (Addendum 370), written before the scored run and before
any of its output exists (2026-10-06). Reads the raw json only and does not import big_eval.py: provenance, counts,
P0, B1-B4 and the per-device table.
    python big_verify.py <run dir>"""
import glob
import json
import os
import statistics
import sys

RUN = sys.argv[1]
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
       "FakeMarrakesh", "FakeAachen")
FAM = ("W1", "W2", "W3", "W4", "W5", "W6")
ARMS = ("A12", "A13", "L3TM")
SHA = dict(script="e29e4e49f11d", a13="6f67f950a936", a12="2227cca2b067", release="bf4630d6356d", hold6="8740a33225f2")
VER = dict(release="2026-10-06.1", a12="2026-10-06.a12", a13="2026-10-06.a13")

F, flags, heads = {}, [], set()
for p in glob.glob(os.path.join(RUN, "big_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    if m["smoke"] or any(not m["sha"][k].startswith(v) for k, v in SHA.items()) or \
            any(m["versions"][k] != v for k, v in VER.items()):
        flags.append(os.path.basename(p))
    F[(m["device"], m["family"])] = r["rows"]
counts = {f: sorted({len(F[(d, f)]) for d in DEV if (d, f) in F}) for f in FAM}
allr = [x for v in F.values() for x in v]
err = sum(1 for x in allr for a in ARMS if a not in x or "error" in x[a])
ok = [x for x in allr if all(a in x and "error" not in x[a] for a in ARMS)]
inexact = sum(1 for x in ok for a in ARMS if x[a]["state_infid"] > 1e-6)
badm = sum(1 for x in ok for a in ARMS if not x[a]["meas_ok"])
p0 = len(F) == 54 and not flags and not err and not inexact and not badm and len(heads) == 1 and \
    all(len(v) == 1 for v in counts.values())
print(f"files {len(F)} of 54, git_head {sorted(heads)}, flags {flags}, circuits per family {counts}")
print(f"P0 {'PASS' if p0 else 'FAIL'}: errors {err}, inexact {inexact}, wrong measurement mapping {badm}")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def V(good, bad):
    return "REFUTED" if bad else ("CONFIRMED" if good else "AMBIGUOUS")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


S = {}
print("\ndevice          sim/all  A12      A13      L3TM     A13/A12  A13/L3TM  meas err diff  time ratio")
for d in DEV:
    rows = [x for f in FAM for x in F[(d, f)]]
    rs = [x for x in rows if all("infid" in x[a] for a in ARMS)]
    m = {a: mean(x[a]["infid"] for x in rs) for a in ARMS}
    e = {a: mean(x[a]["meas_err"] for x in rows) for a in ARMS}
    tr = statistics.median(x["A13"]["compile_s"] for x in rows) / statistics.median(x["A12"]["compile_s"] for x in rows)
    S[d] = (m["A13"] / m["A12"], e["A13"] - e["A12"], m["A13"] / m["L3TM"], tr)
    print(f"{d:14s}  {len(rs):3d}/{len(rows):3d}  {m['A12']:.5f}  {m['A13']:.5f}  {m['L3TM']:.5f}  "
          f"{S[d][0]:.4f}   {S[d][2]:.4f}    {S[d][1]:+.5f}      {tr:.3f}")
b1 = {d: S[d][0] for d in DEV}
b2 = {d: S[d][1] for d in DEV}
b3 = {d: S[d][2] for d in DEV}
b4 = {d: S[d][3] for d in DEV}
out = [("B1", V(sum(v <= 1.0 for v in b1.values()) >= 8, any(v > 1.005 for v in b1.values())), b1),
       ("B2", V(sum(v <= 0 for v in b2.values()) >= 8, sum(v > 0 for v in b2.values()) >= 3), b2),
       ("B3", V(sum(v <= 1.0 for v in b3.values()) >= 7, any(v > 1.02 for v in b3.values())), b3),
       ("B4", V(all(v <= 2.0 for v in b4.values()), any(v > 3.0 for v in b4.values())), b4)]
print()
for q, v, num in out:
    print(q, v, json.dumps(num, default=lambda x: round(x, 5)))
sc = os.path.join(RUN, "score.md")
if os.path.exists(sc):
    txt = open(sc).read()
    same = all(f"**{v}**" in txt.split(f"- {q} (", 1)[1].split("\n", 1)[0] for q, v, _ in out)
    print("verdicts identical to score.md:", same)
