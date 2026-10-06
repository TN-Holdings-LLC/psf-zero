"""a12_verify.py -- independent re-computation of the SPEED run (Addendum 364), written before the scored run and before
any of its output exists (2026-10-06). Reads the raw json only and does not import a12_eval.py: provenance, counts,
P0, S1-S3 and the per-device table.
    python a12_verify.py <run dir>"""
import glob
import json
import os
import statistics
import sys

RUN = sys.argv[1]
CZ, CX, ECR = ("FakeTorino", "FakeKingston"), ("FakeAuckland", "FakeHanoiV2"), ("FakeBrussels", "FakeOsaka")
DEV = CZ + CX + ECR
SHA = dict(script="7c5cc1bd47a4", a12="828c0c9d053c", a11="b619dcd5775c", release="bf4630d6356d")
VER = dict(release="2026-10-06.1", a11="2026-10-05.a11", a12="2026-10-06.a12")

F, flags, heads = {}, [], set()
for p in glob.glob(os.path.join(RUN, "a12_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    if m["smoke"] or any(not m["sha"][k].startswith(v) for k, v in SHA.items()) or \
            any(m["versions"][k] != v for k, v in VER.items()) or len(r["rows"]) != 128:
        flags.append(os.path.basename(p))
    F[m["device"]] = r["rows"]

err = sum(1 for rows in F.values() for r in rows if "error" in r)
ok = [r for rows in F.values() for r in rows if "error" not in r]
inexact = sum(1 for r in ok if r["exact12"] > 1e-6)
off = sum(1 for r in ok if r["off12"])
measured = {d: sum(1 for r in F.get(d, []) if r.get("measured")) for d in DEV}
p0 = sorted(F) == sorted(DEV) and not flags and not err and not inexact and not off and len(heads) == 1 and \
    all(v == 96 for v in measured.values())
print(f"files {len(F)}, git_head {sorted(heads)}, flags {flags}, measured circuits {measured}")
print(f"P0 {'PASS' if p0 else 'FAIL'}: errors {err}, inexact {inexact}, off-target {off}")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def V(good, bad):
    return "REFUTED" if bad else ("CONFIRMED" if good else "AMBIGUOUS")


R = {}
print("\ndevice         identical  med a11  med a12  ratio   (measured / unmeasured ratio)")
for d in DEV:
    rs = F[d]
    m11 = statistics.median(r["t11"] for r in rs)
    m12 = statistics.median(r["t12"] for r in rs)
    R[d] = (sum(r["identical"] for r in rs) / len(rs), m12 / m11)
    rm = statistics.median(r["t12"] for r in rs if r["measured"]) / statistics.median(r["t11"] for r in rs if r["measured"])
    ru = statistics.median(r["t12"] for r in rs if not r["measured"]) / \
        statistics.median(r["t11"] for r in rs if not r["measured"])
    print(f"{d:14s} {sum(r['identical'] for r in rs):3d}/{len(rs)}    {m11:.3f}    {m12:.3f}    {m12 / m11:.3f}   "
          f"({rm:.3f} / {ru:.3f})")
s1 = {d: R[d][0] for d in DEV}
s2 = {d: R[d][1] for d in CZ}
s3 = {d: R[d][1] for d in CX + ECR}
out = [("S1", V(all(v == 1.0 for v in s1.values()), any(v < 1.0 for v in s1.values())), s1),
       ("S2", V(all(v <= 0.75 for v in s2.values()), any(v > 1.0 for v in s2.values())), s2),
       ("S3", V(all(v <= 0.9 for v in s3.values()), any(v > 1.05 for v in s3.values())), s3)]
print()
for q, v, num in out:
    print(q, v, json.dumps(num, default=lambda x: round(x, 4)))
sc = os.path.join(RUN, "score.md")
if os.path.exists(sc):
    txt = open(sc).read()
    same = all(f"**{v}**" in txt.split(f"- {q} (", 1)[1].split("\n", 1)[0] for q, v, _ in out)
    print("verdicts identical to score.md:", same)
