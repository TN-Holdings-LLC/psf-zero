"""track_verify.py -- independent re-computation of the TRACK run (Addendum 387), written before the scored run and
before any of its output exists (2026-10-07). Reads the raw json only and does not import track_eval.py: provenance,
counts, P0, T1-T4 and the per-device table, then compares its verdicts with score.md.
    python track_verify.py <run dir>"""
import glob
import json
import os
import re
import statistics
import sys

RUN = sys.argv[1]
DEV = ("FakeTorino", "FakeKingston", "FakeAuckland", "FakeHanoiV2", "FakeBrussels", "FakeOsaka")
VER = dict(c19="2026-10-07.c19", c22="2026-10-07.c22")
SHA_C19 = "efb4dac3df77"  # candidate c19 as locked in Addendum 383

F, flags, heads, shas = {}, [], set(), set()
for p in glob.glob(os.path.join(RUN, "track_*.json")):
    if p.endswith("_smoke.json"):
        continue
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    shas.add((m["sha"]["script"], m["sha"]["c22"]))
    if m["smoke"] or m["dirty_tracked"] or not m["sha"]["c19"].startswith(SHA_C19) or \
            any(m["versions"][k] != v for k, v in VER.items()) or len(r["rows"]) != 60:
        flags.append(os.path.basename(p))
    F[m["device"]] = r["rows"]

rows = [x for d in F for x in F[d]]
err = [x for x in rows if "error" in x]
good = [x for x in rows if "error" not in x]
off = [x for x in good if x["off22"]]
wide = [x for x in rows if x["wide"]]
checked = [x for x in good if "exact22" in x]
bad = [x for x in checked if x["exact22"] > 1e-6]
want = 9 * len(DEV)  # ring, brick and qft at 8 qubits, 3 circuits each, per device
p0 = sorted(F) == sorted(DEV) and not flags and not err and not off and not wide and not bad and len(heads) == 1 and \
    len(shas) == 1 and len(checked) == want
print(f"files {len(F)}, git_head {sorted(heads)}, one script/c22 hash {len(shas) == 1}, flags {flags}")
print(f"P0 {'PASS' if p0 else 'FAIL'}: errors {len(err)}, off-target {len(off)}, wide inputs {len(wide)}, "
      f"inexact {len(bad)} of {len(checked)} checked (expected {want})")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def ratio_median(rs, sizes):
    return statistics.median([x["t_c22"] / x["t_c19"] for x in rs if x["n"] in sizes])


def V(good_, bad_):
    if bad_:
        return "REFUTED"
    return "CONFIRMED" if good_ else "AMBIGUOUS"


a16, a1214, a20 = {}, {}, {}
print("\ndevice         n   identical  n16    n12-14  n20")
for d in DEV:
    rs = F[d]
    a16[d], a1214[d], a20[d] = ratio_median(rs, {16}), ratio_median(rs, {12, 14}), ratio_median(rs, {20})
    print(f"{d:14s} {len(rs):2d}  {sum(1 for x in rs if x['identical']):2d}/{len(rs):2d}    {a16[d]:.3f}  {a1214[d]:.3f}   "
          f"{a20[d]:.3f}")
ndiff = sum(1 for x in rows if not x["identical"])
mine = {"T1": V(ndiff == 0, ndiff > 2),
        "T2": V(max(a16.values()) <= 0.6, max(a16.values()) > 0.8),
        "T3": V(max(a1214.values()) <= 0.85, max(a1214.values()) > 1.05),
        "T4": V(max(a20.values()) <= 1.15, max(a20.values()) > 1.30)}
print(f"circuits that differ: {ndiff}")
for k, v in mine.items():
    print(f"{k} {v}")
sc = os.path.join(RUN, "score.md")
theirs = dict(re.findall(r"\| (T\d) \| [^|]+ \| \*\*(\w+)\*\* \|", open(sc).read())) if os.path.exists(sc) else {}
print(f"verdicts identical to score.md: {theirs == mine}")
