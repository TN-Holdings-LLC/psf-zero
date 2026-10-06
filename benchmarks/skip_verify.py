"""skip_verify.py -- independent re-computation of the SKIP run (Addendum 379), written before the scored run and before
any of its output exists (2026-10-06). Reads the raw json only and does not import skip_eval.py: provenance, counts,
P0, K1-K4 and the per-device table, then compares its verdicts with score.md.
    python skip_verify.py <run dir>"""
import glob
import json
import os
import re
import statistics
import sys

RUN = sys.argv[1]
DEV = ("FakeTorino", "FakeKingston", "FakeAuckland", "FakeHanoiV2", "FakeBrussels", "FakeOsaka")
N27 = ("FakeAuckland", "FakeHanoiV2")
SHA = dict(script="9559c9aafd0b", release="2a49f611fa99", c17="3547c79b6a67")
VER = dict(release="2026-10-06.3", c17="2026-10-06.c17")

F, flags, heads = {}, [], set()
for p in glob.glob(os.path.join(RUN, "skip_*.json")):
    if p.endswith("_smoke.json"):
        continue
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    n_want = 51 if m["device"] in N27 else 48
    if m["smoke"] or m["dirty_tracked"] or any(not m["sha"][k].startswith(v) for k, v in SHA.items()) or \
            any(m["versions"][k] != v for k, v in VER.items()) or len(r["rows"]) != n_want:
        flags.append(os.path.basename(p))
    F[m["device"]] = r["rows"]

allrows = [x for d in F for x in F[d]]
err = [x for x in allrows if "error" in x]
good = [x for x in allrows if "error" not in x]
off = [x for x in good if x["off17"]]
checked = [x for x in good if "exact17" in x]
bad_exact = [x for x in checked if x["exact17"] > 1e-6]
# every ring/brick/qft circuit of 10 qubits must have been checked for exactness (3 families x 3 circuits per device)
want_checked = 9 * len(DEV)
p0 = sorted(F) == sorted(DEV) and not flags and not err and not off and not bad_exact and len(heads) == 1 and \
    len(checked) == want_checked
print(f"files {len(F)}, git_head {sorted(heads)}, flags {flags}")
print(f"P0 {'PASS' if p0 else 'FAIL'}: errors {len(err)}, off-target {len(off)}, inexact {len(bad_exact)} "
      f"of {len(checked)} checked (expected {want_checked})")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def V(good_, bad_):
    if bad_:
        return "REFUTED"
    return "CONFIRMED" if good_ else "AMBIGUOUS"


hi, lo = {}, {}
skip_right = True
print("\ndevice         n   identical  above16  upto16")
for d in DEV:
    rs = F[d]
    hi[d] = statistics.median([x["t_c17"] / x["t_rel"] for x in rs if x["n"] > 16])
    lo[d] = statistics.median([x["t_c17"] / x["t_rel"] for x in rs if x["n"] <= 16])
    for x in rs:
        expect = int(x["n"] > 16)
        skip_right = skip_right and x["skipped"]["floor"] == expect and x["skipped"]["level3"] == expect
    print(f"{d:14s} {len(rs):2d}  {sum(1 for x in rs if x['identical']):2d}/{len(rs):2d}     {hi[d]:.3f}    {lo[d]:.3f}")
same = all(x["identical"] for x in allrows)
mine = {"K1": V(same, not same),
        "K2": V(skip_right, not skip_right),
        "K3": V(max(hi.values()) <= 0.5, max(hi.values()) > 0.9),
        "K4": V(max(lo.values()) <= 1.10, max(lo.values()) > 1.25)}
for k, v in mine.items():
    print(f"{k} {v}")
sc = os.path.join(RUN, "score.md")
theirs = dict(re.findall(r"\| (K\d) \| [^|]+ \| \*\*(\w+)\*\* \|", open(sc).read())) if os.path.exists(sc) else {}
print(f"verdicts identical to score.md: {theirs == mine}")
