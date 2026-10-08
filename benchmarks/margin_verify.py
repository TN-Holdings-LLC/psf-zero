"""margin_verify.py -- independent re-computation of MARGIN's verdicts (Addendum 405), written before the scored run.
Reads the raw JSON only, imports no harness, and compares with score.md.
    python benchmarks/margin_verify.py <out dir>"""
import glob
import json
import os
import re
import sys

import numpy as np

out = sys.argv[1]
D, metas = {}, []
for p in glob.glob(os.path.join(out, "deploy_*.json")):
    d = json.load(open(p, encoding="utf-8"))
    m = d["meta"]
    metas.append(m)
    D[(m["dataset"], m["n"], m["device"], m["arm"], m["cal"])] = d["rows"]
T = [json.load(open(p, encoding="utf-8"))["meta"] for p in glob.glob(os.path.join(out, "train_*.json"))]
dry = any(m["dry"] for m in metas + T)
DS, NS, DEV, ARMS = ("BC", "D38"), (4, 6), ("FakeAuckland", "FakeTorino"), ("REC", "C24", "RPSF")
ST = ("0", "1", "2")
Ls = sorted({r["L"] for rows in D.values() for r in rows})
infid = max(r["infid"] for rows in D.values() for r in rows)
fw = max(abs(r["z_fullwidth"] - r["z_noisy"]) for rows in D.values() for r in rows if "z_fullwidth" in r)
vers = {a: {m["psf_version"] for m in metas if m["arm"] == a} for a in ARMS}
p0 = (len(D) == 96 and len(T) == 4 and infid <= 1e-6 and fw <= 1e-9
      and vers == {"REC": {"2026-10-07.1"}, "RPSF": {"2026-10-07.1"}, "C24": {"2026-10-07.c24"}}
      and len({m["c12_sha"] for m in metas}) == 1 and len({m["c24_sha"] for m in metas}) == 1
      and all(m["switch_margin"] == (0.05 if m["arm"] == "C24" else None) for m in metas)
      and all((m["stale_t1_changed"] or 0) > 0 for m in metas if m["cal"] != "t")
      and {tuple(m["seeds"]) for m in metas + T} == ({(2, 2)} if dry else {(5, 5)}))
print(f"dry {dry}; P0 {'PASS' if p0 else 'FAIL'}: files {len(D)}/96, train {len(T)}/4, max infidelity {infid:.2e}, "
      f"reduced-vs-whole {fw:.2e}, versions {vers}")


def margin(dev, a, cals):
    """Per calibration: the mean over every (dataset, n, L) cell of the mean of y * z over its test points."""
    return [float(np.mean([np.mean([r["y"] * r["z_noisy"] for r in D[(ds, n, dev, a, c)] if r["L"] == L])
                           for ds in DS for n in NS for L in Ls])) for c in cals]


def differ(dev, cals):
    pairs = [(x["sig"], y["sig"]) for ds in DS for n in NS for c in cals
             for x, y in zip(D[(ds, n, dev, "C24", c)], D[(ds, n, dev, "REC", c)])]
    return sum(a != b for a, b in pairs) / len(pairs)


E = 1e-12
ok = {k: True for k in ("M1", "M2", "M3", "M4", "M5")}
bad = {k: False for k in ok}
rare = []
for dev in DEV:
    cs, rs, ps = margin(dev, "C24", ST), margin(dev, "REC", ST), margin(dev, "RPSF", ST)
    ct, rt, pt = margin(dev, "C24", ("t",))[0], margin(dev, "REC", ("t",))[0], margin(dev, "RPSF", ("t",))[0]
    d1 = np.mean(cs) - np.mean(ps)
    d2 = np.mean(cs) - np.mean(rs)
    d3 = rt - ct
    worst = min(a - b for a, b in zip(cs, ps))
    share = differ(dev, ST)
    if share < 0.05:
        rare.append(dev)
    print(f"{dev}: C24-RPSF stale {d1:+.4f} (worst draw {worst:+.4f}), C24-REC stale {d2:+.4f}, REC-C24 true "
          f"{d3:+.4f}, C24-RPSF true {ct - pt:+.4f}; C24 differs from REC on {share:.3f} of stale rows")
    ok["M1"] &= d1 >= -E
    bad["M1"] |= d1 < -0.002 - E
    ok["M2"] &= d2 >= -0.001 - E
    bad["M2"] |= d2 < -0.005 - E
    ok["M3"] &= d3 <= 0.002 + E
    bad["M3"] |= d3 > 0.005 + E
    if dev == "FakeTorino":
        lo = min(ct - pt, d1)
        ok["M4"] &= lo >= 0.01 - E
        bad["M4"] |= lo < 0.005 - E
    if dev == "FakeAuckland":
        ok["M5"] &= worst >= -0.005 - E
        bad["M5"] |= worst < -0.01 - E
mine = {k: "CONFIRMED" if ok[k] else ("REFUTED" if bad[k] else "AMBIGUOUS") for k in ok}
accept = p0 and mine["M1"] == "CONFIRMED" and mine["M2"] != "REFUTED" and mine["M3"] != "REFUTED" and len(rare) < 2
mine.update(P0="PASS" if p0 else "FAIL", ITEM51="PROPOSE" if accept else "NO")
print("mine", mine)
sc = open(os.path.join(out, "score.md"), encoding="utf-8").read()
theirs = json.loads(re.search(r"SUMMARY (\{.*\})", sc).group(1))
print(f"identical to score.md: {theirs == mine}")
