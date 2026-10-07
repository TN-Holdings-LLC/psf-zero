"""calsplit_verify.py -- independent re-computation of CALSPLIT's verdicts (Addendum 402), written before the scored
run. Reads the raw JSON only (CALSPLIT's and DEPTH-R's), imports no harness, and compares with score.md.
    python benchmarks/calsplit_verify.py <out dir>"""
import glob
import json
import os
import re
import sys

import numpy as np

out = sys.argv[1]
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D, metas = {}, []
for p in glob.glob(os.path.join(out, "deploy_*.json")):
    d = json.load(open(p, encoding="utf-8"))
    m = d["meta"]
    metas.append(m)
    D[(m["dataset"], m["n"], m["device"], m["arm"], m["draw"])] = d["rows"]
dry = any(m["dry"] for m in metas)
REF = os.path.join(REPO, "data", "2026-10-07", "depth_r_dry" if dry else "depth_r")
DS, NS, DEV = ("BC", "D38"), (4, 6), ("FakeAuckland", "FakeTorino")
AW, BL = ("REC", "RPSF", "L3T"), ("DEF", "L3B")
Ls = sorted({r["L"] for rows in D.values() for r in rows})
draws = {a: ("0", "1", "2") if a in AW else ("-",) for a in AW + BL}
infid = max(r["infid"] for rows in D.values() for r in rows)
fw = max(abs(r["z_fullwidth"] - r["z_noisy"]) for rows in D.values() for r in rows if "z_fullwidth" in r)
p0 = len(D) == 88 and infid <= 1e-6 and fw <= 1e-9 and {m["psf_version"] for m in metas} == {"2026-10-07.1"} \
    and len({m.get("c12_sha") for m in metas}) == 1 and all(m["stale_t1_changed"] > 0 for m in metas if m["draw"] != "-")
print(f"dry {dry}; P0 {'PASS' if p0 else 'FAIL'}: files {len(D)}/88, max infidelity {infid:.2e}, reduced-vs-whole {fw:.2e}")


def margin(dev, a):
    return float(np.mean([np.mean([np.mean([r["y"] * r["z_noisy"] for r in D[(ds, n, dev, a, d)] if r["L"] == L])
                                   for ds in DS for n in NS for L in Ls]) for d in draws[a]]))


def flips(dev, a):
    return float(np.mean([sum(int(np.sign(r["z_noisy"]) != np.sign(r["z_ideal"])) for ds in DS for n in NS
                              for r in D[(ds, n, dev, a, d)]) for d in draws[a]]))


def ref(dev):
    m = {}
    for a in ("REC", "RPSF"):
        cells = []
        for ds in DS:
            for n in NS:
                rows = json.load(open(os.path.join(REF, f"deploy_{ds}_n{n}_{dev}_{a}.json"), encoding="utf-8"))["rows"]
                cells += [np.mean([r["y"] * r["z_noisy"] for r in rows if r["L"] == L]) for L in Ls]
        m[a] = float(np.mean(cells))
    return m["REC"] - m["RPSF"]


T = 1e-12
mine, ok, bad = {}, {k: True for k in ("K1", "K2", "K3", "K4", "K5")}, {k: False for k in ("K1", "K2", "K3", "K4", "K5")}
for dev in DEV:
    npts = sum(len(D[(ds, n, dev, "DEF", "-")]) for ds in DS for n in NS)
    bm = max(BL, key=lambda a: margin(dev, a))
    bf = min(BL, key=lambda a: flips(dev, a))
    d1 = margin(dev, "REC") - margin(dev, bm)
    d2 = margin(dev, "REC") - margin(dev, "RPSF")
    d3 = ref(dev) - d2
    d4 = margin(dev, "REC") - margin(dev, "L3T")
    d5 = (flips(dev, "REC") - flips(dev, bf)) / npts
    print(f"{dev}: REC-{bm} {d1:+.4f}, REC-RPSF {d2:+.4f}, same-calibration minus stale {d3:+.4f}, REC-L3T {d4:+.4f}, "
          f"flip-rate REC-{bf} {d5:+.4f}")
    ok["K1"] &= d1 >= 0.002 - T
    bad["K1"] |= d1 < -0.002 - T
    ok["K2"] &= d2 >= 0.003 - T
    bad["K2"] |= d2 < -T
    ok["K3"] &= d3 <= 0.005 + T
    bad["K3"] |= d3 > 0.01 + T
    ok["K4"] &= abs(d4) <= 0.01 + T
    bad["K4"] |= d4 < -0.02 - T
    ok["K5"] &= d5 <= 0.002 + T
    bad["K5"] |= d5 > 0.01 + T
for k in ok:
    mine[k] = "CONFIRMED" if ok[k] else ("REFUTED" if bad[k] else "AMBIGUOUS")
print("mine", mine)
sc = open(os.path.join(out, "score.md"), encoding="utf-8").read()
theirs = json.loads(re.search(r"SUMMARY (\{.*\})", sc).group(1))
their_p0 = re.search(r"P0: (\w+)", sc).group(1)
print(f"identical to score.md: verdicts {theirs == mine}, P0 {their_p0 == ('PASS' if p0 else 'FAIL')}")
