"""depth_r_verify.py -- independent re-computation of DEPTH-R's score (Addendum 399), written before the scored run.
Reads the raw JSON only and imports neither depth_r_eval.py nor depth_eval.py. Recomputes P0, H1-H6 and the gate
from the rows and compares them with score.md.
    python benchmarks/depth_r_verify.py <out dir>"""
import glob
import json
import os
import re
import sys

import numpy as np

out = sys.argv[1]
DS, NS, DEV, ARMS, A = ("BC", "D38"), (4, 6), ("FakeAuckland", "FakeTorino"), ("RPSF", "REC", "L3T"), "FakeAuckland"
D, metas = {}, []
for p in glob.glob(os.path.join(out, "deploy_*.json")):
    d = json.load(open(p, encoding="utf-8"))
    m = d["meta"]
    metas.append(m)
    D[(m["dataset"], m["n"], m["device"], m["arm"])] = d["rows"]
T = {}
for p in glob.glob(os.path.join(out, "train_*.json")):
    d = json.load(open(p, encoding="utf-8"))
    metas.append(d["meta"])
    T[(d["meta"]["dataset"], d["meta"]["n"])] = {r["L"]: r for r in d["rows"]}
F = [json.load(open(p, encoding="utf-8")) for p in glob.glob(os.path.join(out, "finetune_*.json"))]
metas += [f["meta"] for f in F]
dry = any(m["dry"] for m in metas)
Ls = sorted({r["L"] for rows in D.values() for r in rows})
print("dry", dry, "| harness", {m.get("harness_sha") for m in metas}, "| depth_eval", {m.get("depth_eval_sha") for m in metas},
      "| release", {m.get("c12_sha") for m in metas}, "| versions", {m.get("psf_version") for m in metas} - {None},
      "| seeds", {tuple(m.get("seeds", ())) for m in metas})


def g(ds, n, dev, arm, L, k):
    rows = [r for r in D[(ds, n, dev, arm)] if r["L"] == L]
    y = np.array([r["y"] for r in rows])
    zn = np.array([r["z_noisy"] for r in rows])
    zi = np.array([r["z_ideal"] for r in rows])
    return {"n": len(rows), "margin": float(np.mean(y * zn)), "shot": float(np.mean([r["shot_acc"] for r in rows])),
            "flip": float(np.mean(np.sign(zn) != np.sign(zi)))}[k]


infid = max(r["infid"] for rows in D.values() for r in rows)
fw = max(abs(r["z_fullwidth"] - r["z_noisy"]) for rows in D.values() for r in rows if "z_fullwidth" in r)
p0 = len(D) == 24 and (len(F) == 4 or dry) and infid <= 1e-6 and fw <= 1e-9 and \
    {m.get("psf_version") for m in metas} - {None} == {"2026-10-07.1"} and \
    {tuple(m.get("seeds", ())) for m in metas} == ({(2, 2)} if dry else {(4, 4)}) and \
    len({m.get("c12_sha") for m in metas}) == 1
print(f"P0 {'PASS' if p0 else 'FAIL'}: files {len(D)}/24, finetune {len(F)}, max infidelity {infid:.2e}, "
      f"reduced-vs-whole {fw:.2e}")
mine = {}
c1 = [g(ds, 6, A, a, Ls[-1], "margin") < 0.8 * max(g(ds, 6, A, a, L, "margin") for L in Ls) for ds in DS for a in ARMS]
b1 = [max(Ls, key=lambda L: g(ds, 6, A, a, L, "margin")) == Ls[-1] for ds in DS for a in ARMS]
mine["H1"] = "CONFIRMED" if all(c1) else ("REFUTED" if any(b1) else "AMBIGUOUS")
c2, b2 = [], []
for ds in DS:
    nt = g(ds, 6, A, "REC", Ls[0], "n")
    c2.append(all(max(g(ds, 6, A, a, L, "shot") for L in Ls) - g(ds, 6, A, a, Ls[-1], "shot") >= 2 / nt - 1e-12
                  for a in ARMS))
    b2 += [g(ds, 6, A, a, Ls[-1], "shot") >= max(g(ds, 6, A, a, L, "shot") for L in Ls) - 1e-12 for a in ARMS]
mine["H2"] = "CONFIRMED" if any(c2) else ("REFUTED" if all(b2) else "AMBIGUOUS")
ok3 = ok4 = ok5 = True
bad3 = bad4 = bad5 = False
for dev in DEV:
    def pool(a, k):
        return float(np.mean([g(ds, n, dev, a, L, k) for ds in DS for n in NS for L in Ls]))
    d3, d4 = pool("REC", "margin") - pool("RPSF", "margin"), pool("REC", "margin") - pool("L3T", "margin")
    fC, fR = pool("REC", "flip"), pool("RPSF", "flip")
    print(f"{dev}: REC-RPSF {d3:+.4f}, REC-L3T {d4:+.4f}, flip REC {fC:.4f} RPSF {fR:.4f}")
    ok3 &= d3 >= 0.005
    bad3 |= d3 < 0
    ok4 &= abs(d4) <= 0.01
    bad4 |= d4 < -0.02
    ok5 &= fC <= fR
    bad5 |= fC > fR + 0.01
for k, ok, bad in (("H3", ok3, bad3), ("H4", ok4, bad4), ("H5", ok5, bad5)):
    mine[k] = "CONFIRMED" if ok else ("REFUTED" if bad else "AMBIGUOUS")
f12 = [f["result"] for f in F if f["meta"]["L"] == 12]
if f12:
    dN = float(np.mean([r["FTN"]["margin"] - r["DEP"]["margin"] for r in f12]))
    d0 = float(np.mean([r["FT0"]["margin"] - r["DEP"]["margin"] for r in f12]))
    mine["H6"] = "CONFIRMED" if dN >= 0.02 and dN > d0 else ("REFUTED" if dN < -0.01 else "AMBIGUOUS")
    print(f"H6: FTN-DEP {dN:+.4f}, FT0-DEP {d0:+.4f}")
big = [1 for dev in DEV for ds in DS for n in NS for L in Ls
       if abs(g(ds, n, dev, "REC", L, "shot") - g(ds, n, dev, "RPSF", L, "shot")) >= 2 / g(ds, n, dev, "REC", L, "n") - 1e-12]
gate = "GO" if (mine["H1"] == "CONFIRMED" or mine["H2"] == "CONFIRMED") and (mine["H3"] == "CONFIRMED" or big) \
    else "NO-GO"
print("mine", mine, "gate", gate)
sc = open(os.path.join(out, "score.md"), encoding="utf-8").read()
theirs = json.loads(re.search(r"SUMMARY (\{.*\})", sc).group(1))
their_gate = re.search(r"GATE \(stage 2\): (\S+)", sc).group(1)
their_p0 = re.search(r"P0: (\w+)", sc).group(1)
print(f"identical to score.md: verdicts {theirs == mine}, gate {their_gate == gate}, P0 {their_p0 == ('PASS' if p0 else 'FAIL')}")
