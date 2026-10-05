"""depth_verify.py -- independent re-check of the DEPTH stage-1 score (written after the lock, during the run, before
the deployment outputs were read). Reads the raw JSON only; does not import depth_eval.py.
  python depth_verify.py <out dir>"""
import glob, json, os, sys
import numpy as np

out = sys.argv[1]
D = {}
for p in glob.glob(os.path.join(out, "deploy_*.json")):
    d = json.load(open(p)); m = d["meta"]
    assert not m["dry"], p
    D[(m["dataset"], m["n"], m["device"], m["arm"])] = d["rows"]
F = [json.load(open(p)) for p in glob.glob(os.path.join(out, "finetune_*.json"))]
shas = {json.load(open(p))["meta"]["script_sha"] for p in glob.glob(os.path.join(out, "*.json"))}
c12 = {json.load(open(p))["meta"].get("c12_sha") for p in glob.glob(os.path.join(out, "*.json"))} - {None}
print("script sha", shas, "c12 sha", c12)
Ls = sorted({r["L"] for rows in D.values() for r in rows})


def agg(ds, n, dev, arm, L):
    rows = [r for r in D[(ds, n, dev, arm)] if r["L"] == L]
    y = np.array([r["y"] for r in rows]); zn = np.array([r["z_noisy"] for r in rows]); zi = np.array([r["z_ideal"] for r in rows])
    return dict(n=len(rows), margin=float(np.mean(y * zn)), shot=float(np.mean([r["shot_acc"] for r in rows])),
                flip=float(np.mean(np.sign(zn) != np.sign(zi))))


exact = max(abs(r["z_compiled_noiseless"] - r["z_ideal"]) for rows in D.values() for r in rows)
fw = max(abs(r["z_fullwidth"] - r["z_noisy"]) for rows in D.values() for r in rows if "z_fullwidth" in r)
print(f"P0: files {len(D)}/24, finetune {len(F)}/4, exact {exact:.2e}, reduced-vs-whole {fw:.2e} ->",
      "PASS" if len(D) == 24 and len(F) == 4 and exact <= 1e-6 and fw <= 1e-9 else "FAIL")
A, DS, ARMS, DEV = "FakeAuckland", ("BC", "D38"), ("RPSF", "C12", "L3T"), ("FakeAuckland", "FakeTorino")
ok = bad = None
c1 = [agg(ds, 6, A, a, Ls[-1])["margin"] < 0.8 * max(agg(ds, 6, A, a, L)["margin"] for L in Ls) for ds in DS for a in ARMS]
b1 = [max(Ls, key=lambda L: agg(ds, 6, A, a, L)["margin"]) == Ls[-1] for ds in DS for a in ARMS]
print("H1", "CONFIRMED" if all(c1) else ("REFUTED" if any(b1) else "AMBIGUOUS"))
c2 = []; b2 = []
for ds in DS:
    nt = agg(ds, 6, A, "C12", Ls[0])["n"]
    c2.append(all(max(agg(ds, 6, A, a, L)["shot"] for L in Ls) - agg(ds, 6, A, a, Ls[-1])["shot"] >= 2 / nt - 1e-12 for a in ARMS))
    b2 += [agg(ds, 6, A, a, Ls[-1])["shot"] >= max(agg(ds, 6, A, a, L)["shot"] for L in Ls) - 1e-12 for a in ARMS]
print("H2", "CONFIRMED" if any(c2) else ("REFUTED" if all(b2) else "AMBIGUOUS"))
for dev in DEV:
    pool = lambda a, k: float(np.mean([agg(ds, n, dev, a, L)[k] for ds in DS for n in (4, 6) for L in Ls]))
    print(dev, "C12-RPSF margin %+.4f" % (pool("C12", "margin") - pool("RPSF", "margin")),
          "C12-L3T %+.4f" % (pool("C12", "margin") - pool("L3T", "margin")),
          "flip C12 %.4f RPSF %.4f" % (pool("C12", "flip"), pool("RPSF", "flip")))
f12 = [f["result"] for f in F if f["meta"]["L"] == 12]
print("H6 FTN-DEP %+.4f FT0-DEP %+.4f" % (np.mean([r["FTN"]["margin"] - r["DEP"]["margin"] for r in f12]),
                                         np.mean([r["FT0"]["margin"] - r["DEP"]["margin"] for r in f12])))
