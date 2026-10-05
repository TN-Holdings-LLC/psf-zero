"""ai10_verify.py -- re-computation of the MODEL-RO table from the raw JSON (written AFTER the locked score was seen;
it checks arithmetic, it is not blind). python ai10_verify.py <out>"""
import glob, json, sys
import numpy as np
R = {json.load(open(p))["meta"]["device"]: json.load(open(p)) for p in glob.glob(sys.argv[1] + "/ai10_*.json")}
for d, x in sorted(R.items()):
    rr = x["rows"]
    m = {a: (np.mean([r[a]["infid"] for r in rr]), np.mean([r[a]["meas_err"] for r in rr])) for a in ("A9", "A10", "L3TM")}
    print(d, len(rr), " ".join(f"{a} inf {v[0]:.5f} ro {v[1]:.5f}" for a, v in m.items()),
          "A10<=A9 %.3f" % np.mean([r["A10"]["infid"] <= r["A9"]["infid"] + 1e-12 for r in rr]),
          "maxTV", {a: "%.1e" % max(r[a]["tv_noiseless"] for r in rr) for a in ("A9", "A10", "L3TM")})
