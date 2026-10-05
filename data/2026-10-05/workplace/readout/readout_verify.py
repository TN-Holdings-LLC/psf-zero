"""readout_verify.py -- independent re-check of the READOUT score (written after the lock, during the run, before any
scored output was read). Raw JSON only; does not import readout_eval.py.   python readout_verify.py <out>"""
import glob, json, os, sys
import numpy as np
out = sys.argv[1]
R = {json.load(open(p))["meta"]["device"]: json.load(open(p)) for p in glob.glob(os.path.join(out, "readout_*.json"))}
assert not any(d["meta"]["dry"] for d in R.values())
print("shas", {k: {d["meta"][k] for d in R.values()} for k in ("script_sha", "c12_sha", "c13_sha", "depth_eval_sha")})
M = ("C12", "C12M", "C13M", "L3TM")
rows = [r for d in R.values() for r in d["rows"]]
print("rows", {k: len(d["rows"]) for k, d in R.items()})
inf = max(r[a]["infid"] for r in rows for a in M)
mok = all(r[a]["measured"] == [r[a]["fin0"]] for r in rows for a in M[1:])
print("P0", "PASS" if len(R) == 3 and inf <= 1e-6 and mok else "FAIL", f"{inf:.2e}", mok)
m = lambda dev, a, k: float(np.mean([r[a][k] for r in R[dev]["rows"]]))
med = lambda dev, a: float(np.median([r[a]["compile_s"] for r in R[dev]["rows"]]))
T = "FakeTorino"
print("R1 ratio %.4f" % (m(T, "C12M", "meas_err") / m(T, "C12", "meas_err")))
print("R2", {d: round(m(d, "C13M", "meas_err") - m(d, "C12M", "meas_err"), 6) for d in R})
print("R3 %+.4f" % (m(T, "C13M", "eff_margin") - m(T, "C12", "eff_margin")))
print("R4", {d: round(m(d, "C13M", "eff_margin") - m(d, "L3TM", "eff_margin"), 4) for d in R})
print("R5 %.4f" % np.mean([r["c13_same_as_c12"] for r in rows]))
print("R6 max %.3f" % max(med(d, "C13M") / med(d, "C12M") for d in R))
print("R7 %+.4f" % (m(T, "C13M", "acc32") - m(T, "C12", "acc32")))
