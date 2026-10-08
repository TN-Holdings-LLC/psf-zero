"""bp_final_verify.py -- independent re-computation of BP-FINAL's verdicts, written before the scored run. Reads the
raw JSON only, imports no harness, and compares with score.md.
    python benchmarks/bp_final_verify.py <out dir>"""
import json
import math
import os
import re
import sys

import numpy as np

out = sys.argv[1]
fn = os.path.join(out, "bp_final.json")
smoke = not os.path.exists(fn)
d = json.load(open(fn if not smoke else os.path.join(out, "bp_final_smoke.json"), encoding="utf-8"))
m, rows = d["meta"], d["rows"]
T = {}
for x in rows:
    T.setdefault(x["test"], {})[x["arm"]] = x
good = lambda x: "error" not in x  # noqa: E731


def arms(s):
    return {"QK", "REL", "REL2"} | ({"RECR"} if s.endswith("FakeTorino") else set())


p0 = (len(T) == (18 if smoke else 880) and all(set(v) == arms(v["QK"]["stratum"]) for v in T.values())
      and all(x.get("version") == "2026-10-07.1" for x in rows if x["arm"] != "QK" and good(x))
      and m["benchpress"] == "b695f30" and (smoke or not m["dirty_tracked"])
      and m["sha"]["release"] == "73fb2cb0b1acc5870339c23599829b326fbf57aa198945231e8551e55c1884dc"
      and m["sha"]["published_ref"] == "d05231a75695c4fa9cbe2f09891ae8a835db32699e381c4d8de4b2afaaf56877")
r = lambda t, a, b: (T[t][a]["q2"] + 1) / (T[t][b]["q2"] + 1)  # noqa: E731
done = [t for t, v in T.items() if good(v["QK"]) and good(v["REL"])]
logs = np.array([math.log(r(t, "REL", "QK")) for t in done])
g = float(np.exp(logs.mean()))
fams = {"QASMBench": [], "HamLib, abstract": [], "HamLib, FakeTorino": [], "Feynman, FakeTorino": []}
for t in done:
    s = T[t]["QK"]["stratum"]
    k = ("QASMBench" if s.startswith("QASMBench") else s if s in ("HamLib, FakeTorino", "Feynman, FakeTorino")
         else "HamLib, abstract")
    fams[k].append(math.log(r(t, "REL", "QK")))
gf = {k: math.exp(sum(v) / len(v)) for k, v in fams.items() if v}
share = sum(1 for t in done if r(t, "REL", "QK") > 1.10) / len(done)
psf = [T[t][a] for t in T for a in ("REL", "REL2", "RECR") if a in T[t] and good(T[t][a])]
invalid = sum(1 for x in psf if x["valid"] is not True)
chk = [x for x in psf if x.get("checkable") is True and isinstance(x.get("implements"), bool)]
wrong = sum(1 for x in chk if not x["implements"])
recr = [T[t]["RECR"] for t in T if "RECR" in T[t] and good(T[t]["RECR"])]
rf = sum(1 for x in recr if x.get("on_failed", 0) > 0)
qk_ok = [t for t in T if good(T[t]["QK"])]
fs = sum(1 for t in qk_ok if not good(T[t]["REL"])) / len(qk_ok)
v = lambda ok, bad: "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")  # noqa: E731
mine = {"F1": v(g <= 1.06, g > 1.10), "F2": v(all(x <= 1.15 for x in gf.values()), any(x > 1.25 for x in gf.values())),
        "F3": v(share <= 0.15, share > 0.25), "F4": v(not invalid and not wrong and len(chk) >= 20, bool(invalid or wrong)),
        "F5": v(bool(recr) and not rf, bool(rf)), "F6": v(fs <= 0.01, fs > 0.03)}
print(f"smoke {smoke}; P0 {'PASS' if p0 else 'FAIL'}; tests {len(T)}, done {len(done)}; F1 {g:.4f}; families "
      + ", ".join(f"{k} {x:.3f}" for k, x in gf.items()) + f"; share > 1.10 {share:.3f}; invalid {invalid}; "
      f"not implementing {wrong} of {len(chk)}; RECR on failed {rf} of {len(recr)}; REL failures {fs:.3f}")
print("mine", mine)
sc = open(os.path.join(out, "score.md"), encoding="utf-8").read()
if not p0:
    print("identical to score.md:", ("P0 FAIL" in sc))
    sys.exit()
mine.update(P0="PASS", F1_value=round(g, 6))
theirs = json.loads(re.search(r"SUMMARY (\{.*\})", sc).group(1))
same = {k: theirs[k] for k in theirs if k != "F1_value"} == {k: mine[k] for k in mine if k != "F1_value"} \
    and abs(theirs["F1_value"] - mine["F1_value"]) < 1e-9
print(f"identical to score.md: {same}")
