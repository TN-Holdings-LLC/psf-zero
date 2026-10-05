import json, glob, numpy as np
import sys
ARMS = ("RPSF", "C13", sys.argv[2] if len(sys.argv) > 2 else "A11", "L3TM")
for p in sorted(glob.glob((sys.argv[1] if len(sys.argv) > 1 else "out") + "/ecr_*.json")):
    d = json.load(open(p)); rows = d["rows"]
    print(d["device"], len(rows))
    errs = [(r["name"], a, r[a]["error"]) for r in rows for a in ARMS if "error" in r[a]]
    if errs: print("  ERRORS", len(errs), errs[:3])
    for a in ARMS:
        ok = [r[a] for r in rows if "error" not in r[a]]
        print(f"  {a:5s} infid {np.mean([x['infid'] for x in ok]):.5f}  meas_err {np.mean([x['meas_err'] for x in ok]):.4f}  2q {np.mean([x['n2q'] for x in ok]):.2f}"
              f"  max state_infid {max(x['state_infid'] for x in ok):.1e}  off {sum(x['off_target'] for x in ok)}  failed {sum(x['failed_uses'] for x in ok)}  med s {np.median([x['compile_s'] for x in ok]):.3f}")
    for a in [x for x in ARMS if x not in ("RPSF", "L3TM")]:
        print(f"  {a}/L3TM {np.mean([r[a]['infid'] for r in rows])/np.mean([r['L3TM']['infid'] for r in rows]):.3f}  {a}/RPSF {np.mean([r[a]['infid'] for r in rows])/np.mean([r['RPSF']['infid'] for r in rows]):.3f}"
              f"  per-circuit {a}<=L3TM {np.mean([r[a]['infid'] <= r['L3TM']['infid'] + 1e-12 for r in rows]):.2f}")
    # by family
    fam = {}
    for r in rows:
        f = ''.join(c for c in r['name'] if c.isalpha())
        fam.setdefault(f, []).append(r)
    print("  by family A-front/L3TM:", {f: round(np.mean([r[ARMS[2]]['infid'] for r in v]) / np.mean([r['L3TM']['infid'] for r in v]), 3) for f, v in fam.items()})
    print("  by family C13/L3TM:", {f: round(np.mean([r['C13']['infid'] for r in v]) / np.mean([r['L3TM']['infid'] for r in v]), 3) for f, v in fam.items()})
