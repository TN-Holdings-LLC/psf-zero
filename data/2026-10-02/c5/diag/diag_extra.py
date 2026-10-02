"""diag_extra.py -- second view on region_diag.py's output (Addendum 308, exploratory): per device, which arm has the
lowest score under each scoring, and the mean ratio S_avg / S_rep (how far Qiskit's averaged map departs from
the reported per-gate errors).   python diag_extra.py <dir with diag_*.json>"""
import json
import os
import sys

import numpy as np

ARMS = ("C3", "C4", "L3T")
for d in ("FakeAuckland", "FakeTorino", "FakeKingston"):
    rows = json.load(open(os.path.join(sys.argv[1], "diag_%s.json" % d)))["rows"]
    print("## %s (%d circuits)" % (d, len(rows)))
    print("C4 lower than C3 by: S_avg %d, S_rep %d, S_eff %d, measured infidelity %d" % tuple(
        sum(r["C4"][k] < r["C3"][k] for r in rows) for k in ("S_avg", "S_rep", "S_eff", "infid")))
    for k in ("S_avg", "S_rep", "S_eff", "infid"):
        low = {a: 0 for a in ARMS}
        for r in rows:
            low[min(ARMS, key=lambda a: r[a][k])] += 1
        print("lowest arm by %-6s %s" % (k, low))
    print("mean S_avg / S_rep:", {a: round(float(np.mean([r[a]["S_avg"] / max(r[a]["S_rep"], 1e-12) for r in rows])), 2)
                                  for a in ARMS})
    print()
