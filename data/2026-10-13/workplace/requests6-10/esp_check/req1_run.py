"""Request 1 (2026-10-13): a14's estimate on all HW1 circuits; exact (a14 numpy <= 22 active; Aer for all) and rule R6/16."""
import json, math, os, sys, time, warnings
warnings.simplefilter("ignore")
REPO, A14 = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hw1_check.py")).read().split("R = {}")[0]
exec(src)  # builds rel, a14, target t (FakeKingston structure + the day's calibration), man, sc, circs
import est_lc
import numpy as np
R = {}
for p, c in zip(man["pubs"], circs):
    e = man["entries"][p["entry"]]
    if e["part"] != "HW1" or p["repeat"] != 1:
        continue
    key = (e["test"], e["arm"])
    row = dict(active=e["active"], esp=e["esp"], F=sc["F"][f"{e['test']}|{e['arm']}"]["f"])
    t0 = time.time()
    if e["active"] <= 22:
        pos = {q: i for i, q in enumerate(c.qubits)}
        ident = {q: q for q in {pos[q] for i in c.data for q in i.qubits}}
        g = a14._score_weights(a14._state_weights(c), t, ident)
        iw, T = a14.idle_weights(c, t)
        row["np"] = (g, a14.idle_cost(iw, t, ident))
    t1 = time.time()
    row["aer"] = est_lc.estimate(c, t, est_lc.states_aer(c), a14)[:2]
    t2 = time.time()
    st, stats = est_lc.states_rule(c)
    row["rule"] = est_lc.estimate(c, t, st, a14)[:2]
    row["rule_stats"] = stats
    R[key] = row
    print(f"{e['test']:12s} {e['arm']:4s} act {e['active']:2d}  np {row.get('np')}  aer ({row['aer'][0]:.5f}, {row['aer'][1]:.5f})"
          f"  rule ({row['rule'][0]:.5f}, {row['rule'][1]:.5f}) {stats}  [{t1-t0:.0f}s/{t2-t1:.0f}s/{time.time()-t2:.0f}s]", flush=True)
json.dump({f"{k[0]}|{k[1]}": v for k, v in R.items()}, open(sys.argv[3], "w"), indent=1, default=float)
