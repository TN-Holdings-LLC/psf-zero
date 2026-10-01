"""a0_verify.py -- independent check of the A0 score (written after the run). Does not import a0_target_check.py.
Recomputes every floor from Kraus operators (not the closed form), re-derives the below-floor sets, class
statistics and H1-H5/R1-R2 from the raw rows.   python a0_verify.py <a0_raw.json.gz>"""
import gzip, json, math, sys
import numpy as np

def kraus1(t1, t2, t):
    t1 = math.inf if t1 is None else t1
    t2 = 2 * t1 if t2 is None else min(t2, 2 * t1)
    g = 1 - math.exp(-t / t1)
    s = math.exp(-t / t2) / math.exp(-t / (2 * t1))
    A = [np.array([[1, 0], [0, math.sqrt(1 - g)]]), np.array([[0, math.sqrt(g)], [0, 0]])]
    p = (1 + s) / 2
    D = [math.sqrt(p) * np.eye(2), math.sqrt(1 - p) * np.diag([1.0, -1.0])]
    return [d @ a for d in D for a in A]

def floor(t1s, t2s, dur):
    if not dur:
        return 0.0
    K = [np.eye(1)]
    for a, b in zip(t1s, t2s):
        K = [np.kron(k, m) for k in K for m in kraus1(a, b, dur)]
    d = K[0].shape[0]
    return 1 - (sum(abs(np.trace(k)) ** 2 for k in K) + d) / (d * (d + 1))

res = json.load(gzip.open(sys.argv[1], "rt"))
rows = res["rows"]
dmax = max(abs(floor(r["t1"], r["t2"], r["dur"]) - r["floor"]) for r in rows)
print(f"rows {len(rows)}, devices {len(res['devices'])}; Kraus floor vs stored max diff {dmax:.1e}")
med_t2 = {d["dev"]: d["median_t2"] for d in res["devices"]}
two = {}
for d in res["devices"]:
    g = [x for x in ("cx", "ecr", "cz") if x in d["two_q"]]
    two[d["dev"]] = g[0] if len(g) == 1 else ("mixed" if g else "none")
ok = lambda r: r["err"] is not None and 0 < r["err"] < 0.5 and r["dur"] and None not in r["t1"] and None not in r["t2"]
per = {}
for r in rows:
    if r["op"] in ("cx", "ecr", "cz") and ok(r):
        per.setdefault(r["dev"], []).append(r)
frac = {k: sum(r["err"] < r["floor"] for r in v) / len(v) for k, v in per.items()}
for c in ("cx", "ecr", "cz", "mixed"):
    ks = [k for k in per if two[k] == c]
    n = sum(len(per[k]) for k in ks); b = sum(sum(r["err"] < r["floor"] for r in per[k]) for k in ks)
    print(f"{c:5s} devices {len(ks):2d} median {np.median([frac[k] for k in ks]):.3f} pooled {b}/{n} = {b/n:.3f}")
cz_wo = [frac[k] for k in per if two[k] == "cz" and k != "fake_nighthawk"]
print(f"cz without nighthawk: median {np.median(cz_wo):.3f} ({len(cz_wo)} devices)")
below = [(k, r) for k in per for r in per[k] if r["err"] < r["floor"]]
print(f"H4: {sum(min(r['t2']) < med_t2[k] for k, r in below) / len(below):.3f} of {len(below)}")
for k in ("fake_auckland", "fake_torino", "fake_kingston"):
    print(k, sum(r["err"] < r["floor"] for r in per[k]), "of", len(per[k]))
# largest ratios
top = sorted(((r["floor"] / r["err"], k, r["op"], r["q"], r["err"], r["floor"]) for k in per for r in per[k]), reverse=True)[:5]
for t in top:
    print(f"ratio {t[0]:.2f} {t[1]} {t[2]}{t[3]} reported {t[4]:.2e} floor {t[5]:.2e}")
# Kingston below rows: T2 of qubits
kb = [r for r in per["fake_kingston"] if r["err"] < r["floor"]]
print("kingston below: min T2 us", sorted(round(min(r["t2"]) * 1e6) for r in kb), "device median", round(med_t2["fake_kingston"] * 1e6))
