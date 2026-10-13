"""Exploratory (2026-10-13): does ESP carry any signal for which compilation of the same circuit does better on the
device? All usable HW1 points, analysed within each test (arms compared only with arms of the same test).
Uses data/2026-10-12/hw1 (scores, committed circuits) and req1_out.json (a14 estimates, exact)."""
import csv, itertools, json, math, os, sys
import numpy as np
from qiskit import qpy
REPO, REQ1 = sys.argv[1], sys.argv[2]
W = os.path.join(REPO, "data/2026-10-12/hw1")
sc = json.load(open(W + "/hw1_score.json"))["F"]
man = json.load(open(W + "/hw1_manifest.json"))
circs = qpy.load(open(W + "/hw1_circuits.qpy", "rb"))
A = json.load(open(REQ1))
CAL = os.path.join(REPO, "data/2026-10-12/hw_analysis/ibm_kingston_calibrations_2026-10-12T11_15_24Z.csv")
T2 = {}
for r in csv.DictReader(open(CAL, encoding="utf-8")):
    try: T2[int(r["Qubit"])] = float(r["T2 (us)"])
    except ValueError: T2[int(r["Qubit"])] = 1.0
expo = {}
for p, c in zip(man["pubs"], circs):
    e = man["entries"][p["entry"]]
    if e["part"] != "HW1" or p["repeat"] != 1: continue
    pos = {q: i for i, q in enumerate(c.qubits)}; clock, first = {}, {}
    for ins in c.data:
        nm = ins.operation.name; qs = [pos[q] for q in ins.qubits]
        if nm in ("barrier", "measure"): continue
        d = 0.0 if nm == "rz" else (0.068 if len(qs) == 2 else 0.032)
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs: first.setdefault(q, s); clock[q] = s + d
    T = max(clock.values()); expo[f"{e['test']}|{e['arm']}"] = sum((T - first[q]) / T2[q] for q in first)
keys = [k for k in sc if k.split("|")[0] not in ("tfim_n16", "ghz_n12")]
tests = sorted({k.split("|")[0] for k in keys})
groups = {t: [k for k in keys if k.startswith(t + "|")] for t in tests}
logF = {k: math.log(sc[k]["f"]) for k in keys}
pred = {"ESP": {k: math.log(sc[k]["esp"]) for k in keys},
        "a14 (gate+readout+idle)": {k: -sum(A[k]["aer"]) for k in keys},
        "exposure (window/T2)": {k: -expo[k] for k in keys}}
def demean(d):
    out = {}
    for t, ks in groups.items():
        m = np.mean([d[k] for k in ks])
        for k in ks: out[k] = d[k] - m
    return out
yF = demean(logF)
print(f"points: {len(keys)} in {len(tests)} tests; within-test degrees of freedom {len(keys) - len(tests)}")
# measurement noise: repeat-to-repeat spread of log F, against the within-test spread of log F
noise = np.mean([((math.log(sc[k]['reps'][0]) - math.log(sc[k]['reps'][1])) / 2) ** 2 for k in keys])
signal = np.mean([v * v for v in yF.values()])
print(f"within-test variance of log F {signal:.5f}; measurement-noise variance of a mean of two repeats {noise:.5f}"
      f" -> share that is real differences {max(0, 1 - noise / signal):.2f}")
perms = [list(itertools.permutations(groups[t])) for t in tests]
for name, d in pred.items():
    x = demean(d)
    ks = keys
    r = float(np.corrcoef([x[k] for k in ks], [yF[k] for k in ks])[0, 1])
    null = []
    for combo in itertools.product(*perms):
        mp = {}
        for t, perm in zip(tests, combo):
            for a, b in zip(groups[t], perm): mp[a] = b
        null.append(float(np.corrcoef([x[mp[k]] for k in ks], [yF[k] for k in ks])[0, 1]))
    null = np.array(null)
    p_pos = float(np.mean(null >= r - 1e-12)); p_two = float(np.mean(np.abs(null) >= abs(r) - 1e-12))
    n_eff = len(keys) - len(tests) + 3  # Fisher z with the within-test degrees of freedom
    se = 1 / math.sqrt(max(1, n_eff - 3)); z = math.atanh(max(-0.999, min(0.999, r)))
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    print(f"{name:26s} within-test r = {r:+.2f}; permutation p(r >= obs) = {p_pos:.3f}, two-sided {p_two:.3f};"
          f" approx 95% range {lo:+.2f} to {hi:+.2f}  ({len(null)} permutations)")

# ---- the same results as probabilities (added 2026-10-13, reproduces the "Addition" table of
# esp-random-check-2026-10-13.md). Flat prior on Fisher z (se = 1/sqrt(11), the within-test degrees of freedom),
# Bayes factor with rho uniform on (0, 1) against rho = 0, replication = a new test of 25 pairs, hit rate over all
# arm pairs within a test with a uniform prior.
from scipy.stats import norm, beta
se = 1 / math.sqrt(len(keys) - len(tests))
print("\nprobabilities (within-test r as above):")
for name, d in pred.items():
    x = demean(d)
    r = float(np.corrcoef([x[k] for k in keys], [yF[k] for k in keys])[0, 1])
    z = math.atanh(r)
    post = lambda thr: 1 - norm.cdf((math.atanh(thr) - z) / se)
    grid = np.linspace(1e-4, 0.999, 20000)
    bf = float(np.exp(-(z - np.arctanh(grid)) ** 2 / (2 * se * se)).mean() / math.exp(-z * z / (2 * se * se)))
    p_rep = 1 - norm.cdf(-z / math.sqrt(se * se + 1 / (25 - 3)))
    hits = n = 0
    for t in tests:
        for a, b in itertools.combinations(groups[t], 2):
            n += 1
            hits += (d[a] > d[b]) == (logF[a] > logF[b])
    bb = beta(1 + hits, 1 + n - hits)
    print(f"{name:26s} P(rho>0)={post(1e-9):.3f} P(rho>0.2)={post(0.2):.3f} BF+0={bf:.2f} "
          f"P(25-pair replication r>0)={p_rep:.2f}; winner right {hits}/{n}, P(hit rate>0.6)={1 - bb.cdf(0.6):.3f}")
