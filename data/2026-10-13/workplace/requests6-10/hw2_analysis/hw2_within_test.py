"""hw2_within_test.py -- HW2 analysis plan of the workplace (request 8 of 2026-10-13), fixed before HW2's results exist.

The same framework as the workplace's ESP check of HW1 (esp_random_check.py): pairs within a test, exact permutation
tests, Bayesian probabilities and a Bayes factor. Nothing here changes HW2's own verdicts (hw2_real.py score, R1-R6,
S1-S4) or HW2-PLUS's (Q1-Q4); it adds a common, pre-fixed way of stating how strong each predictor's evidence is.

    python hw2_within_test.py HW2_DIR [PREDICTORS_JSONL] [--out OUT.json]

HW2_DIR holds hw2_score.json and hw2_manifest.json (as written by data/2026-10-13/hw2_kit/hw2_real.py).
PREDICTORS_JSONL is hw2_predictors.jsonl (benchmarks/hw2_predictors.py, Addendum 466) for P1-P4; without it only ESP
and R3 are analysed.

Predictors (higher = predicted better), one number per circuit:
  ESP   manifest/score "esp"                       (HW2's reference, R1/R2)
  R3    manifest/score "idle" (idle-aware estimate, HW2 section 2)
  P1-P4 hw2_predictors.jsonl, first element of each value (P1/P2 simulated F, P3 exp(-cost), P4 exp(-exposure))

Scopes, fixed now:
  A (primary)   part R, HW2's seven reading tests, PSF vs QK3 (as R2/R3 and Q1)
  B (reported)  every part R pair with F > 0 for both arms except ghz_n12 and tfim_n16, plus every part S pair
                (S1 vs S0), each pair counted as its own test
A pair enters a scope for a predictor only if both arms have F > 0 and both predictor values are > 0.

Per predictor and scope:
  winners       the predicted winner (ratio > 1) equals the device's winner (F ratio > 1); count of n
  hit rate      Beta(1 + hits, 1 + n - hits): P(rate > 0.5), P(rate > 0.6), 95% interval
  Spearman      rank correlation of predicted ratio and F ratio over the pairs (average ranks for ties);
                exact permutation test over all n! orderings (n <= 9; otherwise 200,000 random ones, seed 466),
                one-sided P(rho_null >= rho) and two-sided
  within-test r correlation of log(predicted ratio) and log(F ratio) through the origin (= the within-test correlation
                of demeaned logs with two arms per test); exact sign-flip permutation test (2^n)
  Bayesian      flat prior on the Fisher z of the Spearman correlation, standard error 1.06 / sqrt(n - 3):
                P(rho > 0), P(rho > 0.2); Bayes factor BF+0 = mean likelihood over rho uniform on (0, 1) / likelihood at 0
                (not computed for n < 4)
  noise share   1 - (mean repeat variance of a two-repeat mean of log F) / (mean square of log F ratio / 2), scope A
"""
import itertools
import json
import math
import os
import sys

import numpy as np

READING = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12")
SKIP_B = ("ghz_n12", "tfim_n16")
PREDICTORS = ("ESP", "R3", "P1", "P2", "P3", "P4")
SEED = 466
N_RANDOM_PERM = 200_000


def avg_ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    a, b = avg_ranks(x), avg_ranks(y)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def perm_spearman(x, y):
    n = len(x)
    obs = spearman(x, y)
    if math.isnan(obs):
        return obs, float("nan"), float("nan")
    ry = avg_ranks(y)
    rx = avg_ranks(x)
    if n <= 9:
        null = np.array([np.corrcoef(rx, ry[list(p)])[0, 1] for p in itertools.permutations(range(n))])
    else:
        rng = np.random.default_rng(SEED)
        null = np.array([np.corrcoef(rx, ry[rng.permutation(n)])[0, 1] for _ in range(N_RANDOM_PERM)])
    return obs, float(np.mean(null >= obs - 1e-12)), float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def within_r(lx, ly):
    lx, ly = np.asarray(lx), np.asarray(ly)
    den = math.sqrt(float(np.dot(lx, lx)) * float(np.dot(ly, ly)))
    if den == 0 or len(lx) < 2:
        return float("nan"), float("nan")
    obs = float(np.dot(lx, ly)) / den
    null = []
    for s in itertools.product((1.0, -1.0), repeat=len(lx)):
        null.append(float(np.dot(lx * np.array(s), ly)) / den)
    return obs, float(np.mean(np.array(null) >= obs - 1e-12))


def bayes(rho, n):
    if n < 4 or math.isnan(rho):
        return dict(p_pos=float("nan"), p_02=float("nan"), bf_plus0=float("nan"))
    from scipy.stats import norm
    se = 1.06 / math.sqrt(n - 3)
    z = math.atanh(max(-0.999, min(0.999, rho)))
    grid = np.linspace(1e-4, 0.999, 20000)
    lik = np.exp(-(z - np.arctanh(grid)) ** 2 / (2 * se * se))
    return dict(p_pos=float(1 - norm.cdf((0 - z) / se)), p_02=float(1 - norm.cdf((math.atanh(0.2) - z) / se)),
                bf_plus0=float(lik.mean() / math.exp(-z * z / (2 * se * se))))


def hit_rate(h, n):
    from scipy.stats import beta
    if n == 0:
        return dict(p_gt_05=float("nan"), p_gt_06=float("nan"), ci=(float("nan"), float("nan")))
    b = beta(1 + h, 1 + n - h)
    return dict(p_gt_05=float(1 - b.cdf(0.5)), p_gt_06=float(1 - b.cdf(0.6)), ci=(float(b.ppf(0.025)), float(b.ppf(0.975))))


def load(hw2_dir, pred_path):
    sc = json.load(open(os.path.join(hw2_dir, "hw2_score.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(hw2_dir, "hw2_manifest.json"), encoding="utf-8"))
    vals = {}  # (part, test, arm) -> {"f", "reps", "ESP", "R3", "P1".. }
    submitted = {(e["part"], e["test"], e["arm"]) for e in man["entries"] if e.get("submitted")}
    for k, v in sc["F"].items():
        part, test, arm = k.split("|")
        if (part, test, arm) not in submitted:
            continue
        vals[(part, test, arm)] = dict(f=v["f"], reps=v.get("reps", []), ESP=v["esp"], R3=v["idle"])
    if pred_path:
        for line in open(pred_path, encoding="utf-8"):
            r = json.loads(line)
            key = (r["part"], r["test"], r["arm"])
            if key in vals:
                for P in ("P1", "P2", "P3", "P4"):
                    if P in r and r[P] is not None and r[P][0] is not None:
                        vals[key][P] = float(r[P][0])
    return vals, man


def pairs_for(vals, scope):
    out = []
    if scope == "A":
        out = [(t, ("R", t, "PSF"), ("R", t, "QK3")) for t in READING]
    else:
        tests_r = sorted({t for (p, t, a) in vals if p == "R" and t not in SKIP_B})
        out = [(f"R:{t}", ("R", t, "PSF"), ("R", t, "QK3")) for t in tests_r]
        tests_s = sorted({t for (p, t, a) in vals if p == "S"})
        out += [(f"S:{t}", ("S", t, "S1"), ("S", t, "S0")) for t in tests_s]
    return [(name, a, b) for name, a, b in out if a in vals and b in vals and vals[a]["f"] > 0 and vals[b]["f"] > 0]


def analyse(vals, scope, P):
    rows = []
    for name, a, b in pairs_for(vals, scope):
        if P in vals[a] and P in vals[b] and vals[a][P] > 0 and vals[b][P] > 0:
            rows.append((name, vals[a][P] / vals[b][P], vals[a]["f"] / vals[b]["f"]))
    n = len(rows)
    x = [r[1] for r in rows]
    y = [r[2] for r in rows]
    hits = sum(1 for u, v in zip(x, y) if (u > 1) == (v > 1))
    rho, p1, p2 = perm_spearman(x, y) if n >= 3 else (float("nan"),) * 3
    wr, wp = within_r([math.log(u) for u in x], [math.log(v) for v in y]) if n >= 2 else (float("nan"),) * 2
    return dict(scope=scope, predictor=P, n=n, pairs=rows, winners=hits, hit_rate=hit_rate(hits, n), spearman=rho,
                perm_p_one=p1, perm_p_two=p2, within_r=wr, within_p_one=wp, **bayes(rho, n))


def noise_share(vals):
    num, den = [], []
    for name, a, b in pairs_for(vals, "A"):
        ra, rb = vals[a]["reps"], vals[b]["reps"]
        if len(ra) == 2 and len(rb) == 2 and min(ra + rb) > 0:
            num += [((math.log(ra[0]) - math.log(ra[1])) / 2) ** 2, ((math.log(rb[0]) - math.log(rb[1])) / 2) ** 2]
            den.append((math.log(vals[a]["f"] / vals[b]["f"])) ** 2 / 2)
    if not den:
        return float("nan")
    return float(1 - np.mean(num) / np.mean(den))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    hw2_dir = args[0]
    pred = args[1] if len(args) > 1 else None
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(hw2_dir, "hw2_within_test.json")
    vals, man = load(hw2_dir, pred)
    res = dict(device=man.get("device"), job_id=man.get("job_id"), noise_share_A=noise_share(vals), results=[])
    print(f"device {res['device']}; job {res['job_id']}; scope A: share of the F-ratio variance that is not repeat "
          f"noise {res['noise_share_A']:.2f}")
    print("| scope | predictor | n | winners | P(hit>0.6) | Spearman | perm p (one-sided) | within r (p) | P(rho>0) | "
          "P(rho>0.2) | BF+0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for scope in ("A", "B"):
        for P in PREDICTORS:
            r = analyse(vals, scope, P)
            if r["n"] == 0:
                continue
            res["results"].append(r)
            print(f"| {scope} | {P} | {r['n']} | {r['winners']} | {r['hit_rate']['p_gt_06']:.2f} | {r['spearman']:+.2f} | "
                  f"{r['perm_p_one']:.3f} | {r['within_r']:+.2f} ({r['within_p_one']:.3f}) | {r['p_pos']:.2f} | "
                  f"{r['p_02']:.2f} | {r['bf_plus0']:.2f} |")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, default=float)
    print("written", out)


if __name__ == "__main__":
    main()
