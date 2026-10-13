"""hw2_holm.py -- one multiplicity-corrected reading of every predictor scored on HW2 (Addendum 472), fixed before HW2
ran. Nine predictors are scored on HW2's seven reading pairs (PSF against QK3):

  ESP, R3                 hw2_score.json (Addendum 460)
  NS2                     ns_hw2.jsonl, f_sim (Addendum 462)
  P1, P2, P3, P4          hw2_predictors.jsonl (Addendum 466)
  P1c, P2c                hw2_predictors_c.jsonl (Addendum 468)

For each: the Spearman correlation of predicted and measured ratios and its one-sided exact permutation p, computed
with the workplace's fixed functions (hw2_within_test.py, Addendum 467, imported unchanged). Holm's step-down
correction over the nine p values; a predictor whose own data are missing counts with p = 1.

Reading (fixed): a predictor "works on HW2" only if its Holm-adjusted p <= 0.05. Each tool's own verdicts (R1-R6,
NS2a-d, Q1-Q4, Q2c, the workplace's "strong / weak / none") stay as they are and are reported beside this one.

    python benchmarks/hw2_holm.py <hw2 dir> <ns2 out>/ns_hw2.jsonl <pred out>/hw2_predictors.jsonl \
        <predc out>/hw2_predictors_c.jsonl --out <file.json>
Any of the three jsonl paths may be given as '-' when that tool has not run (its predictors then count with p = 1)."""
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.join(HERE, "..", "data", "2026-10-13", "workplace", "requests6-10", "hw2_analysis", "hw2_within_test.py")
WT_SHA = "889e1fc3b60cf8aacea0ec892b007cb246a42491ed029bc04700e23995c5f28f"
ORDER = ("ESP", "R3", "NS2", "P1", "P2", "P3", "P4", "P1c", "P2c")
ALPHA = 0.05


def _wt():
    import hashlib
    h = hashlib.sha256(open(WT, "rb").read()).hexdigest()
    if h != WT_SHA:
        raise SystemExit(f"hw2_within_test.py differs from its fixed hash ({h[:16]}): stop")
    spec = importlib.util.spec_from_file_location("hw2_within_test", WT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def holm(ps):
    """Holm-adjusted p values, same order as ps."""
    m = len(ps)
    idx = sorted(range(m), key=lambda i: ps[i])
    adj, run = [1.0] * m, 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def collect(hw2_dir, ns2, pred, predc):
    sc = json.load(open(os.path.join(hw2_dir, "hw2_score.json"), encoding="utf-8"))
    vals = {}
    for k, v in sc["F"].items():
        part, test, arm = k.split("|")
        if part == "R":
            vals[(test, arm)] = dict(f=v["f"], ESP=v["esp"], R3=v["idle"])

    def read(path, fields):
        if path in (None, "-") or not os.path.exists(path):
            return
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("part", "R") != "R":
                continue
            key = (r["test"], r["arm"])
            if key not in vals:
                continue
            for name, get in fields:
                x = get(r)
                if x is not None:
                    vals[key][name] = float(x)
    first = lambda P: (lambda r: r[P][0] if r.get(P) and r[P][0] is not None else None)  # noqa: E731
    read(ns2, [("NS2", lambda r: r.get("f_sim"))])
    read(pred, [(P, first(P)) for P in ("P1", "P2", "P3", "P4")])
    read(predc, [(P, first(P)) for P in ("P1c", "P2c")])
    return vals


def main():
    args = [a for a in sys.argv[1:]]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    hw2_dir, ns2, pred, predc = (args + ["-"] * 4)[:4]
    wt = _wt()
    vals = collect(hw2_dir, ns2, pred, predc)
    rows = []
    for P in ORDER:
        x, y = [], []
        for t in wt.READING:
            a, b = vals.get((t, "PSF")), vals.get((t, "QK3"))
            if a and b and P in a and P in b and min(a[P], b[P], a["f"], b["f"]) > 0:
                x.append(a[P] / b[P])
                y.append(a["f"] / b["f"])
        if len(x) >= 3:
            rho, p1, _ = wt.perm_spearman(x, y)
            p1 = 1.0 if math.isnan(p1) else p1
        else:
            rho, p1 = float("nan"), 1.0
        win = sum(1 for u, v in zip(x, y) if (u > 1) == (v > 1))
        rows.append(dict(predictor=P, n=len(x), winners=win, spearman=rho, p_one=p1))
    adj = holm([r["p_one"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        r["works"] = a <= ALPHA
    lines = ["# HW2: every predictor, Holm-corrected (Addendum 472)", "",
             f"Nine predictors, seven reading pairs (PSF / QK3). 'works' = Holm-adjusted one-sided p <= {ALPHA}.", "",
             "| predictor | pairs | same winner | Spearman | p one-sided | p Holm | works |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['predictor']} | {r['n']} | {r['winners']} | {r['spearman']:+.2f} | {r['p_one']:.3f} | "
              f"{r['p_holm']:.3f} | {'YES' if r['works'] else 'no'} |" for r in rows]
    txt = "\n".join(lines) + "\n"
    print(txt)
    if out:
        json.dump(dict(rows=rows, alpha=ALPHA, order=ORDER), open(out, "w", encoding="utf-8"), indent=1, default=float)
        with open(os.path.splitext(out)[0] + ".md", "w", encoding="utf-8") as fh:
            fh.write(txt)


if __name__ == "__main__":
    main()
