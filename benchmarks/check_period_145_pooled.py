"""Pool several per-iteration compile-time CSVs (the format produced by
converting `test_cumulative_compile_scale.py`'s `.npz` output, three
columns: qiskit, psf_true, psf_false) into one longer series per arm, then
run the same period check as `check_period_145.py`: autocorrelation at lags
1-500, and modular-bin spread compared across several candidate periods
(not just the one requested) so a real periodic effect can be told apart
from a coincidental match at one particular period.

Written for Addendum 23 (2026-09-16), to re-check the ~145-iteration period
using ALL of a matched set of runs at once (higher statistical power than
any single 5000-iteration run) while keeping the runs' experimental
condition uncontrolled data out of the mix -- point it at CSVs from ONE
condition only (e.g. all gc.disable() runs, or all baseline runs); do not
mix conditions in one invocation, since that silently blends two different
underlying processes into one series and produces a result that describes
neither condition on its own. This was written after exactly that mistake
happened by accident locally (a period check picked up an old, differently-
conditioned 50,000-iteration file by default instead of the intended
dataset -- see Addendum 23 Section 3).

Usage:
    python check_period_145_pooled.py --period 145 file1.csv file2.csv ...
"""
import argparse
import csv

import numpy as np

CANDIDATE_PERIODS_DEFAULT = [100, 120, 145, 160, 200]
MAX_LAG = 500


def load_csv(path):
    q, pt, pf = [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            q.append(float(row["qiskit"]))
            pt.append(float(row["psf_true"]))
            pf.append(float(row["psf_false"]))
    return np.array(q), np.array(pt), np.array(pf)


def autocorr_at_lags(x, max_lag):
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x * x)
    out = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        out[lag - 1] = np.sum(x[: n - lag] * x[lag:]) / denom
    return out


def modular_bin_spread(x, period):
    bins = [x[i::period] for i in range(period) if len(x[i::period]) > 0]
    medians = np.array([np.median(b) for b in bins])
    return medians.max() / medians.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="CSV files to pool (qiskit,psf_true,psf_false columns)")
    ap.add_argument("--period", type=int, default=145)
    ap.add_argument("--candidates", type=int, nargs="*", default=None,
                     help="Extra candidate periods to compare against (default: 100 120 145 160 200, "
                          "with --period inserted if not already present)")
    args = ap.parse_args()

    candidates = args.candidates if args.candidates is not None else list(CANDIDATE_PERIODS_DEFAULT)
    if args.period not in candidates:
        candidates = sorted(set(candidates) | {args.period})

    arms = {"qiskit": [], "psf_true": [], "psf_false": []}
    for path in args.files:
        q, pt, pf = load_csv(path)
        arms["qiskit"].append(q)
        arms["psf_true"].append(pt)
        arms["psf_false"].append(pf)

    pooled = {k: np.concatenate(v) for k, v in arms.items()}

    print(f"Pooled {len(args.files)} file(s): {', '.join(args.files)}")
    print(f"n per arm = {len(pooled['qiskit'])}")
    print()

    for name, x in pooled.items():
        print(f"--- {name} (n={len(x)}) ---")
        ac = autocorr_at_lags(x, MAX_LAG)
        top5 = (np.argsort(-ac)[:5] + 1).tolist()
        a_req = ac[args.period - 1]
        rank_req = int((ac > a_req).sum()) + 1
        print(f"  Top 5 lags by autocorrelation (1-{MAX_LAG}): {top5}")
        print(f"  Autocorrelation at lag {args.period}: {a_req:.4f}  (rank {rank_req} of {MAX_LAG})")
        print("  Modular-bin spread (max median / min median) by candidate period:")
        for p in candidates:
            s = modular_bin_spread(x, p)
            marker = f"  <-- {args.period}" if p == args.period else ""
            print(f"    period={p:4d}: spread={s:.3f}x{marker}")
        print()


if __name__ == "__main__":
    main()
