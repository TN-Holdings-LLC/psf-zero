#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checks whether the top-20 outliers' clustering at intervals of ~145 is a
real periodic effect in the full 50,000-iteration dataset, or a coincidence
of looking at only 20 points.

## Why this is needed

`diagnose_outlier_circuits.py` rejected the "near-degenerate block" theory
for why the top-20 Qiskit outliers are slow: their distance-to-landmark
distribution (mean 2.305) is indistinguishable from three baseline indices
(mean 2.367). Sorting the 20 outlier indices by hand instead turned up
something else: 14 of the 20 fall into small clusters with a repeating gap
of almost exactly 145 (10523-10668, 24443-24588-24733,
30533-30678-30823-30968-31113). Twenty points is too few to trust a pattern
found by eye, so this checks the *entire* dataset for the same structure
before treating it as real.

## What this checks

1. **Autocorrelation of the full Qiskit timing series** at lags 1 through
   500. If there is a genuine ~145-iteration period, the autocorrelation
   should show a local peak near lag 145 (and possibly its multiples, 290,
   435, ...) standing out above the surrounding lags -- not just at the
   single lag value spotted by hand.
2. **Modular binning**: bins every iteration index by `index mod 145` (and,
   for comparison, by several nearby moduli to check 145 isn't an artifact
   of the specific bin count chosen) and compares each bin's median Qiskit
   time. If 145 is a real period, iterations landing in the "wrong" phase
   of the cycle should show a visibly higher median than the rest.
3. **A negative control**: the same two checks run on PSF-Zero's own timing
   series (both verify=True and verify=False). If the ~145 period is
   specific to Qiskit (consistent with the earlier finding that Qiskit's
   cumulative-time curve shows slope anomalies PSF-Zero's curves do not),
   it should not show up here, or should be much weaker.

This is diagnostic only -- it does not modify `psf_compile.py`, and finding
a period here does not by itself explain *why* iterations at that phase are
slower; it only establishes whether there is something there worth
explaining.

## Usage

    python check_period_145.py
    python check_period_145.py --npz cumulative_compile_times_50000.npz --period 145
    python check_period_145.py --period 145 --compare-periods 100 120 145 160 200
"""
from __future__ import annotations

import argparse

import numpy as np


def autocorrelation(x, max_lag):
    """Normalized autocorrelation at lags 1..max_lag. Returns an array of
    length max_lag (index 0 = lag 1)."""
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    out = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        out[lag - 1] = np.dot(x[:-lag], x[lag:]) / (n - lag) / var
    return out


def modular_bin_medians(x, period):
    """Median of x for each phase 0..period-1 of index mod period."""
    n = len(x)
    idx = np.arange(n)
    phase = idx % period
    medians = np.empty(period)
    counts = np.empty(period, dtype=int)
    for p in range(period):
        vals = x[phase == p]
        medians[p] = np.median(vals) if len(vals) else np.nan
        counts[p] = len(vals)
    return medians, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="cumulative_compile_times_50000.npz")
    ap.add_argument("--period", type=int, default=145,
                    help="the period spotted by hand in the top-20 outliers")
    ap.add_argument("--max-lag", type=int, default=500)
    ap.add_argument("--compare-periods", type=int, nargs="+",
                    default=[100, 120, 145, 160, 200],
                    help="other candidate periods to check the modular-bin "
                         "test against, so 145 isn't judged in isolation")
    args = ap.parse_args()

    d = np.load(args.npz)

    print("=" * 78)
    print(f"Period check: does ~{args.period} show up as a real periodic "
          f"effect in the full dataset?")
    print("=" * 78)

    for name in ("qiskit", "psf_true", "psf_false"):
        x = d[name]
        print(f"\n--- {name} (n={len(x)}) ---")

        # 1. Autocorrelation
        ac = autocorrelation(x, args.max_lag)
        top5_lags = np.argsort(ac)[::-1][:5] + 1  # +1 because index 0 = lag 1
        print(f"  Top 5 lags by autocorrelation (1-{args.max_lag}): "
              f"{list(top5_lags)}")
        print(f"  Autocorrelation at lag {args.period}: "
              f"{ac[args.period - 1]:.4f}  "
              f"(rank {int(np.sum(ac > ac[args.period - 1])) + 1} of {args.max_lag})")

        # 2. Modular binning across several candidate periods
        print(f"  Modular-bin spread (max median / min median) by candidate period:")
        for p in args.compare_periods:
            medians, counts = modular_bin_medians(x, p)
            valid = ~np.isnan(medians)
            spread = medians[valid].max() / medians[valid].min()
            tag = "  <-- requested period" if p == args.period else ""
            print(f"    period={p:>4}: spread={spread:.3f}x{tag}")

    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    print("If qiskit's autocorrelation at the requested period ranks clearly")
    print("above the surrounding lags, AND its modular-bin spread at that period")
    print("is clearly larger than at the other candidate periods tried, AND")
    print("psf_true/psf_false do not show the same pattern -- that supports a")
    print("real, Qiskit-specific periodic effect at that interval.")
    print("If the autocorrelation rank is unremarkable or the modular-bin spread")
    print("is similar across every candidate period (including ones with no a")
    print("priori reason to matter), the ~145 clustering seen in the top-20")
    print("outliers alone was most likely a coincidence of a 20-point sample.")


if __name__ == "__main__":
    main()
