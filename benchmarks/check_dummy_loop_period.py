#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checks whether the ~187-iteration period found in
`test_cumulative_compile_scale.py`'s Qiskit timings (AMD machine, three
different `PYTHONHASHSEED` values, two different iteration counts, all
giving autocorrelation 0.96-0.99 at lag 187) is specific to Qiskit, or is
a property of the measurement loop itself (OS/Python scheduling, timer
resolution, background activity on this machine) that would show up
regardless of what is being timed.

## Why this is needed

Addendum 20 found a strong ~145-iteration period on an Intel machine that
did not reproduce on an AMD machine. Chasing it further on the AMD machine
found a *different* but equally strong period (~187 iterations,
autocorrelation 0.96-0.99, modular-bin spread ~7.4-7.5x) that survived
changing `PYTHONHASHSEED` (three different values) and changing the total
iteration count (5000 vs 2500) -- ruling out hash randomization and
time-elapsed-based causes, but not yet distinguishing "something about
Qiskit's own execution" from "something about this measurement loop, or
this machine, regardless of what is inside the loop."

## What this does

Runs three arms, each timed the same way `test_cumulative_compile_scale.py`
times Qiskit/PSF-Zero (a `time.perf_counter()` pair around each iteration,
warm-up outside the loop), but with the loop body replaced by something
that does **not** call Qiskit or PSF-Zero at all:

  - `busy`: a fixed amount of pure-Python arithmetic (no I/O, no imports,
    no external calls) calibrated to take roughly the same order of
    magnitude of time as one Qiskit compile in the original benchmark.
  - `sleep`: `time.sleep()` for a fixed duration -- hands control back to
    the OS scheduler every iteration, which `busy` does not.
  - `numpy`: a fixed-size numpy matrix multiplication -- exercises the
    same BLAS/thread-pool machinery Qiskit's own linear algebra relies on,
    without going through Qiskit itself.

If any of these three shows the same ~187-iteration period, that period is
not specific to Qiskit -- it is a property of this loop, this machine, or
this measurement method in general. If none of them show it, that
strengthens (without proving) the case that the period is something about
Qiskit's own execution specifically.

## Usage

    python check_dummy_loop_period.py
    python check_dummy_loop_period.py --iters 5000 --period 187
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def busy_work(n=200_000):
    """Pure-Python arithmetic, no imports inside the loop, no I/O."""
    total = 0
    for i in range(n):
        total += (i * i) % 97
    return total


def numpy_work(size=200):
    a = np.random.default_rng(0).standard_normal((size, size))
    b = np.random.default_rng(1).standard_normal((size, size))
    return a @ b


def calibrate(fn, target_ms, *, max_tries=10):
    """Scales fn's size parameter so one call takes roughly target_ms
    milliseconds, by trying a few sizes and interpolating -- so the dummy
    loop's per-iteration cost is in the same order of magnitude as one
    Qiskit compile in the original benchmark (order of several ms),
    rather than being trivially fast or absurdly slow."""
    size = 100
    for _ in range(max_tries):
        t0 = time.perf_counter()
        fn(size)
        el = (time.perf_counter() - t0) * 1000
        if el <= 0:
            size *= 4
            continue
        ratio = target_ms / el
        if 0.8 <= ratio <= 1.25:
            return size
        size = max(1, int(size * ratio ** 0.5))
    return size


def run_loop(fn, n_iters, arg):
    times = np.empty(n_iters)
    fn(arg)  # warm-up, outside the timed loop
    for i in range(n_iters):
        t0 = time.perf_counter()
        fn(arg)
        times[i] = time.perf_counter() - t0
    return times


def autocorrelation(x, max_lag):
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    out = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        out[lag - 1] = np.dot(x[:-lag], x[lag:]) / (n - lag) / var
    return out


def modular_bin_spread(x, period):
    idx = np.arange(len(x))
    phase = idx % period
    medians = np.array([np.median(x[phase == p]) for p in range(period)
                        if np.any(phase == p)])
    return medians.max() / medians.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--period", type=int, default=187,
                    help="the period found in the Qiskit timings, to check "
                         "against here")
    ap.add_argument("--target-ms", type=float, default=8.0,
                    help="rough per-iteration target time for busy/numpy, "
                         "in the same order of magnitude as one Qiskit "
                         "compile in the original benchmark")
    ap.add_argument("--sleep-s", type=float, default=0.008)
    args = ap.parse_args()

    print("=" * 78)
    print(f"Dummy-loop period check -- does ~{args.period} show up with no "
          f"Qiskit or PSF-Zero involved at all?")
    print("=" * 78)

    busy_n = calibrate(busy_work, args.target_ms)
    numpy_size = calibrate(numpy_work, args.target_ms)
    print(f"\nCalibrated: busy_work(n={busy_n}), numpy_work(size={numpy_size}), "
          f"sleep({args.sleep_s}s)\n")

    arms = [
        ("busy", busy_work, busy_n),
        ("sleep", time.sleep, args.sleep_s),
        ("numpy", numpy_work, numpy_size),
    ]

    for name, fn, arg in arms:
        print(f"--- {name} ---")
        x = run_loop(fn, args.iters, arg)
        print(f"  median={np.median(x)*1000:.3f}ms  "
              f"mean={x.mean()*1000:.3f}ms  std={x.std()*1000:.3f}ms")

        ac = autocorrelation(x, min(500, args.iters // 2))
        top5 = np.argsort(ac)[::-1][:5] + 1
        print(f"  Top 5 lags by autocorrelation: {list(top5)}")
        rank = int(np.sum(ac > ac[args.period - 1])) + 1 if args.period - 1 < len(ac) else None
        ac_at_period = ac[args.period - 1] if args.period - 1 < len(ac) else float("nan")
        print(f"  Autocorrelation at lag {args.period}: {ac_at_period:.4f}  "
              f"(rank {rank} of {len(ac)})")

        spread = modular_bin_spread(x, args.period)
        print(f"  Modular-bin spread at period {args.period}: {spread:.3f}x\n")

    print("=" * 78)
    print("Verdict")
    print("=" * 78)
    print(f"If any arm above shows autocorrelation near the Qiskit-observed")
    print(f"strength (0.96-0.99) at lag {args.period}, or a modular-bin spread")
    print(f"anywhere near 7-7.5x, the {args.period}-iteration period is not")
    print(f"specific to Qiskit -- it is a property of this loop, this machine,")
    print(f"or this measurement method in general.")
    print(f"If every arm's autocorrelation at lag {args.period} stays small")
    print(f"(comparable to the psf_true/psf_false arms already measured, not")
    print(f"the qiskit arm), that strengthens the case that the period is")
    print(f"something about Qiskit's own execution specifically -- though it")
    print(f"does not by itself identify what inside Qiskit causes it.")


if __name__ == "__main__":
    main()