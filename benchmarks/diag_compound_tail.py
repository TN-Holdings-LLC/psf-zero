"""diag_compound_tail.py -- follow-up to Addendum 199 (flag C3 tripped).

In loop_endurance.py Part C (compounding, FakeNighthawk, 120 logical
qubits), the compile-time p99 was 118 ms against a 41 ms median (flag:
p99 <= 2 x median), with a maximum of 199 ms; Part F (fresh circuit each
lap) stayed at p99 55 ms. Only every 50th lap was written to the CSV, so
the tail could not be examined.

Hypothesis, stated before running: the slow laps are Python garbage-
collection pauses. Compounding creates many short-lived objects per lap
(mapping the routed output back to a logical circuit), which triggers
generation-2 collections that land inside the timed compile.

This script records every lap and every collection:
  arm "C, gc on"   -- compounding as in Part C, default garbage collector
  arm "C, gc off"  -- the same with gc.disable() (collected once between arms)
  arm "F, gc on"   -- fresh circuit each lap, as in Part F (control)
For each lap: compile time, time of the back-to-logical step (C arms), and
the number and total duration of collections of each generation that ran
during the timed compile.

Predictions:
  T1: in "C, gc on", at least 80% of laps slower than 2 x median contain a
      generation-2 collection inside the timed compile.
  T2: in "C, gc off", p99 <= 2 x median.
If T1 and T2 fail, the tail has another cause, and this is recorded as
such.

Usage (repository root; loop_endurance.py must be in benchmarks/):
    python -u benchmarks/diag_compound_tail.py 2>&1 | tee diag_compound_tail.txt
"""
from __future__ import annotations

import csv
import gc
import os
import statistics as st
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qiskit import QuantumCircuit

import loop_endurance as le
import psf_compile as pc

LAPS = 1000
OUT_CSV = "diag_compound_tail_2026-09-27.csv"


class GcLog:
    """Records every collection's generation and duration via gc.callbacks."""

    def __init__(self):
        self.events = []  # (generation, start, stop)
        self._start = None

    def __call__(self, phase, info):
        if phase == "start":
            self._start = time.perf_counter()
        elif self._start is not None:
            self.events.append((info["generation"], self._start, time.perf_counter()))
            self._start = None

    def within(self, t0, t1):
        out = {0: [0, 0.0], 1: [0, 0.0], 2: [0, 0.0]}
        for g, s, e in self.events:
            if s >= t0 and e <= t1:
                out[g][0] += 1
                out[g][1] += e - s
        return out


def initial_circuit(n, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, (n // 2, 24))
    qc = QuantumCircuit(n)
    for k in range(n // 2):
        le.add_pair24(qc, 2 * k, 2 * k + 1, theta[k])
    return qc, theta, rng


def run_arm(name, mode, gc_enabled, backend, native, log, rows):
    n = backend.coupling_map.size()
    gc.collect()
    if gc_enabled:
        gc.enable()
    else:
        gc.disable()
    current, theta, rng = initial_circuit(n, 13 if mode == "C" else 11)
    times, back_times, slow_with_g2, lap_g2 = [], [], 0, []
    for lap in range(1, LAPS + 1):
        if mode == "F":
            theta = theta + rng.normal(0.0, le.WALK_SIGMA, theta.shape)
            current = QuantumCircuit(n)
            for k in range(n // 2):
                le.add_pair24(current, 2 * k, 2 * k + 1, theta[k])
        log.events.clear()  # keep only this lap's collections (the log would otherwise grow without bound)
        t0 = time.perf_counter()
        out, el, _phase = le.cfh(current, backend, native)
        t1 = time.perf_counter()
        g = log.within(t0, t1)
        bt = 0.0
        if mode == "C":
            b0 = time.perf_counter()
            current = le.back_to_logical(out, n)
            bt = time.perf_counter() - b0
            back_times.append(bt)
        times.append(el)
        lap_g2.append(g[2][0] > 0)
        rows.append(dict(arm=name, lap=lap, compile_s=el, back_s=bt,
                         gc0_n=g[0][0], gc0_s=g[0][1], gc1_n=g[1][0], gc1_s=g[1][1],
                         gc2_n=g[2][0], gc2_s=g[2][1]))
    gc.enable()
    med = st.median(times)
    p99 = sorted(times)[int(round(0.99 * (len(times) - 1)))]
    slow = [i for i, t in enumerate(times) if t > 2 * med]
    slow_g2 = sum(1 for i in slow if lap_g2[i])
    g2_laps = sum(lap_g2)
    g2_s = sum(r["gc2_s"] for r in rows if r["arm"] == name)
    print(f"{name:10s}: compile median {med * 1000:.1f} ms, p99 {p99 * 1000:.1f} ms, max {max(times) * 1000:.1f} ms; "
          f"laps > 2 x median: {len(slow)} (of which with a gen-2 collection inside: {slow_g2}); "
          f"laps containing a gen-2 collection: {g2_laps}, total gen-2 time {g2_s * 1000:.0f} ms"
          + (f"; back-to-logical median {st.median(back_times) * 1000:.1f} ms" if back_times else ""),
          flush=True)
    return dict(median=med, p99=p99, slow=len(slow), slow_g2=slow_g2)


def main():
    print("LOADED", pc.__file__, pc.VERSION)
    print("LOADED", le.__file__)
    backend, native = le.nighthawk()
    log = GcLog()
    gc.callbacks.append(log)
    rows = []
    res = {}
    res["C, gc on"] = run_arm("C, gc on", "C", True, backend, native, log, rows)
    res["C, gc off"] = run_arm("C, gc off", "C", False, backend, native, log, rows)
    res["F, gc on"] = run_arm("F, gc on", "F", True, backend, native, log, rows)
    gc.callbacks.remove(log)
    on, off = res["C, gc on"], res["C, gc off"]
    t1 = on["slow"] > 0 and on["slow_g2"] >= 0.8 * on["slow"]
    t2 = off["p99"] <= 2 * off["median"]
    print(f"\nT1 (>= 80% of slow laps contain a gen-2 collection, gc on): "
          f"{'CONFIRMED' if t1 else 'NOT CONFIRMED'} ({on['slow_g2']}/{on['slow']})")
    print(f"T2 (gc off: p99 <= 2 x median): {'CONFIRMED' if t2 else 'NOT CONFIRMED'} "
          f"({off['p99'] * 1000:.1f} vs {2 * off['median'] * 1000:.1f} ms)")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
