"""long_loop_100k.py -- Addendum 213: about 100,000 compiles with psf_compile.py
2026-09-27.2 (batched polish, default block_gate_floor 8), unattended.

Addendum 200 ran 1,000 laps per loop with 2026-09-26.4. This repeats the idea at
~100x scale with the current release and the garbage-collection setting of
Addendum 202 (gc.freeze() after set-up; collector off during a lap; one
gc.collect() at each lap boundary), to find what 1,000 laps cannot show:
rare events (core decomposition failures, guard engagements), slow memory
growth, slow-down over time, and whether compounding drift stays linear.

Parts (run in this order; each writes its CSV as it goes, flushed every
CHECK laps, so a run cut short keeps what it did):
  F  fresh cliff circuits, FakeNighthawk, 120 logical qubits, 60 pair24 blocks,
     parameters on a random walk: F_LAPS laps; exact per-pair check every CHECK.
  C  compounding: each lap's output mapped back to logical qubits and compiled
     again: C_LAPS laps; per-pair distance to lap 0 every CHECK.
  E  fresh 12-qubit training circuits (Addendum 199 ansatz, 11 blocks at the new
     default floor), random parameters: E_LAPS compiles; loss against the
     uncompiled reference every E_CHECK.

Per lap: compile time, blocks that fell back to Qiskit's CX synthesis (parsed
from compile()'s summary warning), and, every CHECK laps, RSS, the check value
and psf_compile.GUARD_STATS.

Flags (a tripped flag is a finding to investigate, as in Addendum 199):
  L1 no exception in any part
  L2 RSS growth <= 100 MB in each part, measured from lap CHECK to the end
  L3 each part: compile p99 <= 2 x median and max <= 1 s
  L4 F: every per-pair check <= 1e-12; E: every loss difference <= 1e-10
  L5 C: distance at the last lap <= C_LAPS x 1e-13, and last / half-way in [1.6, 2.4]
  L6 each part: median compile time of the last 10% of laps <= 1.1 x the first 10%
  (exploratory) fallback rate per block, guard counts

Usage (repository root; loop_endurance.py in benchmarks/):
    python -u benchmarks/long_loop_100k.py --smoke     # about 1-2 minutes, 1/100 scale
    nohup python -u benchmarks/long_loop_100k.py > long_loop_100k.txt 2>&1 &
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import os
import platform
import re
import statistics as st
import sys
import time
import traceback
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

import loop_endurance as le
import psf_compile as pc

F_LAPS, C_LAPS, E_LAPS = 50_000, 20_000, 30_000
CHECK, E_CHECK = 1000, 10
WALK_SIGMA = 0.02
FELL = re.compile(r"(\d+) block\(s\) fell back to CX-basis synthesis \((\d+) degenerate/numeric, (\d+) unexpected\)")
PC = time.perf_counter


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


class Writer:
    def __init__(self, path, fields):
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields, restval="")
        self.w.writeheader()

    def row(self, **kw):
        self.w.writerow(kw)

    def flush(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def compiled(fn):
    """Run one compile; returns (output, seconds, fallbacks, degenerate, unexpected)."""
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = PC()
            out = fn()
            el = PC() - t0
    fb = dg = un = 0
    for w in ws:
        m = FELL.search(str(w.message))
        if m:
            fb, dg, un = (int(x) for x in m.groups())
    return out, el, fb, dg, un


def cliff_fn(qc, backend, native):
    def fn():
        pc._CX_CORE_CACHE.clear()
        return pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                       entangling_basis="cx", layout_search=True, on_unsupported="keep",
                                       seed_transpiler=0)
    return fn


def line_fn(qc, cmap):
    def fn():
        pc._CX_CORE_CACHE.clear()
        return pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                                       entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                       on_unsupported="keep", seed_transpiler=0)
    return fn


def run_part(name, laps, make_fn, check_fn, blocks_per_compile, out_csv, summary):
    """Generic loop with the Addendum 202 GC setting. make_fn(lap) -> (compile fn, context);
    check_fn(lap, out, context) -> value or None; may also advance state."""
    w = Writer(out_csv, ["part", "lap", "compile_s", "fallbacks", "degenerate", "unexpected", "rss_mb", "check",
                         "guard", "wall_s"])
    times, fbs, checks, rss, errors = [], 0, [], [], 0
    t_start = time.time()
    print(f"\n=== {name}: {laps} laps ===", flush=True)
    for lap in range(1, laps + 1):
        gc.disable()
        try:
            fn, ctx = make_fn(lap)
            out, el, fb, dg, un = compiled(fn)
            times.append(el)
            fbs += fb
            val = check_fn(lap, out, ctx)
        except Exception as exc:  # noqa: BLE001 -- recorded; the part stops
            errors += 1
            print(f"  lap {lap}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=3)
            gc.enable()
            break
        gc.enable()
        gc.collect()
        row = dict(part=name, lap=lap, compile_s=f"{el:.6f}", fallbacks=fb, degenerate=dg, unexpected=un)
        if val is not None:
            checks.append((lap, val))
            row["check"] = f"{val:.3e}"
        if lap % CHECK == 0 or lap == laps:
            r = le.rss_mb()
            rss.append((lap, r))
            row.update(rss_mb=f"{r:.1f}", guard=str(dict(pc.GUARD_STATS)), wall_s=f"{time.time() - t_start:.0f}")
            w.row(**row)
            w.flush()
            if lap % (CHECK * 10) == 0 or lap == laps:
                print(f"  lap {lap}: median {st.median(times) * 1000:.1f} ms, p99 {pct(times, 0.99) * 1000:.1f} ms, "
                      f"max {max(times) * 1000:.0f} ms; fallbacks so far {fbs}; RSS {r:.0f} MB; "
                      f"last check {checks[-1][1] if checks else None}; {time.time() - t_start:.0f} s", flush=True)
        else:
            w.row(**row)
    w.close()
    n = len(times)
    tenth = max(1, n // 10)
    res = dict(laps_done=n, errors=errors, median=st.median(times), p99=pct(times, 0.99), max=max(times),
               first10=st.median(times[:tenth]), last10=st.median(times[-tenth:]), fallbacks=fbs,
               blocks=n * blocks_per_compile, checks=checks, rss=rss, wall=time.time() - t_start,
               guard=dict(pc.GUARD_STATS))
    summary[name] = res
    print(f"  {name} done: {n} laps in {res['wall']:.0f} s; median {res['median'] * 1000:.1f} ms, "
          f"p99 {res['p99'] * 1000:.1f} ms, max {res['max'] * 1000:.0f} ms; fallbacks {fbs} of about "
          f"{res['blocks']} blocks; guard {res['guard']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1/100 scale, to check the run before leaving it")
    args = ap.parse_args()
    global F_LAPS, C_LAPS, E_LAPS, CHECK
    if args.smoke:
        F_LAPS, C_LAPS, E_LAPS, CHECK = 500, 200, 300, 100
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__} | {'SMOKE' if args.smoke else 'FULL'}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-27.2":
        print("V0 FAILED: psf_compile.py is not 2026-09-27.2. Stopping.")
        return
    tag = "smoke" if args.smoke else "2026-09-27"
    backend, native = le.nighthawk()
    n = backend.coupling_map.size()
    cmap = CouplingMap.from_line(le.E_QUBITS)

    # set-up objects, then freeze (Addendum 202)
    rng_f = np.random.default_rng(101)
    theta_f = rng_f.uniform(-np.pi, np.pi, (n // 2, 24))
    rng_c = np.random.default_rng(13)
    initial = QuantumCircuit(n)
    th = rng_c.uniform(-np.pi, np.pi, (n // 2, 24))
    for k in range(n // 2):
        le.add_pair24(initial, 2 * k, 2 * k + 1, th[k])
    ref_pairs = le.pair_matrices(initial, n)
    rng_e = np.random.default_rng(7)
    target_theta = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
    target_theta[8::15] = 0.0
    target = Statevector(le.e_circuit(target_theta))
    compiled(cliff_fn(initial, backend, native))
    compiled(line_fn(le.e_circuit(target_theta), cmap))
    gc.collect()
    gc.freeze()
    summary = {}

    # ---- F
    state_f = {"theta": theta_f}

    def make_f(lap):
        state_f["theta"] = state_f["theta"] + rng_f.normal(0.0, WALK_SIGMA, state_f["theta"].shape)
        qc = QuantumCircuit(n)
        for k in range(n // 2):
            le.add_pair24(qc, 2 * k, 2 * k + 1, state_f["theta"][k])
        return cliff_fn(qc, backend, native), qc

    def check_f(lap, out, qc):
        if lap % CHECK:
            return None
        ok, worst = le.pair_check(qc, out, n)
        return worst if ok else float("inf")

    run_part("F", F_LAPS, make_f, check_f, n // 2, f"long_loop_F_{tag}.csv", summary)

    # ---- C
    state_c = {"current": initial}

    def make_c(lap):
        return cliff_fn(state_c["current"], backend, native), None

    def check_c(lap, out, _):
        state_c["current"] = le.back_to_logical(out, n)
        if lap % CHECK and lap != C_LAPS // 2:
            return None
        mats = le.pair_matrices(state_c["current"], n)
        return max(le.aligned(ref_pairs[k], mats[k]) for k in ref_pairs)

    run_part("C", C_LAPS, make_c, check_c, n // 2, f"long_loop_C_{tag}.csv", summary)

    # ---- E
    rng_t = np.random.default_rng(202)

    def make_e(lap):
        qc = le.e_circuit(target_theta + rng_t.normal(0.0, 0.5, le.E_NPARAMS))
        return line_fn(qc, cmap), qc

    def check_e(lap, out, qc):
        if lap % E_CHECK:
            return None
        return abs((1 - abs(target.inner(Statevector(out))) ** 2) - (1 - abs(target.inner(Statevector(qc))) ** 2))

    run_part("E", E_LAPS, make_e, check_e, len(le.E_BLOCKS), f"long_loop_E_{tag}.csv", summary)

    # ---- flags
    print("\n=== flags ===")

    def v(ok):
        return "clear" if ok else "TRIPPED"

    errs = {k: r["errors"] for k, r in summary.items()}
    done = {k: r["laps_done"] for k, r in summary.items()}
    print(f"L1 no exception: {errs}; laps done {done} -> {v(all(e == 0 for e in errs.values()))}")
    for k, r in summary.items():
        base = next((x for lap, x in r["rss"] if lap >= CHECK), r["rss"][0][1] if r["rss"] else float("nan"))
        grow = r["rss"][-1][1] - base if r["rss"] else float("nan")
        print(f"L2 {k}: RSS growth {grow:+.0f} MB -> {v(grow <= 100)}")
        print(f"L3 {k}: p99 {r['p99'] * 1000:.1f} vs 2 x median {2 * r['median'] * 1000:.1f} ms; max "
              f"{r['max'] * 1000:.0f} ms -> {v(r['p99'] <= 2 * r['median'] and r['max'] <= 1.0)}")
        print(f"L6 {k}: last-10% median {r['last10'] * 1000:.1f} vs first-10% {r['first10'] * 1000:.1f} ms -> "
              f"{v(r['last10'] <= 1.1 * r['first10'])}")
        print(f"   {k}: fallbacks {r['fallbacks']} of about {r['blocks']} blocks "
              f"({r['fallbacks'] / max(r['blocks'], 1):.2e} per block); guard {r['guard']}")
    wf = max((x for _, x in summary["F"]["checks"]), default=float("nan"))
    we = max((x for _, x in summary["E"]["checks"]), default=float("nan"))
    print(f"L4 F worst per-pair check {wf:.2e} (<= 1e-12); E worst loss difference {we:.2e} (<= 1e-10) -> "
          f"{v(wf <= 1e-12 and we <= 1e-10)}")
    cc = dict(summary["C"]["checks"])
    last_lap = max(cc) if cc else 0
    half = cc.get(C_LAPS // 2)
    if cc and half:
        ratio = cc[last_lap] / half
        print(f"L5 C distance at lap {last_lap} {cc[last_lap]:.3e} (<= {C_LAPS * 1e-13:.1e}); ratio to lap "
              f"{C_LAPS // 2} {ratio:.2f} (1.6-2.4) -> {v(cc[last_lap] <= C_LAPS * 1e-13 and 1.6 <= ratio <= 2.4)}")
    else:
        print("L5 C: not evaluable (part stopped early)")
    print("\nDONE")


if __name__ == "__main__":
    main()
