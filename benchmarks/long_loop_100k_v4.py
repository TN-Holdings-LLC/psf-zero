"""long_loop_100k_v4.py -- pre-registered (home, 2026-09-28, Addendum 243): the
100,000-compile run of Addendum 219 (long_loop_100k_v2.py; workplace v3 with the
fixed core, Addenda 235-236) repeated at home with the candidate release: the
fixed Rust core (CORE_VERSION 2026-09-28.1) together with psf_compile.py
2026-09-28.1 (REFINE_THRESHOLD 1e-14, changelog item 26).

Identical to long_loop_100k_v3.py (workplace) except: the version check
(psf_compile.py 2026-09-28.1 with REFINE_THRESHOLD 1e-14, and CORE_VERSION
2026-09-28.1), the output names (_v4), and the pre-registered lines N1-N7
printed after the v2 flags (they replace v3's M1-M4). Laps, seeds, floor,
checks and the thresholds L1-L6 are unchanged, so every value can be set
against v2 (home, pre-fix core, Addendum 222) and v3 (workplace sandbox).

Pre-registered (Addendum 243):
  N1 E fallbacks <= 50 (v2 43,829; v3 15); refuted >= 500
  N2 exact_rebuilt + psf_rerouted + best_effort <= 10 (v2 151; v3 4); refuted >= 30
  N3 F + C fallbacks == 0; refuted if any
  N4 C drift at lap 20,000 <= 3e-10 (v2 8.33e-10, v3 1.66e-9); refuted >= 8.3e-10
  N5 non-timing flags L1, L2, L4 (E loss <= 1e-13), L5 all clear; refuted if any trips
  N6 timing, home WSL against v2 on the same machine (F 27.4 ms, E 10.6 ms medians):
     E median <= 1.00 x 10.6 ms and F median <= 1.15 x 27.4 ms;
     refuted if E >= 1.15 x or F >= 1.30 x
  N7 cross-environment: C drift at laps 1,000 and 20,000 within 1% of the workplace
     sandbox's REFINE_THRESHOLD 1e-14 arm (6.056e-12 and 1.219e-10, Addendum 239);
     refuted if either differs by more than 10%

Usage (repository root; loop_endurance.py in benchmarks/):
    python -u benchmarks/long_loop_100k_v4.py --smoke     # 1/100 scale, plumbing only
    nohup python -u benchmarks/long_loop_100k_v4.py > long_loop_100k_v4.txt 2>&1 &
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
FLOOR = 8
E_TOL = 1e-13
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
                                       seed_transpiler=0, block_gate_floor=FLOOR)
    return fn


def line_fn(qc, cmap):
    def fn():
        pc._CX_CORE_CACHE.clear()
        return pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                                       entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                       on_unsupported="keep", seed_transpiler=0, block_gate_floor=FLOOR)
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
    ap.add_argument("--floor", type=int, default=8)
    args = ap.parse_args()
    global F_LAPS, C_LAPS, E_LAPS, CHECK, FLOOR
    FLOOR = args.floor
    if args.smoke:
        F_LAPS, C_LAPS, E_LAPS, CHECK = 500, 200, 300, 100
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__} | {'SMOKE' if args.smoke else 'FULL'} | block_gate_floor {FLOOR}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    import psf_zero_core as core
    core_version = getattr(core, "CORE_VERSION", None)
    print("CORE", core.__file__, "CORE_VERSION", core_version)
    print("REFINE_THRESHOLD", pc.REFINE_THRESHOLD, "| psf_compile.CORE_VERSION", getattr(pc, "CORE_VERSION", None))
    if pc.VERSION != "2026-09-28.1" or core_version != "2026-09-28.1" or pc.REFINE_THRESHOLD != 1e-14:
        print("V0 FAILED: need psf_compile.py 2026-09-28.1 (REFINE_THRESHOLD 1e-14) and CORE_VERSION 2026-09-28.1."
              " Stopping.")
        return
    tag = ("smoke" if args.smoke else "2026-09-28") + "_v4"
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
    print(f"   blocks rebuilt exactly (changelog item 23), whole run: {pc.GUARD_STATS.get('exact_rebuilt')}")
    wf = max((x for _, x in summary["F"]["checks"]), default=float("nan"))
    we = max((x for _, x in summary["E"]["checks"]), default=float("nan"))
    print(f"L4 F worst per-pair check {wf:.2e} (<= 1e-12); E worst loss difference {we:.2e} (<= {E_TOL:.0e}) -> "
          f"{v(wf <= 1e-12 and we <= E_TOL)}")
    cc = dict(summary["C"]["checks"])
    last_lap = max(cc) if cc else 0
    half = cc.get(C_LAPS // 2)
    if cc and half:
        ratio = cc[last_lap] / half
        print(f"L5 C distance at lap {last_lap} {cc[last_lap]:.3e} (<= {C_LAPS * 1e-13:.1e}); ratio to lap "
              f"{C_LAPS // 2} {ratio:.2f} (1.6-2.4) -> {v(cc[last_lap] <= C_LAPS * 1e-13 and 1.6 <= ratio <= 2.4)}")
    else:
        print("L5 C: not evaluable (part stopped early)")

    # ---- pre-registered lines for v4 (Addendum 243; thresholds fixed there)
    print("\n=== pre-registered (v4, Addendum 243) ===")
    if args.smoke:
        print("(smoke run: the thresholds below are for the full run; shown for plumbing only)")

    def verdict(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")

    fe = summary["E"]["fallbacks"]
    print(f"N1 E fallbacks {fe} (v2 43829, v3 15; confirmed <= 50, refuted >= 500) -> {verdict(fe <= 50, fe >= 500)}")
    g = dict(pc.GUARD_STATS)
    rep = g.get("exact_rebuilt", 0) + g.get("psf_rerouted", 0) + g.get("best_effort", 0)
    print(f"N2 repair paths: exact_rebuilt {g.get('exact_rebuilt')} + psf_rerouted {g.get('psf_rerouted')} + "
          f"best_effort {g.get('best_effort')} = {rep} (v2 151, v3 4; confirmed <= 10, refuted >= 30) -> "
          f"{verdict(rep <= 10, rep >= 30)}")
    ffc = summary["F"]["fallbacks"] + summary["C"]["fallbacks"]
    print(f"N3 F + C fallbacks {ffc} (confirmed 0, refuted any) -> {verdict(ffc == 0, ffc > 0)}")
    d20 = cc.get(C_LAPS) if cc else None
    if d20 is None:
        print("N4 C drift: not evaluable (part stopped early) -> AMBIGUOUS")
    else:
        print(f"N4 C drift at lap {C_LAPS} {d20:.3e} (v2 8.33e-10, v3 1.66e-9; confirmed <= 3e-10, refuted >= 8.3e-10)"
              f" -> {verdict(d20 <= 3e-10, d20 >= 8.3e-10)}")
    l1 = all(r["errors"] == 0 for r in summary.values())
    l2 = True
    for r in summary.values():
        base = next((x for lap, x in r["rss"] if lap >= CHECK), r["rss"][0][1] if r["rss"] else float("nan"))
        l2 = l2 and bool(r["rss"]) and (r["rss"][-1][1] - base) <= 100
    l4 = wf <= 1e-12 and we <= E_TOL
    l5 = bool(cc and half) and cc[last_lap] <= C_LAPS * 1e-13 and 1.6 <= cc[last_lap] / half <= 2.4
    n5 = l1 and l2 and l4 and l5
    print(f"N5 non-timing flags: L1 {l1}, L2 {l2}, L4 {l4}, L5 {l5} -> {verdict(n5, not n5)}")
    rf = summary["F"]["median"] / 0.0274
    re_ = summary["E"]["median"] / 0.0106
    print(f"N6 timing (home WSL) vs v2: F median {summary['F']['median'] * 1000:.1f} ms = {rf:.3f} x 27.4 "
          f"(confirmed <= 1.15, refuted >= 1.30); E median {summary['E']['median'] * 1000:.1f} ms = {re_:.3f} x 10.6 "
          f"(confirmed <= 1.00, refuted >= 1.15) -> {verdict(rf <= 1.15 and re_ <= 1.00, rf >= 1.30 or re_ >= 1.15)}")
    d1 = cc.get(1000) if cc else None
    if d1 is None or d20 is None:
        print("N7 cross-environment drift: not evaluable -> AMBIGUOUS")
    else:
        e1, e20 = abs(d1 / 6.056e-12 - 1), abs(d20 / 1.219e-10 - 1)
        print(f"N7 C drift vs workplace T14 arm: lap 1000 {d1:.4e} vs 6.056e-12 ({e1:.1%}); lap 20000 {d20:.4e} vs "
              f"1.219e-10 ({e20:.1%}) (confirmed both <= 1%, refuted either > 10%) -> "
              f"{verdict(e1 <= 0.01 and e20 <= 0.01, e1 > 0.10 or e20 > 0.10)}")
    print(f"   whole-run GUARD_STATS {g}")
    print("\nDONE")


if __name__ == "__main__":
    main()
