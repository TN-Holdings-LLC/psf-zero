"""stagetime.py -- STAGETIME (2026-10-10; exploratory, nothing predicted): where c33's default call spends its time,
measured with timers around its stages instead of a profiler (the profiler inflated small Python calls tenfold:
microbench.txt). Nothing in the compiler changes: the stages are wrapped in the module's namespace, where
compile_for_hardware looks them up at each call.

Stages: _unroll_wide, _cancel_candidate (Qiskit's CommutativeCancellation, item 50), compile (PSF-Zero's compression),
smart_vf2_layout (the layout search), generate_preset_pass_manager (building Qiskit's pass manager), the pass
manager's run (layout, routing, translation, optimization; within it _AbsorbRoutingSwaps and VF2PostLayout are timed
too), _failed_elements, prune_coupling_map, _uses_failed. "other" is the call's time minus the outer stages.

Each of BP-FINAL's 106 FakeTorino tests in its own process, on FakeTorino, called twice (cold, warm); Qiskit level 2
timed warm in the same process afterwards, for reference (its cold time is REPL-C33's).

    cd <psf-zero repository>
    python <this file> run --bp <benchpress clone> --out DIR [--par 4]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), HERE, REPO]
C33 = os.path.join(REPO, "patches", "psf_compile_c33_2026-10-10", "psf_compile.py")
OUTER = ("_unroll_wide", "_cancel_candidate", "compile", "smart_vf2_layout", "generate_preset_pass_manager",
         "pm.run", "_failed_elements", "prune_coupling_map", "_uses_failed")
INNER = ("_AbsorbRoutingSwaps", "VF2PostLayout")
TIMEOUT = 900


def instrument(pc, lay, acc):
    def wrap(owner, name, label=None):
        f = getattr(owner, name)

        def g(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                acc[label or name] = acc.get(label or name, 0.0) + time.perf_counter() - t0
        setattr(owner, name, g)
    for n in ("_unroll_wide", "_cancel_candidate", "compile", "_failed_elements", "prune_coupling_map", "_uses_failed"):
        wrap(pc, n)
    wrap(lay, "smart_vf2_layout")
    gp = pc.generate_preset_pass_manager

    def gpm(*a, **k):
        t0 = time.perf_counter()
        pm = gp(*a, **k)
        acc["generate_preset_pass_manager"] = acc.get("generate_preset_pass_manager", 0.0) + time.perf_counter() - t0
        run = pm.run

        def timed_run(*ra, **rk):
            t1 = time.perf_counter()
            try:
                return run(*ra, **rk)
            finally:
                acc["pm.run"] = acc.get("pm.run", 0.0) + time.perf_counter() - t1
        pm.run = timed_run
        return pm
    pc.generate_preset_pass_manager = gpm
    for cls, label in ((pc._AbsorbRoutingSwaps, "_AbsorbRoutingSwaps"), (pc.VF2PostLayout, "VF2PostLayout")):
        wrap(cls, "run", label)


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    import core_fix_c2_eval as H
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    qc, backend = B.build(a.bp, kind, arg)
    lay = H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(C33, "psf_compile")
    pc.WARN_WITHOUT_TARGET = False
    acc = {}
    instrument(pc, lay, acc)
    rec = dict(test=tid, input_qubits=qc.num_qubits)
    for label in ("cold", "warm"):
        acc.clear()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            pc.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True, seed_transpiler=0)
        total = time.perf_counter() - t0
        rec[label] = dict(total=round(total, 5), **{k: round(v, 5) for k, v in acc.items()},
                          other=round(total - sum(acc.get(k, 0.0) for k in OUTER), 5))
    for _ in range(2):  # Qiskit level 2, warm (the process has run Qiskit already): the second call
        t0 = time.perf_counter()
        generate_preset_pass_manager(2, backend, seed_transpiler=0).run(qc)
        rec["qk2_warm"] = round(time.perf_counter() - t0, 5)
    print(json.dumps(rec), flush=True)


def run(a):
    import bp_final as F
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    ts = [t for t in F.population(a.bp) if t[0].endswith("FakeTorino")]
    os.makedirs(a.out)
    path = os.path.join(a.out, "stagetime.jsonl")

    def go(t):
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--job",
                                json.dumps(list(t))], capture_output=True, text=True, timeout=TIMEOUT, cwd=REPO,
                               env=dict(os.environ, PSF_ZERO_REPO=REPO))
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            return json.loads(lines[-1]) if lines else dict(test=t[1], error=(p.stderr or "")[-300:])
        except subprocess.TimeoutExpired:
            return dict(test=t[1], error=f"timeout {TIMEOUT} s")

    recs = []
    with ThreadPoolExecutor(a.par) as ex, open(path, "w", encoding="utf-8", newline="\n") as fh:
        for i, f in enumerate(as_completed([ex.submit(go, t) for t in ts]), 1):
            r = f.result()
            recs.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[{i}/{len(ts)}] {r['test'][-45:]}: " + (r["error"][:60] if "error" in r else
                  f"warm {r['warm']['total']:.3f} s (QK2 {r['qk2_warm']:.3f} s)"), flush=True)
    report(a.out, recs)


def report(out, recs):
    ok = [r for r in recs if "error" not in r]
    L = ["# STAGETIME (exploratory, nothing predicted)", "",
         f"{len(ok)} of {len(recs)} FakeTorino tests; c33's default call with backend=, timers around its stages.", ""]
    for label in ("warm", "cold"):
        for name, sel in (("small (QK2 warm < 0.5 s)", [r for r in ok if r["qk2_warm"] < 0.5]),
                          ("large (QK2 warm >= 0.5 s)", [r for r in ok if r["qk2_warm"] >= 0.5])):
            if not sel:
                continue
            tot = sum(r[label]["total"] for r in sel)
            qk = sum(r["qk2_warm"] for r in sel)
            L += [f"## {label}, {name}: {len(sel)} tests; c33 {tot:.2f} s summed, Qiskit level 2 {qk:.2f} s "
                  f"warm (median per test: c33 {statistics.median(r[label]['total'] for r in sel) * 1000:.1f} ms, "
                  f"QK2 warm {statistics.median(r['qk2_warm'] for r in sel) * 1000:.1f} ms)", "",
                  "| stage | summed (s) | share of c33's time | median per test (ms) |", "|---|---|---|---|"]
            for st in OUTER + ("other",) + INNER:
                v = [r[label].get(st, 0.0) for r in sel]
                L.append(f"| {st}{' (inside pm.run)' if st in INNER else ''} | {sum(v):.3f} | "
                         f"{sum(v) / tot:.1%} | {statistics.median(v) * 1000:.2f} |")
            L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(out, "stagetime.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    a = ap.parse_args()
    {"run": run, "one": one}[a.mode](a)


if __name__ == "__main__":
    main()
