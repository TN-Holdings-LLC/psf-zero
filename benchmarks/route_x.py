"""route_x.py -- ROUTE-X (2026-10-10; exploratory, written down before the run: see the docstring's last paragraph):
does spending more time on routing close release 2026-10-10.3's ESP gap to Qiskit level 3? C35-VAL (Addenda 436-437)
found the plain call with backend= at 0.957 (FakeTorino) and 0.899 (FakeKingston) of level 3's ESP, with about 11%
more two-qubit gates, most of them from routing. The release routes with Qiskit's preset pass manager at
`routing_optimization_level=1`, chosen when compile time mattered; levels 2 and 3 run more layout and routing trials
and more optimisation.

Tests and devices: C35-VAL's (BP-FINAL's 106 FakeTorino tests; FakeTorino and FakeKingston). Arms, each test, device
and arm in its own process, one cold call, 4 at a time:
  R1   psf_compile.py (2026-10-10.3), the plain call with backend= (routing_optimization_level=1, as released)
  R2   the same with routing_optimization_level=2
  R3   the same with routing_optimization_level=3
  QK2  Qiskit level 2 on the backend
  QK3  Qiskit level 3 on the backend
ESP, failed elements and "runnable" (best ESP over the arms >= 0.01) as in C35-VAL (benchmarks/c35_val.py).

Expectations stated before the run (2026-10-10, session): R3's two-qubit count moves toward QK3's and its ESP rises
above R1's; whether it reaches QK3 is open. Levels 2-3 re-run Qiskit's block consolidation over PSF-Zero's output
(psf_compile's docstring), so part of PSF-Zero's own work may be redone by Qiskit. Time is reported, not judged.

    cd <psf-zero repository>
    python <this file> run --bp <benchpress clone> --out DIR [--par 4] [--smoke]
    python <this file> report --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), HERE, REPO]
import c35_val as V  # noqa: E402  (benchmarks/c35_val.py, locked in Addendum 436)

ARMS = ("R1", "R2", "R3", "QK2", "QK3")
LEVEL = {"R1": 1, "R2": 2, "R3": 3}
DEVICES = V.DEVICES
TIMEOUT, BUDGET = 1500, 7200


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    import core_fix_c2_eval as H
    qc, bp_backend = B.build(a.bp, kind, arg)
    backend = V.device(a.device, bp_backend)
    target = backend.target
    rec = dict(stratum=stratum, test=tid, device=a.device, arm=a.arm, input_qubits=qc.num_qubits)
    if a.arm in ("QK2", "QK3"):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        t0 = time.perf_counter()
        out = generate_preset_pass_manager(2 if a.arm == "QK2" else 3, backend).run(qc)
        rec["t"] = round(time.perf_counter() - t0, 3)
    else:
        H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
        pc = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
        rec["version"] = pc.VERSION
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = pc.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True,
                                              seed_transpiler=0, routing_optimization_level=LEVEL[a.arm])
        except Exception as exc:  # noqa: BLE001 - recorded
            rec.update(t=round(time.perf_counter() - t0, 3), error=f"{type(exc).__name__}: {exc}"[:300])
            print(json.dumps(rec), flush=True)
            return
        rec["t"] = round(time.perf_counter() - t0, 3)
    rec["q2"] = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "delay"))
    lg, n_failed, n_missing = V.esp_of(out, target)
    rec.update(esp=None if lg == V.NEG else round(lg, 6), on_failed=n_failed, not_in_target=n_missing)
    print(json.dumps(rec), flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    ts = V.tests(a.bp)
    if a.smoke:
        rows = json.load(open(os.path.join(REPO, "data", "2026-10-08", "bp_final", "bp_final.json"), encoding="utf-8"))
        t_recr = {r["test"]: r["t"] for r in rows["rows"] if r.get("arm") == "RECR" and "t" in r}
        ts = sorted(ts, key=lambda t: t_recr.get(t[1], math.inf))[:1]
    jobs = sorted((hashlib.sha256(("ROUTE-X|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "route_x.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           head_subject=git("log", "-1", "--format=%s")[:80],
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par))) + "\n")
    print(f"ROUTE-X: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
    t_start = time.perf_counter()

    def go(job):
        _, t, d, arm = job
        base = dict(stratum=t[0], test=t[1], device=d, arm=arm)
        if time.perf_counter() - t_start > BUDGET:
            return dict(base, error="not started (budget)")
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--arm", arm,
                                "--device", d, "--job", json.dumps(list(t))], capture_output=True, text=True,
                               timeout=TIMEOUT, cwd=REPO, env=dict(os.environ, PSF_ZERO_REPO=REPO))
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            return json.loads(lines[-1]) if lines else dict(base, error=(p.stderr or "")[-300:], exit=p.returncode)
        except subprocess.TimeoutExpired:
            return dict(base, error=f"timeout {TIMEOUT} s")

    n = 0
    with ThreadPoolExecutor(a.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in jobs]):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:6.0f} s] {r['device'][4:]:8s} {r['arm']:3s} "
                  f"{r['test'][-40:]}: " + (r["error"][:50] if "error" in r else
                                           f"{r['t']} s, q2 {r['q2']}, failed {r['on_failed']}, esp {r['esp']}"),
                  flush=True)
    report(a)


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "route_x.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    lg = lambda r: -math.inf if r.get("esp") is None else r["esp"]  # noqa: E731
    ok = lambda r: r is not None and "error" not in r  # noqa: E731
    L = ["# ROUTE-X (exploratory)", "",
         f"git head {meta['git_head']} ({meta['head_subject']}); {meta['tests']} tests on {', '.join(DEVICES)}; "
         f"{meta['jobs']} jobs.", ""]
    for d in DEVICES:
        keys = sorted(k for k in by if k[0] == d)
        full = [k for k in keys if all(ok(by[k].get(x)) for x in ARMS)]
        run_ = [k for k in full if max(lg(by[k][x]) for x in ARMS) >= -2]
        L += [f"## {d}", "", f"{len(keys)} tests; {len(full)} with every arm; {len(run_)} runnable.", "",
              "| arm | returned | outputs on failed elements | ESP = 0 on runnable | two-qubit / QK3 (gmean, +1) | "
              "time summed (s) | median time (s) |", "|---|---|---|---|---|---|---|"]
        for x in ARMS:
            good = [by[k][x] for k in keys if ok(by[k].get(x))]
            q = [math.log((by[k][x]["q2"] + 1) / (by[k]["QK3"]["q2"] + 1)) for k in full]
            ts = sorted(r["t"] for r in good)
            L.append(f"| {x} | {len(good)} of {len(keys)} | {sum(1 for r in good if r['on_failed'])} | "
                     f"{sum(1 for k in run_ if by[k][x]['esp'] is None)} | "
                     f"{math.exp(sum(q) / len(q)) if q else float('nan'):.3f} | {sum(ts):.0f} | "
                     f"{ts[len(ts) // 2] if ts else float('nan'):.3f} |")
        L += ["", "| ESP ratio, runnable tests, both > 0 | gmean | 10% better | 10% worse | tests |", "|---|---|---|---|---|"]
        for x, o in (("R1", "QK3"), ("R2", "QK3"), ("R3", "QK3"), ("R3", "R1"), ("R2", "R1"), ("R1", "QK2"),
                     ("R3", "QK2")):
            diff = [lg(by[k][x]) - lg(by[k][o]) for k in run_]
            fin = [v for v in diff if math.isfinite(v)]
            L.append(f"| {x} / {o} | {10 ** (sum(fin) / len(fin)) if fin else float('nan'):.3f} | "
                     f"{sum(1 for v in diff if v >= math.log10(1.1))} | {sum(1 for v in diff if v <= -math.log10(1.1))} "
                     f"| {len(fin)} |")
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "route_x.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "report"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--device")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "one": one, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
