"""c27_budget.py -- C27-B (2026-10-09; pre-registered in Addendum 416): candidate 2026-10-09.c27 (changelog item 54,
a work budget shared by the recommended call's estimates and exactness checks) on the 34 development tests with a
target (the FakeTorino and summit tests of C25-ID's 152; those of Addendum 415; none of BP-FINAL's 880).

Arms (each test and arm in its own process; the recommended call; virtual layout clock as C25-ID2; PYTHONHASHSEED=0):
  NB    c27 with work_budget_s=1e6: a budget never reached, so it decides as no budget (c26) does, and counts the work
  NB2   the same again: how often two processes disagree (Addenda 410-412)
  WB    c27 with its default budget (WORK_BUDGET_S = 10)
The three arms of a test are next to each other in the queue, in an order set by the test's hash.

Recorded per job: the output's signature (bp_mock.sig_hash) and two-qubit gate count; the compile time; every draw
on the budget (function, work, made or refused); the wall time of the estimates and checks.

Caps: each job is killed after --job-timeout s (600); no job starts after --budget s (7200). The output folder must
not exist.

    python benchmarks/c27_budget.py run     --bp <benchpress clone> --out DIR [--par 4]
    python benchmarks/c27_budget.py score   --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PATCH = os.path.join(REPO, "patches", "psf_compile_c27_2026-10-09")
sys.path[:0] = [HERE, REPO]
VERSION = "2026-10-09.c27"
ARMS = {"NB": 1e6, "NB2": 1e6, "WB": None}  # work_budget_s; None: the candidate's default
COUNTED = ("excitation_cost", "hybrid_cost", "pauli_cost", "kraus_cost", "_implements", "_same_action")
N_TESTS = 34
CALL_TIME_BOUND_S = 30.0  # B3: the wall time of WB's estimates and checks, per test


def child(args):
    warnings.simplefilter("ignore")
    import bp_mock as B
    import c25_identity as C
    import c25_identity2 as C2
    import core_fix_c2_eval as H
    job = json.loads(args.job)
    qc, backend = B.build(args.bp, job["kind"], job["arg"])
    lay = H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    lay.time = C2._VirtualTime()
    pc = H.load_module(os.path.join(PATCH, "psf_compile.py"), "psf_compile")
    draws, times = [], {f: 0.0 for f in COUNTED}
    orig_draw = pc._work_draw

    def spy(fn, ops, amps):
        ok = orig_draw(fn, ops, amps)
        draws.append((fn, pc.WORK_PER_OP * int(ops) + pc.WORK_PER_AMP[fn] * int(amps), ok))
        return ok

    pc._work_draw = spy

    def timed(name, f):
        @functools.wraps(f)
        def g(*a, **kw):
            t0 = time.perf_counter()
            try:
                return f(*a, **kw)
            finally:
                times[name] += time.perf_counter() - t0
        return g

    for name in COUNTED:
        setattr(pc, name, timed(name, getattr(pc, name)))
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, **C.RECOMMENDED)
    if ARMS[job["arm"]] is not None:
        kw["work_budget_s"] = ARMS[job["arm"]]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    t = time.perf_counter() - t0
    rec = dict(job, version=pc.VERSION, layout_version=lay.LAYOUT_VERSION, clock=type(lay.time).__name__,
               budget_units=int(round((ARMS[job["arm"]] or pc.WORK_BUDGET_S) * pc.WORK_UNITS_PER_S)),
               t=round(t, 3), q2=int(out.count_ops().get(backend.two_q_gate_type, 0)), sig=B.sig_hash(out),
               work_made=sum(w for _, w, ok in draws if ok), refused=[fn for fn, _, ok in draws if not ok],
               draws=len(draws), calls_s={k: round(v, 3) for k, v in times.items() if v})
    print(json.dumps(rec), flush=True)


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def jobs(bp):
    import c25_identity as C
    out = []
    for stratum, tid, kind, arg in C.population(bp):
        if not (stratum.endswith("FakeTorino") or kind == "summit"):
            continue
        h = hashlib.sha256(("C27-B|" + tid).encode()).hexdigest()
        order = sorted(ARMS, key=lambda a: hashlib.sha256((h + a).encode()).hexdigest())
        out += [(h, j, dict(stratum=stratum, test=tid, kind=kind, arg=arg, arm=a)) for j, a in enumerate(order)]
    return [x for _, _, x in sorted(out, key=lambda y: (y[0], y[1]))]


def run(args):
    if os.path.exists(args.out):
        sys.exit(f"STOP: {args.out} exists; a run never replaces an earlier run's files")
    os.makedirs(args.out)
    js = jobs(args.bp)
    path = os.path.join(args.out, "c27_budget.jsonl")
    meta = dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_head=git("rev-parse", "--short",
                "HEAD"), dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                tests=len({j["test"] for j in js}), jobs=len(js), par=args.par, job_timeout=args.job_timeout,
                budget=args.budget, cpus=os.cpu_count(), python=sys.version.split()[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=meta)) + "\n")
    t_start = time.perf_counter()
    env = dict(os.environ, PYTHONHASHSEED="0")

    def go(j):
        if time.perf_counter() - t_start > args.budget:
            return dict(j, error="not started (budget)")
        w0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", "--bp", args.bp, "--job",
                                json.dumps(j)], capture_output=True, text=True, timeout=args.job_timeout, env=env,
                               encoding="utf-8", errors="replace")
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else dict(j, error=(p.stderr or "")[-300:])
        except subprocess.TimeoutExpired:
            rec = dict(j, error=f"killed after {args.job_timeout} s")
        rec["wall"] = round(time.perf_counter() - w0, 2)
        return rec

    n = 0
    with ThreadPoolExecutor(args.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for rec in ex.map(go, js):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n += 1
            print(f"[{n}/{len(js)} {time.perf_counter() - t_start:6.0f} s] {rec['arm']:3s} {rec['test'][:60]}: "
                  + (rec["error"][:60] if "error" in rec else f"{rec['t']:.1f} s, refused {len(rec['refused'])}"),
                  flush=True)
    score(args)


def score(args):
    lines = [json.loads(x) for x in open(os.path.join(args.out, "c27_budget.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault(r["test"], {})[r["arm"]] = r
    ok_rec = [r for r in recs if "error" not in r]
    asrun = (meta["tests"] == N_TESTS and len(recs) == 3 * N_TESTS and not meta["dirty_tracked"]
             and all(r["version"] == VERSION and r["clock"] == "_VirtualTime" for r in ok_rec)
             and all(set(v) == set(ARMS) for v in by.values()))
    budget = 10 * 1_000_000_000  # WORK_BUDGET_S (10) in units
    stable, b1_bad, b2_bad, b2_n, binds = [], [], [], 0, []
    for test, v in sorted(by.items()):
        nb, nb2, wb = v.get("NB", {}), v.get("NB2", {}), v.get("WB", {})
        if any("error" in x or not x for x in (nb, nb2, wb)):
            continue
        if nb["sig"] != nb2["sig"]:
            continue
        stable.append(test)
        if not wb["refused"] and wb["sig"] != nb["sig"]:
            b1_bad.append(test)
        b2_n += 1
        over = nb["work_made"] > budget
        if over != bool(wb["refused"]):
            b2_bad.append(test)
        if wb["refused"]:
            binds.append(test)
    wb_all = [v["WB"] for v in by.values() if "WB" in v]
    wb_fail = [r["test"] for r in wb_all if "error" in r]
    slow = [(r["test"], sum(r["calls_s"].values())) for r in wb_all if "error" not in r
            and sum(r["calls_s"].values()) > CALL_TIME_BOUND_S]
    b1_n = sum(1 for t in stable if not by[t]["WB"]["refused"])
    verdict = lambda ok: "CONFIRMED" if ok else "REFUTED"  # noqa: E731
    L = ["# C27-B", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['tests']} tests (expected {N_TESTS}), {meta['jobs']} jobs, {meta['par']} at a time, job timeout "
         f"{meta['job_timeout']} s, {meta['cpus']} CPUs, Python {meta['python']}", "",
         f"- B0 the run is as locked: **{'PASS' if asrun else 'FAIL'}**",
         f"- tests where NB and NB2 both finished with the same output: {len(stable)} of {len(by)}",
         f"- B1 where WB refused nothing, WB's output is NB's: {b1_n - len(b1_bad)} of {b1_n}: "
         f"**{verdict(not b1_bad)}**" + (f" (differ: {', '.join(t[:50] for t in b1_bad)})" if b1_bad else ""),
         f"- B2 WB refuses a call exactly where NB's work exceeds the budget: {b2_n - len(b2_bad)} of {b2_n}: "
         f"**{verdict(not b2_bad)}**" + (f" (not: {', '.join(t[:50] for t in b2_bad)})" if b2_bad else ""),
         f"- B3 every WB job finishes ({len(wb_all) - len(wb_fail)} of {len(wb_all)}), and its estimates and checks "
         f"take at most {CALL_TIME_BOUND_S:.0f} s per test ({len(slow)} over): **{verdict(not wb_fail and not slow)}**"
         + (f" (failed: {', '.join(t[:50] for t in wb_fail)})" if wb_fail else "")
         + (f" (over: {', '.join(f'{t[:40]} {s:.1f} s' for t, s in slow)})" if slow else ""), "",
         "## Where the budget binds (reported without prediction)", "",
         "| test | NB work (s of units) | WB refused | q2 NB / NB2 / WB | compile s NB / NB2 / WB | "
         "estimates and checks s NB / WB |", "|---|---|---|---|---|---|"]
    for test, v in sorted(by.items()):
        wb = v.get("WB", {})
        if not (wb.get("refused") or "error" in wb or any("error" in v.get(a, {}) for a in ARMS)):
            continue

        def cell(a, key, fmt="{}"):
            r = v.get(a, {})
            return "x" if "error" in r or key not in r else fmt.format(r[key])

        def calls(a):
            r = v.get(a, {})
            return "x" if "error" in r or "calls_s" not in r else f"{sum(r['calls_s'].values()):.1f}"

        nbw = v.get("NB", {}).get("work_made")
        L.append(f"| {test[:60]} | {'x' if nbw is None else f'{nbw / 1e9:.1f}'} | "
                 f"{len(wb.get('refused', []))} | {cell('NB', 'q2')} / {cell('NB2', 'q2')} / {cell('WB', 'q2')} | "
                 f"{cell('NB', 't', '{:.1f}')} / {cell('NB2', 't', '{:.1f}')} / {cell('WB', 't', '{:.1f}')} | "
                 f"{calls('NB')} / {calls('WB')} |")
    L += ["", "x: the job did not finish (killed or error).", "",
          f"VERDICT {'B0-B3 HOLD' if asrun and not b1_bad and not b2_bad and not wb_fail and not slow else 'SEE ABOVE'}"]
    txt = "\n".join(L) + "\n"
    out = os.path.join(args.out, "c27_budget.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "score", "child"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--job-timeout", type=int, default=600)
    ap.add_argument("--budget", type=int, default=7200)
    a = ap.parse_args()
    {"run": run, "score": score, "child": child}[a.mode](a)


if __name__ == "__main__":
    main()
