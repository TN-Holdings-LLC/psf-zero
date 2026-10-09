"""c27_probe.py -- exploratory (2026-10-09; not pre-registered, nothing predicted): checks two explanations Addendum
417 gave for C27-B's results, on the six tests where the budget bound.

1. B3's cause. Addendum 417 says the 37.8 s that WB spent on hwb10's refused calls went into listing the circuit's
   instructions before the count. Here, on each WB output, the parts of a refused call are timed separately:
     data_len     len(circ.data)
     count_ops    circ.count_ops() (what a cheaper pre-count would use)
     listing      the list of (operation, qubit indices) that excitation_cost builds before it counts
     touched      _touched_qubits(circ) (what the checks build before they count)
     refused      a whole excitation_cost call with no budget left
2. Where the quality goes. For each refused call, the function that asked for it (its caller: _select_resynthesis,
   _choose_lazy, _compare_level3, ...) is recorded, with the counters of the decisions (RESYNTH_STATS, COMPARE_STATS,
   EXACT_STATS, SKIP_STATS) for NB (budget never reached) and WB (default budget), and the two-qubit gate counts.

Candidate 2026-10-09.c27, the recommended call, virtual layout clock and PYTHONHASHSEED=0, as C27-B. NB is not run on
hwb10 (it does not finish in 600 s).

Caps: each job is killed after --job-timeout s (900); no job starts after --budget s (3600); 3 jobs at a time. The
output folder must not exist.

    python benchmarks/c27_probe.py run --bp <benchpress clone> --out DIR [--par 3]
    python benchmarks/c27_probe.py report --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
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
TESTS = ("hwb10.qasm]", "ham_ham_JW-14]", "ham_enc_gray_dvalues_4-4-", "ham_enc_gray_dvalues_8-8-8]", "ham_ham_JW-10]",
         "ham_ham_parity10]")
ARMS = {"NB": 1e6, "WB": None}
STATS = ("RESYNTH_STATS", "COMPARE_STATS", "EXACT_STATS", "SKIP_STATS", "WORK_STATS")


def _timed(f, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        r = f()
    return r, (time.perf_counter() - t0) / reps


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
    draws = []
    orig = pc._work_draw

    def spy(fn, ops, amps):
        ok = orig(fn, ops, amps)
        caller = sys._getframe(2).f_code.co_name  # 0: spy, 1: the estimate or check, 2: who asked for it
        draws.append([caller, fn, pc.WORK_PER_OP * int(ops) + pc.WORK_PER_AMP[fn] * int(amps), ok])
        return ok

    pc._work_draw = spy
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, **C.RECOMMENDED)
    if ARMS[job["arm"]] is not None:
        kw["work_budget_s"] = ARMS[job["arm"]]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    rec = dict(job, version=pc.VERSION, t=round(time.perf_counter() - t0, 2),
               q2=int(out.count_ops().get(backend.two_q_gate_type, 0)), draws=draws,
               stats={s: dict(getattr(pc, s)) for s in STATS})
    if job["arm"] == "WB":
        n, t_len = _timed(lambda: len(out.data))
        _, t_co = _timed(lambda: out.count_ops())
        _, t_list = _timed(lambda: [(ins.operation, tuple(out.find_bit(b).index for b in ins.qubits))
                                    for ins in out.data if ins.operation.name not in ("barrier", "measure", "delay")])
        _, t_touch = _timed(lambda: pc._touched_qubits(out))
        pc._WORK["left"] = 0
        try:
            r, t_ref = _timed(lambda: pc.excitation_cost(out, backend.target))
        finally:
            pc._WORK["left"] = None
        rec["probe"] = dict(instructions=n, data_len_s=round(t_len, 6), count_ops_s=round(t_co, 4),
                            listing_s=round(t_list, 3), touched_s=round(t_touch, 3), refused_call_s=round(t_ref, 3),
                            refused_result=r)
    print(json.dumps(rec), flush=True)


def run(args):
    import c25_identity as C
    if os.path.exists(args.out):
        sys.exit(f"STOP: {args.out} exists; a run never replaces an earlier run's files")
    os.makedirs(args.out)
    pop = C.population(args.bp)
    js = []
    for key in TESTS:
        hits = [t for t in pop if key in t[1] and (t[0].endswith("FakeTorino") or t[2] == "summit")]
        if len(hits) != 1:
            sys.exit(f"STOP: {len(hits)} tests match {key}")
        stratum, tid, kind, arg = hits[0]
        for arm in ARMS:
            if arm == "NB" and key.startswith("hwb10"):
                continue
            js.append(dict(stratum=stratum, test=tid, kind=kind, arg=arg, arm=arm))
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    path = os.path.join(args.out, "c27_probe.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=head.strip(), dirty_tracked=dirty, jobs=len(js), par=args.par,
                                           job_timeout=args.job_timeout, budget=args.budget))) + "\n")
    env = dict(os.environ, PYTHONHASHSEED="0")
    t_start = time.perf_counter()

    def go(j):
        if time.perf_counter() - t_start > args.budget:
            return dict(j, error="not started (budget)")
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", "--bp", args.bp, "--job",
                                json.dumps(j)], capture_output=True, text=True, timeout=args.job_timeout, env=env,
                               encoding="utf-8", errors="replace")
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            return json.loads(lines[-1]) if lines else dict(j, error=(p.stderr or "")[-300:])
        except subprocess.TimeoutExpired:
            return dict(j, error=f"killed after {args.job_timeout} s")

    with ThreadPoolExecutor(args.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for n, rec in enumerate(ex.map(go, js), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{n}/{len(js)} {time.perf_counter() - t_start:5.0f} s] {rec['arm']} {rec['test'][-45:]}: "
                  + (rec["error"][:60] if "error" in rec else f"{rec['t']} s, q2 {rec['q2']}"), flush=True)
    report(args)


def report(args):
    lines = [json.loads(x) for x in open(os.path.join(args.out, "c27_probe.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    L = ["# c27_probe (exploratory, nothing predicted)", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['jobs']} jobs, {meta['par']} at a time", "",
         "## 1. The parts of a refused call (on WB's output)", "",
         "| test | instructions | count_ops s | listing s | touched s | refused excitation_cost s |",
         "|---|---|---|---|---|---|"]
    for r in recs:
        if r["arm"] == "WB" and "probe" in r:
            p = r["probe"]
            L.append(f"| {r['test'][-45:]} | {p['instructions']} | {p['count_ops_s']} | {p['listing_s']} | "
                     f"{p['touched_s']} | {p['refused_call_s']} |")
    L += ["", "## 2. Who asked for each call, and what was refused", ""]
    for key in TESTS:
        rs = [r for r in recs if key in r["test"]]
        if not rs:
            continue
        L += [f"### {rs[0]['test'][-60:]}", ""]
        for r in rs:
            if "error" in r:
                L.append(f"- {r['arm']}: {r['error'][:80]}")
                continue
            L.append(f"- {r['arm']}: q2 {r['q2']}, {r['t']} s")
            for caller, fn, w, ok in r["draws"]:
                L.append(f"  - {caller} -> {fn}: {w / 1e9:.2f} s of work, {'made' if ok else 'REFUSED'}")
            for s in STATS:
                nz = {k: v for k, v in r["stats"][s].items() if v}
                if nz:
                    L.append(f"  - {s}: {nz}")
        L.append("")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "c27_probe.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "report", "child"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--job-timeout", type=int, default=900)
    ap.add_argument("--budget", type=int, default=3600)
    a = ap.parse_args()
    {"run": run, "report": report, "child": child}[a.mode](a)


if __name__ == "__main__":
    main()
