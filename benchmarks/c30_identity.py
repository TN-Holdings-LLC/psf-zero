"""c30_identity.py -- C30-ID (2026-10-10; pre-registered in Addendum 426): candidate 2026-10-10.c30 (changelog item 57b:
the estimates' and checks' per-gate loops in psf_zero_core57) against release 2026-10-10.1 on the 152 development
tests of C25-ID2, whole compiled outputs by value, with a control arm. C29-ID's script (benchmarks/c29_identity.py)
with the arms and predictions changed; its `value_sig` is used unchanged.

Arms (each test, call and arm in its own process; one compile; the layout search's clock virtual, PYTHONHASHSEED=0):
  REL   psf_compile.py, benchmarks/psf_smart_layout.py (release 2026-10-10.1, layout 2026-10-01.1)
  REL2  REL again, in its own process (control)
  C30   patches/psf_compile_c30_2026-10-10/psf_compile.py, benchmarks/psf_smart_layout.py, with psf_zero_core57
        (patches/psf_zero_core57_2026-10-10) installed
Calls as C25-ID2: "default" on every test, "recommended" (with the backend's target) on the FakeTorino and summit tests.

Caps: a job is killed after 3,600 s; no job starts after 14,400 s; 4 jobs at a time. The output folder must not exist.

    python benchmarks/c29_identity.py list    --bp <benchpress clone>
    python benchmarks/c29_identity.py run     --bp <benchpress clone> --out DIR [--par 4]
    python benchmarks/c29_identity.py compare --out DIR
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PATCH = os.path.join(REPO, "patches", "psf_compile_c30_2026-10-10")
sys.path[:0] = [HERE, REPO]
import bp_mock as B           # noqa: E402  (BP-MOCK, unchanged)
import c25_identity2 as C2    # noqa: E402  (C25-ID2's population, calls and virtual clock, unchanged)
from c29_identity import value_sig  # noqa: E402  (C29-ID's comparison by value, unchanged)

LAYOUT = os.path.join(HERE, "psf_smart_layout.py")
ARMS = {"REL": os.path.join(REPO, "psf_compile.py"), "REL2": os.path.join(REPO, "psf_compile.py"),
        "C30": os.path.join(PATCH, "psf_compile.py")}
VERSIONS = {"REL": "2026-10-10.1", "REL2": "2026-10-10.1", "C30": "2026-10-10.c30"}
CORE57 = "2026-10-10.c30"
LAYOUT_VERSION = "2026-10-01.1"
STATS = ("COMPARE_STATS", "EXACT_STATS", "RESYNTH_STATS", "SKIP_STATS", "FEASIBILITY_STATS", "CORE57_STATS")
TIMEOUT, BUDGET = 3600, 14_400
N_TESTS = 152
RATIO_MAX = 0.85  # RC3


def jobs(bp):
    out = []
    for stratum, tid, kind, arg in C2.population(bp):
        calls = ("default", "recommended") if stratum.endswith("FakeTorino") or kind == "summit" else ("default",)
        h = hashlib.sha256(("C30-ID|" + tid).encode()).hexdigest()
        arms = [("REL", "REL2", "C30"), ("C30", "REL", "REL2"), ("REL2", "C30", "REL")][int(h, 16) % 3]
        for call in calls:
            out += [(h, dict(stratum=stratum, test=tid, kind=kind, arg=arg, call=call, arm=a)) for a in arms]
    return [j for _, j in sorted(out, key=lambda x: x[0])]


def one(args):
    warnings.simplefilter("ignore")
    job = json.loads(args.job)
    qc, backend = B.build(args.bp, job["kind"], job["arg"])
    import core_fix_c2_eval as H
    lay = H.load_module(LAYOUT, "psf_smart_layout")
    lay.time = C2._VirtualTime()
    pc = H.load_module(ARMS[job["arm"]], "psf_compile")
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0)
    if job["call"] == "recommended":
        kw.update(target=backend.target, **C2.RECOMMENDED)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    rec = dict(job, version=pc.VERSION, layout_version=lay.LAYOUT_VERSION, clock=type(lay.time).__name__,
               t=round(time.perf_counter() - t0, 3), q2=int(out.count_ops().get(backend.two_q_gate_type, 0)),
               sig=B.sig_hash(out), vsig=value_sig(out),
               stats={s: {k: v for k, v in dict(getattr(pc, s)).items() if v} for s in STATS if hasattr(pc, s)},
               core57=getattr(pc, "CORE57_VERSION", None))
    print(json.dumps(rec), flush=True)


def run(args):
    if os.path.exists(args.out):
        sys.exit(f"STOP: {args.out} exists; a run never replaces an earlier run's files")
    js = jobs(args.bp)
    os.makedirs(args.out)
    path = os.path.join(args.out, "c30_identity.jsonl")
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout
    meta = dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_head=head.strip(),
                dirty_tracked=dirty.strip(), tests=len({j["test"] for j in js}), jobs=len(js), par=args.par,
                cpus=os.cpu_count(), python=sys.version.split()[0], timeout=TIMEOUT, budget=BUDGET)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=meta)) + "\n")
    env = dict(os.environ, PYTHONHASHSEED="0")
    t_start = time.perf_counter()

    def go(j):
        if time.perf_counter() - t_start > BUDGET:
            return dict(j, error="not started (budget)")
        cmd = [sys.executable, os.path.abspath(__file__), "one", "--bp", args.bp, "--job", json.dumps(j)]
        w0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, encoding="utf-8",
                               errors="replace", env=env)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else dict(j, error=(p.stderr or "")[-400:], exit=p.returncode)
        except subprocess.TimeoutExpired:
            rec = dict(j, error=f"timeout {TIMEOUT} s")
        rec["wall"] = round(time.perf_counter() - w0, 2)
        return rec

    n = 0
    with ThreadPoolExecutor(args.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in js]):
            rec = f.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            print(f"[{n}/{len(js)} {time.perf_counter() - t_start:6.0f} s] {rec['arm']} {rec['call']} "
                  f"{rec['test'][:60]}" + (" ERROR " + rec["error"][:40] if "error" in rec else f" {rec['wall']} s"),
                  flush=True)
    compare(args)


def compare(args):
    lines = [json.loads(x) for x in open(os.path.join(args.out, "c30_identity.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault((r["test"], r["call"]), {})[r["arm"]] = r
    arms = ("REL", "REL2", "C30")
    ok = lambda r: r is not None and "error" not in r  # noqa: E731
    both = lambda k, x, y: ok(by[k].get(x)) and ok(by[k].get(y))  # noqa: E731
    keys = sorted(by)
    ctrl_done = [k for k in keys if both(k, "REL", "REL2")]
    ctrl_diff = [k for k in ctrl_done if by[k]["REL"]["vsig"] != by[k]["REL2"]["vsig"]]
    scored = [k for k in ctrl_done if k not in ctrl_diff and ok(by[k].get("C30"))]
    cand_diff = [k for k in scored if by[k]["C30"]["vsig"] != by[k]["REL"]["vsig"]]
    unscored = [k for k in keys if k not in scored]
    fail = {a: sorted(k for k in keys if a in by[k] and "error" in by[k][a]) for a in arms}
    new_fail = [k for k in fail["C30"] if not (k in fail["REL"] and k in fail["REL2"])]
    present = all(a in by[k] for k in keys for a in arms)
    versions_ok = all(r.get("version") == VERSIONS[r["arm"]] and r.get("layout_version") == LAYOUT_VERSION
                      and r.get("clock") == "_VirtualTime" and (r["arm"] != "C30" or r.get("core57") == CORE57)
                      for r in recs if "error" not in r)
    RC0 = meta["tests"] == N_TESTS and not meta["dirty_tracked"] and versions_ok and present

    def estimates(r):
        s = r["stats"]
        return sum(s.get("COMPARE_STATS", {}).values()) + sum(s.get("RESYNTH_STATS", {}).values())

    # RC3's set: recommended test-calls finished in all three arms on which REL made at least one estimate
    est = [k for k in keys if k[1] == "recommended" and all(ok(by[k].get(a)) for a in arms)
           and estimates(by[k]["REL"]) > 0]

    def gmean(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

    g = gmean([by[k]["C30"]["t"] / by[k]["REL"]["t"] for k in est])
    g_ctrl = gmean([by[k]["REL2"]["t"] / by[k]["REL"]["t"] for k in est])
    tsum = {a: sum(by[k][a]["t"] for k in est) for a in arms}
    rust = sum(by[k]["C30"]["stats"].get("CORE57_STATS", {}).get("rust", 0) for k in keys if ok(by[k].get("C30")))
    py = sum(by[k]["C30"]["stats"].get("CORE57_STATS", {}).get("python", 0) for k in keys if ok(by[k].get("C30")))
    RC1 = bool(scored) and not cand_diff
    RC2 = not new_fail
    RC3 = len(est) >= 5 and g <= RATIO_MAX
    v = lambda c: "CONFIRMED" if c else "REFUTED"  # noqa: E731
    L = ["# C30-ID", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['tests']} tests (expected {N_TESTS}), {len(keys)} test-calls, {meta['jobs']} jobs, {meta['par']} at "
         f"a time, {meta['cpus']} CPUs, Python {meta['python']}", "",
         "| ID | prediction | value | verdict |", "|---|---|---|---|",
         f"| RC0 | the run is as locked | {N_TESTS} tests, every arm on every test-call, versions, virtual clock and "
         f"psf_zero_core57 {CORE57} in every record, no uncommitted change: {RC0} | **{'PASS' if RC0 else 'FAIL'}** |",
         f"| RC1 | C30 = REL by value wherever REL2 = REL | {len(scored) - len(cand_diff)} identical of {len(scored)} "
         f"scored | **{v(RC1)}** |",
         f"| RC2 | C30 fails nowhere that REL and REL2 do not | C30 {len(fail['C30'])} failures, {len(new_fail)} new | "
         f"**{v(RC2)}** |",
         f"| RC3 | recommended calls with estimates: geometric mean of C30 / REL compile time <= {RATIO_MAX} | "
         f"{g:.3f} on {len(est)} test-calls | **{v(RC3)}** |", "",
         "Reported without prediction:", "",
         f"- control: REL2 = REL by value on {len(ctrl_done) - len(ctrl_diff)} of {len(ctrl_done)} test-calls both "
         f"finished; on RC3's set, geometric mean of REL2 / REL time {g_ctrl:.3f}",
         f"- not scored ({len(unscored)}): REL2 differs from REL, or an arm failed",
         f"- RC3's set, summed compile time: REL {tsum['REL']:.0f} s, REL2 {tsum['REL2']:.0f} s, "
         f"C30 {tsum['C30']:.0f} s",
         f"- C30's estimate and check calls: {rust} in psf_zero_core57, {py} in Python"]
    L += [f"  - C30 differs: {t} ({c})" for t, c in cand_diff[:30]]
    L += [f"  - REL2 differs (not scored): {t} ({c})" for t, c in ctrl_diff[:30]]
    L += [f"  - RC3: {t[-50:]}: REL {by[(t, c)]['REL']['t']:.1f} s, C30 {by[(t, c)]['C30']['t']:.1f} s "
          f"({by[(t, c)]['C30']['t'] / by[(t, c)]['REL']['t']:.2f})" for t, c in est]
    L += [f"  - failed ({a}): {t} ({c}): {(by[(t, c)][a]['error'].strip().splitlines() or ['no message'])[-1][-80:]}"
          for a in arms for t, c in fail[a][:20]]
    L += ["", "SUMMARY " + json.dumps(dict(RC0="PASS" if RC0 else "FAIL", RC1=v(RC1), RC2=v(RC2), RC3=v(RC3)))]
    txt = "\n".join(L) + "\n"
    open(os.path.join(args.out, "compare.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("list", "run", "one", "compare"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "list":
        js = jobs(a.bp)
        print(f"{len({j['test'] for j in js})} tests, {len({(j['test'], j['call']) for j in js})} test-calls, "
              f"{len(js)} jobs")
        return
    {"run": run, "one": one, "compare": compare}[a.mode](a)


if __name__ == "__main__":
    main()
