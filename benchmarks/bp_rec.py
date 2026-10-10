"""bp_rec.py -- BP-REC (2026-10-10; exploratory, nothing predicted): the recommended call of release 2026-10-10.2
(items 53, 56 and 57b, with psf_zero_core57) against release 2026-10-07.1 on BP-FINAL's FakeTorino tests (HamLib and
Feynman on FakeTorino; BP-FINAL's population, bp_final.population, restricted to the FakeTorino strata).

Per test, in its own process each, one after the other in an order set by the test's hash, as a user would run it
(the real clock, no fixed hash seed):
  R071  patches/psf_compile_release_2026-10-07.1/psf_compile.py, the README's recommended call with the target
  R102  psf_compile.py (2026-10-10.2), the same call
Recorded: compile time, two-qubit count, two-qubit depth, the output by value (c29_identity.value_sig), two-qubit
gates on FakeTorino's failed elements, Benchpress's validation, and R102's CORE57_STATS. BP-FINAL's Qiskit level 2
(QK) two-qubit counts are read from data/2026-10-08/bp_final/bp_final.json for the ratios.

Caps: a job is killed after 1,500 s (BP-FINAL's limit); no job starts after 7,200 s; 4 at a time. The output folder
must not exist.

    cd <psf-zero repository>
    python <this file> run --bp <benchpress clone> --out DIR [--par 4] [--smoke: one fast test only]
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
import statistics
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), HERE, REPO]
ARMS = {"R071": os.path.join(REPO, "patches", "psf_compile_release_2026-10-07.1", "psf_compile.py"),
        "R102": os.path.join(REPO, "psf_compile.py")}
VERSIONS = {"R071": "2026-10-07.1", "R102": "2026-10-10.2"}
TIMEOUT, BUDGET = 1500, 7200


def tests(bp):
    import bp_final as F
    return [t for t in F.population(bp) if t[0].endswith("FakeTorino")]


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_final as F
    import bp_mock as B
    import core_fix_c2_eval as H
    from c29_identity import value_sig
    qc, backend = B.build(a.bp, kind, arg)
    from benchpress.qiskit_gym.utils.validation import qiskit_circuit_validation
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(ARMS[a.arm], "psf_compile")
    two_q = backend.two_q_gate_type
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, **F.RECOMMENDED)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    rec = dict(stratum=stratum, test=tid, arm=a.arm, version=pc.VERSION, t=round(time.perf_counter() - t0, 3),
               q2=int(out.count_ops().get(two_q, 0)), d2=out.depth(filter_function=lambda x: x.operation.name == two_q),
               vsig=value_sig(out), input_qubits=qc.num_qubits,
               core57=dict(getattr(pc, "CORE57_STATS", {})) or None)
    edges, qubits = pc._failed_elements(backend.target, 0.5)
    rec["on_failed"] = F.on_failed(out, edges, qubits)
    try:
        qiskit_circuit_validation(out, backend)
        rec["valid"] = True
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["valid"] = f"{type(exc).__name__}: {exc}"[:200]
    print(json.dumps(rec), flush=True)


def core57():
    try:
        import psf_zero_core57 as m
        return getattr(m, "CORE57_VERSION", "?")
    except ImportError:
        return None


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    ts = tests(a.bp)
    if a.smoke:  # the smoke run: the test BP-FINAL's recommended call compiled fastest
        rows = json.load(open(os.path.join(REPO, "data", "2026-10-08", "bp_final", "bp_final.json"), encoding="utf-8"))
        t_recr = {r["test"]: r["t"] for r in rows["rows"] if r.get("arm") == "RECR" and "t" in r}
        ts = sorted(ts, key=lambda t: t_recr.get(t[1], math.inf))[:1]
    jobs = sorted(((hashlib.sha256(("BP-REC|" + t[1] + arm).encode()).hexdigest(), t, arm) for t in ts for arm in ARMS))
    os.makedirs(a.out)
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    path = os.path.join(a.out, "bp_rec.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=head.strip(), dirty_tracked=dirty, tests=len(ts), jobs=len(jobs),
                                           par=a.par, timeout=TIMEOUT, budget=BUDGET, cpus=os.cpu_count(),
                                           core57=core57()))) + "\n")
    t_start = time.perf_counter()
    print(f"BP-REC: {len(ts)} tests, {len(jobs)} jobs, par {a.par}, head {head.strip()}, core57 {core57()}", flush=True)

    def go(job):
        _, t, arm = job
        if time.perf_counter() - t_start > BUDGET:
            return dict(stratum=t[0], test=t[1], arm=arm, error="not started (budget)")
        w0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--arm", arm,
                                "--job", json.dumps(list(t))], capture_output=True, text=True, timeout=TIMEOUT,
                               cwd=REPO, env=dict(os.environ, PSF_ZERO_REPO=REPO))
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else dict(stratum=t[0], test=t[1], arm=arm,
                                                           error=(p.stderr or "")[-300:], exit=p.returncode)
        except subprocess.TimeoutExpired:
            rec = dict(stratum=t[0], test=t[1], arm=arm, error=f"timeout {TIMEOUT} s")
        rec["wall"] = round(time.perf_counter() - w0, 2)
        return rec

    n = 0
    with ThreadPoolExecutor(a.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in jobs]):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:6.0f} s] {r['arm']} {r['test'][-45:]}: "
                  + (r["error"][:50] if "error" in r else f"{r['t']} s, q2 {r['q2']}"), flush=True)
    report(a)


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "bp_rec.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault(r["test"], {})[r["arm"]] = r
    rows = json.load(open(os.path.join(REPO, "data", "2026-10-08", "bp_final", "bp_final.json"), encoding="utf-8"))["rows"]
    qk = {r["test"]: r for r in rows if r.get("arm") == "QK" and "q2" in r}
    recr = {r["test"]: r for r in rows if r.get("arm") == "RECR" and "q2" in r}
    ok = lambda r: r is not None and "error" not in r  # noqa: E731
    both = [t for t, v in by.items() if ok(v.get("R071")) and ok(v.get("R102"))]
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")  # noqa: E731
    ratio = [by[t]["R102"]["t"] / by[t]["R071"]["t"] for t in both] or [float("nan")]
    r071 = [t for t, v in by.items() if ok(v.get("R071")) and t in recr]
    same_recr = sum(by[t]["R071"]["q2"] == recr[t]["q2"] for t in r071)
    same_v = sum(by[t]["R102"]["vsig"] == by[t]["R071"]["vsig"] for t in both)
    same_q = sum(by[t]["R102"]["q2"] == by[t]["R071"]["q2"] for t in both)
    withqk = [t for t in both if t in qk]
    fails = {arm: sorted(t for t, v in by.items() if arm in v and "error" in v[arm]) for arm in ARMS}
    L = ["# BP-REC (exploratory, nothing predicted)", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['tests']} FakeTorino tests of BP-FINAL, {meta['jobs']} jobs, {meta['par']} at a time, "
         f"{meta['cpus']} CPUs; psf_zero_core57 {meta.get('core57')}; the real clock", "",
         f"- both finished: {len(both)}; failed: R071 {len(fails['R071'])}, R102 {len(fails['R102'])}",
         f"- R102 / R071 compile time: geometric mean {gm(ratio):.3f}, median {statistics.median(ratio):.3f}; "
         f"summed {sum(by[t]['R071']['t'] for t in both):.0f} s -> {sum(by[t]['R102']['t'] for t in both):.0f} s",
         f"- the same output by value: {same_v} of {len(both)}; the same two-qubit count: {same_q}",
         f"- R071's two-qubit count equals BP-FINAL's RECR (the same release, work PC) on {same_recr} of {len(r071)}",
         f"- R102's estimate and check calls: " + ", ".join(f"{k} {sum((by[t]['R102'].get('core57') or {}).get(k, 0) for t in both)}"
                                                       for k in ("rust", "python", "fallback")),
         f"- two-qubit count / Qiskit level 2 (BP-FINAL's QK), geometric mean of (q2 + 1) ratios on {len(withqk)}: "
         f"R071 {gm([(by[t]['R071']['q2'] + 1) / (qk[t]['q2'] + 1) for t in withqk]):.3f}, "
         f"R102 {gm([(by[t]['R102']['q2'] + 1) / (qk[t]['q2'] + 1) for t in withqk]):.3f}",
         f"- tests with gates on failed elements: R071 {sum(1 for t in both if by[t]['R071']['on_failed'])}, "
         f"R102 {sum(1 for t in both if by[t]['R102']['on_failed'])}",
         f"- not valid (Benchpress): R071 {sum(1 for t in both if by[t]['R071']['valid'] is not True)}, "
         f"R102 {sum(1 for t in both if by[t]['R102']['valid'] is not True)}", "",
         "| test | qubits | R071 s | R102 s | R102 / R071 | q2 R071 / R102 | same value |", "|---|---|---|---|---|---|---|"]
    for t in sorted(both, key=lambda t: -by[t]["R071"]["t"]):
        x, y = by[t]["R071"], by[t]["R102"]
        L.append(f"| {t.split('[')[-1].rstrip(']')[:45]} | {x['input_qubits']} | {x['t']:.2f} | {y['t']:.2f} | "
                 f"{y['t'] / x['t']:.2f} | {x['q2']} / {y['q2']} | {'yes' if x['vsig'] == y['vsig'] else 'no'} |")
    L += [f"- failed ({arm}): {t}: {by[t][arm]['error'][:80]}" for arm in ARMS for t in fails[arm]]
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "bp_rec.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "report"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "one": one, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
