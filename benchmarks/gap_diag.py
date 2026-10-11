"""gap_diag.py -- GAP-DIAG (2026-10-11; a diagnosis, nothing predicted): where does release 2026-10-11.1's remaining ESP gap
to Qiskit level 3 come from? C36-VAL (Addendum 441) found the plain call with backend= at 0.976 and 0.963 of level 3's
ESP on the runnable tests where both outputs are clean, with about the same two-qubit count (Addendum 439). ESP is a
product over operations, so its logarithm splits exactly into parts:

  log10 ESP = - (two-qubit part + one-qubit part + measurement part),  each part = sum over its operations of
              -log10(1 - error)

and the two-qubit part splits further into "how many" and "how good":

  two-qubit part = (number of two-qubit gates) x (their mean -log10(1 - error))

Tests: BP-FINAL's 106 FakeTorino tests. Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez. Arms, each test,
device and arm in its own process, one cold call, 4 at a time:
  REL   psf_compile.py (2026-10-11.1), the plain call with backend=
  QK3   Qiskit level 3 on the backend
For each output: the counts and parts above, the number of distinct physical qubits and couplers used, and whether an
operation is on a failed element (error >= 0.5). The report compares REL with QK3 on the runnable tests (best ESP of
the two >= 0.01) where neither uses a failed element.

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
import statistics
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), HERE, REPO]
ARMS = ("REL", "QK3")
DEVICES = ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez")
FAILED = 0.5
TIMEOUT, BUDGET = 1500, 7200


def tests(bp):
    import bp_final as F
    return [t for t in F.population(bp) if t[0].endswith("FakeTorino")]


def breakdown(out, target):
    r = dict(n2=0, n1=0, nm=0, s2=0.0, s1=0.0, sm=0.0, failed=0, missing=0)
    qubits, couplers = set(), set()
    for ins in out.data:
        name = ins.operation.name
        if name in ("barrier", "delay"):
            continue
        qa = tuple(out.find_bit(q).index for q in ins.qubits)
        qubits.update(qa)
        try:
            p = target[name][qa]
        except KeyError:
            r["missing"] += 1
            continue
        e = p.error if p is not None and p.error is not None else 0.0
        if e >= FAILED:
            r["failed"] += 1
        cost = -math.log10(1.0 - min(e, 1.0 - 1e-15))
        if name == "measure":
            r["nm"] += 1
            r["sm"] += cost
        elif len(qa) == 2:
            r["n2"] += 1
            r["s2"] += cost
            couplers.add(tuple(sorted(qa)))
        elif name != "rz":
            r["n1"] += 1
            r["s1"] += cost
    r.update(qubits=len(qubits), couplers=len(couplers), s2=round(r["s2"], 7), s1=round(r["s1"], 7),
             sm=round(r["sm"], 7))
    return r


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    import core_fix_c2_eval as H
    from qiskit_ibm_runtime import fake_provider
    qc, _ = B.build(a.bp, kind, arg)
    backend = getattr(fake_provider, a.device)()
    target = backend.target
    rec = dict(stratum=stratum, test=tid, device=a.device, arm=a.arm, input_qubits=qc.num_qubits)
    t0 = time.perf_counter()
    if a.arm == "QK3":
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        out = generate_preset_pass_manager(3, backend).run(qc)
    else:
        H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
        pc = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
        rec["version"] = pc.VERSION
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = pc.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True,
                                              seed_transpiler=0)
        except Exception as exc:  # noqa: BLE001 - recorded
            rec.update(t=round(time.perf_counter() - t0, 3), error=f"{type(exc).__name__}: {exc}"[:300])
            print(json.dumps(rec), flush=True)
            return
    rec["t"] = round(time.perf_counter() - t0, 3)
    rec.update(breakdown(out, target))
    print(json.dumps(rec), flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    ts = tests(a.bp)
    if a.smoke:
        rows = json.load(open(os.path.join(REPO, "data", "2026-10-08", "bp_final", "bp_final.json"), encoding="utf-8"))
        t_recr = {r["test"]: r["t"] for r in rows["rows"] if r.get("arm") == "RECR" and "t" in r}
        ts = sorted(ts, key=lambda t: t_recr.get(t[1], math.inf))[:1]
    jobs = sorted((hashlib.sha256(("GAP-DIAG|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "gap_diag.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par))) + "\n")
    print(f"GAP-DIAG: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
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
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:6.0f} s] {r['device'][4:]:9s} {r['arm']:3s} "
                  f"{r['test'][-40:]}: " + (r["error"][:50] if "error" in r else
                                           f"{r['t']} s, n2 {r['n2']}, s2 {r['s2']:.4f}, s1 {r['s1']:.4f}, "
                                           f"sm {r['sm']:.4f}, failed {r['failed']}"), flush=True)
    report(a)


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "gap_diag.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        if "error" not in r:
            by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    L = ["# GAP-DIAG (a diagnosis, nothing predicted)", "",
         f"git head {meta['git_head']}; {meta['tests']} tests on {', '.join(DEVICES)}; {meta['jobs']} jobs; errors "
         f"{sum(1 for r in recs if 'error' in r)}.", "",
         "Parts are sums of -log10(1 - error); log10 ESP = -(two-qubit + one-qubit + measurement). Means over the "
         "runnable tests (best ESP of the two >= 0.01) where neither output uses a failed element; a positive "
         "difference is REL's extra loss.", "",
         "| device | tests | ESP REL / QK3 (gmean) | two-qubit part: REL - QK3 | ... from the count | ... from the "
         "error per gate | one-qubit part | measurement part | two-qubit gates REL / QK3 | mean error per two-qubit "
         "gate REL / QK3 | qubits used REL / QK3 | REL worse by 10% / better by 10% |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in DEVICES:
        ks = [k for k in by if k[0] == d and all(x in by[k] for x in ARMS)]
        tot = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731
        sel = [k for k in ks if not by[k]["REL"]["failed"] and not by[k]["QK3"]["failed"]
               and min(tot(by[k]["REL"]), tot(by[k]["QK3"])) <= 2]
        if not sel:
            continue
        P, Q = [by[k]["REL"] for k in sel], [by[k]["QK3"] for k in sel]
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        d2 = mean([p["s2"] - q["s2"] for p, q in zip(P, Q)])
        cnt = mean([(p["n2"] - q["n2"]) * (q["s2"] / q["n2"] if q["n2"] else 0.0) for p, q in zip(P, Q)])
        per = d2 - cnt
        d1 = mean([p["s1"] - q["s1"] for p, q in zip(P, Q)])
        dm = mean([p["sm"] - q["sm"] for p, q in zip(P, Q)])
        ratio = 10 ** (-mean([tot(p) - tot(q) for p, q in zip(P, Q)]))
        e2p = sum(p["s2"] for p in P) / max(1, sum(p["n2"] for p in P))
        e2q = sum(q["s2"] for q in Q) / max(1, sum(q["n2"] for q in Q))
        worse = sum(1 for p, q in zip(P, Q) if tot(p) - tot(q) >= math.log10(1.1))
        better = sum(1 for p, q in zip(P, Q) if tot(q) - tot(p) >= math.log10(1.1))
        L.append(f"| {d} | {len(sel)} | {ratio:.3f} | {d2:+.5f} | {cnt:+.5f} | {per:+.5f} | {d1:+.5f} | {dm:+.5f} | "
                 f"{sum(p['n2'] for p in P)} / {sum(q['n2'] for q in Q)} | {e2p:.5f} / {e2q:.5f} | "
                 f"{statistics.median(p['qubits'] for p in P)} / {statistics.median(q['qubits'] for q in Q)} | "
                 f"{worse} / {better} |")
    L += ["", "Per test (all devices), the 15 where REL loses most:", "",
          "| device | test | log10 ESP REL - QK3 | two-qubit REL / QK3 | two-qubit part | one-qubit part | measurement part |",
          "|---|---|---|---|---|---|---|"]
    rows = []
    for k, v in by.items():
        if all(x in v for x in ARMS) and not v["REL"]["failed"] and not v["QK3"]["failed"]:
            p, q = v["REL"], v["QK3"]
            tp, tq = p["s2"] + p["s1"] + p["sm"], q["s2"] + q["s1"] + q["sm"]
            if min(tp, tq) <= 2:
                rows.append((tq - tp, k, p, q))
    for diff, k, p, q in sorted(rows, key=lambda x: x[0])[:15]:
        L.append(f"| {k[0][4:]} | {k[1].split('[')[-1].rstrip(']')[:40]} | {diff:+.4f} | {p['n2']} / {q['n2']} | "
                 f"{p['s2'] - q['s2']:+.4f} | {p['s1'] - q['s1']:+.4f} | {p['sm'] - q['sm']:+.4f} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "gap_diag.md"), "w", encoding="utf-8", newline="\n").write(txt)
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
