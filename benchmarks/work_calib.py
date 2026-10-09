"""work_calib.py -- exploratory (2026-10-09; not pre-registered, nothing predicted): how the time of the recommended
call's estimates and exactness checks grows with the size of what they simulate. It prepares changelog item 54, a
budget for that work counted in operations instead of seconds.

Each estimate and check applies gate matrices to a state of 2^k amplitudes (k touched qubits). Two counts describe
one call:
  ops   the number of gate applications (Python-level work per gate: matrix lookup, tensordot, moveaxis)
  amps  the sum over those applications of 2^k (numpy work, growing with the state)
This records, for every call, its function, ops, amps, k and wall time, on the recommended call of the development
tests (the FakeTorino and summit tests of C25-ID's 152; none of BP-FINAL's 880), with candidate 2026-10-09.c26 (item
53), on which item 54 would be built. It also times, without counts, every call psf_compile makes to Qiskit's
`transpile` (level 3's among them) and the re-synthesis candidate, so a long compile can be attributed. Then `fit`
fits time = a * ops + b * amps per function.

Caps (every loop is bounded):
  - each test runs in its own process, killed after --job-timeout seconds (default 600);
  - no new test starts after --budget seconds (default 5400); the run ends at most one job-timeout later;
  - every call writes a "start" line before it runs and an "end" line after, flushed, so a call that never ends is
    visible in the records of a killed job.
The output folder must not exist: a run never replaces an earlier run's files.

    python benchmarks/work_calib.py run --bp <benchpress clone> --out DIR [--par 4] [--job-timeout 600] [--budget 5400]
    python benchmarks/work_calib.py fit --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import math
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PATCH = os.path.join(REPO, "patches", "psf_compile_c26_2026-10-09")
sys.path[:0] = [HERE, REPO]
VERSION = "2026-10-09.c26"
SKIP = ("barrier", "measure", "delay")
COUNTED = ("excitation_cost", "hybrid_cost", "pauli_cost", "kraus_cost", "_implements", "_same_action")
TIMED = ("transpile", "_resynthesis_candidate")


# ---------------------------------------------------------------------------------------------------------- counts
def _shape(circ):
    """(instructions other than barrier/measure/delay, of which on two or more qubits, touched qubit indices)."""
    n_all = n_multi = 0
    touched = set()
    for ins in circ.data:
        if ins.operation.name in SKIP:
            continue
        n_all += 1
        n_multi += len(ins.qubits) >= 2
        touched.update(circ.find_bit(b).index for b in ins.qubits)
    return n_all, n_multi, touched


def counts(fn, args):
    """(ops, amps, k) for one call, as the function would do it if it runs to the end (it may stop earlier).
    excitation_cost and hybrid_cost apply only multi-qubit gates to the state (item 49); pauli_cost and kraus_cost
    apply every gate; the checks apply both circuits from two seeds. Matrix building (_ops_of) is in `ops`."""
    if fn in ("excitation_cost", "hybrid_cost", "pauli_cost", "kraus_cost"):
        n_all, n_multi, t = _shape(args[0])
        k = max(len(t), 1)
        n_state = n_multi if fn in ("excitation_cost", "hybrid_cost") else n_all
        return n_all, n_state * 2 ** k, k
    if fn == "_same_action":
        a, b = _shape(args[0]), _shape(args[1])
        k = len(a[2] | b[2])
        return 2 * (a[0] + b[0]), 2 * (a[0] + b[0]) * 2 ** k, k
    if fn == "_implements":
        qc, out = args[0], args[1]
        a, b = _shape(qc), _shape(out)
        n = qc.num_qubits
        k = max(len(b[2]), n)  # out's touched qubits and the layouts' positions; at least n
        return 2 * (a[0] + b[0]), 2 * (a[0] * 2 ** n + b[0] * 2 ** k), k
    return None, None, None


# ----------------------------------------------------------------------------------------------------------- child
def child(args):
    warnings.simplefilter("ignore")
    import bp_mock as B
    import c25_identity as C
    import core_fix_c2_eval as H
    job = json.loads(args.job)
    qc, backend = B.build(args.bp, job["kind"], job["arg"])
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(os.path.join(PATCH, "psf_compile.py"), "psf_compile")
    log = open(args.log, "a", encoding="utf-8", newline="\n")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    seq = [0]

    def wrap(name, f):
        @functools.wraps(f)
        def g(*a, **kw):
            seq[0] += 1
            s = seq[0]
            ops, amps, k = counts(name, a) if name in COUNTED else (None, None, None)
            write(dict(ev="start", seq=s, fn=name, ops=ops, amps=amps, k=k))
            t0 = time.perf_counter()
            r = f(*a, **kw)
            t = time.perf_counter() - t0
            res = None if name in TIMED else (r if r is None or isinstance(r, bool) else "value")
            write(dict(ev="end", seq=s, fn=name, t=round(t, 6), result=res))
            return r
        return g

    for name in COUNTED + TIMED:  # module globals: the module's own calls go through the wrappers
        setattr(pc, name, wrap(name, getattr(pc, name)))
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, **C.RECOMMENDED)
    write(dict(ev="compile_start", version=pc.VERSION, num_qubits=qc.num_qubits, size=qc.size()))
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    write(dict(ev="compile_end", t=round(time.perf_counter() - t0, 3),
               q2=int(out.count_ops().get(backend.two_q_gate_type, 0))))
    log.close()


# ------------------------------------------------------------------------------------------------------------- run
def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def run(args):
    import c25_identity as C
    if os.path.exists(args.out):
        sys.exit(f"STOP: {args.out} exists; a run never replaces an earlier run's files")
    os.makedirs(os.path.join(args.out, "jobs"))
    tests = [t for t in C.population(args.bp) if t[0].endswith("FakeTorino") or t[2] == "summit"]
    meta = dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_head=git("rev-parse", "--short",
                "HEAD"), dirty_tracked=git("status", "--porcelain", "--untracked-files=no"), version=VERSION,
                tests=len(tests), par=args.par, job_timeout=args.job_timeout, budget=args.budget,
                cpus=os.cpu_count(), python=sys.version.split()[0])
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    t_start = time.perf_counter()

    def go(item):
        idx, (stratum, tid, kind, arg) = item
        if time.perf_counter() - t_start > args.budget:
            return idx, tid, "not started (budget)", 0.0
        log = os.path.join(args.out, "jobs", f"{idx:03d}.jsonl")
        with open(log, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(dict(ev="job", stratum=stratum, test=tid, kind=kind, arg=arg)) + "\n")
        job = json.dumps(dict(kind=kind, arg=arg))
        w0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", "--bp", args.bp, "--job", job,
                                "--log", log], capture_output=True, text=True, timeout=args.job_timeout,
                               encoding="utf-8", errors="replace")
            status = "ok" if p.returncode == 0 else "error: " + (p.stderr or "")[-200:].replace("\n", " ")
        except subprocess.TimeoutExpired:
            status = f"killed after {args.job_timeout} s"
        with open(log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(dict(ev="job_end", status=status, wall=round(time.perf_counter() - w0, 2))) + "\n")
        return idx, tid, status, time.perf_counter() - w0

    done = 0
    with ThreadPoolExecutor(args.par) as ex:
        for idx, tid, status, w in ex.map(go, list(enumerate(tests))):
            done += 1
            print(f"[{done}/{len(tests)} {time.perf_counter() - t_start:6.0f} s] {tid[:60]}: {status[:60]} "
                  f"({w:.1f} s)", flush=True)
    fit(args)


# ------------------------------------------------------------------------------------------------------------- fit
def load(out):
    """[(test, status, compile time or None, [calls])], a call = dict(fn, ops, amps, k, t or None if it never ended)."""
    res = []
    for name in sorted(os.listdir(os.path.join(out, "jobs"))):
        test, status, ct, calls, open_ = None, "no end line", None, [], {}
        for ln in open(os.path.join(out, "jobs", name), encoding="utf-8"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue  # a line cut by a kill
            ev = r.get("ev")
            if ev == "job":
                test = r["test"]
            elif ev == "start":
                open_[r["seq"]] = dict(fn=r["fn"], ops=r["ops"], amps=r["amps"], k=r["k"], t=None)
                calls.append(open_[r["seq"]])
            elif ev == "end" and r["seq"] in open_:
                open_[r["seq"]]["t"] = r["t"]
                open_[r["seq"]]["result"] = r.get("result")
            elif ev == "compile_end":
                ct = r["t"]
            elif ev == "job_end":
                status = r["status"]
        res.append((test, status, ct, calls))
    return res


def lstsq2(rows):
    """Least squares t = a * ops + b * amps (no intercept); (a, b, R^2) or None."""
    import numpy as np
    if len(rows) < 3:
        return None
    X = np.array([[o, m] for o, m, _ in rows], dtype=float)
    y = np.array([t for _, _, t in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss = float(((y - X @ coef) ** 2).sum())
    tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    return float(coef[0]), float(coef[1]), 1.0 - ss / tot


def fit(args):
    res = load(args.out)
    meta = json.load(open(os.path.join(args.out, "meta.json"), encoding="utf-8"))
    L = ["# work_calib (exploratory, nothing predicted)", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"candidate {meta['version']}; {meta['tests']} tests, {meta['par']} at a time; job timeout "
         f"{meta['job_timeout']} s; budget {meta['budget']} s; {meta['cpus']} CPUs; Python {meta['python']}", ""]
    st = {}
    for _, s, _, _ in res:
        key = "ok" if s == "ok" else s.split(":")[0] if s.startswith("error") else s
        st[key] = st.get(key, 0) + 1
    L += ["jobs: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())), ""]
    L += ["## Per function: time = a * ops + b * amps (calls that ended)", "",
          "| function | calls | never ended | total s | a (us per op) | b (ns per amplitude) | R^2 | largest k |",
          "|---|---|---|---|---|---|---|---|"]
    allc = [(test, c) for test, _, _, calls in res for c in calls]
    for fn in COUNTED + TIMED:
        cs = [c for _, c in allc if c["fn"] == fn]
        if not cs:
            continue
        ended = [c for c in cs if c["t"] is not None]
        tot = sum(c["t"] for c in ended)
        f = lstsq2([(c["ops"], c["amps"], c["t"]) for c in ended]) if fn in COUNTED else None
        ks = [c["k"] for c in cs if c["k"] is not None]
        L.append(f"| {fn} | {len(cs)} | {len(cs) - len(ended)} | {tot:.1f} | "
                 + (f"{f[0] * 1e6:.2f} | {f[1] * 1e9:.3f} | {f[2]:.3f}" if f else "- | - | -")
                 + f" | {max(ks) if ks else '-'} |")
    L += ["", "## The 20 longest calls (a call that never ended is listed first)", "",
          "| test | function | k | ops | amps | s |", "|---|---|---|---|---|---|"]
    longest = sorted(allc, key=lambda x: (x[1]["t"] is not None, -(x[1]["t"] or 0)))[:20]
    for test, c in longest:
        secs = "never ended" if c["t"] is None else f"{c['t']:.2f}"
        L.append(f"| {(test or '?')[:60]} | {c['fn']} | {c['k']} | {c['ops']} | {c['amps']} | {secs} |")
    L += ["", "## Calls a work cap would stop (counted calls only; work = ops * R + amps, in amplitudes; R from the pooled fit)", ""]
    pooled = lstsq2([(c["ops"], c["amps"], c["t"]) for _, c in allc if c["fn"] in COUNTED and c["t"] is not None])
    if pooled and pooled[1] > 0:
        a, b, r2 = pooled
        ratio = max(a / b, 1.0)
        L.append(f"pooled fit over all counted calls: a {a * 1e6:.2f} us per op, b {b * 1e9:.3f} ns per amplitude, "
                 f"R^2 {r2:.3f}; one op costs as much as {ratio:.0f} amplitudes")
        L += ["", "| cap (work units) | calls over | of which never ended | their total s | tests touched |",
              "|---|---|---|---|---|"]
        counted = [(test, c) for test, c in allc if c["fn"] in COUNTED and c["ops"] is not None]
        for e in range(20, 33, 2):
            cap = 2 ** e
            over = [(test, c) for test, c in counted if c["ops"] * ratio + c["amps"] > cap]
            L.append(f"| 2^{e} | {len(over)} | {sum(c['t'] is None for _, c in over)} | "
                     f"{sum(c['t'] or 0 for _, c in over):.1f} | {len({t for t, _ in over})} |")
    else:
        L.append("no pooled fit (too few calls, or no amplitude term)")
    L += ["", "## Whole compiles", "", "| test | status | compile s | counted calls s | transpile s | resynthesis s |",
          "|---|---|---|---|---|---|"]
    # unfinished compiles first, then the longest
    for test, s, ct, calls in sorted(res, key=lambda x: (x[2] is not None, -(x[2] or 0)))[:40]:
        cs = sum(c["t"] or 0 for c in calls if c["fn"] in COUNTED)
        tr = sum(c["t"] or 0 for c in calls if c["fn"] == "transpile")
        rs = sum(c["t"] or 0 for c in calls if c["fn"] == "_resynthesis_candidate")
        cts = "-" if ct is None else f"{ct:.1f}"
        L.append(f"| {(test or '?')[:60]} | {s[:30]} | {cts} | {cs:.1f} | {tr:.1f} | {rs:.1f} |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "work_calib.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "fit", "child"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--job")
    ap.add_argument("--log")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--job-timeout", type=int, default=600)
    ap.add_argument("--budget", type=int, default=5400)
    a = ap.parse_args()
    {"run": run, "fit": fit, "child": child}[a.mode](a)


if __name__ == "__main__":
    main()
