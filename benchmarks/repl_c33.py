"""repl_c33.py -- REPL-C33 (2026-10-10; exploratory, nothing predicted): does replacing work make PSF-Zero faster?
Candidate c33 (changelog item 61: with a Target, one compile on the map without the failed elements, instead of
item 31's compile and recompile) against c32, and both against Qiskit level 2, on BP-FINAL's 106 FakeTorino tests,
on FakeTorino and FakeKingston.

Arms, each test, device and arm in its own process, the call made twice in it (cold: the first call in the process,
which a user always pays; warm: the second):
  QK2   Qiskit level 2 on the backend (generate_preset_pass_manager(2, backend).run)
  C32D  patches/psf_compile_c32_2026-10-10/psf_compile.py, the default call with backend=
  C33D  patches/psf_compile_c33_2026-10-10/psf_compile.py, the same call
Recorded per output: both times, the two-qubit count, the operations on failed elements, log10 ESP (esp_ft.py's
definition), and the output's value signature (c29_identity.value_sig) to compare C33D with C32D.

`profile` mode: cProfile of C33D's cold and warm call on the ten tests where BP-FINAL's recommended call was fastest
and on two mid-size tests, the top functions by own time and by cumulative time.

Caps: a job is killed after 1,500 s; no job starts after 7,200 s; 4 at a time. The output folder must not exist.

    cd <psf-zero repository>
    python <this file> run --bp <benchpress clone> --out DIR [--par 4] [--smoke]
    python <this file> profile --bp <benchpress clone> --out DIR
    python <this file> report --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import cProfile
import hashlib
import io
import json
import math
import os
import pstats
import statistics
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), HERE, REPO]
ARMS = ("QK2", "C32D", "C33D")
FILES = {"C32D": os.path.join(REPO, "patches", "psf_compile_c32_2026-10-10", "psf_compile.py"),
         "C33D": os.path.join(REPO, "patches", "psf_compile_c33_2026-10-10", "psf_compile.py")}
DEVICES = ("FakeTorino", "FakeKingston")
FAILED = 0.5
TIMEOUT, BUDGET = 1500, 7200
PROFILE_MID = ("ham_tsp_prob-fl417_Ncity-16_enc-stdbinary", "ham_bh_graph-2D-grid-nonpbc-qubitnodes")


def tests(bp):
    import bp_final as F
    return [t for t in F.population(bp) if t[0].endswith("FakeTorino")]


def device(name, bp_backend):
    if name == "FakeTorino":
        return bp_backend
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


def esp_of(out, target):
    """(log10 ESP or None when an operation has error 1, operations on failed elements)."""
    lg, n_failed, dead = 0.0, 0, False
    for ins in out.data:
        name = ins.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(out.find_bit(q).index for q in ins.qubits)
        try:
            props = target[name][qargs]
        except KeyError:
            continue
        e = props.error if props is not None and props.error is not None else 0.0
        if e >= FAILED:
            n_failed += 1
        if e >= 1.0:
            dead = True
        else:
            lg += math.log10(1.0 - e)
    return (None if dead else round(lg, 6)), n_failed


def compiler(arm, backend):
    """A function that compiles a circuit as the arm does."""
    if arm == "QK2":
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        return lambda qc: generate_preset_pass_manager(2, backend).run(qc), None
    import core_fix_c2_eval as H
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(FILES[arm], "psf_compile")
    pc.WARN_WITHOUT_TARGET = False

    def run(qc):
        with contextlib.redirect_stdout(io.StringIO()):
            return pc.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True,
                                           seed_transpiler=0)
    return run, pc


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    from c29_identity import value_sig
    qc, bp_backend = B.build(a.bp, kind, arg)
    backend = device(a.device, bp_backend)
    rec = dict(stratum=stratum, test=tid, device=a.device, arm=a.arm, input_qubits=qc.num_qubits)
    run, pc = compiler(a.arm, backend)
    if pc is not None:
        rec["version"] = pc.VERSION
    times = []
    for _ in range(2):
        t0 = time.perf_counter()
        try:
            out = run(qc)
        except Exception as exc:  # noqa: BLE001 - recorded (item 58's refusal among them)
            rec.update(error=f"{type(exc).__name__}: {exc}"[:300], times=[round(t, 4) for t in times])
            print(json.dumps(rec), flush=True)
            return
        times.append(time.perf_counter() - t0)
    rec["t_cold"], rec["t_warm"] = round(times[0], 4), round(times[1], 4)
    rec["q2"] = sum(1 for ins in out.data if len(ins.qubits) == 2 and ins.operation.name not in ("barrier", "delay"))
    rec["esp"], rec["on_failed"] = esp_of(out, backend.target)
    rec["vsig"] = value_sig(out)
    if pc is not None:
        rec["prune"] = dict(pc.PRUNE_STATS)
    print(json.dumps(rec), flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


def fastest(ts, n):
    rows = json.load(open(os.path.join(REPO, "data", "2026-10-08", "bp_final", "bp_final.json"), encoding="utf-8"))
    t_recr = {r["test"]: r["t"] for r in rows["rows"] if r.get("arm") == "RECR" and "t" in r}
    return sorted(ts, key=lambda t: t_recr.get(t[1], math.inf))[:n]


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    ts = tests(a.bp)
    if a.smoke:
        ts = fastest(ts, 1)
    jobs = sorted((hashlib.sha256(("REPL-C33|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "repl_c33.jsonl")
    import qiskit
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par, cpus=os.cpu_count(),
                                           qiskit=qiskit.__version__))) + "\n")
    print(f"REPL-C33: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
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
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:6.0f} s] {r['device'][4:]:8s} {r['arm']:4s} "
                  f"{r['test'][-40:]}: " + (r["error"][:50] if "error" in r else
                                           f"{r['t_cold']} / {r['t_warm']} s, q2 {r['q2']}, failed {r['on_failed']}"),
                  flush=True)
    report(a)


def profile(a):
    """cProfile of C33D's cold and warm call, each test in its own process (mode profile_one)."""
    os.makedirs(a.out, exist_ok=True)
    ts = tests(a.bp)
    chosen = fastest(ts, 10) + [next(t for t in ts if t[1].split("[")[-1].startswith(m)) for m in PROFILE_MID]
    for t in chosen:
        name = t[1].split("[")[-1].rstrip("]")[:60].replace("/", "_")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "profile_one", "--bp", a.bp, "--job",
                            json.dumps(list(t))], capture_output=True, text=True, timeout=TIMEOUT, cwd=REPO,
                           env=dict(os.environ, PSF_ZERO_REPO=REPO))
        open(os.path.join(a.out, f"profile_{name}.txt"), "w", encoding="utf-8", newline="\n").write(
            p.stdout + ("\n--- stderr ---\n" + p.stderr[-2000:] if p.returncode else ""))
        print(f"profiled {name} (exit {p.returncode})", flush=True)


def profile_one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    qc, backend = B.build(a.bp, kind, arg)
    run_, _ = compiler("C33D", backend)
    for label in ("cold", "warm"):
        prof = cProfile.Profile()
        t0 = time.perf_counter()
        prof.enable()
        run_(qc)
        prof.disable()
        print(f"===== {tid} -- C33D {label} call: {time.perf_counter() - t0:.4f} s =====")
        for key in ("tottime", "cumulative"):
            s = io.StringIO()
            pstats.Stats(prof, stream=s).strip_dirs().sort_stats(key).print_stats(25)
            print(f"--- top 25 by {key} ---")
            print("\n".join(ln for ln in s.getvalue().splitlines() if ln.strip())[:6000])


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "repl_c33.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    ok = lambda r: r is not None and "error" not in r  # noqa: E731
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")  # noqa: E731
    lg = lambda r: -math.inf if r["esp"] is None else r["esp"]  # noqa: E731
    L = ["# REPL-C33 (exploratory, nothing predicted)", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['tests']} tests on {', '.join(DEVICES)}; {meta['jobs']} jobs, {meta['par']} at a time; Qiskit "
         f"{meta.get('qiskit')}. Cold: the first call in the process; warm: the second.", ""]
    for d in DEVICES:
        keys = [k for k in by if k[0] == d]
        full = [k for k in keys if all(ok(by[k].get(x)) for x in ARMS)]
        small = [k for k in full if by[k]["QK2"]["t_warm"] < 0.5]
        L += [f"## {d}", "", f"{len(full)} of {len(keys)} tests with every arm; failed jobs: "
              + ", ".join(f"{x} {sum(1 for k in keys if not ok(by[k].get(x)))}" for x in ARMS), "",
              "| arm | time cold (s, summed) | time warm (s, summed) | cold / QK2 cold (gmean) | warm / QK2 warm (gmean) "
              "| small tests: median cold / warm (s) | q2 / QK2 (gmean, +1) | outputs on failed elements |",
              "|---|---|---|---|---|---|---|---|"]
        for x in ARMS:
            rs = [by[k][x] for k in full]
            L.append(f"| {x} | {sum(r['t_cold'] for r in rs):.0f} | {sum(r['t_warm'] for r in rs):.0f} | "
                     f"{gm([max(r['t_cold'], 1e-3) / max(by[k]['QK2']['t_cold'], 1e-3) for k, r in zip(full, rs)]):.2f} | "
                     f"{gm([max(r['t_warm'], 1e-3) / max(by[k]['QK2']['t_warm'], 1e-3) for k, r in zip(full, rs)]):.2f} | "
                     f"{statistics.median(by[k][x]['t_cold'] for k in small):.3f} / "
                     f"{statistics.median(by[k][x]['t_warm'] for k in small):.3f} | "
                     f"{gm([(r['q2'] + 1) / (by[k]['QK2']['q2'] + 1) for k, r in zip(full, rs)]):.3f} | "
                     f"{sum(1 for r in rs if r['on_failed'])} |")
        same = sum(by[k]["C33D"]["vsig"] == by[k]["C32D"]["vsig"] for k in full)
        rat = [by[k]["C33D"]["t_cold"] / max(by[k]["C32D"]["t_cold"], 1e-3) for k in full]
        run_ = [k for k in full if max(lg(by[k][x]) for x in ARMS) >= -2]
        diff = [lg(by[k]["C33D"]) - lg(by[k]["C32D"]) for k in run_]
        fin = [v for v in diff if math.isfinite(v)]
        pf = sum(1 for k in full if by[k]["C33D"].get("prune", {}).get("pruned_first"))
        rc = sum(1 for k in full if by[k]["C32D"].get("prune", {}).get("recompiled"))
        L += ["", f"- C33D compiled once on the pruned map on {pf} tests; C32D recompiled on {rc}.",
              f"- C33D / C32D cold time: geometric mean {gm(rat):.3f}, median {statistics.median(rat):.3f}; "
              f"the same output by value on {same} of {len(full)}.",
              f"- ESP where the best arm's >= 0.01 ({len(run_)} tests): C33D / C32D geometric mean "
              f"{10 ** (sum(fin) / len(fin)) if fin else float('nan'):.3f}; 10% better on "
              f"{sum(1 for v in diff if v >= math.log10(1.1))}, 10% worse on {sum(1 for v in diff if v <= -math.log10(1.1))}.",
              ""]
        L += [f"- failed job ({k[1]}, {x}): {by[k][x]['error'][:100]}" for k in keys for x in ARMS
              if x in by[k] and not ok(by[k][x])][:30]
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "repl_c33.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "report", "profile", "profile_one"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--device")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "one": one, "report": report, "profile": profile, "profile_one": profile_one}[a.mode](a)


if __name__ == "__main__":
    main()
