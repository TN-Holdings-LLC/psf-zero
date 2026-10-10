"""esp_c32.py -- ESP-C32 (2026-10-10; exploratory, nothing predicted): candidate c32 (changelog item 60) against c31, the
default call with backend=, on ESP-FT's tests, devices and measures (esp_ft.py, unchanged otherwise).

Devices: FakeTorino and FakeKingston, as in ESP-FT. Arms, each test and arm in its own process:
  C31D  patches/psf_compile_c31_2026-10-10/psf_compile.py, the default call with backend= (item 58)
  C32D  patches/psf_compile_c32_2026-10-10/psf_compile.py, the same call (item 60: placement by the errors)
A c31 call that raises FailedElementsError is recorded as such (error "FailedElementsError: ..."): it returned no
circuit rather than one that uses a failed element.

ESP of a circuit = product over its gates and measurements of (1 - error), errors as the device's Target reports
them (rz: 0; delays and barriers ignored; no idling term). An element is "failed" when its error is >= 0.5 (as
psf_compile's _failed_elements). Recorded per output: log10 ESP over all operations ("esp"; -inf if an operation has
error 1), log10 ESP over the operations on working elements ("esp_ok"), the number of operations on failed
elements, the two-qubit count and depth, the compile time, and whether every operation is in the Target.

Caps: a job is killed after 1,500 s; no job starts after 7,200 s; 4 at a time. The output folder must not exist.

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
ARMS = ("C31D", "C32D")
FILES = {"C31D": os.path.join(REPO, "patches", "psf_compile_c31_2026-10-10", "psf_compile.py"),
         "C32D": os.path.join(REPO, "patches", "psf_compile_c32_2026-10-10", "psf_compile.py")}
DEVICES = ("FakeTorino", "FakeKingston")
FAILED = 0.5
TIMEOUT, BUDGET = 1500, 7200
NEG = -1e300  # log10 ESP when an operation has error 1 (written as null in the records)


def tests(bp):
    import bp_final as F
    return [t for t in F.population(bp) if t[0].endswith("FakeTorino")]


def device(name, bp_backend):
    if name == "FakeTorino":
        return bp_backend
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


def esp_of(out, target):
    """(log10 ESP, log10 ESP over working elements, operations on failed elements, operations not in the Target)."""
    lg, lg_ok, n_failed, n_missing = 0.0, 0.0, 0, 0
    for ins in out.data:
        name = ins.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(out.find_bit(q).index for q in ins.qubits)
        try:
            props = target[name][qargs]
        except KeyError:
            n_missing += 1
            continue
        e = props.error if props is not None and props.error is not None else 0.0
        if e >= FAILED:
            n_failed += 1
        else:
            lg_ok += math.log10(1.0 - e)
        lg = NEG if (lg == NEG or e >= 1.0) else lg + math.log10(1.0 - e)
    return lg, lg_ok, n_failed, n_missing


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    import core_fix_c2_eval as H
    qc, bp_backend = B.build(a.bp, kind, arg)
    backend = device(a.device, bp_backend)
    target = backend.target
    rec = dict(stratum=stratum, test=tid, device=a.device, arm=a.arm, input_qubits=qc.num_qubits)
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(FILES[a.arm], "psf_compile")
    import bp_final as F
    rec["version"] = pc.VERSION
    pc.WARN_WITHOUT_TARGET = False
    kw = dict(backend=backend, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = pc.compile_for_hardware(qc, **kw)
    except Exception as exc:  # noqa: BLE001 - item 58's refusal, recorded
        rec.update(t=round(time.perf_counter() - t0, 3), error=f"{type(exc).__name__}: {exc}"[:300])
        print(json.dumps(rec), flush=True)
        return
    rec["t"] = round(time.perf_counter() - t0, 3)
    rec["prune"] = dict(pc.PRUNE_STATS)
    two = [ins for ins in out.data if len(ins.qubits) == 2 and ins.operation.name not in ("barrier", "delay")]
    rec["q2"] = len(two)
    rec["d2"] = out.depth(filter_function=lambda x: len(x.qubits) == 2 and x.operation.name != "barrier")
    lg, lg_ok, n_failed, n_missing = esp_of(out, target)
    rec.update(esp=None if lg == NEG else round(lg, 6), esp_ok=round(lg_ok, 6), on_failed=n_failed,
               not_in_target=n_missing, measured=sum(1 for ins in out.data if ins.operation.name == "measure"))
    print(json.dumps(rec), flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    ts = tests(a.bp)
    if a.smoke:  # one fast test (BP-FINAL's quickest recommended call), every device and arm
        rows = json.load(open(os.path.join(REPO, "data", "2026-10-08", "bp_final", "bp_final.json"), encoding="utf-8"))
        t_recr = {r["test"]: r["t"] for r in rows["rows"] if r.get("arm") == "RECR" and "t" in r}
        ts = sorted(ts, key=lambda t: t_recr.get(t[1], math.inf))[:1]
    jobs = sorted((hashlib.sha256(("ESP-C32|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "esp_c32.jsonl")
    try:
        import qiskit, qiskit_ibm_runtime  # noqa: E401
        versions = dict(qiskit=qiskit.__version__, qiskit_ibm_runtime=qiskit_ibm_runtime.__version__)
    except ImportError:
        versions = {}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par, timeout=TIMEOUT, budget=BUDGET,
                                           cpus=os.cpu_count(), versions=versions))) + "\n")
    print(f"ESP-C32: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
    t_start = time.perf_counter()

    def go(job):
        _, t, d, arm = job
        base = dict(stratum=t[0], test=t[1], device=d, arm=arm)
        if time.perf_counter() - t_start > BUDGET:
            return dict(base, error="not started (budget)")
        w0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--arm", arm,
                                "--device", d, "--job", json.dumps(list(t))], capture_output=True, text=True,
                               timeout=TIMEOUT, cwd=REPO, env=dict(os.environ, PSF_ZERO_REPO=REPO))
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else dict(base, error=(p.stderr or "")[-300:], exit=p.returncode)
        except subprocess.TimeoutExpired:
            rec = dict(base, error=f"timeout {TIMEOUT} s")
        rec["wall"] = round(time.perf_counter() - w0, 2)
        return rec

    n = 0
    with ThreadPoolExecutor(a.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in jobs]):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:6.0f} s] {r['device'][4:]:8s} {r['arm']:4s} "
                  f"{r['test'][-40:]}: " + (r["error"][:50] if "error" in r else
                                           f"{r['t']} s, q2 {r['q2']}, failed {r['on_failed']}, esp {r['esp']}"),
                  flush=True)
    report(a)


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "esp_c32.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    ep = os.path.join(os.path.dirname(os.path.abspath(a.out)), "esp_ft", "esp_ft.jsonl")
    for x in [json.loads(y) for y in open(ep, encoding="utf-8")][1:]:  # Qiskit's and the release's, from ESP-FT
        if x["arm"] in ("QK2", "QK3", "PSFR"):
            by.setdefault((x["device"], x["test"]), {})[x["arm"]] = x
    lg = lambda r: -math.inf if r.get("esp") is None else r["esp"]  # noqa: E731
    ok = lambda r: r is not None and "error" not in r  # noqa: E731
    L = ["# ESP-C32 (exploratory, nothing predicted)", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['tests']} tests on {', '.join(DEVICES)}; {meta['jobs']} jobs; {meta.get('versions')}. QK2, QK3 and "
         "PSFR (the release's recommended call) are ESP-FT's records.", ""]
    others = ("QK2", "QK3", "PSFR")
    for d in DEVICES:
        keys = [k for k in by if k[0] == d and all(ok(by[k].get(x)) for x in ARMS + others)]
        run_ = [k for k in keys if max(lg(by[k][x]) for x in ARMS + others) >= -2]
        L += [f"## {d}", "", f"{len(keys)} tests with every arm; {len(run_)} where the best ESP >= 0.01.", "",
              "| arm | ops on failed elements (tests) | ESP = 0 where the best >= 0.01 | q2 / QK2 (gmean, +1) | time (s, summed) |",
              "|---|---|---|---|---|"]
        for x in ARMS + others:
            rs = [by[k][x] for k in keys]
            g = math.exp(sum(math.log((r["q2"] + 1) / (by[k]["QK2"]["q2"] + 1)) for k, r in zip(keys, rs)) / len(rs))
            L.append(f"| {x} | {sum(1 for r in rs if r['on_failed'])} | {sum(1 for k in run_ if by[k][x]['esp'] is None)} "
                     f"| {g:.3f} | {sum(r['t'] for r in rs):.0f} |")
        L += ["", "ESP ratios where the best ESP >= 0.01 (geometric mean over the tests where both are > 0; wins and losses "
              "by 10% or more; 'dead' = the other's ESP is 0 and this arm's is not):", "",
              "| arm / other | gmean ratio | wins | losses | dead |", "|---|---|---|---|---|"]
        for x in ARMS:
            for o in ("QK2", "QK3", "PSFR", "C31D"):
                if o == x:
                    continue
                diff = [lg(by[k][x]) - lg(by[k][o]) for k in run_]
                fin = [v for v in diff if math.isfinite(v)]
                L.append(f"| {x} / {o} | {10 ** (sum(fin) / len(fin)):.3f} | {sum(1 for v in diff if v >= math.log10(1.1))} | "
                         f"{sum(1 for v in diff if v <= -math.log10(1.1))} | {sum(1 for v in diff if v == math.inf)} |")
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "esp_c32.md"), "w", encoding="utf-8", newline="\n").write(txt)
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
