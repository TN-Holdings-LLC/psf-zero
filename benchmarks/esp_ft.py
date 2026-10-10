"""esp_ft.py -- ESP-FT (2026-10-10; exploratory, nothing predicted): what placing gates on failed elements costs, as the
estimated success probability (ESP) of each compiled circuit, on BP-FINAL's FakeTorino tests.

Devices: FakeTorino (Benchpress's device for these tests) and FakeKingston (qiskit-ibm-runtime's newest Heron r2
snapshot), both as qiskit-ibm-runtime builds their Target. Arms, each test and arm in its own process:
  QK2   Qiskit level 2 on the backend (generate_preset_pass_manager(2, backend); BP-FINAL's QK)
  QK3   Qiskit level 3 on the backend
  PSFD  psf_compile.py (2026-10-10.2), the default call (coupling map and basis, no target)
  PSFR  the same file, the README's recommended call with the target

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
ARMS = ("QK2", "QK3", "PSFD", "PSFR")
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
    if a.arm in ("QK2", "QK3"):
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        t0 = time.perf_counter()
        out = generate_preset_pass_manager(2 if a.arm == "QK2" else 3, backend).run(qc)
        rec["t"] = round(time.perf_counter() - t0, 3)
    else:
        H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
        pc = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
        import bp_final as F
        rec["version"] = pc.VERSION
        basis = [g for g in target.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
        kw = dict(coupling_map=target.build_coupling_map(), basis_gates=basis, entangling_basis="cx",
                  layout_search=True, seed_transpiler=0)
        if a.arm == "PSFR":
            kw.update(target=target, **F.RECOMMENDED)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            out = pc.compile_for_hardware(qc, **kw)
        rec["t"] = round(time.perf_counter() - t0, 3)
        rec["core57"] = dict(getattr(pc, "CORE57_STATS", {})) or None
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
    jobs = sorted((hashlib.sha256(("ESP-FT|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "esp_ft.jsonl")
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
    print(f"ESP-FT: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
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
    lines = [json.loads(x) for x in open(os.path.join(a.out, "esp_ft.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    ok = lambda r: r is not None and "error" not in r  # noqa: E731
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")  # noqa: E731
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    lg = lambda r: -math.inf if r["esp"] is None else r["esp"]  # noqa: E731
    L = ["# ESP-FT (exploratory, nothing predicted)", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['tests']} FakeTorino tests of BP-FINAL on {', '.join(DEVICES)}; {meta['jobs']} jobs, {meta['par']} at a "
         f"time; {meta.get('versions')}", "",
         "ESP: product of (1 - error) over gates and measurements as the Target reports them; no idling term. "
         f"Failed element: error >= {FAILED}.", ""]
    for d in DEVICES:
        keys = [k for k in by if k[0] == d]
        full = [k for k in keys if all(ok(by[k].get(arm)) for arm in ARMS)]
        L += [f"## {d}", "", f"- tests with all four arms finished: {len(full)} of {len(keys)}; failed jobs: "
              + ", ".join(f"{arm} {sum(1 for k in keys if not ok(by[k].get(arm)))}" for arm in ARMS), ""]
        L += ["| arm | tests with ops on failed elements | ops on failed elements | ESP = 0 (an op with error 1) | "
              "not in Target | q2 / QK2 (gmean, +1) | time (s, summed) |", "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            rs = [by[k][arm] for k in full]
            L.append(f"| {arm} | {sum(1 for r in rs if r['on_failed'])} | {sum(r['on_failed'] for r in rs)} | "
                     f"{sum(1 for r in rs if r['esp'] is None)} | {sum(r['not_in_target'] for r in rs)} | "
                     f"{gm([(r['q2'] + 1) / (by[k]['QK2']['q2'] + 1) for k, r in zip(full, rs)]):.3f} | "
                     f"{sum(r['t'] for r in rs):.0f} |")
        for floor in (1e-2, 1e-6):
            run_ = [k for k in full if max(lg(by[k][arm]) for arm in ARMS) >= math.log10(floor)]
            L += ["", f"**Tests where the best arm's ESP >= {floor:g}: {len(run_)}.** ESP ratios on them "
                  "(PSFR / other; an ESP of 0 counts as a ratio over 10^6):", "",
                  "| other | median ratio | ratio >= 2 | ratio >= 10 | ratio <= 0.5 | ratio in (0.5, 2) |",
                  "|---|---|---|---|---|---|"]
            for other in ("QK2", "QK3", "PSFD"):
                rat = []
                for k in run_:
                    x, y = lg(by[k]["PSFR"]), lg(by[k][other])
                    rat.append(6.0 if y == -math.inf and x > -math.inf else
                               (-6.0 if x == -math.inf and y > -math.inf else
                                (0.0 if x == y == -math.inf else min(6.0, max(-6.0, x - y)))))
                L.append(f"| {other} | {10 ** med(rat):.3g} | {sum(1 for v in rat if v >= math.log10(2))} | "
                         f"{sum(1 for v in rat if v >= 1)} | {sum(1 for v in rat if v <= -math.log10(2))} | "
                         f"{sum(1 for v in rat if -math.log10(2) < v < math.log10(2))} |")
        L += ["", "Per test (log10 ESP; '0' = an op with error 1; failed = ops on failed elements):", "",
              "| test | qubits | QK2 | QK3 | PSFD | PSFR | failed QK2 / QK3 / PSFD / PSFR |", "|---|---|---|---|---|---|---|"]
        fmt = lambda r: "0" if r["esp"] is None else f"{r['esp']:.2f}"  # noqa: E731
        for k in sorted(full, key=lambda k: -max(lg(by[k][arm]) for arm in ARMS)):
            v = by[k]
            L.append(f"| {k[1].split('[')[-1].rstrip(']')[:44]} | {v['QK2']['input_qubits']} | "
                     + " | ".join(fmt(v[arm]) for arm in ARMS) + " | "
                     + " / ".join(str(v[arm]["on_failed"]) for arm in ARMS) + " |")
        L += [f"- failed job ({k[1]}, {arm}): {by[k][arm]['error'][:80]}" for k in keys for arm in ARMS
              if arm in by[k] and not ok(by[k][arm])]
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "esp_ft.md"), "w", encoding="utf-8", newline="\n").write(txt)
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
