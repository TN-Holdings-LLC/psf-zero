"""c36_val.py -- C36-VAL (pre-registered in Addendum 440, 2026-10-11): should candidate 2026-10-10.c36 (changelog item 64:
routing at Qiskit's level 3 whenever the device's Target is given) replace release 2026-10-10.3? Measured as C35-VAL
measured (benchmarks/c35_val.py): failed elements first, then the estimated success probability (ESP), the two-qubit
count and the compile time.

Tests: BP-FINAL's 106 FakeTorino tests (HamLib and Feynman), as ESP-FT built them. Devices: FakeMarrakesh and FakeFez
(qiskit-ibm-runtime's Heron r2 snapshots, 156 qubits, both with couplers at error 1), which neither C35-VAL nor
ROUTE-X (Addendum 439, where item 64 was found) used. Arms, each test, device and arm in its own process, one cold
call, 4 at a time:
  QK2   Qiskit level 2 on the backend
  QK3   Qiskit level 3 on the backend
  REL3D release 2026-10-10.3 (psf_compile.py), the plain call with backend=
  REL3R release 2026-10-10.3, backend= and bp_final.RECOMMENDED
  REL3N release 2026-10-10.3, without the Target (coupling map and basis only)
  C36D  c36 (patches/psf_compile_c36_2026-10-10), the plain call with backend=
  C36R  c36, backend= and bp_final.RECOMMENDED
  C36N  c36, without the Target
Common: entangling_basis="cx", layout_search=True, seed_transpiler=0; psf_smart_layout from benchmarks/.

Predictions (Addendum 440), each on both devices:
  W0  every arm returns a circuit on every test (a FailedElementsError counts against it)
  W1  C36D and C36R: no output with an operation on a failed element
  W2  ESP C36D / REL3D >= 1.03 (confirmed); < 1.00 refuted
  W3  ESP C36D / QK3 >= 0.96 (confirmed); < 0.93 refuted
  W4  C36N's two-qubit count equals REL3N's on every test (confirmed); refuted otherwise
  W5  compile time summed, C36D / REL3D <= 1.5 (confirmed); > 2.0 refuted
  W6  ESP C36R / REL3R >= 1.00 (confirmed); < 0.97 refuted
Release rule (Addendum 440): c36 becomes the release if W0, W1, W2 and W4 are not refuted on either device.

Caps: a job is killed after 1,500 s; no job starts after 7,200 s. The output folder must not exist.

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
ARMS = ("QK2", "QK3", "REL3D", "REL3R", "REL3N", "C36D", "C36R", "C36N")
C36 = os.path.join(REPO, "patches", "psf_compile_c36_2026-10-10")
LAYOUT = os.path.join(HERE, "psf_smart_layout.py")
FILES = {a: (os.path.join(REPO, "psf_compile.py"), LAYOUT) for a in ("REL3D", "REL3R", "REL3N")}
FILES.update({a: (os.path.join(C36, "psf_compile.py"), LAYOUT) for a in ("C36D", "C36R", "C36N")})
DEVICES = ("FakeMarrakesh", "FakeFez")
FAILED = 0.5
TIMEOUT, BUDGET = 1500, 7200
NEG = -1e300
BASIS = ("cx", "cz", "ecr", "rz", "sx", "x", "id")


def tests(bp):
    import bp_final as F
    return [t for t in F.population(bp) if t[0].endswith("FakeTorino")]


def device(name, bp_backend):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def esp_of(out, target):
    lg, n_failed, n_missing = 0.0, 0, 0
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
        n_failed += e >= FAILED
        lg = NEG if (lg == NEG or e >= 1.0) else lg + math.log10(1.0 - e)
    return lg, n_failed, n_missing


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
        import bp_final as F
        f_pc, f_lay = FILES[a.arm]
        H.load_module(f_lay, "psf_smart_layout")
        pc = H.load_module(f_pc, "psf_compile")
        rec["version"] = pc.VERSION
        if hasattr(pc, "WARN_WITHOUT_TARGET"):
            pc.WARN_WITHOUT_TARGET = False
        common = dict(entangling_basis="cx", layout_search=True, seed_transpiler=0)
        no_target = dict(coupling_map=target.build_coupling_map(),
                         basis_gates=[g for g in target.operation_names if g in BASIS])
        kw = {"D": dict(backend=backend), "R": dict(backend=backend, **F.RECOMMENDED), "N": no_target}[a.arm[-1]]
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = pc.compile_for_hardware(qc, **common, **kw)
        except Exception as exc:  # noqa: BLE001 - recorded (FailedElementsError is item 58's refusal)
            rec.update(t=round(time.perf_counter() - t0, 3), error=f"{type(exc).__name__}: {exc}"[:300])
            print(json.dumps(rec), flush=True)
            return
        rec["t"] = round(time.perf_counter() - t0, 3)
    two = [ins for ins in out.data if len(ins.qubits) == 2 and ins.operation.name not in ("barrier", "delay")]
    rec["q2"] = len(two)
    lg, n_failed, n_missing = esp_of(out, target)
    rec.update(esp=None if lg == NEG else round(lg, 6), on_failed=n_failed, not_in_target=n_missing)
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
    jobs = sorted((hashlib.sha256(("C36-VAL|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "c36_val.jsonl")
    try:
        import qiskit, qiskit_ibm_runtime  # noqa: E401
        versions = dict(qiskit=qiskit.__version__, qiskit_ibm_runtime=qiskit_ibm_runtime.__version__)
    except ImportError:
        versions = {}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           head_subject=git("log", "-1", "--format=%s")[:80],
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par, timeout=TIMEOUT, budget=BUDGET,
                                           cpus=os.cpu_count(), versions=versions, smoke=a.smoke))) + "\n")
    print(f"C36-VAL: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
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
                                           f"{r['t']} s, q2 {r['q2']}, failed {r['on_failed']}, esp {r['esp']}"),
                  flush=True)
    report(a)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "c36_val.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    lg = lambda r: -math.inf if r.get("esp") is None else r["esp"]  # noqa: E731
    ok = lambda r: r is not None and "error" not in r  # noqa: E731
    L = ["# C36-VAL (pre-registered in Addendum 440)", "",
         f"git head {meta['git_head']} ({meta['head_subject']}), uncommitted tracked changes: "
         f"{'none' if not meta['dirty_tracked'] else 'YES'}; {meta['tests']} tests on {', '.join(DEVICES)}; "
         f"{meta['jobs']} jobs, {meta['par']} at a time; {meta.get('versions')}; smoke {meta.get('smoke')}.", ""]
    score = {}
    for d in DEVICES:
        keys = sorted(k for k in by if k[0] == d)
        full = [k for k in keys if all(ok(by[k].get(x)) for x in ARMS)]
        run_ = [k for k in full if max(lg(by[k][x]) for x in ARMS) >= -2]
        L += [f"## {d}", "", f"{len(keys)} tests; {len(full)} with every arm; {len(run_)} runnable (best ESP >= 0.01).",
              "", "| arm | returned | errors (FailedElementsError / other) | outputs on failed elements | ESP = 0 on "
              "runnable tests | two-qubit / QK2 (gmean, +1) | time summed (s) |", "|---|---|---|---|---|---|---|"]
        for x in ARMS:
            rs = [by[k].get(x) for k in keys]
            good = [r for r in rs if ok(r)]
            fe = sum(1 for r in rs if r is not None and "FailedElementsError" in r.get("error", ""))
            other = sum(1 for r in rs if r is None or ("error" in r and "FailedElementsError" not in r["error"]))
            q = [math.log((by[k][x]["q2"] + 1) / (by[k]["QK2"]["q2"] + 1)) for k in full]
            L.append(f"| {x} | {len(good)} of {len(keys)} | {fe} / {other} | {sum(1 for r in good if r['on_failed'])} | "
                     f"{sum(1 for k in run_ if by[k][x]['esp'] is None)} | "
                     f"{math.exp(sum(q) / len(q)) if q else float('nan'):.3f} | {sum(r['t'] for r in good):.0f} |")

        def ratio(x, o):
            diff = [lg(by[k][x]) - lg(by[k][o]) for k in run_]
            fin = [v for v in diff if math.isfinite(v)]
            return (10 ** (sum(fin) / len(fin)) if fin else float("nan"), sum(1 for v in diff if v >= math.log10(1.1)),
                    sum(1 for v in diff if v <= -math.log10(1.1)), len(fin))
        L += ["", "| ESP ratio (runnable tests, both > 0) | gmean | 10% better | 10% worse | tests |", "|---|---|---|---|---|"]
        for x, o in (("C36D", "REL3D"), ("C36D", "QK3"), ("C36D", "QK2"), ("C36R", "REL3R"), ("C36R", "QK3"),
                     ("REL3D", "QK3"), ("REL3R", "QK3")):
            g, w, l_, n = ratio(x, o)
            L.append(f"| {x} / {o} | {g:.3f} | {w} | {l_} | {n} |")
        t = lambda x: sum(by[k][x]["t"] for k in full)  # noqa: E731
        eq = sum(1 for k in full if by[k]["C36N"]["q2"] == by[k]["REL3N"]["q2"])
        v0_bad = sum(1 for k in keys for x in ARMS if not ok(by[k].get(x)))
        v1_bad = sum(1 for k in keys for x in ("C36D", "C36R") if ok(by[k].get(x)) and by[k][x]["on_failed"])
        g2, g3, g6 = ratio("C36D", "REL3D")[0], ratio("C36D", "QK3")[0], ratio("C36R", "REL3R")[0]
        tr = t("C36D") / t("REL3D") if full else float("nan")
        score[d] = {
            "W0": (verdict(v0_bad == 0, v0_bad > 0), f"{v0_bad} arm-tests without a circuit"),
            "W1": (verdict(v1_bad == 0, v1_bad > 0), f"{v1_bad} C36D/C36R outputs on failed elements"),
            "W2": (verdict(g2 >= 1.03, g2 < 1.00), f"ESP C36D / REL3D = {g2:.3f}"),
            "W3": (verdict(g3 >= 0.96, g3 < 0.93), f"ESP C36D / QK3 = {g3:.3f}"),
            "W4": (verdict(eq == len(full), eq != len(full)), f"C36N two-qubit = REL3N on {eq} of {len(full)}"),
            "W5": (verdict(tr <= 1.5, tr > 2.0), f"time C36D / REL3D = {t('C36D'):.0f} / {t('REL3D'):.0f} s = {tr:.3f}"),
            "W6": (verdict(g6 >= 1.00, g6 < 0.97), f"ESP C36R / REL3R = {g6:.3f}"),
        }
        L.append("")
    L += ["## Scoring (Addendum 440)", "", "| ID | " + " | ".join(DEVICES) + " |", "|---|" + "---|" * len(DEVICES)]
    for vid in ("W0", "W1", "W2", "W3", "W4", "W5", "W6"):
        L.append(f"| {vid} | " + " | ".join(f"**{score[d][vid][0]}**: {score[d][vid][1]}" for d in DEVICES) + " |")
    release = all(score[d][v][0] != "REFUTED" for d in DEVICES for v in ("W0", "W1", "W2", "W4"))
    L += ["", f"Release rule (W0, W1, W2, W4 not refuted on either device): "
          f"**{'MET: c36 may replace release 2026-10-10.3' if release else 'NOT MET: release 2026-10-10.3 stays'}**", ""]
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "c36_val.md"), "w", encoding="utf-8", newline="\n").write(txt)
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
