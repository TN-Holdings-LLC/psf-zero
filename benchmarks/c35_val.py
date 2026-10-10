"""c35_val.py -- C35-VAL (pre-registered in Addendum 436, 2026-10-10): should candidate 2026-10-10.c35 (changelog
items 58-63) replace release 2026-10-10.2? Measured as a device user meets it: whether an output uses a failed
element, its estimated success probability (ESP), its two-qubit count and the compile time.

Tests: BP-FINAL's 106 FakeTorino tests (HamLib and Feynman), as ESP-FT built them. Devices: FakeTorino and
FakeKingston (qiskit-ibm-runtime's snapshots; both keep couplers and qubits at error 1 in their Targets, as the live
ibm_kingston Target did on 2026-09-28: Addendum 434). Arms, each test, device and arm in its own process, one cold
call, 4 at a time:
  QK2   Qiskit level 2 on the backend (generate_preset_pass_manager(2, backend))
  QK3   Qiskit level 3 on the backend
  RELR  release 2026-10-10.2 (psf_compile.py), the README's recommended call: coupling map and basis from the Target,
        target=, bp_final.RECOMMENDED
  RELD  release 2026-10-10.2, the default call without the Target (coupling map and basis only)
  C35D  c35 (patches/psf_compile_c35_2026-10-10), the default call with backend=
  C35R  c35, backend= and bp_final.RECOMMENDED
  C35N  c35, the default call without the Target (coupling map and basis only), as RELD
Common: entangling_basis="cx", layout_search=True, seed_transpiler=0. The PSF-Zero arms load psf_smart_layout.py from
their own folder (benchmarks/ for the release, the candidate's folder for c35).

ESP = product over the output's gates and measurements of (1 - error) as the Target reports them (rz 0; delays and
barriers ignored; no idling term); log10 ESP is null when an operation has error 1. A failed element has error
>= 0.5. "Runnable" tests: the best ESP over the seven arms is >= 0.01. ESP ratios are geometric means over the
runnable tests where both arms' ESP > 0. A c35 call that raises FailedElementsError is recorded as such.

Predictions (Addendum 436), each on both devices:
  V0  every arm returns a circuit on every test (C35D and C35R may raise FailedElementsError; counted, not excused)
  V1  C35D and C35R: no output with an operation on a failed element
  V2  ESP C35D / RELR >= 0.95 (confirmed); < 0.90 refuted
  V3  ESP C35D / QK2 >= 1.05 (confirmed); < 1.00 refuted
  V4  compile time summed, C35D / RELR <= 0.60 (confirmed); > 0.80 refuted
  V5  C35N's two-qubit count equals RELD's on every test (confirmed); summed within 0.5% (ambiguous); refuted otherwise
  V6  ESP C35R / RELR >= 0.97 (confirmed); < 0.93 refuted
Release rule (Addendum 436): c35 becomes the release if V0, V1, V2, V3 and V5 are not refuted on either device.

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
ARMS = ("QK2", "QK3", "RELR", "RELD", "C35D", "C35R", "C35N")
C35 = os.path.join(REPO, "patches", "psf_compile_c35_2026-10-10")
FILES = {"RELR": (os.path.join(REPO, "psf_compile.py"), os.path.join(HERE, "psf_smart_layout.py")),
         "RELD": (os.path.join(REPO, "psf_compile.py"), os.path.join(HERE, "psf_smart_layout.py")),
         "C35D": (os.path.join(C35, "psf_compile.py"), os.path.join(C35, "psf_smart_layout.py")),
         "C35R": (os.path.join(C35, "psf_compile.py"), os.path.join(C35, "psf_smart_layout.py")),
         "C35N": (os.path.join(C35, "psf_compile.py"), os.path.join(C35, "psf_smart_layout.py"))}
DEVICES = ("FakeTorino", "FakeKingston")
FAILED = 0.5
TIMEOUT, BUDGET = 1500, 7200
NEG = -1e300
BASIS = ("cx", "cz", "ecr", "rz", "sx", "x", "id")


def tests(bp):
    import bp_final as F
    return [t for t in F.population(bp) if t[0].endswith("FakeTorino")]


def device(name, bp_backend):
    if name == "FakeTorino":
        return bp_backend
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


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
        kw = {"RELR": dict(no_target, target=target, **F.RECOMMENDED), "RELD": no_target,
              "C35D": dict(backend=backend), "C35R": dict(backend=backend, **F.RECOMMENDED),
              "C35N": no_target}[a.arm]
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
    jobs = sorted((hashlib.sha256(("C35-VAL|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "c35_val.jsonl")
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
    print(f"C35-VAL: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
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
    lines = [json.loads(x) for x in open(os.path.join(a.out, "c35_val.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    lg = lambda r: -math.inf if r.get("esp") is None else r["esp"]  # noqa: E731
    ok = lambda r: r is not None and "error" not in r  # noqa: E731
    L = ["# C35-VAL (pre-registered in Addendum 436)", "",
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
        for x, o in (("C35D", "RELR"), ("C35D", "QK2"), ("C35D", "QK3"), ("C35R", "RELR"), ("C35R", "QK3"),
                     ("RELR", "QK3"), ("RELD", "QK2")):
            g, w, l_, n = ratio(x, o)
            L.append(f"| {x} / {o} | {g:.3f} | {w} | {l_} | {n} |")
        t = lambda x: sum(by[k][x]["t"] for k in full)  # noqa: E731
        eq = sum(1 for k in full if by[k]["C35N"]["q2"] == by[k]["RELD"]["q2"])
        s_n, s_d = sum(by[k]["C35N"]["q2"] for k in full), sum(by[k]["RELD"]["q2"] for k in full)
        v0_bad = sum(1 for k in keys for x in ARMS if not ok(by[k].get(x)))
        v1_bad = sum(1 for k in keys for x in ("C35D", "C35R") if ok(by[k].get(x)) and by[k][x]["on_failed"])
        g2, g3, g6 = ratio("C35D", "RELR")[0], ratio("C35D", "QK2")[0], ratio("C35R", "RELR")[0]
        tr = t("C35D") / t("RELR") if full else float("nan")
        score[d] = {
            "V0": (verdict(v0_bad == 0, v0_bad > 0), f"{v0_bad} arm-tests without a circuit"),
            "V1": (verdict(v1_bad == 0, v1_bad > 0), f"{v1_bad} C35D/C35R outputs on failed elements"),
            "V2": (verdict(g2 >= 0.95, g2 < 0.90), f"ESP C35D / RELR = {g2:.3f}"),
            "V3": (verdict(g3 >= 1.05, g3 < 1.00), f"ESP C35D / QK2 = {g3:.3f}"),
            "V4": (verdict(tr <= 0.60, tr > 0.80), f"time C35D / RELR = {t('C35D'):.0f} / {t('RELR'):.0f} s = {tr:.3f}"),
            "V5": (verdict(eq == len(full), eq != len(full) and abs(s_n - s_d) > 0.005 * s_d),
                   f"C35N two-qubit = RELD on {eq} of {len(full)}; summed {s_n} / {s_d}"),
            "V6": (verdict(g6 >= 0.97, g6 < 0.93), f"ESP C35R / RELR = {g6:.3f}"),
        }
        L.append("")
    L += ["## Scoring (Addendum 436)", "", "| ID | " + " | ".join(DEVICES) + " |", "|---|" + "---|" * len(DEVICES)]
    for vid in ("V0", "V1", "V2", "V3", "V4", "V5", "V6"):
        L.append(f"| {vid} | " + " | ".join(f"**{score[d][vid][0]}**: {score[d][vid][1]}" for d in DEVICES) + " |")
    release = all(score[d][v][0] != "REFUTED" for d in DEVICES for v in ("V0", "V1", "V2", "V3", "V5"))
    L += ["", f"Release rule (V0, V1, V2, V3, V5 not refuted on either device): "
          f"**{'MET: c35 may replace release 2026-10-10.2' if release else 'NOT MET: release 2026-10-10.2 stays'}**", ""]
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "c35_val.md"), "w", encoding="utf-8", newline="\n").write(txt)
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
