"""ablate_c34.py -- ABLATE-C34 (2026-10-10; exploratory, nothing predicted): what PSF-Zero's own steps are worth.
STAGETIME found most of the default call's time in PSF-Zero's own steps (the SWAP absorption after routing and the
compression), not in Qiskit's. Each arm is c34's default call with backend= on BP-FINAL's 106 FakeTorino tests, on
FakeTorino, called twice in its own process (cold, warm):
  C33D  candidate c33 with the release's psf_smart_layout (the baseline)
  C34D  candidate c34 at its defaults, with c34's psf_smart_layout (rustworkx): expected to give C33D's outputs
  C34Q  c34 with ABSORB_SYNTH = "qiskit" (the absorbed blocks synthesised by Qiskit's Rust decomposer)
  C34N  c34 with post_routing_resynthesis=False (no SWAP absorption)
  C34X  c34 with COMPRESS = False (no compression before routing)
Recorded per output: both times, the two-qubit count, ESP and operations on failed elements (esp_ft.py's
definitions), and the output's value signature.

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
ARMS = ("C33D", "C34D", "C34Q", "C34N", "C34X")
C33 = os.path.join(REPO, "patches", "psf_compile_c33_2026-10-10")
C34 = os.path.join(REPO, "patches", "psf_compile_c34_2026-10-10")
DEVICES = ("FakeTorino",)
FAILED = 0.5
TIMEOUT, BUDGET = 1500, 7200


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
    import core_fix_c2_eval as H
    if arm == "C33D":
        H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
        pc = H.load_module(os.path.join(C33, "psf_compile.py"), "psf_compile")
    else:
        H.load_module(os.path.join(C34, "psf_smart_layout.py"), "psf_smart_layout")
        pc = H.load_module(os.path.join(C34, "psf_compile.py"), "psf_compile")
        if arm == "C34Q":
            pc.ABSORB_SYNTH = "qiskit"
        if arm == "C34X":
            pc.COMPRESS = False
    pc.WARN_WITHOUT_TARGET = False
    extra = dict(post_routing_resynthesis=False) if arm == "C34N" else {}

    def run(qc):
        with contextlib.redirect_stdout(io.StringIO()):
            return pc.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True,
                                           seed_transpiler=0, **extra)
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
    jobs = sorted((hashlib.sha256(("ABLATE-C34|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "ablate_c34.jsonl")
    import qiskit
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par, cpus=os.cpu_count(),
                                           qiskit=qiskit.__version__))) + "\n")
    print(f"ABLATE-C34: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
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


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "ablate_c34.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    by = {}
    for r in recs:
        by.setdefault(r["test"], {})[r["arm"]] = r
    ep = os.path.join(os.path.dirname(os.path.abspath(a.out)), "esp_ft", "esp_ft.jsonl")
    qk = {}
    if os.path.exists(ep):
        for x in [json.loads(y) for y in open(ep, encoding="utf-8")][1:]:
            if x.get("device") == "FakeTorino" and x.get("arm") in ("QK2", "QK3") and "error" not in x:
                qk[(x["test"], x["arm"])] = x
    ok = lambda r: r is not None and "error" not in r  # noqa: E731
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")  # noqa: E731
    lg = lambda r: -math.inf if r.get("esp") is None else r["esp"]  # noqa: E731
    full = [t for t in by if all(ok(by[t].get(x)) for x in ARMS)]
    small = [t for t in full if by[t]["C33D"]["t_warm"] < 0.5]
    run_ = [t for t in full if max(lg(by[t][x]) for x in ARMS) >= -2]
    L = ["# ABLATE-C34 (exploratory, nothing predicted)", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['tests']} FakeTorino tests on FakeTorino; {meta['jobs']} jobs, {meta['par']} at a time; Qiskit "
         f"{meta.get('qiskit')}. {len(full)} tests with every arm; {len(run_)} where the best ESP >= 0.01.", "",
         "| arm | cold (s, summed) | warm (s, summed) | small tests: median cold / warm (ms) | q2 / C33D (gmean, +1) | "
         "q2 / QK3 | ESP / C33D (gmean, both > 0) | ESP 10% better / worse than C33D | same output as C33D | "
         "outputs on failed elements |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x in ARMS:
        rs = [by[t][x] for t in full]
        diff = [lg(by[t][x]) - lg(by[t]["C33D"]) for t in run_]
        fin = [v for v in diff if math.isfinite(v)]
        q3 = [(by[t][x]["q2"] + 1) / (qk[(t, "QK3")]["q2"] + 1) for t in full if (t, "QK3") in qk]
        L.append(f"| {x} | {sum(r['t_cold'] for r in rs):.1f} | {sum(r['t_warm'] for r in rs):.1f} | "
                 f"{statistics.median(by[t][x]['t_cold'] for t in small) * 1000:.1f} / "
                 f"{statistics.median(by[t][x]['t_warm'] for t in small) * 1000:.1f} | "
                 f"{gm([(r['q2'] + 1) / (by[t]['C33D']['q2'] + 1) for t, r in zip(full, rs)]):.4f} | {gm(q3):.3f} | "
                 f"{10 ** (sum(fin) / len(fin)) if fin else float('nan'):.3f} | "
                 f"{sum(1 for v in diff if v >= math.log10(1.1))} / {sum(1 for v in diff if v <= -math.log10(1.1))} | "
                 f"{sum(by[t][x]['vsig'] == by[t]['C33D']['vsig'] for t in full)} of {len(full)} | "
                 f"{sum(1 for r in rs if r['on_failed'])} |")
    if qk:
        rs = [qk[(t, "QK2")] for t in full if (t, "QK2") in qk]
        L += ["", f"For scale, ESP-FT's Qiskit level 2 on the same tests: {sum(r['t'] for r in rs):.1f} s (one call, "
              "cold, in its own process)."]
    L += [f"- failed job ({t}, {x}): {by[t][x]['error'][:100]}" for t in by for x in ARMS
          if x in by[t] and not ok(by[t][x])][:30]
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "ablate_c34.md"), "w", encoding="utf-8", newline="\n").write(txt)
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
