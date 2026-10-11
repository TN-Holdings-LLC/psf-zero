"""c37_x.py -- C37-X (2026-10-11; exploratory, not pre-registered): release 2026-10-11.1 (REL) against candidate c37
(C37: items 65-66) and Qiskit level 3 (QK3), with GAP-DIAG's breakdown of log10 ESP into two-qubit, one-qubit and
measurement parts, on BP-FINAL's 106 FakeTorino tests and FakeTorino, FakeKingston, FakeMarrakesh, FakeFez (the
same tests and devices as GAP-DIAG, so this is a development measurement, not a validation).

One process per test and device runs QK3, REL and C37 in that order (times are indicative only).

    cd <psf-zero repository>
    [C37X_ARMS=QK3,REL,C37] [C37X_DEVICES=FakeTorino,...] python benchmarks/c37_x.py run --bp <benchpress> --out DIR
        [--par 2] [--only SUBSTR] [--select-from <an earlier run's jsonl>]
    python benchmarks/c37_x.py report --bp <benchpress> --out DIR
Arms: QK3 (Qiskit level 3, unseeded), QK3x4 (level 3 with seeds 0-3, kept by the same rule as item 67), REL
(psf_compile.py), C37 (the candidate; C37_FILE, default patches/psf_compile_c37_2026-10-11/psf_compile.py; each run
keeps a copy as run).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
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
C37 = os.environ.get("C37_FILE", os.path.join(REPO, "patches", "psf_compile_c37_2026-10-11", "psf_compile.py"))
GAP = os.path.join(REPO, "benchmarks")  # gap_diag.py
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"),
                REPO, GAP]
DEVICES = tuple(os.environ.get("C37X_DEVICES", "FakeTorino,FakeKingston,FakeMarrakesh,FakeFez").split(","))
ARMS = tuple(os.environ.get("C37X_ARMS", "QK3,REL,C37").split(","))
TIMEOUT, BUDGET = 1500, 14400


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def one(a):
    warnings.simplefilter("ignore")
    import bp_mock as B
    import gap_diag as G
    from qiskit_ibm_runtime import fake_provider
    from qiskit.transpiler import generate_preset_pass_manager
    stratum, tid, kind, arg = json.loads(a.job)
    qc, _ = B.build(a.bp, kind, arg)
    backend = getattr(fake_provider, a.device)()
    target = backend.target
    load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    mods = {"REL": load(os.path.join(REPO, "psf_compile.py"), "psf_rel"), "C37": load(C37, "psf_c37")}
    for arm in ARMS:
        rec = dict(stratum=stratum, test=tid, device=a.device, arm=arm, input_qubits=qc.num_qubits)
        t0 = time.perf_counter()
        try:
            if arm == "QK3":
                out = generate_preset_pass_manager(3, backend).run(qc)
            elif arm == "QK3x4":  # Qiskit level 3 with seeds 0-3, kept by the same rule as item 67
                best = None
                for s in range(4):
                    o = generate_preset_pass_manager(3, backend, seed_transpiler=s).run(qc)
                    b = G.breakdown(o, target)
                    c = math.inf if b["failed"] else b["s2"] + b["s1"] + b["sm"]
                    if best is None or c < best[0]:
                        best = (c, o)
                out = best[1]
            else:
                rec["version"] = mods[arm].VERSION
                with contextlib.redirect_stdout(io.StringIO()):
                    out = mods[arm].compile_for_hardware(qc, backend=backend, entangling_basis="cx",
                                                         layout_search=True, seed_transpiler=0)
        except Exception as exc:  # noqa: BLE001 - recorded
            rec.update(t=round(time.perf_counter() - t0, 3), error=f"{type(exc).__name__}: {exc}"[:300])
            print(json.dumps(rec), flush=True)
            continue
        rec["t"] = round(time.perf_counter() - t0, 3)
        rec.update(G.breakdown(out, target))
        if arm == "C37":
            rec["final_stats"] = dict(mods["C37"].FINAL_STATS)
            rec["tries_stats"] = dict(getattr(mods["C37"], "TRIES_STATS", {}))
        print(json.dumps(rec), flush=True)


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    import bp_final as F
    ts = [t for t in F.population(a.bp) if t[0].endswith("FakeTorino") and (a.only is None or a.only in t[1])]
    jobs = [(t, d) for t in ts for d in DEVICES]
    if a.select_from:  # only the (test, device) pairs that were runnable there: best log10 ESP of any arm >= -2
        prev = {}
        for ln in open(a.select_from, encoding="utf-8"):
            r = json.loads(ln)
            if "s2" in r:
                c = r["s2"] + r["s1"] + r["sm"]
                prev[(r["test"], r["device"])] = min(prev.get((r["test"], r["device"]), math.inf), c)
        jobs = [(t, d) for t, d in jobs if prev.get((t[1], d), math.inf) <= 2]
    os.makedirs(a.out)
    import hashlib
    import shutil
    frozen = os.path.join(a.out, "psf_compile_c37.py")  # the candidate as run: later edits do not reach this run
    shutil.copyfile(C37, frozen)
    path = os.path.join(a.out, "c37_x.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           tests=len(ts), jobs=len(jobs), par=a.par, arms=ARMS,
                                           select_from=a.select_from,
                                           c37_sha256=hashlib.sha256(open(frozen, "rb").read()).hexdigest()))) + "\n")
    t_start = time.perf_counter()

    def go(job):
        t, d = job
        base = dict(stratum=t[0], test=t[1], device=d)
        if time.perf_counter() - t_start > BUDGET:
            return [dict(base, arm=x, error="not started (budget)") for x in ARMS]
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--device", d,
                                "--job", json.dumps(list(t))], capture_output=True, text=True, timeout=TIMEOUT,
                               cwd=REPO, env=dict(os.environ, C37_FILE=frozen))
            recs = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]
            got = {r["arm"] for r in recs}
            return recs + [dict(base, arm=x, error=(p.stderr or "")[-300:]) for x in ARMS if x not in got]
        except subprocess.TimeoutExpired:
            return [dict(base, arm=x, error=f"timeout {TIMEOUT} s") for x in ARMS]

    n = 0
    with ThreadPoolExecutor(a.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in jobs]):
            for r in f.result():
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:6.0f} s]", flush=True)
    if set(ARMS) == {"QK3", "REL", "C37"}:
        report(a)


def report(a):
    rows = [json.loads(x) for x in open(os.path.join(a.out, "c37_x.jsonl"), encoding="utf-8")][1:]
    by = {}
    for r in rows:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    L = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731
    out = ["# C37-X (exploratory; same tests and devices as GAP-DIAG)", ""]
    errs = [r for r in rows if "error" in r]
    out.append(f"jobs with an error: {len(errs)}")
    for r in errs[:10]:
        out.append(f"- {r['device']} {r['arm']} {r['test'][-50:]}: {r['error'][:120]}")
    out += ["", "## Failed elements (outputs with an operation at error >= 0.5)", "",
            "| device | QK3 | REL | C37 |", "|---|---|---|---|"]
    for d in DEVICES:
        cells = [str(sum(1 for (dv, _), x in by.items() if dv == d and arm in x and x[arm].get("failed")))
                 for arm in ARMS]
        out.append(f"| {d} | " + " | ".join(cells) + " |")
    out += ["", "## Runnable tests (best ESP of the three >= 0.01), all outputs clean: geometric-mean ESP ratios and "
            "mean part differences (log10; positive = the first arm loses)", "",
            "| device | tests | C37/QK3 | REL/QK3 | C37/REL | 2q part C37-QK3 | 1q part C37-QK3 | meas part C37-QK3 | "
            "1q part C37-REL | meas part C37-REL | non-rz 1q C37/REL/QK3 | time C37/REL/QK3 s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ess = {}
    for d in DEVICES:
        sel = []
        for (dv, t), x in by.items():
            if dv != d or any(arm not in x or "error" in x[arm] for arm in ARMS):
                continue
            if any(x[arm]["failed"] for arm in ARMS):
                continue
            if min(L(x[arm]) for arm in ARMS) > 2:
                continue
            sel.append((t, x))
        n = len(sel)
        if not n:
            continue

        def gm(p, q):
            return 10 ** (sum(L(x[q]) - L(x[p]) for _, x in sel) / n)

        def part(k, p, q):
            return sum(x[p][k] - x[q][k] for _, x in sel) / n
        n1 = [sum(x[arm]["n1"] for _, x in sel) for arm in ("C37", "REL", "QK3")]
        tt = [sum(x[arm]["t"] for _, x in sel) for arm in ("C37", "REL", "QK3")]
        out.append(f"| {d} | {n} | {gm('C37', 'QK3'):.3f} | {gm('REL', 'QK3'):.3f} | {gm('C37', 'REL'):.3f} | "
                   f"{part('s2', 'C37', 'QK3'):+.5f} | {part('s1', 'C37', 'QK3'):+.5f} | {part('sm', 'C37', 'QK3'):+.5f}"
                   f" | {part('s1', 'C37', 'REL'):+.5f} | {part('sm', 'C37', 'REL'):+.5f} | "
                   f"{n1[0]}/{n1[1]}/{n1[2]} | {tt[0]:.0f}/{tt[1]:.0f}/{tt[2]:.0f} |")
        ess[d] = sel
    out += ["", "## Essential: runnable tests where QK3's own failed elements count against it (ESP of a failed "
            "output = 0)", "", "| device | runnable | QK3 ESP 0 | C37 ESP 0 | C37/QK3 on both > 0 |", "|---|---|---|---|---|"]
    for d in DEVICES:
        sel = []
        for (dv, t), x in by.items():
            if dv != d or any(arm not in x or "error" in x[arm] for arm in ARMS):
                continue
            esp = {arm: (0.0 if x[arm]["failed"] else 10 ** -L(x[arm])) for arm in ARMS}
            if max(esp.values()) < 0.01:
                continue
            sel.append(esp)
        both = [e for e in sel if e["C37"] > 0 and e["QK3"] > 0]
        r = 10 ** (sum(math.log10(e["C37"] / e["QK3"]) for e in both) / len(both)) if both else float("nan")
        out.append(f"| {d} | {len(sel)} | {sum(1 for e in sel if e['QK3'] == 0)} | "
                   f"{sum(1 for e in sel if e['C37'] == 0)} | {r:.3f} |")
    out += ["", "## Largest C37 losses to QK3 (clean runnable tests, log10 ESP)", ""]
    worst = []
    for d, sel in ess.items():
        for t, x in sel:
            worst.append((L(x["QK3"]) - L(x["C37"]), d, t, x))
    worst.sort(key=lambda w: w[0])
    for w, d, t, x in worst[:12]:
        out.append(f"- {d} {t[-48:]}: {w:+.4f} (2q {x['C37']['n2']} vs {x['QK3']['n2']}; parts C37 "
                   f"{x['C37']['s2']:.3f}/{x['C37']['s1']:.3f}/{x['C37']['sm']:.3f}, QK3 {x['QK3']['s2']:.3f}/"
                   f"{x['QK3']['s1']:.3f}/{x['QK3']['sm']:.3f})")
    stats = {}
    for r in rows:
        for k, v in (r.get("final_stats") or {}).items():
            stats[k] = stats.get(k, 0) + v
    out += ["", f"FINAL_STATS summed over C37 calls: {stats}", ""]
    txt = "\n".join(out)
    open(os.path.join(a.out, "c37_x.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "report"))
    ap.add_argument("--bp", required=True, help="a Benchpress clone")
    ap.add_argument("--out")
    ap.add_argument("--device")
    ap.add_argument("--job")
    ap.add_argument("--only")
    ap.add_argument("--select-from")
    ap.add_argument("--par", type=int, default=2)
    a = ap.parse_args()
    {"run": run, "one": one, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
