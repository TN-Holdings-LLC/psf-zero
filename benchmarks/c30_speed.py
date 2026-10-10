"""c30_speed.py -- exploratory, nothing predicted (2026-10-10): candidate c30 (items 53, 56, 57b) against candidate
c29 (items 53, 56) on the slow development tests of type A (weakness report) and a few small ones: the recommended
call's time, two-qubit gates, whether the outputs are equal by value (`c29_identity.value_sig`), and how many
estimate and check calls each way went through psf_zero_core57 (CORE57_STATS).

Each test and arm in its own process, one after another (nothing in parallel, for the times); the layout search's
clock virtual and PYTHONHASHSEED=0. A job is killed after 1,200 s; no job starts after 3,600 s. The output folder
must not exist.

    cd <psf-zero repository>
    python <this file> --bp <benchpress clone> --c30 <folder with c30's psf_compile.py> --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))  # run from the psf-zero repository root
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [HERE, REPO]
TESTS = ("ham_ham_JW-14]", "ham_enc_gray_dvalues_8-8-8]", "ham_ham_JW-10]", "ham_ham_parity10]", "ham_ham_JW-6]",
         "grover_5.qasm]", "mod_red_21.qasm]")
JOB_S, BUDGET_S = 1200, 3600


def child(a):
    import warnings
    warnings.simplefilter("ignore")
    import bp_mock as B
    import c25_identity2 as C2
    import core_fix_c2_eval as H
    from c29_identity import value_sig
    job = json.loads(a.job)
    qc, backend = B.build(a.bp, job["kind"], job["arg"])
    lay = H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    lay.time = C2._VirtualTime()
    path = (os.path.join(REPO, "patches", "psf_compile_c29_2026-10-10", "psf_compile.py") if job["arm"] == "C29"
            else os.path.join(a.c30, "psf_compile.py"))
    pc = H.load_module(path, "psf_compile")
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, **C2.RECOMMENDED)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    print(json.dumps(dict(job, version=pc.VERSION, t=round(time.perf_counter() - t0, 2),
                          q2=int(out.count_ops().get(backend.two_q_gate_type, 0)), vsig=value_sig(out),
                          core57=dict(getattr(pc, "CORE57_STATS", {})))), flush=True)


def run(a):
    import c25_identity2 as C2
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    os.makedirs(a.out)
    pop = C2.population(a.bp)
    jobs = []
    for key in TESTS:
        hit = [t for t in pop if key in t[1] and t[0].endswith("FakeTorino")]
        if len(hit) != 1:
            sys.exit(f"STOP: {len(hit)} tests match {key}")
        jobs += [dict(test=hit[0][1], kind=hit[0][2], arg=hit[0][3], arm=arm) for arm in ("C29", "C30")]
    env = dict(os.environ, PYTHONHASHSEED="0", PSF_ZERO_REPO=REPO)
    t_start = time.perf_counter()
    res = {}
    with open(os.path.join(a.out, "c30_speed.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for n, j in enumerate(jobs, 1):
            if time.perf_counter() - t_start > BUDGET_S:
                r = dict(j, error="not started (budget)")
            else:
                try:
                    p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", "--bp", a.bp, "--c30",
                                        a.c30, "--job", json.dumps(j)], capture_output=True, text=True,
                                       timeout=JOB_S, env=env)
                    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                    r = json.loads(lines[-1]) if lines else dict(j, error=(p.stderr or "")[-300:])
                except subprocess.TimeoutExpired:
                    r = dict(j, error=f"over {JOB_S} s")
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            res[(r["test"], r["arm"])] = r
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:5.0f} s] {r['arm']} {r['test'][-40:]}: "
                  + (r["error"][:70] if "error" in r else f"{r['t']} s, q2 {r['q2']}, core57 {r['core57']}"),
                  flush=True)
    L = ["| test | C29 s / 2q | C30 s / 2q | C30 / C29 time | same by value | C30's calls in Rust / Python |",
         "|---|---|---|---|---|---|"]
    for key in TESTS:
        tid = next(t for t, _ in res if key in t)
        x, y = res[(tid, "C29")], res[(tid, "C30")]
        if "error" in x or "error" in y:
            L.append(f"| {tid.split('[')[-1].rstrip(']')} | {x.get('error', x.get('t'))} | "
                     f"{y.get('error', y.get('t'))} | | | |")
            continue
        L.append(f"| {tid.split('[')[-1].rstrip(']')} | {x['t']:.1f} / {x['q2']} | {y['t']:.1f} / {y['q2']} | "
                 f"{y['t'] / max(x['t'], 1e-9):.2f} | {'yes' if x['vsig'] == y['vsig'] else 'NO'} | "
                 f"{y['core57'].get('rust', 0)} / {y['core57'].get('python', 0)} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "c30_speed.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=("run", "child"))
    ap.add_argument("--bp", required=True)
    ap.add_argument("--c30", required=True)
    ap.add_argument("--out")
    ap.add_argument("--job")
    a = ap.parse_args()
    {"run": run, "child": child}[a.mode](a)


if __name__ == "__main__":
    main()
