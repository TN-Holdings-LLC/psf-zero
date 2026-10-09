"""c28_try.py -- exploratory, nothing predicted (2026-10-09): candidate c28 (items 53, 56, 57a) against the release.

  func   excitation_cost and hybrid_cost of c28 against c26 on 200 random circuits transpiled for FakeTorino: the
         largest relative difference (item 57 sums in another order), and the time of each
  demo   the recommended call of REL, C26 and C28 on nine development tests: time, two-qubit gates, and whether the
         output's signature is REL's. Each test and arm in its own process; killed after 1,200 s; 3 at a time.

    python c28_try.py func --repo ~/psf_zero_fresh_test --c28 DIR
    python c28_try.py demo --repo ~/psf_zero_fresh_test --c28 DIR --bp ~/benchpress --out OUT
"""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

TESTS = ("grover_5.qasm]", "mod_red_21.qasm]", "barenco_tof_10.qasm]", "ham_ham_JW-6]", "ham_ham_JW-10]",
         "ham_ham_parity10]", "ham_enc_gray_dvalues_8-8-8]", "ham_ham_JW-14]", "hwb10.qasm]")
JOB_S, BUDGET_S, PAR = 1200, 3000, 3


def paths(a):
    return {"REL": os.path.join(a.repo, "psf_compile.py"),
            "C26": os.path.join(a.repo, "patches", "psf_compile_c26_2026-10-09", "psf_compile.py"),
            "C28": os.path.join(a.c28, "psf_compile.py")}


def func(a):
    import warnings
    warnings.simplefilter("ignore")
    sys.path[:0] = [os.path.join(a.repo, "benchmarks"), a.repo]
    import core_fix_c2_eval as H
    from qiskit import transpile
    from qiskit.circuit.random import random_circuit
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    H.load_module(os.path.join(a.repo, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    p = paths(a)
    c26 = H.load_module(p["C26"], "pc26")
    c28 = H.load_module(p["C28"], "pc28")
    target = FakeTorino().target
    worst, times = {}, {}
    for k in range(200):
        n = 2 + k % 11
        qc = random_circuit(n, 4 + (k * 7) % 20, max_operands=2, measure=False, seed=57_000 + k)
        out = transpile(qc, target=target, optimization_level=1, seed_transpiler=k)
        for f in ("excitation_cost", "hybrid_cost"):
            vals = {}
            for name, mod in (("C26", c26), ("C28", c28)) if k % 2 == 0 else (("C28", c28), ("C26", c26)):
                t0 = time.perf_counter()
                vals[name] = getattr(mod, f)(out, target)
                times[(f, name)] = times.get((f, name), 0.0) + time.perf_counter() - t0
            a_, b_ = vals["C26"], vals["C28"]
            d = 0.0 if a_ == b_ else abs(a_ - b_) / max(abs(a_), 1e-300)
            worst[f] = max(worst.get(f, 0.0), d)
    for f in ("excitation_cost", "hybrid_cost"):
        print(f"{f}: largest relative difference {worst[f]:.2e}; time C26 {times[(f, 'C26')]:.2f} s, "
              f"C28 {times[(f, 'C28')]:.2f} s ({times[(f, 'C28')] / times[(f, 'C26')]:.2f})")


def child(a):
    import warnings
    warnings.simplefilter("ignore")
    sys.path[:0] = [os.path.join(a.repo, "benchmarks"), a.repo]
    import bp_mock as B
    import c25_identity as C
    import c25_identity2 as C2
    import core_fix_c2_eval as H
    job = json.loads(a.job)
    qc, backend = B.build(a.bp, job["kind"], job["arg"])
    lay = H.load_module(os.path.join(a.repo, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    lay.time = C2._VirtualTime()
    pc = H.load_module(paths(a)[job["arm"]], "psf_compile")
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, **C.RECOMMENDED)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    print(json.dumps(dict(job, version=pc.VERSION, t=round(time.perf_counter() - t0, 2),
                          q2=int(out.count_ops().get(backend.two_q_gate_type, 0)), sig=B.sig_hash(out))), flush=True)


def demo(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    os.makedirs(a.out)
    sys.path[:0] = [os.path.join(a.repo, "benchmarks"), a.repo]
    import c25_identity as C
    pop = C.population(a.bp)
    jobs = []
    for key in TESTS:
        hit = [t for t in pop if key in t[1] and t[0].endswith("FakeTorino")]
        if len(hit) != 1:
            sys.exit(f"STOP: {len(hit)} tests match {key}")
        jobs += [dict(test=hit[0][1], kind=hit[0][2], arg=hit[0][3], arm=arm) for arm in a.arms.split(",")]
    t_start = time.perf_counter()
    env = dict(os.environ, PYTHONHASHSEED="0")

    def go(j):
        if time.perf_counter() - t_start > BUDGET_S:
            return dict(j, error="not started (budget)")
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", "--repo", a.repo, "--c28", a.c28,
                                "--bp", a.bp, "--job", json.dumps(j)], capture_output=True, text=True, timeout=JOB_S,
                               env=env)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            return json.loads(lines[-1]) if lines else dict(j, error=(p.stderr or "")[-200:])
        except subprocess.TimeoutExpired:
            return dict(j, error=f"over {JOB_S} s")

    res = {}
    if a.prev:  # records of an earlier demo run for the arms not run now
        for ln in open(a.prev, encoding="utf-8"):
            r = json.loads(ln)
            if r["arm"] not in a.arms.split(","):
                res[(r["test"], r["arm"])] = r
    with ThreadPoolExecutor(PAR) as ex, open(os.path.join(a.out, "c28_try.jsonl"), "w", encoding="utf-8") as fh:
        for n, r in enumerate(ex.map(go, jobs), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            res[(r["test"], r["arm"])] = r
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:5.0f} s] {r['arm']} {r['test'][-40:]}: "
                  + (r["error"][:60] if "error" in r else f"{r['t']} s, q2 {r['q2']}"), flush=True)
    L = ["| test | REL s / 2q | C26 s / 2q | C28 s / 2q | C28 = REL? |", "|---|---|---|---|---|"]
    for key in TESTS:
        tid = next(t for t, _ in res if key in t)
        rs = [res.get((tid, arm), {"error": "missing"}) for arm in ("REL", "C26", "C28")]
        cells = ["x" if "error" in r else f"{r['t']:.1f} / {r['q2']}" for r in rs]
        same = "-" if any("error" in r for r in (rs[0], rs[2])) else ("same" if rs[0]["sig"] == rs[2]["sig"] else "DIFFERS")
        L.append(f"| {tid.split('[')[-1].rstrip(']')} | " + " | ".join(cells) + f" | {same} |")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "c28_try.md"), "w", encoding="utf-8").write(txt)
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("func", "demo", "child"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--c28", required=True)
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--job")
    ap.add_argument("--arms", default="REL,C26,C28")
    ap.add_argument("--prev", help="an earlier demo's c28_try.jsonl, for the arms not run now")
    a = ap.parse_args()
    {"func": func, "demo": demo, "child": child}[a.mode](a)


if __name__ == "__main__":
    main()
