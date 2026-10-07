"""cancel_eval.py -- test CANCEL (2026-10-07): does candidate c23 (changelog item 50: commutative cancellation tried
when it removes two-qubit gates) leave the default call's circuit unchanged where the cancellation removes nothing,
never use more two-qubit gates where it removes some, stay exact, and cost little time? Addendum 394 has the
predictions.

Tests: the 140 Benchpress tests of BP-MOCK (92) and BP-MOCK2 (48), built as there (bp_mock.py, bp_mock2.py, imported
unchanged). Item 50 was written after BP-MOCK, from one of its tests (BV-like), and after BP-MOCK2; it was not tuned on
the others. Arms (default call, no target): C22, C22 again in its own process (C22B: the release-like pipeline does not
always reproduce itself, Addenda 390, 392), and C23. Each test and arm in its own process, 1,500 s limit (C23 may run
the pipeline twice), six at a time.

    python benchmarks/cancel_eval.py run --bp <benchpress clone> --out DIR [--par N] [--smoke]
    python benchmarks/cancel_eval.py score --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
sys.path[:0] = [os.path.join(WORK, "depth1"), HERE, REPO]
import bp_mock as B  # noqa: E402  (locked in Addendum 389)
import bp_mock2 as B2  # noqa: E402  (locked in Addendum 391)

C23_PATH = os.path.join(REPO, "patches", "psf_compile_c23_2026-10-07", "psf_compile.py")
ARMS = ("C22", "C22B", "C23")
VERSIONS = {"C22": "2026-10-07.c22", "C22B": "2026-10-07.c22", "C23": "2026-10-07.c23"}
TIMEOUT = 1500


def tests(bp, smoke=False):
    t = B.sample(bp) + B2.sample(bp)
    if smoke:  # one test per stratum, BV-like included
        seen, out = set(), []
        for x in t:
            if x[0] not in seen or "BVlike" in x[1]:
                seen.add(x[0])
                out.append(x)
        return out
    return t


def one(args):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(args.job)
    qc, backend = B.build(args.bp, kind, arg)
    from qiskit import transpile
    from benchpress.qiskit_gym.utils.validation import qiskit_circuit_validation
    import core_fix_c2_eval as H
    two_q = backend.two_q_gate_type
    rec = dict(stratum=stratum, test=tid, arm=args.arm, input_qubits=qc.num_qubits, wide=B.wide(qc))
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(C23_PATH if args.arm == "C23" else B.C22_PATH, "psf_compile")
    rec["version"] = pc.VERSION
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0)
    rec["t"] = time.perf_counter() - t0
    if hasattr(pc, "CANCEL_STATS"):
        rec["cancel"] = dict(pc.CANCEL_STATS)
    ops = out.count_ops()
    rec["q2"] = int(ops.get(two_q, 0))
    rec["d2"] = out.depth(filter_function=lambda x: x.operation.name == two_q)
    rec["sig"] = B.sig_hash(out)
    try:
        qiskit_circuit_validation(out, backend)
        rec["valid"] = True
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["valid"] = f"{type(exc).__name__}: {exc}"[:200]
    if qc.num_qubits <= B2.CHECK_MAX_QUBITS:
        bare = qc.copy()
        bare.remove_final_measurements(inplace=True)
        ref = transpile(bare, basis_gates=["u", "cx"], optimization_level=0)
        chk = H.load_module(B.C22_PATH, "psf_compile_check")
        try:
            rec["implements"] = bool(chk._implements(ref, out))
            rec["checkable"] = chk.EXACT_STATS["not_checkable"] == 0
        except Exception as exc:  # noqa: BLE001
            rec["implements"] = f"n/a: {type(exc).__name__}"[:80]
            rec["checkable"] = False
    print(json.dumps(rec))


def run(args):
    ts = tests(args.bp, args.smoke)
    jobs = [(t, arm) for t in ts for arm in ARMS]
    os.makedirs(args.out, exist_ok=True)
    head, dirty = B.git_state()
    import qiskit
    meta = dict(smoke=bool(args.smoke), git_head=head, dirty_tracked=dirty, qiskit=qiskit.__version__,
                python=sys.version.split()[0], par=args.par, tests=len(ts), jobs=len(jobs),
                sha=dict(script=B.norm_sha(os.path.abspath(__file__)), c22=B.norm_sha(B.C22_PATH),
                         c23=B.norm_sha(C23_PATH), bp_mock=B.norm_sha(os.path.join(HERE, "bp_mock.py")),
                         bp_mock2=B.norm_sha(os.path.join(HERE, "bp_mock2.py"))),
                start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps(meta), flush=True)
    t00 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1",
               QISKIT_PARALLEL="FALSE", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def go(job):
        t, arm = job
        cmd = [sys.executable, os.path.abspath(__file__), "one", "--bp", args.bp, "--arm", arm, "--job", json.dumps(t)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=REPO, env=env,
                               encoding="utf-8", errors="replace")
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else dict(stratum=t[0], test=t[1], arm=arm,
                                                           error=(p.stderr or "")[-400:])
        except subprocess.TimeoutExpired:
            rec = dict(stratum=t[0], test=t[1], arm=arm, error=f"timeout {TIMEOUT} s")
        show = {k: rec[k] for k in ("q2", "valid", "implements", "cancel", "error") if k in rec}
        print(f"{int(time.time() - t00):5d} s  {t[1][:58]:58s} {arm:4s} {show}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.par) as ex:
        rows = list(ex.map(go, jobs))
    meta["wall_s"] = round(time.time() - t00, 1)
    fn = "cancel_smoke.json" if args.smoke else "cancel.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, fn), "w"), indent=1)
    print(f"wrote {fn}: {len(ts)} tests, {len(rows)} jobs, {meta['wall_s']} s")


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    fn = os.path.join(args.out, "cancel.json")
    smoke = not os.path.exists(fn)
    r = json.load(open(fn if not smoke else os.path.join(args.out, "cancel_smoke.json")))
    m, rows = r["meta"], r["rows"]
    by = {}
    for x in rows:
        by.setdefault(x["test"], {})[x["arm"]] = x
    ts = list(by)

    def ok(x):
        return "error" not in x

    n_expected = m["tests"] if smoke else 140
    bad = [t for t in ts if (not ok(by[t]["C23"]) and ok(by[t]["C22"])) or
           (ok(by[t]["C23"]) and by[t]["C23"]["valid"] is not True)]
    vers = all(x.get("version") == VERSIONS[x["arm"]] for x in rows if ok(x))
    meta_ok = m["smoke"] == smoke and (smoke or not m["dirty_tracked"])
    p0 = len(ts) == n_expected and not bad and vers and meta_ok
    errs = {a: sum(1 for t in ts if "error" in by[t][a]) for a in ARMS}
    out = [f"# CANCEL score{' (SMOKE: not counted)' if smoke else ''}\n",
           f"git_head {m['git_head']}, tests {len(ts)} (expected {n_expected}), jobs {len(rows)}",
           f"P0 {'PASS' if p0 else 'FAIL'}: C23 failed where C22 finished, or invalid: {len(bad)}; versions ok {vers}; "
           f"meta ok {meta_ok}", f"errors or timeouts per arm: {errs}"]
    if not p0:
        out.append("Nothing below is scored.")
        print("\n".join(out))
        open(os.path.join(args.out, "score.md"), "w").write("\n".join(out) + "\n")
        return
    done = [t for t in ts if all(ok(by[t][a]) for a in ARMS)]
    tried = [t for t in done if by[t]["C23"]["cancel"]["tried"] > 0]
    quiet = [t for t in done if by[t]["C23"]["cancel"]["tried"] == 0]
    stable = [t for t in quiet if by[t]["C22"]["sig"] == by[t]["C22B"]["sig"]]
    k1 = [t for t in stable if by[t]["C23"]["sig"] != by[t]["C22"]["sig"]]
    k2 = [t for t in tried if by[t]["C23"]["q2"] > by[t]["C22"]["q2"]]
    chk = [t for t in done if by[t]["C23"].get("checkable") is True and isinstance(by[t]["C23"].get("implements"), bool)]
    k4 = [t for t in chk if not by[t]["C23"]["implements"]]
    tr = [max(by[t]["C23"]["t"], 1e-4) / max(by[t]["C22"]["t"], 1e-4) for t in quiet]
    k5 = statistics.median(tr) if tr else float("nan")
    res = [("K1", "no two-qubit gate cancels, C22 reproduces itself: C23 returns C22's circuit",
            f"{len(k1)} of {len(stable)} differ ({len(quiet) - len(stable)} where C22 did not reproduce)",
            verdict(not k1, len(k1) >= 2)),
           ("K2", "some cancel: C23 has no more two-qubit gates than C22", f"{len(k2)} of {len(tried)} have more",
            verdict(bool(tried) and not k2, bool(k2))),
           ("K4", "every checkable C23 output implements its input", f"{len(k4)} of {len(chk)} do not",
            verdict(len(chk) >= 5 and not k4, bool(k4))),
           ("K5", "no two-qubit gate cancels: median time C23 / C22 <= 1.15", f"{k5:.3f} on {len(tr)}",
            verdict(k5 <= 1.15, k5 > 1.5))]
    out += ["", f"tests finished by all arms: {len(done)}; cancellation tried on {len(tried)}", "",
            "| | prediction | value | verdict |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {v} | **{c}** |" for a, b, v, c in res]
    out += ["", "Reported without prediction: the tests where cancellation was tried",
            "", "| test | C22 two-qubit | C23 two-qubit | C23 kept | C23 / C22 time |", "|---|---|---|---|---|"]
    for t in tried:
        c = by[t]["C23"]["cancel"]
        kept = "cancelled" if c.get("cancelled_kept") else "original"
        out.append(f"| {t} | {by[t]['C22']['q2']} | {by[t]['C23']['q2']} | {kept} | "
                   f"{by[t]['C23']['t'] / max(by[t]['C22']['t'], 1e-4):.2f} |")
    print("\n".join(out))
    open(os.path.join(args.out, "score.md"), "w").write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "score"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "one": one, "score": score}[a.mode](a)


if __name__ == "__main__":
    main()
