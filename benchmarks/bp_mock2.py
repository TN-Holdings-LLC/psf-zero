"""bp_mock2.py -- BP-MOCK2 (2026-10-07): the pre-registered re-test of item 48 (instructions on three or more qubits
expanded first; candidate c22) after BP-MOCK (Addenda 389-390), on a new sample of Benchpress, with the two things BP-MOCK
got wrong put right: the release's own non-reproducibility (REL is compiled twice, in two processes) and the
equivalence check (item 39's `_implements` and the workplace probe's state check, against the input expanded through
its definitions). Addendum 391 has the predictions.

Sample (fixed by rule, before any run): BP-MOCK's strata without the 100-qubit tests (all of which BP-PROBE or BP-MOCK
used); in each, the published test ids minus the 12 BP-PROBE ran and the 92 BP-MOCK ran, ordered by SHA-256 of
"BP-MOCK2|" + id, and the first K taken. Building, arms QK/REL/C22 and metrics as BP-MOCK (bp_mock.py, imported
unchanged); REL2 is REL again in its own process. No recommended call.

    python benchmarks/bp_mock2.py sample --bp <benchpress clone>
    python benchmarks/bp_mock2.py run --bp <benchpress clone> --out DIR [--par N] [--smoke]
    python benchmarks/bp_mock2.py score --out DIR
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
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
sys.path[:0] = [os.path.join(WORK, "depth1"), HERE, REPO]
import bp_mock as B  # noqa: E402  (BP-MOCK's script, locked in Addendum 389, unchanged)

K = {"QASMBench small": 3, "QASMBench medium": 2, "QASMBench large": 2, "HamLib": 3, "HamLib, FakeTorino": 4,
     "Feynman, FakeTorino": 4}
ARMS = ("QK", "REL", "REL2", "C22")
CHECK_MAX_QUBITS = 10  # inputs checked for equivalence (as BP-MOCK: at most 10 qubits)


def sample(bp, smoke=False):
    used = {t[1] for t in B.sample(bp)}
    ref = json.load(open(B.REF))
    out = []
    for name, tests in B.strata(bp).items():
        if name not in K and name.split(",")[0] not in K:
            continue
        k = K.get(name, K.get(name.split(",")[0]))
        keep = sorted((t for t in tests if t[0] in ref and t[0] not in B.PROBED and t[0] not in used),
                      key=lambda t: hashlib.sha256(("BP-MOCK2|" + t[0]).encode()).hexdigest())
        out += [(name,) + tuple(t) for t in keep[:1 if smoke else k]]
    return out


def one(args):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(args.job)
    qc, backend = B.build(args.bp, kind, arg)
    from qiskit import transpile
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from benchpress.qiskit_gym.utils.validation import qiskit_circuit_validation
    import core_fix_c2_eval as H
    two_q = backend.two_q_gate_type
    rec = dict(stratum=stratum, test=tid, arm=args.arm, input_qubits=qc.num_qubits, wide=B.wide(qc))
    if args.arm == "QK":
        t0 = time.perf_counter()
        out = generate_preset_pass_manager(2, backend).run(qc)
        rec["t"] = time.perf_counter() - t0
    else:
        H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
        pc = H.load_module(B.REL_PATH if args.arm.startswith("REL") else B.C22_PATH, "psf_compile")
        rec["version"] = pc.VERSION
        basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=basis,
                                          entangling_basis="cx", layout_search=True, seed_transpiler=0)
        rec["t"] = time.perf_counter() - t0
    ops = out.count_ops()
    rec["q2"] = int(ops.get(two_q, 0))
    rec["d2"] = out.depth(filter_function=lambda x: x.operation.name == two_q)
    rec["sig"] = B.sig_hash(out)
    try:
        qiskit_circuit_validation(out, backend)
        rec["valid"] = True
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["valid"] = f"{type(exc).__name__}: {exc}"[:200]
    if qc.num_qubits <= CHECK_MAX_QUBITS:
        # the input with final measurements removed, expanded through its definitions (a PauliEvolutionGate becomes
        # the product formula every compiler builds)
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
        try:
            RE = H.load_module(os.path.join(WORK, "readout", "readout_eval.py"), "readout_eval_bpm2")
            rec["infid0"] = RE.state_infid(ref, RE.strip_measure(out))
        except Exception as exc:  # noqa: BLE001
            rec["infid0"] = f"n/a: {type(exc).__name__}"[:80]
    print(json.dumps(rec))


def run(args):
    tests = sample(args.bp, args.smoke)
    jobs = [(t, arm) for t in tests for arm in ARMS]
    os.makedirs(args.out, exist_ok=True)
    head, dirty = B.git_state()
    bphead = subprocess.run(["git", "-C", args.bp, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    import qiskit
    meta = dict(smoke=bool(args.smoke), git_head=head, dirty_tracked=dirty, benchpress=bphead,
                qiskit=qiskit.__version__, python=sys.version.split()[0], par=args.par, tests=len(tests),
                jobs=len(jobs), sha=dict(script=B.norm_sha(os.path.abspath(__file__)),
                                         bp_mock=B.norm_sha(os.path.join(HERE, "bp_mock.py")),
                                         release=B.norm_sha(B.REL_PATH), c22=B.norm_sha(B.C22_PATH)),
                start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps(meta), flush=True)
    t00 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1",
               QISKIT_PARALLEL="FALSE", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def go(job):
        t, arm = job
        cmd = [sys.executable, os.path.abspath(__file__), "one", "--bp", args.bp, "--arm", arm, "--job", json.dumps(t)]
        w0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=B.TIMEOUT, cwd=REPO, env=env,
                               encoding="utf-8", errors="replace")
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else dict(stratum=t[0], test=t[1], arm=arm,
                                                           error=(p.stderr or "")[-400:])
        except subprocess.TimeoutExpired:
            rec = dict(stratum=t[0], test=t[1], arm=arm, error=f"timeout {B.TIMEOUT} s")
        rec["wall"] = round(time.perf_counter() - w0, 2)
        show = {k: rec[k] for k in ("q2", "d2", "valid", "implements", "error") if k in rec}
        print(f"{int(time.time() - t00):5d} s  {t[1][:60]:60s} {arm:4s} {show}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.par) as ex:
        rows = list(ex.map(go, jobs))
    meta["wall_s"] = round(time.time() - t00, 1)
    fn = "bp_mock2_smoke.json" if args.smoke else "bp_mock2.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, fn), "w"), indent=1)
    print(f"wrote {fn}: {len(tests)} tests, {len(rows)} jobs, {meta['wall_s']} s")


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    fn = os.path.join(args.out, "bp_mock2.json")
    smoke = not os.path.exists(fn)
    r = json.load(open(fn if not smoke else os.path.join(args.out, "bp_mock2_smoke.json")))
    m, rows = r["meta"], r["rows"]
    by = {}
    for x in rows:
        by.setdefault(x["test"], {})[x["arm"]] = x
    tests = list(by)
    n_expected = 18 if smoke else 4 * (3 + 2 + 2 + 3) + 4 + 4

    def ok(x):
        return "error" not in x

    c22_bad = [t for t in tests if (not ok(by[t]["C22"]) and ok(by[t]["REL"])) or
               (ok(by[t]["C22"]) and by[t]["C22"]["valid"] is not True)]
    vers_ok = all(x.get("version") == ("2026-10-06.4" if x["arm"].startswith("REL") else "2026-10-07.c22")
                  for x in rows if x["arm"] != "QK" and ok(x))
    meta_ok = m["smoke"] == smoke and (smoke or not m["dirty_tracked"]) and m["benchpress"] == "b695f30"
    p0 = len(tests) == n_expected and not c22_bad and vers_ok and meta_ok
    errs = {a: sum(1 for t in tests if "error" in by[t][a]) for a in ARMS}
    out = [f"# BP-MOCK2 score{' (SMOKE: not counted)' if smoke else ''}\n",
           f"git_head {m['git_head']}, Benchpress {m['benchpress']}, tests {len(tests)} (expected {n_expected}), "
           f"jobs {len(rows)}",
           f"P0 {'PASS' if p0 else 'FAIL'}: C22 failed where REL finished, or invalid: {len(c22_bad)}; versions ok "
           f"{vers_ok}; meta ok {meta_ok}", f"errors or timeouts per arm: {errs}"]
    if not p0:
        out.append("Nothing below is scored.")
        print("\n".join(out))
        open(os.path.join(args.out, "score.md"), "w").write("\n".join(out) + "\n")
        return
    done = [t for t in tests if all(ok(by[t][a]) for a in ARMS)]
    flat = [t for t in done if by[t]["C22"]["wide"] == 0]
    wides = [t for t in done if by[t]["C22"]["wide"] > 0]
    rel_unstable = [t for t in flat if by[t]["REL"]["sig"] != by[t]["REL2"]["sig"]]
    m1_bad = [t for t in flat if t not in rel_unstable and by[t]["C22"]["sig"] != by[t]["REL"]["sig"]]

    def ratio(t, a, b):
        return (by[t][a]["q2"] + 1) / (by[t][b]["q2"] + 1)

    g_wide = gmean([ratio(t, "C22", "REL") for t in wides])
    g_qk = gmean([ratio(t, "C22", "QK") for t in done])
    chk = [t for t in done if by[t]["C22"].get("checkable") is True and isinstance(by[t]["C22"].get("implements"),
                                                                                    bool)]
    m5_bad = [t for t in chk if not by[t]["C22"]["implements"]]
    res = [("M1", "flat inputs where REL reproduces itself: C22 returns REL's circuit",
            f"{len(m1_bad)} of {len(flat) - len(rel_unstable)} differ ({len(rel_unstable)} where REL did not reproduce)",
            verdict(not m1_bad, len(m1_bad) >= 2)),
           ("M2", "wide inputs: geometric mean C22/REL two-qubit count <= 0.85", f"{g_wide:.3f} on {len(wides)}",
            verdict(g_wide <= 0.85, g_wide > 1.00)),
           ("M3", "all tests: geometric mean C22/QK two-qubit count <= 1.15", f"{g_qk:.3f} on {len(done)}",
            verdict(g_qk <= 1.15, g_qk > 1.30)),
           ("M5", "every checkable C22 output implements its input", f"{len(m5_bad)} of {len(chk)} do not",
            verdict(len(chk) >= 5 and not m5_bad, bool(m5_bad)))]
    out += ["", f"tests finished by all arms: {len(done)} ({len(flat)} flat, {len(wides)} wide)", "",
            "| | prediction | value | verdict |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {v} | **{c}** |" for a, b, v, c in res]
    out += ["", "Reported without prediction:",
            f"- REL/QK two-qubit count {gmean([ratio(t, 'REL', 'QK') for t in done]):.3f}; REL2 differs from REL on "
            f"{sum(1 for t in done if by[t]['REL']['sig'] != by[t]['REL2']['sig'])} of {len(done)} tests "
            f"(two-qubit count differs on {sum(1 for t in done if by[t]['REL']['q2'] != by[t]['REL2']['q2'])})",
            f"- wide inputs where C22 has more two-qubit gates than REL: "
            f"{[t for t in wides if by[t]['C22']['q2'] > by[t]['REL']['q2']]}",
            f"- REL outputs checked: {sum(1 for t in done if by[t]['REL'].get('checkable') is True)}, not implementing: "
            f"{sum(1 for t in done if by[t]['REL'].get('checkable') is True and by[t]['REL'].get('implements') is False)};"
            f" QK outputs not implementing (state check, not comparable after measurement-aware passes): "
            f"{sum(1 for t in done if by[t]['QK'].get('checkable') is True and by[t]['QK'].get('implements') is False)}",
            "", "| stratum | tests | C22/QK | REL/QK | C22/REL |", "|---|---|---|---|---|"]
    for s in dict.fromkeys(by[t]["QK"]["stratum"] for t in tests):
        ts = [t for t in done if by[t]["QK"]["stratum"] == s]
        if ts:
            out.append(f"| {s} | {len(ts)} | {gmean([ratio(t, 'C22', 'QK') for t in ts]):.3f} | "
                       f"{gmean([ratio(t, 'REL', 'QK') for t in ts]):.3f} | {gmean([ratio(t, 'C22', 'REL') for t in ts]):.3f} |")
    print("\n".join(out))
    open(os.path.join(args.out, "score.md"), "w").write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sample", "run", "one", "score"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.mode == "sample":
        for s, tid, kind, arg in sample(a.bp, a.smoke):
            print(f"{s:28s} {tid}")
    else:
        {"run": run, "one": one, "score": score}[a.mode](a)


if __name__ == "__main__":
    main()
