"""bp_final.py -- BP-FINAL (2026-10-08): the final, pre-registered Benchpress test of release 2026-10-07.1 against
Qiskit level 2 as Benchpress calls it, on every published Benchpress transpilation test that no earlier test of this
project has used (880 tests). Pre-registration: the Addendum that locks this file.

Population (fixed by rule, no sampling): BP-MOCK's strata (bp_mock.strata, imported unchanged); in each, the published
test ids (data/2026-10-06/bp_probe/published_ref.json) minus the 12 BP-PROBE ran, the 92 BP-MOCK ran and the 48
BP-MOCK2 ran. The 100-qubit stratum is empty (all 9 used). Jobs are ordered within a stratum by SHA-256 of
"BP-FINAL|" + id (order only). The smoke run uses tests that were already used (the first BP-MOCK2 test of each
stratum), so that none of the 880 is compiled before the lock.

Building, backends and metrics as BP-MOCK and BP-MOCK2 (bp_mock.py, bp_mock2.py, unchanged). Arms, each test and arm
in its own process, 1,500 s limit for every arm:
  QK    generate_preset_pass_manager(2, backend).run(circuit) -- Benchpress's Qiskit call, not seeded
  REL   release 2026-10-07.1 (psf_compile.py), default call (coupling map and basis only)
  REL2  REL again, in its own process (reproducibility)
  RECR  release 2026-10-07.1, the README's recommended call with the backend's target (FakeTorino tests only)

    python benchmarks/bp_final.py list  --bp <benchpress clone> [--smoke]
    python benchmarks/bp_final.py run   --bp <benchpress clone> --out DIR [--par N] [--smoke]
    python benchmarks/bp_final.py score --out DIR
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
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
sys.path[:0] = [os.path.join(WORK, "depth1"), HERE, REPO]
import bp_mock as B    # noqa: E402  (BP-MOCK, Addendum 389, unchanged)
import bp_mock2 as B2  # noqa: E402  (BP-MOCK2, Addendum 391, unchanged)

RELEASE_VERSION = "2026-10-07.1"
RELEASE_SHA = "73fb2cb0b1acc5870339c23599829b326fbf57aa198945231e8551e55c1884dc"
REF_SHA = "d05231a75695c4fa9cbe2f09891ae8a835db32699e381c4d8de4b2afaaf56877"
BENCHPRESS = "b695f30"
N_TESTS, N_SMOKE = 880, 18
TIMEOUT = 1500
CHECK_MAX_QUBITS = 10
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
FAMILIES = {"QASMBench": lambda s: s.startswith("QASMBench"),
            "HamLib, abstract": lambda s: s.startswith("HamLib") and not s.endswith("FakeTorino"),
            "HamLib, FakeTorino": lambda s: s == "HamLib, FakeTorino",
            "Feynman, FakeTorino": lambda s: s == "Feynman, FakeTorino"}
BOOT, BOOT_SEED = 10000, 20261008


def arms_for(stratum):
    return ("QK", "REL", "REL2") + (("RECR",) if stratum.endswith("FakeTorino") else ())


def population(bp, smoke=False):
    """[(stratum, test id, kind, arg)]: every unused published test (or, for the smoke run, used ones)."""
    if smoke:
        return B2.sample(bp, smoke=True)
    ref = set(json.load(open(B.REF)))
    used = B.PROBED | {t[1] for t in B.sample(bp)} | {t[1] for t in B2.sample(bp)}
    out = []
    for name, tests in B.strata(bp).items():
        keep = sorted((t for t in tests if t[0] in ref and t[0] not in used),
                      key=lambda t: hashlib.sha256(("BP-FINAL|" + t[0]).encode()).hexdigest())
        out += [(name,) + tuple(t) for t in keep]
    return out


def on_failed(out, edges, qubits):
    n = 0
    for ins in out.data:
        if len(ins.qubits) == 2 and ins.operation.name != "barrier":
            q = tuple(out.find_bit(b).index for b in ins.qubits)
            if q in edges or q[::-1] in edges or q[0] in qubits or q[1] in qubits:
                n += 1
    return n


def one(args):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(args.job)
    qc, backend = B.build(args.bp, kind, arg)
    from qiskit import transpile
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from benchpress.qiskit_gym.utils.validation import qiskit_circuit_validation
    import core_fix_c2_eval as H
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(B.REL_PATH, "psf_compile")
    two_q = backend.two_q_gate_type
    rec = dict(stratum=stratum, test=tid, arm=args.arm, input_qubits=qc.num_qubits, wide=B.wide(qc))
    if args.arm == "QK":
        t0 = time.perf_counter()
        out = generate_preset_pass_manager(2, backend).run(qc)
        rec["t"] = time.perf_counter() - t0
    else:
        rec["version"] = pc.VERSION
        basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
        kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
                  seed_transpiler=0)
        if args.arm == "RECR":
            kw.update(target=backend.target, **RECOMMENDED)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            out = pc.compile_for_hardware(qc, **kw)
        rec["t"] = time.perf_counter() - t0
    ops = out.count_ops()
    rec["q2"] = int(ops.get(two_q, 0))
    rec["d2"] = out.depth(filter_function=lambda x: x.operation.name == two_q)
    rec["sig"] = B.sig_hash(out)
    if stratum.endswith("FakeTorino"):
        edges, qubits = pc._failed_elements(backend.target, 0.5)
        rec["on_failed"] = on_failed(out, edges, qubits)
    try:
        qiskit_circuit_validation(out, backend)
        rec["valid"] = True
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["valid"] = f"{type(exc).__name__}: {exc}"[:200]
    if qc.num_qubits <= CHECK_MAX_QUBITS and args.arm != "QK":
        # as BP-MOCK2: the input without final measurements, expanded through its definitions
        bare = qc.copy()
        bare.remove_final_measurements(inplace=True)
        ref = transpile(bare, basis_gates=["u", "cx"], optimization_level=0)
        chk = H.load_module(B.REL_PATH, "psf_compile_check")
        try:
            rec["implements"] = bool(chk._implements(ref, out))
            rec["checkable"] = chk.EXACT_STATS["not_checkable"] == 0
        except Exception as exc:  # noqa: BLE001
            rec["implements"] = f"n/a: {type(exc).__name__}"[:80]
            rec["checkable"] = False
    print(json.dumps(rec))


def run(args):
    tests = population(args.bp, args.smoke)
    want = N_SMOKE if args.smoke else N_TESTS
    if len(tests) != want:
        raise SystemExit(f"STOP: {len(tests)} tests, expected {want}")
    jobs = [(t, arm) for t in tests for arm in arms_for(t[0])]
    os.makedirs(args.out, exist_ok=True)
    head, dirty = B.git_state()
    bphead = subprocess.run(["git", "-C", args.bp, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    import numpy
    import qiskit
    import qiskit_ibm_runtime
    meta = dict(smoke=bool(args.smoke), git_head=head, dirty_tracked=dirty, benchpress=bphead,
                qiskit=qiskit.__version__, qiskit_ibm_runtime=qiskit_ibm_runtime.__version__,
                numpy=numpy.__version__, python=sys.version.split()[0], cpus=os.cpu_count(), par=args.par,
                tests=len(tests), jobs=len(jobs), timeout_s=TIMEOUT,
                sha=dict(script=B.norm_sha(os.path.abspath(__file__)), bp_mock=B.norm_sha(os.path.join(HERE, "bp_mock.py")),
                         bp_mock2=B.norm_sha(os.path.join(HERE, "bp_mock2.py")), release=B.norm_sha(B.REL_PATH),
                         bp_probe=B.norm_sha(os.path.join(HERE, "bp_probe.py")), published_ref=B.norm_sha(B.REF)),
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
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=REPO, env=env,
                               encoding="utf-8", errors="replace")
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else dict(stratum=t[0], test=t[1], arm=arm,
                                                           error=(p.stderr or "")[-400:])
        except subprocess.TimeoutExpired:
            rec = dict(stratum=t[0], test=t[1], arm=arm, error=f"timeout {TIMEOUT} s")
        rec["wall"] = round(time.perf_counter() - w0, 2)
        show = {k: rec[k] for k in ("q2", "valid", "implements", "on_failed", "error") if k in rec}
        print(f"{int(time.time() - t00):6d} s  {t[1][:58]:58s} {arm:4s} {show}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.par) as ex:
        rows = list(ex.map(go, jobs))
    meta["wall_s"] = round(time.time() - t00, 1)
    fn = "bp_final_smoke.json" if args.smoke else "bp_final.json"
    with open(os.path.join(args.out, fn), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)
    print(f"wrote {fn}: {len(tests)} tests, {len(rows)} jobs, {meta['wall_s']} s")


# ---------------------------------------------------------------- scoring

def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def ratio(by, t, a, b, key="q2"):
    return (by[t][a][key] + 1) / (by[t][b][key] + 1)  # +1: a test may have no two-qubit gate


def boot_ci(by, done):
    """95% interval of the geometric mean REL/QK: tests resampled with replacement within each stratum."""
    import numpy as np
    rng = np.random.default_rng(BOOT_SEED)
    groups = {}
    for t in done:
        groups.setdefault(by[t]["QK"]["stratum"], []).append(math.log(ratio(by, t, "REL", "QK")))
    arrs = [np.array(v) for _, v in sorted(groups.items())]
    n = sum(len(a) for a in arrs)
    means = np.zeros(BOOT)
    for a in arrs:
        idx = rng.integers(0, len(a), size=(BOOT, len(a)))
        means += a[idx].sum(axis=1)
    means = np.exp(means / n)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(args):
    fn = os.path.join(args.out, "bp_final.json")
    smoke = not os.path.exists(fn)
    with open(fn if not smoke else os.path.join(args.out, "bp_final_smoke.json"), encoding="utf-8") as f:
        r = json.load(f)
    m, rows = r["meta"], r["rows"]
    by = {}
    for x in rows:
        by.setdefault(x["test"], {})[x["arm"]] = x
    tests = list(by)

    def ok(x):
        return "error" not in x

    n_exp = N_SMOKE if smoke else N_TESTS
    complete = all(set(by[t]) == set(arms_for(by[t]["QK"]["stratum"])) for t in tests)
    vers_ok = all(x.get("version") == RELEASE_VERSION for x in rows if x["arm"] != "QK" and ok(x))
    meta_ok = (m["smoke"] == smoke and (smoke or not m["dirty_tracked"]) and m["benchpress"] == BENCHPRESS
               and m["sha"]["release"] == RELEASE_SHA and m["sha"]["published_ref"] == REF_SHA)
    p0 = len(tests) == n_exp and complete and vers_ok and meta_ok
    errs = {a: sum(1 for t in tests if a in by[t] and not ok(by[t][a])) for a in ("QK", "REL", "REL2", "RECR")}
    out = [f"# BP-FINAL score{' (SMOKE: tests already used, not counted)' if smoke else ''}", "",
           f"git_head {m['git_head']}, Benchpress {m['benchpress']}, Python {m['python']}, Qiskit {m['qiskit']}, "
           f"tests {len(tests)} (expected {n_exp}), jobs {len(rows)}, {m.get('wall_s')} s",
           f"P0 {'PASS' if p0 else 'FAIL'}: every arm present {complete}; versions {vers_ok}; release, reference and "
           f"Benchpress as locked, no uncommitted change {meta_ok}", f"errors or timeouts per arm: {errs}"]
    if not p0:
        out.append("Nothing below is scored.")
        _write(args, out)
        return
    done = [t for t in tests if ok(by[t]["QK"]) and ok(by[t]["REL"])]
    g = gmean([ratio(by, t, "REL", "QK") for t in done])
    lo, hi = boot_ci(by, done)
    fam = {k: [t for t in done if f(by[t]["QK"]["stratum"])] for k, f in FAMILIES.items()}
    gf = {k: gmean([ratio(by, t, "REL", "QK") for t in ts]) for k, ts in fam.items() if ts}
    worse = [t for t in done if ratio(by, t, "REL", "QK") > 1.10]
    share = len(worse) / len(done)
    psf = [(t, a) for t in tests for a in ("REL", "REL2", "RECR") if a in by[t] and ok(by[t][a])]
    invalid = [(t, a) for t, a in psf if by[t][a]["valid"] is not True]
    checked = [(t, a) for t, a in psf if by[t][a].get("checkable") is True and isinstance(by[t][a].get("implements"),
                                                                                           bool)]
    wrong = [(t, a) for t, a in checked if not by[t][a]["implements"]]
    recr = [t for t in tests if "RECR" in by[t] and ok(by[t]["RECR"])]
    recr_failed = [t for t in recr if by[t]["RECR"].get("on_failed", 0) > 0]
    qk_ok = [t for t in tests if ok(by[t]["QK"])]
    rel_fail = [t for t in qk_ok if not ok(by[t]["REL"])]
    fail_share = len(rel_fail) / len(qk_ok) if qk_ok else float("nan")
    res = [("F1", "all tests: geometric mean REL/QK two-qubit count <= 1.06 (REFUTED > 1.10)",
            f"{g:.3f} (95% {lo:.3f}-{hi:.3f}) on {len(done)}", verdict(g <= 1.06, g > 1.10)),
           ("F2", "every family <= 1.15 (REFUTED if any > 1.25)",
            "; ".join(f"{k} {v:.3f} ({len(fam[k])})" for k, v in gf.items()),
            verdict(all(v <= 1.15 for v in gf.values()), any(v > 1.25 for v in gf.values()))),
           ("F3", "share of tests with more than 1.10 times QK's count <= 0.15 (REFUTED > 0.25)",
            f"{share:.3f} ({len(worse)} of {len(done)})", verdict(share <= 0.15, share > 0.25)),
           ("F4", "every REL, REL2 and RECR output passes Benchpress's validator, and every checkable one implements "
                  "its input (at least 20 checked)",
            f"{len(invalid)} invalid of {len(psf)}; {len(wrong)} not implementing of {len(checked)}",
            verdict(not invalid and not wrong and len(checked) >= 20, bool(invalid or wrong))),
           ("F5", "the recommended call places no two-qubit gate on a failed coupler or qubit of FakeTorino",
            f"{len(recr_failed)} of {len(recr)} tests", verdict(bool(recr) and not recr_failed, bool(recr_failed))),
           ("F6", "REL fails (error or timeout) on <= 1% of the tests QK finishes (REFUTED > 3%)",
            f"{fail_share:.3f} ({len(rel_fail)} of {len(qk_ok)})", verdict(fail_share <= 0.01, fail_share > 0.03))]
    out += ["", f"tests finished by QK and REL: {len(done)} of {len(tests)}", "",
            "| | prediction | value | verdict |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {v} | **{c}** |" for a, b, v, c in res]
    both = [t for t in done if ok(by[t]["REL2"])]
    tq = [max(by[t]["REL"]["t"], 1e-4) / max(by[t]["QK"]["t"], 1e-4) for t in done]
    recq = [t for t in recr if ok(by[t]["QK"])]
    out += ["", "Reported without prediction:",
            f"- two-qubit depth REL/QK {gmean([ratio(by, t, 'REL', 'QK', 'd2') for t in done]):.3f}; families "
            f"weighted equally {gmean(list(gf.values())):.3f}",
            f"- fewer two-qubit gates than QK on {sum(1 for t in done if by[t]['REL']['q2'] < by[t]['QK']['q2'])}, "
            f"as many on {sum(1 for t in done if by[t]['REL']['q2'] == by[t]['QK']['q2'])}, more on "
            f"{sum(1 for t in done if by[t]['REL']['q2'] > by[t]['QK']['q2'])}",
            f"- compile time REL/QK: median {statistics.median(tq):.2f}, geometric mean {gmean(tq):.2f} (jobs run "
            f"{m['par']} at a time)",
            f"- REL2 differs from REL on {sum(1 for t in both if by[t]['REL']['sig'] != by[t]['REL2']['sig'])} of "
            f"{len(both)} tests (two-qubit count on {sum(1 for t in both if by[t]['REL']['q2'] != by[t]['REL2']['q2'])})",
            f"- FakeTorino: RECR/QK two-qubit count {gmean([ratio(by, t, 'RECR', 'QK') for t in recq]):.3f} on "
            f"{len(recq)}; tests with gates on failed elements: QK "
            f"{sum(1 for t in recq if by[t]['QK'].get('on_failed', 0) > 0)}, REL "
            f"{sum(1 for t in recq if ok(by[t]['REL']) and by[t]['REL'].get('on_failed', 0) > 0)}, RECR "
            f"{len(recr_failed)}",
            f"- REL outputs checked for equivalence: {sum(1 for t, a in checked if a == 'REL')}",
            "", "| stratum | tests | REL/QK | REL/QK depth | time REL/QK (median) |", "|---|---|---|---|---|"]
    for s in dict.fromkeys(by[t]["QK"]["stratum"] for t in tests):
        ts = [t for t in done if by[t]["QK"]["stratum"] == s]
        if ts:
            out.append(f"| {s} | {len(ts)} | {gmean([ratio(by, t, 'REL', 'QK') for t in ts]):.3f} | "
                       f"{gmean([ratio(by, t, 'REL', 'QK', 'd2') for t in ts]):.3f} | "
                       f"{statistics.median(max(by[t]['REL']['t'], 1e-4) / max(by[t]['QK']['t'], 1e-4) for t in ts):.2f} |")
    out += ["", "SUMMARY " + json.dumps(dict({a: c for a, _, _, c in res}, P0="PASS", F1_value=round(g, 6)))]
    _write(args, out)


def _write(args, out):
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(args.out, "score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("list", "run", "one", "score"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.mode == "list":
        ts = population(a.bp, a.smoke)
        for s, tid, kind, arg in ts:
            print(f"{s:28s} {tid}")
        print(f"{len(ts)} tests")
    else:
        {"run": run, "one": one, "score": score}[a.mode](a)


if __name__ == "__main__":
    main()
