"""rec_probe.py -- REC-PROBE (2026-10-07, exploratory, not pre-registered): what release 2026-10-07.1's items 48 and
50 do inside the recommended call. Both act at the start of the pipeline without a target, so also in the
recommended call's first compile, in its recompile on the pruned map and in its floor candidate; BP-MOCK's M4
checked item 48 there, item 50 was not part of a pre-registered test (Addendum 397, section 4).

Tests: the 20 FakeTorino tests of BP-MOCK (HamLib 8, Feynman 6, 100-qubit 6), built as there (bp_mock.build,
Benchpress b695f30). Arms, each test and arm in its own process (600 s limit):
  R4  2026-10-06.4 (the kept copy), recommended call -- BP-MOCK's RELR, again on this machine;
  R1  2026-10-07.1, recommended call;
  D1  2026-10-07.1, default call.
Recorded: two-qubit count and depth, validator, the circuit's hash, time, the counters' changes (CANCEL_STATS,
UNROLL_STATS, PRUNE_STATS, COMPARE_STATS, SKIP_STATS), the number of two-qubit gates on couplers FakeTorino reports
as failed (error >= 0.5), and on inputs of at most 16 qubits item 39's `_implements` against the input expanded
through its definitions (as BP-MOCK2).

    python benchmarks/rec_probe.py run --bp <benchpress clone> --out data/2026-10-07/rec_probe [--par 4] [--smoke]
    python benchmarks/rec_probe.py score --out data/2026-10-07/rec_probe
"""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, REPO]
import bp_mock  # noqa: E402  (the sample and the builders; nothing in it is changed)

PATHS = {"R4": os.path.join(REPO, "patches", "psf_compile_release_2026-10-06.4", "psf_compile.py"),
         "R1": os.path.join(REPO, "psf_compile.py"), "D1": os.path.join(REPO, "psf_compile.py")}
VERSIONS = {"R4": "2026-10-06.4", "R1": "2026-10-07.1", "D1": "2026-10-07.1"}
STATS = ("CANCEL_STATS", "UNROLL_STATS", "PRUNE_STATS", "COMPARE_STATS", "SKIP_STATS")
TIMEOUT = 600


def tests(bp, smoke):
    t = [x for x in bp_mock.sample(bp) if x[0].endswith("FakeTorino")]
    if smoke:
        keep = {"test_BVlike_simplification_transpile", "test_feynman_transpile[mod5_4.qasm]"}
        t = [x for x in t if x[1] in keep]
    return t


def one(args):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(args.job)
    qc, backend = bp_mock.build(args.bp, kind, arg)
    from benchpress.qiskit_gym.utils.validation import qiskit_circuit_validation
    import core_fix_c2_eval as H
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(PATHS[args.arm], "psf_compile")
    two_q = backend.two_q_gate_type
    rec = dict(stratum=stratum, test=tid, arm=args.arm, version=pc.VERSION, input_qubits=qc.num_qubits,
               wide=bp_mock.wide(qc))
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0)
    if args.arm.startswith("R"):
        kw.update(target=backend.target, **bp_mock.RECOMMENDED)
    before = {s: dict(getattr(pc, s)) for s in STATS if hasattr(pc, s)}
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    rec["t"] = time.perf_counter() - t0
    rec["stats"] = {s: {k: getattr(pc, s)[k] - v for k, v in d.items() if getattr(pc, s)[k] != v}
                    for s, d in before.items()}
    rec["q2"] = int(out.count_ops().get(two_q, 0))
    rec["d2"] = out.depth(filter_function=lambda x: x.operation.name == two_q)
    rec["sig"] = bp_mock.sig_hash(out)
    edges, _ = pc._failed_elements(backend.target, 0.5)
    rec["on_failed"] = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier"
                           and tuple(out.find_bit(q).index for q in i.qubits) in edges)
    try:
        qiskit_circuit_validation(out, backend)
        rec["valid"] = True
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["valid"] = f"{type(exc).__name__}: {exc}"[:200]
    if qc.num_qubits <= 16:
        try:
            from qiskit import transpile
            bare = qc.copy()
            bare.remove_final_measurements(inplace=True)
            ref = transpile(bare, basis_gates=["u", "cx"], optimization_level=0)
            rec["implements"] = bool(pc._implements(ref, bp_mock.strip_measure(out)))
        except Exception as exc:  # noqa: BLE001 - recorded
            rec["implements"] = f"n/a: {type(exc).__name__}"[:80]
    print(json.dumps(rec))


def run(args):
    ts = tests(args.bp, args.smoke)
    jobs = [(t, arm) for t in ts for arm in ("R4", "R1", "D1")]
    os.makedirs(args.out, exist_ok=True)
    head, dirty = bp_mock.git_state()
    bphead = subprocess.run(["git", "-C", args.bp, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    import qiskit
    meta = dict(smoke=bool(args.smoke), git_head=head, dirty_tracked=dirty, benchpress=bphead,
                qiskit=qiskit.__version__, python=sys.version.split()[0], par=args.par, tests=len(ts), jobs=len(jobs),
                sha=dict(script=bp_mock.norm_sha(os.path.abspath(__file__)),
                         **{a: bp_mock.norm_sha(p) for a, p in PATHS.items() if a != "D1"}),
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
        show = {k: rec[k] for k in ("q2", "on_failed", "valid", "implements", "error") if k in rec}
        if rec.get("stats", {}).get("CANCEL_STATS"):
            show["cancel"] = rec["stats"]["CANCEL_STATS"]
        print(f"{int(time.time() - t00):5d} s  {t[1][:58]:58s} {arm} {show}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.par) as ex:
        rows = list(ex.map(go, jobs))
    meta["wall_s"] = round(time.time() - t00, 1)
    fn = "rec_probe_smoke.json" if args.smoke else "rec_probe.json"
    with open(os.path.join(args.out, fn), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)
    print(f"wrote {fn}: {len(ts)} tests, {len(rows)} jobs, {meta['wall_s']} s")


def score(args):
    fn = os.path.join(args.out, "rec_probe.json")
    if not os.path.exists(fn):
        fn = os.path.join(args.out, "rec_probe_smoke.json")
    r = json.load(open(fn, encoding="utf-8"))
    by = {}
    for x in r["rows"]:
        by.setdefault(x["test"], {})[x["arm"]] = x
    ref = {}
    for x in json.load(open(os.path.join(REPO, "data", "2026-10-07", "bp_mock", "bp_mock.json")))["rows"]:
        ref.setdefault(x["test"], {})[x["arm"]] = x
    ok = lambda x: x is not None and "error" not in x  # noqa: E731
    out = [f"# REC-PROBE ({os.path.basename(fn)}; exploratory)", "",
           f"git_head {r['meta']['git_head']}, Benchpress {r['meta']['benchpress']}, {r['meta']['tests']} tests, "
           f"{r['meta']['jobs']} jobs, {r['meta'].get('wall_s')} s", "",
           "| test | QK (BP-MOCK) | R4 | R1 | D1 | R4 as in BP-MOCK | R1 cancel tried/kept | R1 recompiled | "
           "on failed R4/R1/D1 | R1/R4 time | valid, implements (R1) |", "|" + "---|" * 11]
    for t, a in by.items():
        q = lambda k: a[k]["q2"] if ok(a.get(k)) else "err"  # noqa: E731
        c = a["R1"].get("stats", {}).get("CANCEL_STATS", {}) if ok(a.get("R1")) else {}
        p = a["R1"].get("stats", {}).get("PRUNE_STATS", {}) if ok(a.get("R1")) else {}
        same = ok(a.get("R4")) and a["R4"]["q2"] == ref.get(t, {}).get("RELR", {}).get("q2")
        fl = "/".join(str(a[k].get("on_failed")) if ok(a.get(k)) else "err" for k in ("R4", "R1", "D1"))
        tr = f"{a['R1']['t'] / max(a['R4']['t'], 1e-4):.2f}" if ok(a.get("R1")) and ok(a.get("R4")) else ""
        vi = f"{a['R1'].get('valid')}, {a['R1'].get('implements', '-')}" if ok(a.get("R1")) else ""
        out.append(f"| {t} | {ref.get(t, {}).get('QK', {}).get('q2', '?')} | {q('R4')} | {q('R1')} | {q('D1')} | "
                   f"{same} | {c.get('tried', 0)}/{c.get('cancelled_kept', 0)} | {p.get('recompiled', 0)} | {fl} | "
                   f"{tr} | {vi} |")
    text = "\n".join(out) + "\n"
    with open(os.path.join(args.out, "score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "score"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "one": one, "score": score}[a.mode](a)


if __name__ == "__main__":
    main()
