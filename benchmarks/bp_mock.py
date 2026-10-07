"""bp_mock.py -- BP-MOCK (2026-10-07): a pre-registered mock exam on a stratified sample of Benchpress's 1,032
transpilation tests (Benchpress commit b695f30). Three default calls are compared on the same inputs and backends:
Qiskit level 2 as Benchpress calls it (QK), release 2026-10-06.4 (REL) and candidate 2026-10-07.c22 (C22, items 46-49);
on the FakeTorino tests also the README's recommended call of both (RELR, C22R). Addendum 388 has the predictions.

Sample (fixed by rule, before any run): the 19 strata below; in each, the published test ids (published_ref.json)
minus the 12 ids BP-PROBE ran (Addendum 377), ordered by SHA-256 of "BP-MOCK|" + id, and the first K taken.

Every test is built as Benchpress's Qiskit gym builds it (bp_probe.py's builders, plus the 100-qubit "summit" tests):
the same input circuit and backend (FakeTorino, or FlexibleBackend for an abstract topology; basis id/sx/x/rz/cz), the
same metrics (two-qubit gate count and depth of the backend's two-qubit gate) and Benchpress's structural validator.
Each (test, arm) runs in its own process, 600 s limit; jobs run in parallel (--par), which changes only times.

    python benchmarks/bp_mock.py sample --bp <benchpress clone>
    python benchmarks/bp_mock.py run --bp <benchpress clone> --out DIR [--par N] [--smoke]
    python benchmarks/bp_mock.py score --out DIR
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
REF = os.path.join(REPO, "data", "2026-10-06", "bp_probe", "published_ref.json")
REL_PATH = os.path.join(REPO, "psf_compile.py")
C22_PATH = os.path.join(REPO, "patches", "psf_compile_c22_2026-10-07", "psf_compile.py")
VERSIONS = dict(REL="2026-10-06.4", C22="2026-10-07.c22")
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
TOPOS = ("all-to-all", "square", "heavy-hex", "linear")
SUMMIT = ("test_BV_100_transpile", "test_BVlike_simplification_transpile", "test_QAOA_100_transpile",
          "test_QFT_100_transpile", "test_QV_100_transpile", "test_circSU2_100_transpile", "test_circSU2_89_transpile",
          "test_clifford_100_transpile", "test_square_heisenberg_100_transpile")
PROBED = {  # the test ids BP-PROBE ran (data/2026-10-06/bp_probe/*/bp_probe.json)
    "test_QASMBench_large[32-linear]", "test_QASMBench_medium[bigadder_n18-square]",
    "test_QASMBench_medium[bv_n14-square]", "test_QASMBench_small[adder_n4-all-to-all]",
    "test_QASMBench_small[basis_trotter_n4-all-to-all]", "test_QFT_100_transpile", "test_QV_100_transpile",
    "test_circSU2_100_transpile", "test_feynman_transpile[barenco_tof_10.qasm]",
    "test_hamiltonians[ham_0-uuf100-0810.cnf-40-res-heavy-hex]",
    "test_hamlib_hamiltonians_transpile[ham_bh_graph-1D-grid-nonpbc-qubitnodes_Lx-12_U-40_enc-unary_d-4]",
    "test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4-4-4-4-4-4]"}
K = {"QASMBench small": 5, "QASMBench medium": 4, "QASMBench large": 4, "HamLib": 5, "HamLib, FakeTorino": 8,
     "Feynman, FakeTorino": 6, "100-qubit, FakeTorino": 6}
TIMEOUT = 600


# ---------------------------------------------------------------- the sample (no Qiskit needed)

def strata(bp):
    """{stratum: [(test id, kind, builder argument)]} for every Benchpress transpilation test."""
    root = os.path.join(bp, "benchpress")

    def qasm(sub):
        out = {}
        for r, _, files in os.walk(os.path.join(root, "qasm", sub)):
            for f in files:
                if f.endswith(".qasm") and "transpiled" not in f:
                    out[f.split(".")[0]] = os.path.join(r, f)
        return out

    S = {}
    for size in ("small", "medium", "large"):
        names = qasm("qasmbench-" + size)
        for t in TOPOS:
            S[f"QASMBench {size}, {t}"] = [(f"test_QASMBench_{size}[{n}-{t}]", "qasmbench", [p, t])
                                           for n, p in names.items()]
    hams = json.load(open(os.path.join(root, "hamiltonian", "hamlib", "100_representative.json")))
    for t in TOPOS:
        S[f"HamLib, {t}"] = [(f"test_hamiltonians[ham_{h['ham_instance'][1:-1]}-{t}]", "ham_abstract",
                              [h["ham_instance"], t]) for h in hams]
    S["HamLib, FakeTorino"] = [(f"test_hamlib_hamiltonians_transpile[ham_{h['ham_instance'][1:-1]}]", "ham_device",
                                h["ham_instance"]) for h in hams]
    S["Feynman, FakeTorino"] = [(f"test_feynman_transpile[{f}]", "feynman", f)
                                for f in os.listdir(os.path.join(root, "qasm", "feynman")) if f.endswith(".qasm")]
    S["100-qubit, FakeTorino"] = [(i, "summit", i) for i in SUMMIT]
    return S


def sample(bp, smoke=False):
    """[(stratum, test id, kind, arg)] in stratum order; smoke: the first test of each stratum only."""
    ref = json.load(open(REF))
    out = []
    for name, tests in strata(bp).items():
        k = K.get(name, K.get(name.split(",")[0]))
        keep = sorted((t for t in tests if t[0] in ref and t[0] not in PROBED),
                      key=lambda t: hashlib.sha256(("BP-MOCK|" + t[0]).encode()).hexdigest())
        out += [(name,) + tuple(t) for t in keep[:1 if smoke else k]]
    return out


# ---------------------------------------------------------------- building a test as Benchpress does

def build(bp, kind, arg):
    sys.path[:0] = [bp, HERE, REPO]
    import bp_probe
    cfg = bp_probe.bp_setup(bp)
    if kind == "summit":
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import EfficientSU2, QuantumVolume
        from benchpress.qiskit_gym.circuits import bv_all_ones, trivial_bvlike_circuit
        q = cfg.get_qasm_dir
        qc = {"test_BV_100_transpile": lambda: bv_all_ones(100),
              "test_BVlike_simplification_transpile": lambda: trivial_bvlike_circuit(100),
              "test_QAOA_100_transpile": lambda: QuantumCircuit.from_qasm_file(
                  q("qaoa") + "qaoa_barabasi_albert_N100_3reps.qasm"),
              "test_QFT_100_transpile": lambda: QuantumCircuit.from_qasm_file(q("qft") + "qft_N100.qasm"),
              "test_QV_100_transpile": lambda: QuantumVolume(100, 100, seed=12345),
              "test_circSU2_100_transpile": lambda: EfficientSU2(100, reps=3, entanglement="circular"),
              "test_circSU2_89_transpile": lambda: EfficientSU2(89, reps=3, entanglement="circular"),
              "test_clifford_100_transpile": lambda: QuantumCircuit.from_qasm_file(
                  q("clifford") + "clifford_100_12345.qasm"),
              "test_square_heisenberg_100_transpile": lambda: QuantumCircuit.from_qasm_file(
                  q("square-heisenberg") + "square_heisenberg_N100.qasm")}[arg]()
        return qc, cfg.backend()
    return bp_probe.build(cfg, kind, tuple(arg) if isinstance(arg, list) else arg)


def wide(qc):
    return sum(1 for i in qc.data if len(i.qubits) > 2 and i.operation.name != "barrier")


def strip_measure(out):
    """`out` without measurements, keeping its layout (so that Operator.from_circuit undoes the permutations)."""
    c = out.copy_empty_like()
    for ins in out.data:
        if ins.operation.name != "measure":
            c.append(ins.operation, ins.qubits, ins.clbits)
    c._layout = out._layout
    return c


def sig_hash(c):
    s = [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
          [repr(p) for p in i.operation.params]] for i in c.data]
    lay = getattr(c, "layout", None)
    s += [repr(c.global_phase), list(lay.initial_index_layout(filter_ancillas=True)) if lay else None,
          list(lay.final_index_layout(filter_ancillas=True)) if lay else None]
    return hashlib.sha256(json.dumps(s).encode()).hexdigest()


def one(args):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(args.job)
    qc, backend = build(args.bp, kind, arg)
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from benchpress.qiskit_gym.utils.validation import qiskit_circuit_validation
    two_q = backend.two_q_gate_type
    rec = dict(stratum=stratum, test=tid, arm=args.arm, input_qubits=qc.num_qubits, wide=wide(qc))
    if args.arm == "QK":
        t0 = time.perf_counter()
        out = generate_preset_pass_manager(2, backend).run(qc)
        rec["t"] = time.perf_counter() - t0
    else:
        import core_fix_c2_eval as H
        H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
        pc = H.load_module(REL_PATH if args.arm.startswith("REL") else C22_PATH, "psf_compile")
        rec["version"] = pc.VERSION
        basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
        kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
                  seed_transpiler=0)
        if args.arm.endswith("R"):
            kw.update(target=backend.target, **RECOMMENDED)
        before = {s: dict(getattr(pc, s)) for s in ("COMPARE_STATS", "SKIP_STATS", "UNROLL_STATS") if hasattr(pc, s)}
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            out = pc.compile_for_hardware(qc, **kw)
        rec["t"] = time.perf_counter() - t0
        rec["stats"] = {s: {k: getattr(pc, s)[k] - v for k, v in d.items() if getattr(pc, s)[k] != v}
                        for s, d in before.items()}
    ops = out.count_ops()
    rec["q2"] = int(ops.get(two_q, 0))
    rec["d2"] = out.depth(filter_function=lambda x: x.operation.name == two_q)
    rec["other_2q"] = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name not in (two_q, "barrier"))
    rec["sig"] = sig_hash(out)
    try:
        qiskit_circuit_validation(out, backend)
        rec["valid"] = True
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["valid"] = f"{type(exc).__name__}: {exc}"[:200]
    if out.num_qubits <= 10:
        try:
            from qiskit.quantum_info import Operator
            a = qc.copy()
            a.remove_final_measurements(inplace=True)
            rec["equivalent"] = bool(Operator(a).equiv(Operator.from_circuit(strip_measure(out))))
        except Exception as exc:  # noqa: BLE001 - mid-circuit measurement, reset, ...
            rec["equivalent"] = f"n/a: {type(exc).__name__}"[:80]
    print(json.dumps(rec))


def git_state():
    def git(*a):
        try:
            return subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return "unknown"
    return git("rev-parse", "--short", "HEAD"), git("status", "--porcelain", "--untracked-files=no")


def norm_sha(path):
    t = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in t.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def run(args):
    tests = sample(args.bp, args.smoke)
    jobs = [(t, arm) for t in tests for arm in ("QK", "REL", "C22") + (("RELR", "C22R") if t[0].endswith("FakeTorino")
                                                                         else ())]
    os.makedirs(args.out, exist_ok=True)
    head, dirty = git_state()
    bphead = subprocess.run(["git", "-C", args.bp, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    import qiskit
    meta = dict(smoke=bool(args.smoke), git_head=head, dirty_tracked=dirty, benchpress=bphead,
                qiskit=qiskit.__version__, python=sys.version.split()[0], par=args.par, tests=len(tests),
                jobs=len(jobs), sha=dict(script=norm_sha(os.path.abspath(__file__)), release=norm_sha(REL_PATH),
                                         c22=norm_sha(C22_PATH), bp_probe=norm_sha(os.path.join(HERE, "bp_probe.py")),
                                         published_ref=norm_sha(REF)),
                start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps(meta), flush=True)
    rows, t00 = [], time.time()
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
        show = {k: rec[k] for k in ("q2", "d2", "valid", "equivalent", "error") if k in rec}
        print(f"{int(time.time() - t00):5d} s  {t[1][:60]:60s} {arm:4s} {show}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.par) as ex:
        rows = list(ex.map(go, jobs))
    meta["wall_s"] = round(time.time() - t00, 1)
    fn = "bp_mock_smoke.json" if args.smoke else "bp_mock.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, fn), "w"), indent=1)
    print(f"wrote {fn}: {len(tests)} tests, {len(rows)} jobs, {meta['wall_s']} s")


# ---------------------------------------------------------------- scoring

def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    fn = os.path.join(args.out, "bp_mock.json")
    smoke = not os.path.exists(fn)
    r = json.load(open(fn if not smoke else os.path.join(args.out, "bp_mock_smoke.json")))
    m, rows = r["meta"], r["rows"]
    by = {}
    for x in rows:
        by.setdefault(x["test"], {})[x["arm"]] = x
    tests = list(by)
    n_expected = 19 if smoke else 4 * (5 + 4 + 4 + 5) + 8 + 6 + 6
    c22_fail = [t for t in tests if "error" in by[t]["C22"] and "error" not in by[t]["REL"]]
    c22_invalid = [t for t in tests if "error" not in by[t]["C22"] and by[t]["C22"]["valid"] is not True]
    c22r_bad = [t for t in tests if "C22R" in by[t] and ("error" in by[t]["C22R"] and "error" not in by[t]["RELR"]
                                                         or "error" not in by[t]["C22R"] and by[t]["C22R"]["valid"] is not True)]
    flags = []
    if m["smoke"] != smoke or (m["dirty_tracked"] and not smoke) or m["benchpress"] != "b695f30":
        flags.append("meta")
    vers = {x.get("version") for x in rows if x["arm"] in ("REL", "RELR") and "version" in x}, \
        {x.get("version") for x in rows if x["arm"] in ("C22", "C22R") and "version" in x}
    if vers[0] - {VERSIONS["REL"]} or vers[1] - {VERSIONS["C22"]}:
        flags.append(f"versions {vers}")
    p0 = len(tests) == n_expected and not c22_fail and not c22_invalid and not c22r_bad and not flags
    out = [f"# BP-MOCK score{' (SMOKE: not counted)' if smoke else ''}\n",
           f"git_head {m['git_head']}, Benchpress {m['benchpress']}, tests {len(tests)} (expected {n_expected}), "
           f"jobs {len(rows)}, flags {flags}",
           f"P0 {'PASS' if p0 else 'FAIL'}: C22 failed where REL finished {len(c22_fail)}, C22 invalid "
           f"{len(c22_invalid)}, C22R failed or invalid {len(c22r_bad)}"]
    errs = {a: sum(1 for t in tests if a in by[t] and "error" in by[t][a]) for a in ("QK", "REL", "C22", "RELR", "C22R")}
    out.append(f"errors or timeouts per arm: {errs}")
    if not p0:
        out.append("Nothing below is scored.")
        print("\n".join(out))
        open(os.path.join(args.out, "score.md"), "w").write("\n".join(out) + "\n")
        return
    done = [t for t in tests if all("error" not in by[t][a] for a in ("QK", "REL", "C22"))]
    flat = [t for t in done if by[t]["C22"]["wide"] == 0]
    wides = [t for t in done if by[t]["C22"]["wide"] > 0]
    differ_flat = [t for t in flat if by[t]["C22"]["sig"] != by[t]["REL"]["sig"]]

    def ratio(t, a, b, key="q2"):
        x, y = by[t][a][key], by[t][b][key]
        return (x + 1) / (y + 1)  # +1: a test may have no two-qubit gate (BV-like)

    g_wide = gmean([ratio(t, "C22", "REL") for t in wides])
    g_qk = gmean([ratio(t, "C22", "QK") for t in done])
    rr = [t for t in tests if "C22R" in by[t] and "error" not in by[t]["C22R"] and "error" not in by[t]["RELR"]]
    g_r = gmean([ratio(t, "C22R", "RELR") for t in rr])
    eq = [t for t in done if isinstance(by[t]["C22"].get("equivalent"), bool)]
    noneq = [t for t in eq if not by[t]["C22"]["equivalent"]]
    res = [("M1", "inputs without wide instructions: C22's default call returns REL's circuit",
            verdict(not differ_flat, bool(differ_flat))),
           ("M2", "inputs with wide instructions: geometric mean C22/REL two-qubit count <= 0.85",
            verdict(g_wide <= 0.85, g_wide > 1.00)),
           ("M3", "all tests: geometric mean C22/QK two-qubit count <= 1.15", verdict(g_qk <= 1.15, g_qk > 1.30)),
           ("M4", "FakeTorino tests: geometric mean C22R/RELR two-qubit count <= 1.00", verdict(g_r <= 1.00, g_r > 1.05)),
           ("M5", "every C22 output checked for equivalence (<= 10 qubits) is equivalent",
            verdict(bool(eq) and not noneq, bool(noneq)))]
    out += ["", f"tests finished by QK, REL and C22: {len(done)} ({len(flat)} without wide instructions, "
                f"{len(wides)} with); C22R and RELR both finished: {len(rr)}; equivalence checked on {len(eq)}",
            "", "| | prediction | value | verdict |", "|---|---|---|---|"]
    vals = [f"{len(differ_flat)} of {len(flat)} differ", f"{g_wide:.3f}", f"{g_qk:.3f}", f"{g_r:.3f}",
            f"{len(noneq)} of {len(eq)} not equivalent"]
    out += [f"| {a} | {b} | {v} | **{c}** |" for (a, b, c), v in zip(res, vals)]
    out += ["", "Reported without prediction:",
            f"- REL/QK two-qubit count, all tests: {gmean([ratio(t, 'REL', 'QK') for t in done]):.3f}; "
            f"two-qubit depth C22/QK {gmean([ratio(t, 'C22', 'QK', 'd2') for t in done]):.3f}, "
            f"REL/QK {gmean([ratio(t, 'REL', 'QK', 'd2') for t in done]):.3f}",
            "", "| stratum | tests | C22/QK q2 | REL/QK q2 | C22/QK d2 | C22 time / QK time |", "|---|---|---|---|---|---|"]
    for s in dict.fromkeys(by[t]["QK"]["stratum"] for t in tests):
        ts = [t for t in done if by[t]["QK"]["stratum"] == s]
        if ts:
            out.append(f"| {s} | {len(ts)} | {gmean([ratio(t, 'C22', 'QK') for t in ts]):.3f} | "
                       f"{gmean([ratio(t, 'REL', 'QK') for t in ts]):.3f} | "
                       f"{gmean([ratio(t, 'C22', 'QK', 'd2') for t in ts]):.3f} | "
                       f"{gmean([max(by[t]['C22']['t'], 1e-4) / max(by[t]['QK']['t'], 1e-4) for t in ts]):.2f} |")
    qk_eq = [t for t in done if isinstance(by[t]["QK"].get("equivalent"), bool)]
    out += ["", f"- QK outputs not equivalent: {sum(1 for t in qk_eq if not by[t]['QK']['equivalent'])} of {len(qk_eq)}; "
                f"REL: {sum(1 for t in eq if by[t]['REL'].get('equivalent') is False)} of {len(eq)}",
            f"- C22R/RELR time: {gmean([max(by[t]['C22R']['t'], 1e-4) / max(by[t]['RELR']['t'], 1e-4) for t in rr]):.2f} "
            f"(geometric mean); C22R/QK two-qubit count {gmean([ratio(t, 'C22R', 'QK') for t in rr if t in done]):.3f}"]
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
