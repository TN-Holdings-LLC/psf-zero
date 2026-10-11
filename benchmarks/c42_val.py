"""c42_val.py -- C42-VAL: prototype 2026-10-11.c42 (item 74 on c41: an output that uses a near-failed element, error
in [0.1, 0.5), is compared with the input compiled with those elements avoided too; the better by ESP is kept), on
Benchpress QASM files that C41-VAL did not use (QAOA and Heisenberg sizes outside its lists, QFT, quantum volume, every
Feynman and QASMBench file outside its tests, dtc, clifford_100) and random circuits with new seeds ("C42-VAL|...").
Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
Arms, each test and device in its own process: QK3 (Qiskit level 3, seed 0), R1 (release 2026-10-11.1), C41, C42;
the psf arms with the plain call with backend=. A circuit without measurements gets measure_all().

    cd <psf-zero repository>
    python <this file> run --bp <benchpress> --out DIR [--par 2] [--smoke]
    python <this file> report --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
C42 = os.environ.get("C42_FILE", os.path.join(REPO, "patches", "psf_compile_c42_2026-10-11", "psf_compile.py"))
C41 = os.path.join(REPO, "patches", "psf_compile_c41_2026-10-11", "psf_compile.py")
R1 = os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.1", "psf_compile.py")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
DEVICES = ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez", "FakeAachen", "FakeBoston")
ARMS = ("QK3", "R1", "C41", "C42")
FAILED = 0.5
TIMEOUT, BUDGET = 1800, 8 * 3600
QAOA_N = (7, 11, 18, 35, 50, 80, 120)
HEIS_N = (121, 144)
QFT_N = (8, 16, 24, 32, 48, 64, 96)
QV_N = (8, 12, 16, 24, 32)
MAX_OPS = 50_000  # run 2: QASM inputs with more instructions are left out (run 1 stopped on 170k-660k-gate files)
C41_USED = ("qft/qft_N010.qasm", "qft/qft_N020.qasm", "qv/qv_N010_12345.qasm", "clifford/clifford_20_12345.qasm",
            "feynman/adder.qasm", "feynman/gf2^4_mult.qasm")


# ------------------------------------------------------------------ circuits

def _label(n, paulis):
    lb = ["I"] * n
    for q, p in paulis.items():
        lb[n - 1 - q] = p
    return "".join(lb)


def _regular3(n, rng):
    """A random 3-regular simple graph on n (even) nodes, by repeated random matchings."""
    while True:
        stubs = [v for v in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            e = (min(a, b), max(a, b))
            if a == b or e in edges:
                ok = False
                break
            edges.add(e)
        if ok:
            return sorted(edges)


def qasm_root(bp):
    return os.path.join(bp, "benchpress", "qasm")


def _load_qasm(bp, rel):
    from qiskit import QuantumCircuit
    return QuantumCircuit.from_qasm_file(os.path.join(qasm_root(bp), rel))


def tests(bp):
    """{name: (family, QASM path or None)}, by fixed rules: Benchpress QAOA with N in QAOA_N and square-lattice Heisenberg
    with N in HEIS_N (sizes C41-VAL did not use), QFT with N in QFT_N, quantum volume with N in QV_N, dtc (cx, cz),
    clifford_100, and every Feynman and QASMBench file (not *_transpiled) with <= 156 qubits that C41-VAL did not use
    (C41-VAL took those that item 73 reads, and the controls in C41_USED); random circuits n = 10, 20, 40. QASM inputs
    with more than MAX_OPS instructions are left out."""
    out = {}
    for n in QAOA_N:
        out[f"qaoa_N{n}"] = ("qaoa_qasm", f"qaoa/qaoa_barabasi_albert_N{n}_3reps.qasm")
    for n in HEIS_N:
        out[f"heis_N{n}"] = ("heisenberg_qasm", f"square-heisenberg/square_heisenberg_N{n}.qasm")
    for n in QFT_N:
        out[f"qft_N{n}"] = ("qft", f"qft/qft_N{n:03d}.qasm")
    for n in QV_N:
        out[f"qv_N{n}"] = ("qv", f"qv/qv_N{n:03d}_12345.qasm")
    for rel in ("dtc/dtc_100_cx_12345.qasm", "dtc/dtc_100_cz_12345.qasm", "clifford/clifford_100_12345.qasm"):
        out["other_" + os.path.basename(rel)[:-5]] = ("other", rel)
    c41 = _c41()
    for fam, pat in (("qasmbench", "qasmbench-*/*/*.qasm"), ("feynman", "feynman/*.qasm")):
        for f in sorted(glob.glob(os.path.join(qasm_root(bp), pat))):
            rel = os.path.relpath(f, qasm_root(bp)).replace(os.sep, "/")
            if "transpiled" in rel or rel in C41_USED:
                continue
            try:
                qc = _load_qasm(bp, rel)
            except Exception:  # noqa: BLE001 - not loadable: not a test
                continue
            if qc.num_qubits <= 156 and c41._st_parse(qc) is None and c41._st_parse_gates(qc) is None:
                out[fam + "_" + os.path.basename(rel)[:-5]] = (fam, rel)
    for n in (10, 20, 40):
        out[f"gen_random_n{n}"] = ("gen_random", None)
    return {k: v for k, v in out.items() if v[1] is None or len(_load_qasm(bp, v[1]).data) <= MAX_OPS}


def _c41():
    if "psf_c41_sel" not in sys.modules:
        load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
        load(C41, "psf_c41_sel")  # C41-VAL's selection rule (item 73's reader), to keep its tests out
    return sys.modules["psf_c41_sel"]


def make_generated(name):
    from qiskit.circuit.random import random_circuit
    rng = random.Random("C42-VAL|" + name)
    n = int(name.split("_n")[1])
    return random_circuit(n, 10, max_operands=2, measure=True, seed=rng.randrange(10 ** 6))


def build(bp, test):
    fam, rel = tests_cached(bp)[test]
    qc = make_generated(test) if rel is None else _load_qasm(bp, rel)
    if not any(i.operation.name == "measure" for i in qc.data):
        qc.measure_all()
    return qc


_TESTS = {}


def tests_cached(bp):
    if bp not in _TESTS:
        path = os.environ.get("C42_TESTS")
        _TESTS[bp] = json.load(open(path)) if path and os.path.exists(path) else tests(bp)
    return _TESTS[bp]


def jobs(bp):
    from qiskit_ibm_runtime import fake_provider
    nq = {d: getattr(fake_provider, d)().target.num_qubits for d in DEVICES}
    out = []
    for t in tests_cached(bp):
        n = build(bp, t).num_qubits
        out += [(t, None, d) for d in DEVICES if n <= nq[d]]
    return out


# ------------------------------------------------------------------ one job

def breakdown(out, target):
    r = dict(n2=0, n1=0, nm=0, s2=0.0, s1=0.0, sm=0.0, failed=0, missing=0)
    for ins in out.data:
        name = ins.operation.name
        if name in ("barrier", "delay"):
            continue
        qa = tuple(out.find_bit(q).index for q in ins.qubits)
        try:
            p = target[name][qa]
        except KeyError:
            r["missing"] += 1
            continue
        e = p.error if p is not None and p.error is not None else 0.0
        r["failed"] += e >= FAILED
        cost = -math.log10(1.0 - min(e, 1.0 - 1e-15))
        if name == "measure":
            r["nm"] += 1
            r["sm"] += cost
        elif len(qa) == 2:
            r["n2"] += 1
            r["s2"] += cost
        elif name != "rz":
            r["n1"] += 1
            r["s1"] += cost
    r.update(s2=round(r["s2"], 7), s1=round(r["s1"], 7), sm=round(r["sm"], 7))
    return r


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def one(a):
    warnings.simplefilter("ignore")
    from qiskit_ibm_runtime import fake_provider
    from qiskit.transpiler import generate_preset_pass_manager
    from c29_identity import value_sig
    test, arg = json.loads(a.job)
    qc = build(a.bp, test)
    backend = getattr(fake_provider, a.device)()
    target = backend.target
    load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    mods = {"R1": load(R1, "psf_r1"), "C41": load(C41, "psf_c41"), "C42": load(C42, "psf_c42")}
    sigs = {}
    for arm in ARMS:
        rec = dict(test=test, device=a.device, arm=arm, input_qubits=qc.num_qubits)
        t0 = time.perf_counter()
        try:
            if arm == "QK3":
                out = generate_preset_pass_manager(3, backend, seed_transpiler=0).run(qc)
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
        rec.update(breakdown(out, target))
        if arm != "QK3":
            sigs[arm] = hashlib.sha256(repr(value_sig(out)).encode()).hexdigest()[:16]
            rec["sig"] = sigs[arm]
        if arm in ("C41", "C42"):
            rec["steiner_stats"] = dict(mods[arm].STEINER_STATS)
        if arm == "C42":
            rec["near_stats"] = dict(mods["C42"].NEAR_STATS)
            rec["same_as_c41"] = sigs.get("C42") == sigs.get("C41")
        print(json.dumps(rec), flush=True)


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    os.makedirs(a.out)
    import shutil
    shutil.copyfile(C42, os.path.join(a.out, "psf_compile_c42_as_run.py"))
    os.environ["C42_FILE"] = os.path.join(a.out, "psf_compile_c42_as_run.py")
    json.dump(tests(a.bp), open(os.path.join(a.out, "c42_tests.json"), "w"), indent=0, sort_keys=True)
    os.environ["C42_TESTS"] = os.path.join(a.out, "c42_tests.json")
    js = jobs(a.bp)
    if a.smoke:
        js = [j for j in js if j[0] == "gen_random_n10"][:4]
    js.sort(key=lambda j: hashlib.sha256(("C42-VAL|" + j[0] + j[2]).encode()).hexdigest())
    path = os.path.join(a.out, "c42_val.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), jobs=len(js),
                                           par=a.par, c42_sha256=hashlib.sha256(open(C42, "rb").read()).hexdigest(),
                                           c41_sha256=hashlib.sha256(open(C41, "rb").read()).hexdigest(),
                                           r1_sha256=hashlib.sha256(open(R1, "rb").read()).hexdigest()))) + "\n")
    t_start = time.perf_counter()

    def go(job):
        test, arg, d = job
        base = dict(test=test, device=d)
        if time.perf_counter() - t_start > BUDGET:
            return [dict(base, arm=x, error="not started (budget)") for x in ARMS]
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--device", d,
                                "--job", json.dumps([test, arg])], capture_output=True, text=True, timeout=TIMEOUT,
                               cwd=REPO, env=dict(os.environ))
            recs = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]
            got = {r["arm"] for r in recs}
            return recs + [dict(base, arm=x, error=(p.stderr or "")[-300:]) for x in ARMS if x not in got]
        except subprocess.TimeoutExpired:
            return [dict(base, arm=x, error=f"timeout {TIMEOUT} s") for x in ARMS]

    n = 0
    with ThreadPoolExecutor(a.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in js]):
            for r in f.result():
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
            print(f"[{n}/{len(js)} {time.perf_counter() - t_start:6.0f} s]", flush=True)
    report(a)


# ------------------------------------------------------------------ report

def report(a):
    rows = [json.loads(x) for x in open(os.path.join(a.out, "c42_val.jsonl"), encoding="utf-8")][1:]
    by = {}
    for r in rows:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    L = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731

    def esp(r):
        return 0.0 if r["failed"] else 10 ** -L(r)

    def gm(p, q, xs):
        ok = [x for x in xs if esp(x[p]) > 0 and esp(x[q]) > 0]
        return (10 ** (sum(L(x[q]) - L(x[p]) for x in ok) / len(ok)), len(ok)) if ok else (float("nan"), 0)
    tried = lambda x: (x["C42"].get("near_stats") or {}).get("tried", 0) > 0  # noqa: E731
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")  # noqa: E731
    out = ["# C42-VAL", "", f"jobs: {len(by)}; records with an error: {sum(1 for r in rows if 'error' in r)}", ""]
    for r in [r for r in rows if "error" in r][:20]:
        out.append(f"- {r['device']} {r['arm']} {r['test'][-50:]}: {r['error'][:150]}")
    out += ["", "| device | W1 C42 on failed | W2 C42 >= C41 every test | W3 untried: C42 = C41 by value | W5 tried / tests "
            "| W4 tried, runnable: ESP C42/C41 | alternative kept | W6 tried: median time C42/C41 | W7 all runnable: ESP "
            "C42/R1, C42/QK3 |", "|" + "---|" * 9]
    res = {}
    for d in DEVICES + ("all",):
        sel = [x for (dv, _), x in by.items() if (d == "all" or dv == d) and all(k in x and "error" not in x[k]
                                                                                for k in ARMS)]
        w1 = sum(x["C42"]["failed"] > 0 for x in sel)
        w2 = (sum(1 for x in sel if L(x["C42"]) <= L(x["C41"]) + 1e-9), len(sel))
        un = [x for x in sel if not tried(x)]
        w3 = (sum(1 for x in un if x["C42"].get("same_as_c41")), len(un))
        tr = [x for x in sel if tried(x)]
        runn = lambda xs: [x for x in xs if max(esp(x[k]) for k in ARMS) >= 0.01]  # noqa: E731
        w4 = gm("C42", "C41", runn(tr))
        kept = sum(1 for x in tr if (x["C42"].get("near_stats") or {}).get("kept_alternative", 0) > 0)
        w6 = med([x["C42"]["t"] / max(x["C41"]["t"], 1e-3) for x in tr])
        allr = runn(sel)
        w7 = (gm("C42", "R1", allr), gm("C42", "QK3", allr))
        res[d] = dict(w1=w1, w2=w2, w3=w3, tried=(len(tr), len(sel)), w4=w4, kept=kept, w6=w6, w7=w7)
        out.append(f"| {d} | {w1} | {w2[0]} / {w2[1]} | {w3[0]} / {w3[1]} | {len(tr)} / {len(sel)} | {w4[0]:.3f} "
                   f"({w4[1]}) | {kept} | {w6:.2f} | {w7[0][0]:.3f}, {w7[1][0]:.3f} ({w7[0][1]}) |")
    out += ["", "## Tests where item 74 tried (log10 ESP C42 - C41; ESP C41 -> C42; two-qubit C41 -> C42; kept?)", ""]
    for (d, t), x in sorted(by.items(), key=lambda z: (z[0][1], z[0][0])):
        if all(k in x and "error" not in x[k] for k in ARMS) and tried(x):
            k = "alternative" if x["C42"]["near_stats"].get("kept_alternative") else "first"
            out.append(f"- {t} {d[4:]}: {L(x['C41']) - L(x['C42']):+.4f}; {esp(x['C41']):.3g} -> {esp(x['C42']):.3g}; "
                       f"{x['C41']['n2']} -> {x['C42']['n2']}; {k}")
    txt = "\n".join(out) + "\n"
    open(os.path.join(a.out, "c42_val.md"), "w", encoding="utf-8", newline="\n").write(txt)
    json.dump(res, open(os.path.join(a.out, "c42_val_summary.json"), "w"), default=str)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "report"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--device")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "one": one, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
