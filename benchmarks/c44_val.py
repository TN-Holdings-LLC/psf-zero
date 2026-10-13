"""c44_val.py -- C44-VAL: prototype 2026-10-11.c44 (on c43 = release 2026-10-11.4): item 75 revised (three search orders
in the deterministic re-placement) and item 77 (no time budget in the layout search); with an ablation of item 74
(near-failed elements) to decide whether it stays on by default. Tests not used in C41-VAL to C43-VAL: Benchpress QFT,
QAOA and quantum-volume sizes outside their lists, and generated circuits with seeds "C44-VAL|..." (random circuits,
Bernstein-Vazirani, GHZ).
Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
Arms, each test and device in its own process: QK3 (Qiskit level 3, seed 0), R3 (release 2026-10-11.3), R4 (release
2026-10-11.4), C44, C44off (c44 with near_failed_error=None); then C44 again in the same process (C44r).
The psf arms with the plain call with backend=. A circuit without measurements gets measure_all().

    cd <psf-zero repository>
    python <this file> run --bp <benchpress> --out DIR [--par 2] [--smoke]
    python <this file> report --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
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
C44 = os.environ.get("C44_FILE", os.path.join(REPO, "patches", "psf_compile_c44_2026-10-11", "psf_compile.py"))
R4 = os.path.join(REPO, "psf_compile.py")
R3 = os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.3", "psf_compile.py")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
DEVICES = ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez", "FakeAachen", "FakeBoston")
ARMS = ("QK3", "R3", "R4", "C44", "C44off")
REPEATS = ("C44",)
FAILED = 0.5
TIMEOUT, BUDGET = 1800, 8 * 3600
QFT_N = (11, 13, 15, 18, 22, 26, 30, 36)  # none in C42-VAL's or C43-VAL's lists
QAOA_N = (14, 19, 24, 32, 38, 55)  # none in C41-VAL's to C43-VAL's lists
QV_N = (7, 11, 18, 22)  # none in C41-VAL's to C43-VAL's lists
GEN = ("random_n10", "random_n20", "random_n30", "random_n40", "random_n48", "random_n56", "bv_n30", "bv_n60",
       "bv_n100", "ghz_n30", "ghz_n80")


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
    """{name: (family, QASM path or None)}, by fixed rules (none of them in C41-VAL to C43-VAL)."""
    out = {}
    for n in QFT_N:
        out[f"qft_N{n}"] = ("qft_qasm", f"qft/qft_N{n:03d}.qasm")
    for n in QAOA_N:
        out[f"qaoa_N{n}"] = ("qaoa_qasm", f"qaoa/qaoa_barabasi_albert_N{n}_3reps.qasm")
    for n in QV_N:
        out[f"qv_N{n}"] = ("qv", f"qv/qv_N{n:03d}_12345.qasm")
    for g in GEN:
        out["gen_" + g] = ("gen_" + g.split("_")[0], None)
    return out


def make_generated(name):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.circuit.random import random_circuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter
    rng = random.Random("C44-VAL|" + name)
    n = int(name.split("_n")[1])
    if "random" in name:
        return random_circuit(n, 10, max_operands=2, measure=True, seed=rng.randrange(10 ** 6))
    if "ghz" in name:
        qc = QuantumCircuit(n)
        qc.h(0)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        qc.measure_all()
        return qc
    if "bv" in name:
        qc = QuantumCircuit(n + 1, n)
        qc.x(n)
        qc.h(range(n + 1))
        for q in range(n):
            if rng.random() < 0.5:
                qc.cx(q, n)
        qc.h(range(n))
        qc.measure(range(n), range(n))
        return qc
    if "tfim" in name:
        terms = [(_label(n, {q: "Z", (q + 1) % n: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
        terms += [(_label(n, {q: "X"}), rng.uniform(-1, 1)) for q in range(n)]
        syn = LieTrotter(reps=2)
    elif "heis" in name:
        terms = [(_label(n, {q: p, (q + 1) % n: p}), rng.uniform(-1, 1)) for q in range(n) for p in "XYZ"]
        syn = None
    else:
        d = {}
        for _ in range(4 * n):
            qs = rng.sample(range(n), rng.choice((2, 2, 3)))
            d[_label(n, {q: rng.choice("XYZ") for q in qs})] = rng.uniform(-1, 1)
        terms = sorted(d.items())
        syn = None
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=0.7, synthesis=syn), range(n))
    qc = transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h"], optimization_level=1, seed_transpiler=0)
    qc.measure_all()
    return qc


def build(bp, test):
    fam, rel = tests_cached(bp)[test]
    qc = make_generated(test) if rel is None else _load_qasm(bp, rel)
    if not any(i.operation.name == "measure" for i in qc.data):
        qc.measure_all()
    return qc


_TESTS = {}


def tests_cached(bp):
    if bp not in _TESTS:
        path = os.environ.get("C44_TESTS")
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
    mods = {"R3": load(R3, "psf_r3"), "R4": load(R4, "psf_r4"), "C44": load(C44, "psf_c44")}
    mods["C44off"] = mods["C44"]
    sigs = {}
    for arm in ARMS + tuple(r + "r" for r in REPEATS):
        mod = mods.get(arm[:-1] if arm.endswith("r") else arm)
        rec = dict(test=test, device=a.device, arm=arm, input_qubits=qc.num_qubits)
        stats = {k: dict(getattr(mod, k)) for k in ("STEINER_STATS", "NEAR_STATS", "DET_STATS")
                 if mod is not None and hasattr(mod, k)}
        t0 = time.perf_counter()
        try:
            if arm == "QK3":
                out = generate_preset_pass_manager(3, backend, seed_transpiler=0).run(qc)
            else:
                rec["version"] = mod.VERSION
                kw = dict(near_failed_error=None) if arm == "C44off" else {}
                with contextlib.redirect_stdout(io.StringIO()):
                    out = mod.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True,
                                                   seed_transpiler=0, **kw)
        except Exception as exc:  # noqa: BLE001 - recorded
            rec.update(t=round(time.perf_counter() - t0, 3), error=f"{type(exc).__name__}: {exc}"[:300])
            print(json.dumps(rec), flush=True)
            continue
        rec["t"] = round(time.perf_counter() - t0, 3)
        rec.update(breakdown(out, target))
        if arm != "QK3":
            sigs[arm] = hashlib.sha256(repr(value_sig(out)).encode()).hexdigest()[:16]
            rec["sig"] = sigs[arm]
            for k, before in stats.items():  # this call's counts
                rec[k.lower()] = {x: getattr(mod, k)[x] - before.get(x, 0) for x in getattr(mod, k)}
        if arm.endswith("r"):
            rec["repeats"] = sigs.get(arm) == sigs.get(arm[:-1])
        if arm == "C44":
            p = mod._st_parse(qc)
            p = p if p is not None else mod._st_parse_gates(qc)
            rec["read"] = p is not None
        print(json.dumps(rec), flush=True)


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    os.makedirs(a.out)
    import shutil
    shutil.copyfile(C44, os.path.join(a.out, "psf_compile_c44_as_run.py"))
    os.environ["C44_FILE"] = os.path.join(a.out, "psf_compile_c44_as_run.py")
    json.dump(tests(a.bp), open(os.path.join(a.out, "c44_tests.json"), "w"), indent=0, sort_keys=True)
    os.environ["C44_TESTS"] = os.path.join(a.out, "c44_tests.json")
    js = jobs(a.bp)
    if a.smoke:
        js = [j for j in js if j[0] == "gen_random_n10"][:4]
    js.sort(key=lambda j: hashlib.sha256(("C44-VAL|" + j[0] + j[2]).encode()).hexdigest())
    path = os.path.join(a.out, "c44_val.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), jobs=len(js),
                                           par=a.par, c44_sha256=hashlib.sha256(open(C44, "rb").read()).hexdigest(),
                                           r4_sha256=hashlib.sha256(open(R4, "rb").read()).hexdigest(),
                                           r3_sha256=hashlib.sha256(open(R3, "rb").read()).hexdigest()))) + "\n")
    t_start = time.perf_counter()

    def go(job):
        test, arg, d = job
        base = dict(test=test, device=d)
        if time.perf_counter() - t_start > BUDGET:
            return [dict(base, arm=x, error="not started (budget)") for x in ARMS + ("C44r",)]
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--device", d,
                                "--job", json.dumps([test, arg])], capture_output=True, text=True, timeout=TIMEOUT,
                               cwd=REPO, env=dict(os.environ))
            recs = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]
            got = {r["arm"] for r in recs}
            return recs + [dict(base, arm=x, error=(p.stderr or "")[-300:]) for x in ARMS + ("C44r",)
                           if x not in got]
        except subprocess.TimeoutExpired:
            return [dict(base, arm=x, error=f"timeout {TIMEOUT} s") for x in ARMS + ("C44r",)]

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
    rows = [json.loads(x) for x in open(os.path.join(a.out, "c44_val.jsonl"), encoding="utf-8")][1:]
    by = {}
    for r in rows:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    L = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731
    ALL = ARMS + ("C44r",)

    def esp(r):
        return 0.0 if r["failed"] else 10 ** -L(r)

    def gm(p, q, xs):
        ok = [x for x in xs if esp(x[p]) > 0 and esp(x[q]) > 0]
        return (10 ** (sum(L(x[q]) - L(x[p]) for x in ok) / len(ok)), len(ok)) if ok else (float("nan"), 0)
    runn = lambda xs: [x for x in xs if max(esp(x[k]) for k in ARMS) >= 0.01]  # noqa: E731
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")  # noqa: E731
    tried = lambda x: (x["C44"].get("near_stats") or {}).get("tried", 0) > 0  # noqa: E731
    out = ["# C44-VAL", "", f"jobs: {len(by)}; records with an error: {sum(1 for r in rows if 'error' in r)}", ""]
    for r in [r for r in rows if "error" in r][:30]:
        out.append(f"- {r['device']} {r['arm']} {r['test'][-50:]}: {r['error'][:150]}")
    out += ["", "| device | U1 C44 on failed | U2 C44 repeats | U3 ESP C44/R4 | U4 below 0.95 x R4 | U5 ESP C44/R3; below 0.95 "
            "x R3 | U6 median time C44/R4 | U7 R3 raised, C44 compiled; C44 raised, QK3 clean | D: item 74 tried: ESP "
            "C44/C44off, median time | all runnable C44/QK3 |", "|" + "---|" * 10]
    res = {}
    for d in DEVICES + ("all",):
        allx = [x for (dv, _), x in by.items() if d == "all" or dv == d]
        sel = [x for x in allx if all(k in x and "error" not in x[k] for k in ALL)]
        u1 = sum(1 for x in allx if "C44" in x and "error" not in x["C44"] and x["C44"]["failed"] > 0)
        u2 = (sum(1 for x in allx if x.get("C44r", {}).get("repeats")),
              sum(1 for x in allx if "C44" in x and "error" not in x["C44"]))
        rr = runn(sel)
        u3 = gm("C44", "R4", rr)
        u4 = (sum(1 for x in rr if L(x["C44"]) - L(x["R4"]) > -math.log10(0.95)), len(rr))
        u5 = (gm("C44", "R3", rr), sum(1 for x in rr if L(x["C44"]) - L(x["R3"]) > -math.log10(0.95)))
        u6 = med([x["C44"]["t"] / max(x["R4"]["t"], 1e-3) for x in sel])
        u7a = sum(1 for x in allx if "error" in x.get("R3", {}) and "C44" in x and "error" not in x["C44"])
        u7b = sum(1 for x in allx if "error" in x.get("C44", {}) and "QK3" in x and "error" not in x["QK3"]
                  and x["QK3"]["failed"] == 0)
        tr = [x for x in sel if tried(x)]
        dd = (gm("C44", "C44off", runn(tr)), med([x["C44"]["t"] / max(x["C44off"]["t"], 1e-3) for x in tr]), len(tr))
        qk = gm("C44", "QK3", rr)
        res[d] = dict(u1=u1, u2=u2, u3=u3, u4=u4, u5=u5, u6=u6, u7=(u7a, u7b), d=dd, qk=qk)
        out.append(f"| {d} | {u1} | {u2[0]} / {u2[1]} | {u3[0]:.3f} ({u3[1]}) | {u4[0]} / {u4[1]} | {u5[0][0]:.3f}; "
                   f"{u5[1]} | {u6:.2f} | {u7a}; {u7b} | {dd[0][0]:.3f} ({dd[0][1]} of {dd[2]}), {dd[1]:.2f} | "
                   f"{qk[0]:.3f} ({qk[1]}) |")
    out += ["", "## Per test (log10 ESP C44 - R4, C44 - R3, C44 - QK3; C44 repeats)", ""]
    for (d, t), x in sorted(by.items(), key=lambda z: (z[0][1], z[0][0])):
        if all(k in x and "error" not in x[k] for k in ALL):
            out.append(f"- {t} {d[4:]}: {L(x['R4']) - L(x['C44']):+.4f}, {L(x['R3']) - L(x['C44']):+.4f}, "
                       f"{L(x['QK3']) - L(x['C44']):+.4f}; {x['C44r'].get('repeats')}")
    txt = "\n".join(out) + "\n"
    open(os.path.join(a.out, "c44_val.md"), "w", encoding="utf-8", newline="\n").write(txt)
    json.dump(res, open(os.path.join(a.out, "c44_val_summary.json"), "w"), default=str)
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
