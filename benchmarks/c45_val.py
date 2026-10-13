"""c45_val.py -- C45-VAL: prototype 2026-10-12.c45 (on c44): item 78 (item 75's re-placement at Qiskit's routing stage, in
item 80's compile), item 79 (basis gates from a Target given alone) and item 80 (Qiskit level 3, deterministic, as a
candidate by ESP). Tests not used in C41-VAL to C44-VAL: Benchpress QFT, QAOA and quantum-volume sizes outside their
lists, clifford_20, and generated circuits with seeds "C45-VAL|..." (random circuits, Bernstein-Vazirani, GHZ).
Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
Arms, each test and device in its own process: QK3 (Qiskit level 3, seed 0), R4 (release 2026-10-11.4), C44, C45, C45n
(c45 with level3_candidate=False); then C45 again in the same process (C45r).
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
C45 = os.environ.get("C45_FILE", os.path.join(REPO, "patches", "psf_compile_c45_2026-10-12", "psf_compile.py"))
C44 = os.path.join(REPO, "patches", "psf_compile_c44_2026-10-11", "psf_compile.py")
R4 = os.path.join(REPO, "psf_compile.py")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
DEVICES = ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez", "FakeAachen", "FakeBoston")
ARMS = ("QK3", "R4", "C44", "C45", "C45n")
REPEATS = ("C45",)
FAILED = 0.5
TIMEOUT, BUDGET = 1800, 8 * 3600
QFT_N = (17, 21, 25, 33, 44)  # none in C42-VAL's to C44-VAL's lists
QAOA_N = (16, 23, 29, 42, 65)  # none in C41-VAL's to C44-VAL's lists
QV_N = (13, 15, 19, 26)  # none in C41-VAL's to C44-VAL's lists
OTHER = ("clifford/clifford_20_12345.qasm",)  # C42-VAL used clifford_100 only
GEN = ("random_n14", "random_n26", "random_n36", "random_n44", "random_n52", "bv_n50", "bv_n90", "ghz_n40",
       "ghz_n100")


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
    """{name: (family, QASM path or None)}, by fixed rules (none of them in C41-VAL to C44-VAL)."""
    out = {}
    for n in QFT_N:
        out[f"qft_N{n}"] = ("qft_qasm", f"qft/qft_N{n:03d}.qasm")
    for n in QAOA_N:
        out[f"qaoa_N{n}"] = ("qaoa_qasm", f"qaoa/qaoa_barabasi_albert_N{n}_3reps.qasm")
    for n in QV_N:
        out[f"qv_N{n}"] = ("qv", f"qv/qv_N{n:03d}_12345.qasm")
    for rel in OTHER:
        out["other_" + os.path.basename(rel)[:-5]] = ("other", rel)
    for g in GEN:
        out["gen_" + g] = ("gen_" + g.split("_")[0], None)
    return out


def make_generated(name):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.circuit.random import random_circuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter
    rng = random.Random("C45-VAL|" + name)
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
        path = os.environ.get("C45_TESTS")
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
    mods = {"R4": load(R4, "psf_r4"), "C44": load(C44, "psf_c44"), "C45": load(C45, "psf_c45")}
    mods["C45n"] = mods["C45"]
    sigs = {}
    for arm in ARMS + tuple(r + "r" for r in REPEATS):
        mod = mods.get(arm[:-1] if arm.endswith("r") else arm)
        rec = dict(test=test, device=a.device, arm=arm, input_qubits=qc.num_qubits)
        stats = {k: dict(getattr(mod, k)) for k in ("STEINER_STATS", "NEAR_STATS", "DET_STATS", "L3C_STATS")
                 if mod is not None and hasattr(mod, k)}
        t0 = time.perf_counter()
        try:
            if arm == "QK3":
                out = generate_preset_pass_manager(3, backend, seed_transpiler=0).run(qc)
            else:
                rec["version"] = mod.VERSION
                kw = dict(level3_candidate=False) if arm == "C45n" else {}
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
        if arm == "C45":
            p = mod._st_parse(qc)
            p = p if p is not None else mod._st_parse_gates(qc)
            rec["read"] = p is not None
        print(json.dumps(rec), flush=True)


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    os.makedirs(a.out)
    import shutil
    shutil.copyfile(C45, os.path.join(a.out, "psf_compile_c45_as_run.py"))
    os.environ["C45_FILE"] = os.path.join(a.out, "psf_compile_c45_as_run.py")
    json.dump(tests(a.bp), open(os.path.join(a.out, "c45_tests.json"), "w"), indent=0, sort_keys=True)
    os.environ["C45_TESTS"] = os.path.join(a.out, "c45_tests.json")
    js = jobs(a.bp)
    if a.smoke:
        js = [j for j in js if j[0] == "gen_random_n14"][:4]
    js.sort(key=lambda j: hashlib.sha256(("C45-VAL|" + j[0] + j[2]).encode()).hexdigest())
    path = os.path.join(a.out, "c45_val.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), jobs=len(js),
                                           par=a.par, c45_sha256=hashlib.sha256(open(C45, "rb").read()).hexdigest(),
                                           c44_sha256=hashlib.sha256(open(C44, "rb").read()).hexdigest(),
                                           r4_sha256=hashlib.sha256(open(R4, "rb").read()).hexdigest()))) + "\n")
    t_start = time.perf_counter()

    def go(job):
        test, arg, d = job
        base = dict(test=test, device=d)
        if time.perf_counter() - t_start > BUDGET:
            return [dict(base, arm=x, error="not started (budget)") for x in ARMS + ("C45r",)]
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--device", d,
                                "--job", json.dumps([test, arg])], capture_output=True, text=True, timeout=TIMEOUT,
                               cwd=REPO, env=dict(os.environ))
            recs = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]
            got = {r["arm"] for r in recs}
            return recs + [dict(base, arm=x, error=(p.stderr or "")[-300:]) for x in ARMS + ("C45r",)
                           if x not in got]
        except subprocess.TimeoutExpired:
            return [dict(base, arm=x, error=f"timeout {TIMEOUT} s") for x in ARMS + ("C45r",)]

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
    rows = [json.loads(x) for x in open(os.path.join(a.out, "c45_val.jsonl"), encoding="utf-8")][1:]
    by = {}
    for r in rows:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    L = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731
    ALL = ARMS + ("C45r",)

    def esp(r):
        return 0.0 if r["failed"] else 10 ** -L(r)

    def gm(p, q, xs):
        ok = [x for x in xs if esp(x[p]) > 0 and esp(x[q]) > 0]
        return (10 ** (sum(L(x[q]) - L(x[p]) for x in ok) / len(ok)), len(ok)) if ok else (float("nan"), 0)
    runn = lambda xs: [x for x in xs if max(esp(x[k]) for k in ARMS) >= 0.01]  # noqa: E731
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")  # noqa: E731
    lc = lambda x: x["C45"].get("l3c_stats") or {}  # noqa: E731
    out = ["# C45-VAL", "", f"jobs: {len(by)}; records with an error: {sum(1 for r in rows if 'error' in r)}", ""]
    for r in [r for r in rows if "error" in r][:30]:
        out.append(f"- {r['device']} {r['arm']} {r['test'][-50:]}: {r['error'][:150]}")
    out += ["", "| device | X1 C45 on failed | X2 C45 repeats | X3 ESP C45/QK3; below 0.98 x QK3 | X4 ESP C45/C44; "
            "below C44 | X5 C45n = C44 by value | X6 median time C45/C44 | X7 level 3 kept / tried (unverified) | "
            "C45/R4 | C45n/QK3 |", "|" + "---|" * 10]
    res = {}
    for d in DEVICES + ("all",):
        allx = [x for (dv, _), x in by.items() if d == "all" or dv == d]
        sel = [x for x in allx if all(k in x and "error" not in x[k] for k in ALL)]
        x1 = sum(1 for x in allx if "C45" in x and "error" not in x["C45"] and x["C45"]["failed"] > 0)
        x2 = (sum(1 for x in allx if x.get("C45r", {}).get("repeats")),
              sum(1 for x in allx if "C45" in x and "error" not in x["C45"]))
        rr = runn(sel)
        x3 = (gm("C45", "QK3", rr), sum(1 for x in rr if L(x["C45"]) - L(x["QK3"]) > -math.log10(0.98)), len(rr))
        x4 = (gm("C45", "C44", rr), sum(1 for x in sel if L(x["C45"]) - L(x["C44"]) > 1e-9))
        both = [x for x in allx if all(k in x and "error" not in x[k] for k in ("C44", "C45n"))]
        x5 = (sum(1 for x in both if x["C45n"].get("sig") == x["C44"].get("sig")), len(both))
        x6 = med([x["C45"]["t"] / max(x["C44"]["t"], 1e-3) for x in sel])
        x7 = (sum(lc(x).get("kept_level3", 0) for x in sel), sum(lc(x).get("tried", 0) for x in sel),
              sum(lc(x).get("unverified", 0) for x in sel))
        r4, nq = gm("C45", "R4", rr), gm("C45n", "QK3", rr)
        res[d] = dict(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, x7=x7, r4=r4, nq=nq,
                      tmax=max([x["C45"]["t"] for x in sel] or [0]))
        out.append(f"| {d} | {x1} | {x2[0]} / {x2[1]} | {x3[0][0]:.3f} ({x3[0][1]}); {x3[1]} of {x3[2]} | "
                   f"{x4[0][0]:.3f}; {x4[1]} | {x5[0]} / {x5[1]} | {x6:.2f} | {x7[0]} / {x7[1]} ({x7[2]}) | "
                   f"{r4[0]:.3f} | {nq[0]:.3f} |")
    fam = lambda t: t.split("_")[0] if not t.startswith("gen_") else t.split("_")[1]  # noqa: E731
    out += ["", "## By family (runnable; ESP C45/QK3, C45/C44, C45n/QK3)", ""]
    allsel = runn([x for x in by.values() if all(k in x and "error" not in x[k] for k in ALL)])
    for f in sorted({fam(x["C45"]["test"]) for x in allsel}):
        xs = [x for x in allsel if fam(x["C45"]["test"]) == f]
        out.append(f"- {f}: {gm('C45', 'QK3', xs)[0]:.3f}, {gm('C45', 'C44', xs)[0]:.3f}, "
                   f"{gm('C45n', 'QK3', xs)[0]:.3f} ({len(xs)})")
    out += ["", "## Per test (log10 ESP C45 - QK3, C45 - C44, C45n - QK3; level 3 kept; C45 repeats)", ""]
    for (d, t), x in sorted(by.items(), key=lambda z: (z[0][1], z[0][0])):
        if all(k in x and "error" not in x[k] for k in ALL):
            out.append(f"- {t} {d[4:]}: {L(x['QK3']) - L(x['C45']):+.4f}, {L(x['C44']) - L(x['C45']):+.4f}, "
                       f"{L(x['QK3']) - L(x['C45n']):+.4f}; {lc(x).get('kept_level3', 0)}; {x['C45r'].get('repeats')}")
    txt = "\n".join(out) + "\n"
    open(os.path.join(a.out, "c45_val.md"), "w", encoding="utf-8", newline="\n").write(txt)
    json.dump(res, open(os.path.join(a.out, "c45_val_summary.json"), "w"), default=str)
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
