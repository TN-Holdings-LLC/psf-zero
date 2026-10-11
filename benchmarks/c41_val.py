"""c41_val.py -- C41-VAL: prototype 2026-10-11.c41 (item 73 on c40: circuits written as gates are read as Pauli
rotations by carrying their Clifford gates to the end, and STEINER6 is tried on them under item 69's choice), on
Benchpress's QASM files (QAOA, square-lattice Heisenberg, QASMBench, Feynman) that item 73 reads, generated gate-level
circuits with new seeds ("C41-VAL|..."), and controls it does not read.
Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
Arms, each test and device in its own process: QK3 (Qiskit level 3, seed 0), R1 (release 2026-10-11.1), C40, C41;
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
C41 = os.environ.get("C41_FILE", os.path.join(REPO, "patches", "psf_compile_c41_2026-10-11", "psf_compile.py"))
C40 = os.path.join(REPO, "patches", "psf_compile_c40_2026-10-11", "psf_compile.py")
R1 = os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.1", "psf_compile.py")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
DEVICES = ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez", "FakeAachen", "FakeBoston")
ARMS = ("QK3", "R1", "C40", "C41")
FAILED = 0.5
TIMEOUT, BUDGET = 1800, 8 * 3600
QAOA_N = (5, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100)
HEIS_N = (4, 9, 16, 25, 36, 49, 64, 81, 100)
CONTROLS = ("qft/qft_N010.qasm", "qft/qft_N020.qasm", "qv/qv_N010_12345.qasm", "clifford/clifford_20_12345.qasm",
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
    """{name: (family, QASM path or None)}, by the fixed rules: Benchpress QAOA with N in QAOA_N, square-lattice
    Heisenberg with N in HEIS_N, every QASMBench and Feynman file (not the *_transpiled copies) that item 73 reads and
    whose qubits fit the largest device; generated circuits; controls."""
    out = {}
    for n in QAOA_N:
        out[f"qaoa_N{n}"] = ("qaoa_qasm", f"qaoa/qaoa_barabasi_albert_N{n}_3reps.qasm")
    for n in HEIS_N:
        out[f"heis_N{n}"] = ("heisenberg_qasm", f"square-heisenberg/square_heisenberg_N{n}.qasm")
    c41 = _c41()
    for fam, pat in (("qasmbench", "qasmbench-*/*/*.qasm"), ("feynman", "feynman/*.qasm")):
        for f in sorted(glob.glob(os.path.join(qasm_root(bp), pat))):
            rel = os.path.relpath(f, qasm_root(bp)).replace(os.sep, "/")
            if "transpiled" in rel:
                continue
            try:
                qc = _load_qasm(bp, rel)
            except Exception:  # noqa: BLE001 - not loadable: not a test
                continue
            if qc.num_qubits <= 156 and c41._st_parse(qc) is None and c41._st_parse_gates(qc) is not None:
                out[fam + "_" + os.path.basename(rel)[:-5]] = (fam, rel)
    for name in ("export_tfim_n10", "export_tfim_n16", "export_heis_n10", "export_heis_n16", "export_randpauli_n10"):
        out["gen_" + name] = ("gen_export", None)
    for n in (12, 20, 28):
        out[f"gen_qaoagates_n{n}_p2"] = ("gen_qaoa_gates", None)
    for rel in CONTROLS:
        out["control_" + os.path.basename(rel)[:-5]] = ("control", rel)
    for n in (6, 8):
        out[f"control_random_n{n}"] = ("control", None)
    return out


def _c41():
    if "psf_c41_sel" not in sys.modules:
        load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
        load(C41, "psf_c41_sel")
    return sys.modules["psf_c41_sel"]


def make_generated(name):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.circuit.random import random_circuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter
    rng = random.Random("C41-VAL|" + name)
    n = int(name.split("_n")[1].split("_")[0])
    if name.startswith("control_random"):
        return random_circuit(n, 10, max_operands=2, measure=True, seed=rng.randrange(10 ** 6))
    if name.startswith("gen_qaoagates"):
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        edges = _regular3(n, rng)
        for _ in range(2):
            g, beta = rng.uniform(0, 3), rng.uniform(0, 3)
            for a, b in edges:
                qc.cx(a, b)
                qc.rz(g * rng.uniform(-1, 1), b)
                qc.cx(a, b)
            for q in range(n):
                qc.rx(beta, q)
        qc.measure_all()
        return qc
    if "tfim" in name:
        terms = [(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a, b in _regular3(n, rng)]
        terms += [(_label(n, {q: "X"}), rng.uniform(-1, 1)) for q in range(n)]
        syn = LieTrotter(reps=2)
    elif "heis" in name:
        terms = [(_label(n, {a: p, b: p}), rng.uniform(-1, 1)) for a, b in _regular3(n, rng) for p in "XYZ"]
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
        path = os.environ.get("C41_TESTS")
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
    mods = {"R1": load(R1, "psf_r1"), "C40": load(C40, "psf_c40"), "C41": load(C41, "psf_c41")}
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
        if arm in ("C40", "C41"):
            rec["steiner_stats"] = dict(mods[arm].STEINER_STATS)
        if arm == "C41":
            rec["same_as_r1"] = sigs.get("C41") == sigs.get("R1")
            rec["same_as_c40"] = sigs.get("C41") == sigs.get("C40")
            p = mods["C41"]._st_parse(qc)
            p = p if p is not None else mods["C41"]._st_parse_gates(qc)
            rec["parsed_terms"] = None if p is None else len(mods["C41"]._st_merge(p[2]))
        print(json.dumps(rec), flush=True)


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    os.makedirs(a.out)
    import shutil
    shutil.copyfile(C41, os.path.join(a.out, "psf_compile_c41_as_run.py"))
    os.environ["C41_FILE"] = os.path.join(a.out, "psf_compile_c41_as_run.py")
    json.dump(tests(a.bp), open(os.path.join(a.out, "c41_tests.json"), "w"), indent=0, sort_keys=True)
    os.environ["C41_TESTS"] = os.path.join(a.out, "c41_tests.json")
    js = jobs(a.bp)
    if a.smoke:
        js = [j for j in js if j[0] in ("qaoa_N8", "control_random_n6")][:4]
    js.sort(key=lambda j: hashlib.sha256(("C41-VAL|" + j[0] + j[2]).encode()).hexdigest())
    path = os.path.join(a.out, "c41_val.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), jobs=len(js),
                                           par=a.par, c41_sha256=hashlib.sha256(open(C41, "rb").read()).hexdigest(),
                                           c40_sha256=hashlib.sha256(open(C40, "rb").read()).hexdigest(),
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
    rows = [json.loads(x) for x in open(os.path.join(a.out, "c41_val.jsonl"), encoding="utf-8")][1:]
    fams_of = json.load(open(os.path.join(a.out, "c41_tests.json")))
    by = {}
    for r in rows:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    L = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731

    def esp(r):
        return 0.0 if r["failed"] else 10 ** -L(r)

    def gm(p, q, xs):
        ok = [x for x in xs if esp(x[p]) > 0 and esp(x[q]) > 0]
        return (10 ** (sum(L(x[q]) - L(x[p]) for x in ok) / len(ok)), len(ok)) if ok else (float("nan"), 0)

    def g2(p, q, xs):
        return math.exp(sum(math.log((x[p]["n2"] + 1) / (x[q]["n2"] + 1)) for x in xs) / len(xs)) if xs else float("nan")
    fam = lambda x: fams_of[x["C41"]["test"]][0]  # noqa: E731
    out = ["# C41-VAL", "", f"jobs: {len(by)}; records with an error: {sum(1 for r in rows if 'error' in r)}", ""]
    for r in [r for r in rows if "error" in r][:20]:
        out.append(f"- {r['device']} {r['arm']} {r['test'][-50:]}: {r['error'][:150]}")
    fams = ("qaoa_qasm", "heisenberg_qasm", "qasmbench", "feynman", "gen_export", "gen_qaoa_gates")
    out += ["", "| device | Z1 C41 on failed | Z2 C41 >= R1 every test | Z3 C41 >= C40 every test | Z4 controls C41 = C40 = "
            "R1 | " + " | ".join(f"{f}: ESP C41/R1 (runnable), 2q C41/R1" for f in fams) + " | ESP C41/QK3 (all runnable)"
            " | median time C41 / R1 s (read) |", "|" + "---|" * (7 + len(fams))]
    res = {}
    for d in DEVICES + ("all",):
        sel = [x for (dv, _), x in by.items() if (d == "all" or dv == d) and all(k in x and "error" not in x[k]
                                                                                for k in ARMS)]
        z1 = sum(x["C41"]["failed"] > 0 for x in sel)
        z2 = (sum(1 for x in sel if L(x["C41"]) <= L(x["R1"]) + 1e-9), len(sel))
        z3 = (sum(1 for x in sel if L(x["C41"]) <= L(x["C40"]) + 1e-9), len(sel))
        ctl = [x for x in sel if fam(x) == "control"]
        z4 = (sum(1 for x in ctl if x["C41"].get("same_as_r1") and x["C41"].get("same_as_c40")), len(ctl))
        cells = []
        r = dict(z1=z1, z2=z2, z3=z3, z4=z4)
        for f in fams:
            xs = [x for x in sel if fam(x) == f]
            runn = [x for x in xs if max(esp(x[k]) for k in ARMS) >= 0.01]
            e, k = gm("C41", "R1", runn)
            r[f] = dict(esp=e, n=k, q2=g2("C41", "R1", xs), tests=len(xs))
            cells.append(f"{e:.3f} ({k}/{len(xs)}), {r[f]['q2']:.3f}")
        allrun = [x for x in sel if fam(x) != "control" and max(esp(x[k]) for k in ARMS) >= 0.01]
        qk = gm("C41", "QK3", allrun)
        read = [x for x in sel if x["C41"].get("parsed_terms")]
        med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")  # noqa: E731
        r.update(qk=qk, t=(med([x["C41"]["t"] for x in read]), med([x["R1"]["t"] for x in read])),
                 tmax=max([x["C41"]["t"] for x in read], default=float("nan")))
        res[d] = r
        out.append(f"| {d} | {z1} | {z2[0]} / {z2[1]} | {z3[0]} / {z3[1]} | {z4[0]} / {z4[1]} | " + " | ".join(cells)
                   + f" | {qk[0]:.3f} ({qk[1]}) | {r['t'][0]:.2f} / {r['t'][1]:.2f} (max {r['tmax']:.1f}) |")
    out += ["", "## Per test (log10 ESP C41 - R1, C41 - QK3; two-qubit C41 / R1 / QK3; STEINER chosen?; ESP C41)", ""]
    for (d, t), x in sorted(by.items(), key=lambda z: (z[0][1], z[0][0])):
        if all(k in x and "error" not in x[k] for k in ARMS):
            st = x["C41"].get("steiner_stats") or {}
            ch = "steiner" if st.get("chosen") else ("usual" if st.get("kept_usual") else "-")
            out.append(f"- {t} {d[4:]}: {L(x['R1']) - L(x['C41']):+.4f}, {L(x['QK3']) - L(x['C41']):+.4f}; "
                       f"{x['C41']['n2']} / {x['R1']['n2']} / {x['QK3']['n2']}; {ch}; {esp(x['C41']):.3g}")
    txt = "\n".join(out) + "\n"
    open(os.path.join(a.out, "c41_val.md"), "w", encoding="utf-8", newline="\n").write(txt)
    json.dump(res, open(os.path.join(a.out, "c41_val_summary.json"), "w"), default=str)
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
