"""c40_val.py -- C40-VAL: prototype 2026-10-11.c40 (items 71-72 on c39: STEINER6 tried on every circuit it can read as
Pauli rotations, commuting or not, kept where its ESP is higher; multi-step product formulas, QAOA, rotation-gate
circuits; exact merging of repeated terms), on circuits generated with new seeds ("C40-VAL|...", `generated()`):
QAOA (QAOAAnsatz, MaxCut on 3-regular graphs, p = 1-3), multi-step non-commuting product formulas (transverse-field
Ising, LieTrotter 2 steps and SuzukiTrotter order 2), multi-step commuting ones (3 steps), single-step non-commuting
Hamiltonians (Heisenberg, XY, random Pauli), rotation-gate circuits, and ordinary random circuits (controls).
Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
Arms, each test and device in its own process: QK3 (Qiskit level 3, seed 0), R1 (release 2026-10-11.1), C39, C40;
the psf arms with the plain call with backend=.

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
C40 = os.environ.get("C40_FILE", os.path.join(REPO, "patches", "psf_compile_c40_2026-10-11", "psf_compile.py"))
C39 = os.path.join(REPO, "patches", "psf_compile_c39_2026-10-11", "psf_compile.py")
R1 = os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.1", "psf_compile.py")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
DEVICES = ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez", "FakeAachen", "FakeBoston")
ARMS = ("QK3", "R1", "C39", "C40")
FAILED = 0.5
TIMEOUT, BUDGET = 1800, 6 * 3600


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


def generated():
    """{name: (kind, family, n, parameters)}: built by `make()`; all seeds fixed."""
    out = {}
    for n in (12, 20, 28):
        for p in (1, 2, 3):
            out[f"gen_qaoa_n{n}_p{p}"] = ("qaoa", "qaoa", n, p)
    for n in (10, 16, 22):
        out[f"gen_tfim_lie2_n{n}"] = ("tfim", "trotter_noncommuting", n, "lie2")
        out[f"gen_tfim_suzuki2_n{n}"] = ("tfim", "trotter_noncommuting", n, "suzuki2")
    for n in (12, 20):
        out[f"gen_ising_lie3_n{n}"] = ("ising", "trotter_commuting", n, "lie3")
    out["gen_sat3_lie2_n14"] = ("sat3", "trotter_commuting", 14, "lie2")
    for n in (10, 16):
        out[f"gen_heisenberg3reg_n{n}"] = ("heis", "single_noncommuting", n, None)
        out[f"gen_randpauli_n{n}"] = ("randpauli", "single_noncommuting", n, None)
    out["gen_xyring_n12"] = ("xy", "single_noncommuting", 12, None)
    for n in (12, 20):
        out[f"gen_rotations_n{n}"] = ("rot", "rotation_gates", n, None)
    for n in (6, 8):
        out[f"gen_random_n{n}"] = ("random", "control", n, None)
    return out


def make(test):
    """The circuit of a generated test (final measurements on all qubits, as a user would run it)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate, QAOAAnsatz
    from qiskit.circuit.random import random_circuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter, SuzukiTrotter
    kind, _, n, par = generated()[test]
    rng = random.Random("C40-VAL|" + test)
    syn = {"lie2": LieTrotter(reps=2), "lie3": LieTrotter(reps=3), "suzuki2": SuzukiTrotter(order=2, reps=1),
           None: None}[par if kind != "qaoa" else None]
    if kind == "qaoa":
        cost = SparsePauliOp.from_list([(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a, b in _regular3(n, rng)])
        qc = QAOAAnsatz(cost, reps=par)
        qc = qc.assign_parameters([rng.uniform(0, 3) for _ in range(qc.num_parameters)])
        qc.measure_all()
        return qc
    if kind == "random":
        return random_circuit(n, 10, max_operands=2, measure=True, seed=rng.randrange(10 ** 6))
    if kind == "rot":
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for layer in range(2):
            for a, b in _regular3(n, rng):
                qc.rzz(rng.uniform(-2, 2), a, b)
            for q in range(n):
                qc.rx(rng.uniform(-2, 2), q)
        qc.measure_all()
        return qc
    if kind in ("tfim", "ising"):
        terms = [(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a, b in _regular3(n, rng)]
        terms += [(_label(n, {q: "X" if kind == "tfim" else "Z"}), rng.uniform(-1, 1)) for q in range(n)]
    elif kind == "sat3":
        d = {}
        for _ in range(3 * n):
            a, b, c = rng.sample(range(n), 3)
            for sub in ((a,), (b,), (c,), (a, b), (a, c), (b, c), (a, b, c)):
                lb = _label(n, {q: "Z" for q in sub})
                d[lb] = d.get(lb, 0.0) + rng.uniform(-1, 1)
        terms = sorted(d.items())
    elif kind == "heis":
        terms = [(_label(n, {a: p, b: p}), rng.uniform(-1, 1)) for a, b in _regular3(n, rng) for p in "XYZ"]
    elif kind == "xy":
        terms = [(_label(n, {q: p, (q + 1) % n: p}), rng.uniform(-1, 1)) for q in range(n) for p in "XY"]
    else:  # randpauli: 2- and 3-local random Paulis
        d = {}
        for _ in range(4 * n):
            qs = rng.sample(range(n), rng.choice((2, 2, 3)))
            d[_label(n, {q: rng.choice("XYZ") for q in qs})] = rng.uniform(-1, 1)
        terms = sorted(d.items())
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=0.7, synthesis=syn), range(n))
    qc.measure_all()
    return qc


def hamlib_new(bp):
    """[(test id, builder argument)]: Benchpress's HamLib FakeTorino tests that are not in BP-FINAL's 67."""
    import bp_mock as B
    import steiner3 as S3
    dev = {t[1] for t in S3.tests(bp)}
    return [(t[0], t[2]) for t in B.strata(bp)["HamLib, FakeTorino"] if t[0] not in dev]


def build(bp, test, arg):
    if test.startswith("gen_"):
        return make(test)
    import bp_mock as B
    qc, _ = B.build(bp, "ham_device", arg)
    return qc


def jobs(bp):
    from qiskit_ibm_runtime import fake_provider
    tests = [(g, None) for g in generated()]
    sizes = {}
    for t, a in tests:
        if t.startswith("gen_"):
            sizes[t] = generated()[t][2]
        else:
            import bp_mock as B
            sizes[t] = B.build(bp, "ham_device", a)[0].num_qubits
    nq = {d: getattr(fake_provider, d)().target.num_qubits for d in DEVICES}
    return [(t, a, d) for t, a in tests for d in DEVICES if sizes[t] <= nq[d]]


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
    qc = build(a.bp, test, arg)
    backend = getattr(fake_provider, a.device)()
    target = backend.target
    load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    mods = {"R1": load(R1, "psf_r1"), "C39": load(C39, "psf_c39"), "C40": load(C40, "psf_c40")}
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
        if arm in ("C39", "C40"):
            rec["steiner_stats"] = dict(mods[arm].STEINER_STATS)
        if arm == "C40":
            rec["same_as_r1"] = sigs.get("C40") == sigs.get("R1")
            rec["parsed_terms"] = (lambda p: None if p is None else len(p[2]))(mods["C40"]._st_parse(qc))
        print(json.dumps(rec), flush=True)


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    js = jobs(a.bp)
    if a.smoke:
        js = [j for j in js if j[0] in ("gen_qaoa_n12_p1", "gen_random_n6")][:4]
    js.sort(key=lambda j: hashlib.sha256(("C40-VAL|" + j[0] + j[2]).encode()).hexdigest())
    os.makedirs(a.out)
    import shutil
    shutil.copyfile(C40, os.path.join(a.out, "psf_compile_c40_as_run.py"))
    path = os.path.join(a.out, "c40_val.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), jobs=len(js),
                                           par=a.par, c40_sha256=hashlib.sha256(open(C40, "rb").read()).hexdigest(),
                                           c39_sha256=hashlib.sha256(open(C39, "rb").read()).hexdigest(),
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
                               cwd=REPO, env=dict(os.environ, C40_FILE=os.path.join(a.out,
                                                                                    "psf_compile_c40_as_run.py")))
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
    rows = [json.loads(x) for x in open(os.path.join(a.out, "c40_val.jsonl"), encoding="utf-8")][1:]
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
    out = ["# C40-VAL", "", f"jobs: {len(by)}; records with an error: {sum(1 for r in rows if 'error' in r)}", ""]
    for r in [r for r in rows if "error" in r][:10]:
        out.append(f"- {r['device']} {r['arm']} {r['test'][-50:]}: {r['error'][:150]}")
    fams = ("qaoa", "trotter_noncommuting", "trotter_commuting", "single_noncommuting", "rotation_gates")
    out += ["", "| device | Y1 C40 on failed | Y2 C40 >= R1 every test | Y3 C40 >= C39 every test | Y7 controls C40 = R1 | "
            + " | ".join(f"{f}: ESP C40/R1 (runnable), 2q C40/R1" for f in fams) + " | ESP C40/QK3 (all runnable) | "
            "Y8 median time C40 / R1 s (readable) |", "|" + "---|" * (7 + len(fams))]
    res = {}
    for d in DEVICES + ("all",):
        sel = [x for (dv, _), x in by.items() if (d == "all" or dv == d) and all(k in x and "error" not in x[k]
                                                                                for k in ARMS)]
        fam = lambda x: generated()[x["C40"]["test"]][1]  # noqa: E731
        y1 = sum(x["C40"]["failed"] > 0 for x in sel)
        y2 = (sum(1 for x in sel if L(x["C40"]) <= L(x["R1"]) + 1e-9), len(sel))
        y3 = (sum(1 for x in sel if L(x["C40"]) <= L(x["C39"]) + 1e-9), len(sel))
        ctl = [x for x in sel if fam(x) == "control"]
        y7 = (sum(1 for x in ctl if x["C40"].get("same_as_r1")), len(ctl))
        cells = []
        r = dict(y1=y1, y2=y2, y3=y3, y7=y7)
        for f in fams:
            xs = [x for x in sel if fam(x) == f]
            runn = [x for x in xs if max(esp(x[k]) for k in ARMS) >= 0.01]
            e, k = gm("C40", "R1", runn)
            r[f] = dict(esp=e, n=k, q2=g2("C40", "R1", xs))
            cells.append(f"{e:.3f} ({k}), {r[f]['q2']:.3f}")
        allrun = [x for x in sel if fam(x) != "control" and max(esp(x[k]) for k in ARMS) >= 0.01]
        qk = gm("C40", "QK3", allrun)
        readable = sorted(x["C40"]["t"] for x in sel if x["C40"].get("parsed_terms"))
        r1t = sorted(x["R1"]["t"] for x in sel if x["C40"].get("parsed_terms"))
        med = lambda v: v[len(v) // 2] if v else float("nan")  # noqa: E731
        r.update(qk=qk, t=(med(readable), med(r1t)))
        res[d] = r
        out.append(f"| {d} | {y1} | {y2[0]} / {y2[1]} | {y3[0]} / {y3[1]} | {y7[0]} / {y7[1]} | " + " | ".join(cells)
                   + f" | {qk[0]:.3f} ({qk[1]}) | {r['t'][0]:.2f} / {r['t'][1]:.2f} |")
    out += ["", "## Per test (log10 ESP C40 - R1, C40 - QK3; two-qubit C40 / R1 / QK3; STEINER chosen?)", ""]
    for (d, t), x in sorted(by.items(), key=lambda z: (z[0][1], z[0][0])):
        if all(k in x and "error" not in x[k] for k in ARMS):
            st = x["C40"].get("steiner_stats") or {}
            ch = "steiner" if st.get("chosen") else ("usual" if st.get("kept_usual") else "-")
            out.append(f"- {t} {d[4:]}: {L(x['R1']) - L(x['C40']):+.4f}, {L(x['QK3']) - L(x['C40']):+.4f}; "
                       f"{x['C40']['n2']} / {x['R1']['n2']} / {x['QK3']['n2']}; {ch}")
    txt = "\n".join(out) + "\n"
    open(os.path.join(a.out, "c40_val.md"), "w", encoding="utf-8", newline="\n").write(txt)
    json.dump(res, open(os.path.join(a.out, "c40_val_summary.json"), "w"), default=str)
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
