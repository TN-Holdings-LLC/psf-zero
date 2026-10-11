"""c39_val.py -- C39-VAL: candidate 2026-10-11.c39 (changelog items 69-70: STEINER6's output kept only where its ESP is
above the usual path's; a 2000-term limit) on circuits generated with new seeds, among them the type item 68 lost on
(tsp in unary encoding).

Circuits (all generated, fixed seeds "C39-VAL|...", `generated()`):
  - commuting: tsp-type in unary encoding (k = 3, 4, 5 cities: 9, 16, 25 qubits), dense random graphs (Z + ZZ,
    G(n, 0.4)), MaxCut type on random 3-regular graphs, 3-SAT type (Z + ZZ + ZZZ);
  - not commuting (controls): transverse-field Ising on random 3-regular graphs, a Heisenberg ring.
Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
Arms, each test and device in its own process, in this order: QK3 (Qiskit level 3, seed 0), R1 (release 2026-10-11.1,
the usual path: patches/psf_compile_release_2026-10-11.1/), R2 (release 2026-10-11.2, item 68: psf_compile.py), C39
(the candidate); the psf arms with the plain call with backend=.

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
C39 = os.environ.get("C39_FILE", os.path.join(REPO, "patches", "psf_compile_c39_2026-10-11", "psf_compile.py"))
R1 = os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.1", "psf_compile.py")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
DEVICES = ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez", "FakeAachen", "FakeBoston")
ARMS = ("QK3", "R1", "R2", "C39")
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
    """{name: (n, [(label, coefficient)], commuting?)}, all from fixed seeds."""
    out = {}
    for k in (3, 4, 5):
        for sd in (1, 2, 3):
            rng = random.Random(f"C39-VAL|tsp|{k}|{sd}")
            n = k * k
            terms = {}
            def add(qs, w):
                lb = _label(n, {q: "Z" for q in qs})
                terms[lb] = terms.get(lb, 0.0) + w
            for c in range(k):
                for t in range(k):
                    add((c * k + t,), rng.uniform(-1, 1))
                    for t2 in range(t + 1, k):
                        add((c * k + t, c * k + t2), rng.uniform(0.5, 1.5))
                        add((t * k + c, t2 * k + c), rng.uniform(0.5, 1.5))
            d = [[rng.uniform(0.1, 1.0) for _ in range(k)] for _ in range(k)]
            for a in range(k):
                for b in range(k):
                    if a != b:
                        for t in range(k):
                            add((a * k + t, b * k + (t + 1) % k), d[a][b])
            out[f"gen_tspunary_k{k}_s{sd}"] = (n, sorted(terms.items()), True)
    for n in (10, 16, 22):
        for sd in (1, 2):
            rng = random.Random(f"C39-VAL|gnp|{n}|{sd}")
            terms = [(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a in range(n) for b in range(a + 1, n)
                     if rng.random() < 0.4]
            terms += [(_label(n, {q: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
            out[f"gen_densegnp_n{n}_s{sd}"] = (n, terms, True)
    for n in (12, 24, 36):
        for sd in (1, 2):
            rng = random.Random(f"C39-VAL|maxcut|{n}|{sd}")
            terms = [(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a, b in _regular3(n, rng)]
            terms += [(_label(n, {q: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
            out[f"gen_maxcut3reg_n{n}_s{sd}"] = (n, terms, True)
    for n in (14, 24):
        for sd in (1, 2):
            rng = random.Random(f"C39-VAL|sat3|{n}|{sd}")
            terms = {}
            for _ in range(3 * n):
                a, b, c = rng.sample(range(n), 3)
                for sub in ((a,), (b,), (c,), (a, b), (a, c), (b, c), (a, b, c)):
                    lb = _label(n, {q: "Z" for q in sub})
                    terms[lb] = terms.get(lb, 0.0) + rng.uniform(-1, 1)
            out[f"gen_sat3_n{n}_s{sd}"] = (n, sorted(terms.items()), True)
    for n in (12, 24):
        rng = random.Random(f"C39-VAL|tfim|{n}")
        terms = [(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a, b in _regular3(n, rng)]
        terms += [(_label(n, {q: "X"}), rng.uniform(-1, 1)) for q in range(n)]
        out[f"gen_tfim3reg_n{n}"] = (n, terms, False)
    rng = random.Random("C39-VAL|heis|10")
    out["gen_heisenberg_ring_n10"] = (10, [(_label(10, {q: p, (q + 1) % 10: p}), rng.uniform(-1, 1)) for q in range(10)
                                           for p in "XYZ"], False)
    return out


def hamlib_new(bp):
    """[(test id, builder argument)]: Benchpress's HamLib FakeTorino tests that are not in BP-FINAL's 67."""
    import bp_mock as B
    import steiner3 as S3
    dev = {t[1] for t in S3.tests(bp)}
    return [(t[0], t[2]) for t in B.strata(bp)["HamLib, FakeTorino"] if t[0] not in dev]


def build(bp, test, arg):
    if test.startswith("gen_"):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        n, terms, _ = generated()[test]
        qc = QuantumCircuit(n)
        qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=1.0), range(n))
        return qc
    import bp_mock as B
    qc, _ = B.build(bp, "ham_device", arg)
    return qc


def jobs(bp):
    from qiskit_ibm_runtime import fake_provider
    tests = [(g, None) for g in generated()]
    sizes = {}
    for t, a in tests:
        if t.startswith("gen_"):
            sizes[t] = generated()[t][0]
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
    mods = {"R1": load(R1, "psf_r1"), "R2": load(os.path.join(REPO, "psf_compile.py"), "psf_r2"),
            "C39": load(C39, "psf_c39")}
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
        if arm in ("R2", "C39"):
            rec["steiner_stats"] = dict(mods[arm].STEINER_STATS)
        if arm == "C39":
            rec["same_as_r1"] = sigs.get("C39") == sigs.get("R1")
            rec["same_as_r2"] = sigs.get("C39") == sigs.get("R2")
        print(json.dumps(rec), flush=True)


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    js = jobs(a.bp)
    if a.smoke:
        js = [j for j in js if j[0] in ("gen_tspunary_k3_s1", "gen_tfim3reg_n12")][:4]
    js.sort(key=lambda j: hashlib.sha256(("C39-VAL|" + j[0] + j[2]).encode()).hexdigest())
    os.makedirs(a.out)
    import shutil
    shutil.copyfile(C39, os.path.join(a.out, "psf_compile_c39_as_run.py"))
    path = os.path.join(a.out, "c39_val.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), jobs=len(js),
                                           par=a.par, c39_sha256=hashlib.sha256(open(C39, "rb").read()).hexdigest(),
                                           r2_sha256=hashlib.sha256(open(os.path.join(REPO, "psf_compile.py"), "rb")
                                                                    .read()).hexdigest(),
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
                               cwd=REPO, env=dict(os.environ, C39_FILE=os.path.join(a.out,
                                                                                    "psf_compile_c39_as_run.py")))
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
    rows = [json.loads(x) for x in open(os.path.join(a.out, "c39_val.jsonl"), encoding="utf-8")][1:]
    by = {}
    for r in rows:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    L = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731

    def esp(r):
        return 0.0 if r["failed"] else 10 ** -L(r)
    out = ["# C39-VAL", "", f"jobs: {len(by)}; records with an error: {sum(1 for r in rows if 'error' in r)}", ""]
    for r in [r for r in rows if "error" in r][:10]:
        out.append(f"- {r['device']} {r['arm']} {r['test'][-50:]}: {r['error'][:150]}")
    out += ["", "| device | commuting | X1 C39 on failed | X2 non-commuting C39 = R1 = R2 | X3 C39 >= R1 every commuting test "
            "| X4 C39 >= R2 every commuting test | runnable commuting | X5 ESP C39/R1 | ESP R2/R1 | ESP C39/QK3 | "
            "X6 tsp-type: C39/R1, R2/R1 | X7 median time C39 / R1 s (commuting) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for d in DEVICES + ("all",):
        sel = [x for (dv, _), x in by.items() if (d == "all" or dv == d) and all(k in x and "error" not in x[k]
                                                                                for k in ARMS)]
        comm = [x for x in sel if generated()[x["C39"]["test"]][2]]
        nonc = [x for x in sel if not generated()[x["C39"]["test"]][2]]
        x1 = sum(x["C39"]["failed"] > 0 for x in sel)
        x2 = (sum(1 for x in nonc if x["C39"].get("same_as_r1") and x["C39"].get("same_as_r2")), len(nonc))
        x3 = (sum(1 for x in comm if L(x["C39"]) <= L(x["R1"]) + 1e-9), len(comm))
        x4 = (sum(1 for x in comm if L(x["C39"]) <= L(x["R2"]) + 1e-9), len(comm))
        runn = [x for x in comm if max(esp(x[k]) for k in ARMS) >= 0.01]

        def gm(p, q, xs):
            ok = [x for x in xs if esp(x[p]) > 0 and esp(x[q]) > 0]
            return (10 ** (sum(L(x[q]) - L(x[p]) for x in ok) / len(ok)), len(ok)) if ok else (float("nan"), 0)
        tsp = [x for x in runn if x["C39"]["test"].startswith("gen_tspunary")]
        t39 = sorted(x["C39"]["t"] for x in comm)
        t1 = sorted(x["R1"]["t"] for x in comm)
        med = lambda v: v[len(v) // 2] if v else float("nan")  # noqa: E731
        r = dict(x1=x1, x2=x2, x3=x3, x4=x4, x5=gm("C39", "R1", runn), r2=gm("R2", "R1", runn),
                 qk=gm("C39", "QK3", runn), tsp39=gm("C39", "R1", tsp), tsp2=gm("R2", "R1", tsp), t=(med(t39), med(t1)))
        res[d] = r
        out.append(f"| {d} | {len(comm)} | {x1} | {x2[0]} / {x2[1]} | {x3[0]} / {x3[1]} | {x4[0]} / {x4[1]} | {len(runn)} | "
                   f"{r['x5'][0]:.3f} ({r['x5'][1]}) | {r['r2'][0]:.3f} | {r['qk'][0]:.3f} | "
                   f"{r['tsp39'][0]:.3f}, {r['tsp2'][0]:.3f} ({r['tsp39'][1]}) | {r['t'][0]:.2f} / {r['t'][1]:.2f} |")
    out += ["", "## Runnable commuting tests (log10 ESP C39 - R1, R2 - R1, C39 - QK3; two-qubit C39 / R1 / R2; chosen)", ""]
    for (d, t), x in sorted(by.items()):
        if all(k in x and "error" not in x[k] for k in ARMS) and generated()[t][2] and max(esp(x[k]) for k in ARMS) >= 0.01:
            st = x["C39"].get("steiner_stats") or {}
            ch = "steiner" if st.get("chosen") else ("usual" if st.get("kept_usual") else "-")
            out.append(f"- {d[4:]} {t}: {L(x['R1']) - L(x['C39']):+.4f}, {L(x['R1']) - L(x['R2']):+.4f}, "
                       f"{L(x['QK3']) - L(x['C39']):+.4f}; {x['C39']['n2']} / {x['R1']['n2']} / {x['R2']['n2']}; {ch}")
    txt = "\n".join(out) + "\n"
    open(os.path.join(a.out, "c39_val.md"), "w", encoding="utf-8", newline="\n").write(txt)
    json.dump(res, open(os.path.join(a.out, "c39_val_summary.json"), "w"), default=str)
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
