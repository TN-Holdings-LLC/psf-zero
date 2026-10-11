"""c38_val.py -- C38-VAL: should a Pauli evolution whose terms all commute be compiled by STEINER6 (candidate
2026-10-11.c38, changelog item 68)? Tested on circuits STEINER6 was never developed on.

Circuits:
  - HamLib: the instances of Benchpress's 100_representative.json that are not among BP-FINAL's 67 HamLib tests
    (STEINER6's development set) and fit on the device, built as Benchpress's FakeTorino HamLib test builds them;
  - generated (fixed seeds, `generated()`): MaxCut-type Z + ZZ on random 3-regular graphs (commuting), Z + ZZ + ZZZ
    "3-SAT-type" (commuting); controls that do not commute: transverse-field Ising on random 3-regular graphs, and
    Heisenberg rings.
Devices: FakeTorino, FakeKingston, FakeMarrakesh, FakeFez, FakeAachen, FakeBoston.
Arms, each test and device in its own process, in this order: QK3 (Qiskit level 3, seed 0), REL (psf_compile.py, the
plain call with backend=), C38 (the candidate, the same call).

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
C38 = os.environ.get("C38_FILE", os.path.join(REPO, "patches", "psf_compile_c38_2026-10-11", "psf_compile.py"))
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
DEVICES = ("FakeTorino", "FakeKingston", "FakeMarrakesh", "FakeFez", "FakeAachen", "FakeBoston")
ARMS = ("QK3", "REL", "C38")
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
    for n in (10, 16, 22, 28, 34, 40):
        for s in (1, 2):
            rng = random.Random(f"C38-VAL|maxcut|{n}|{s}")
            terms = [(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a, b in _regular3(n, rng)]
            terms += [(_label(n, {q: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
            out[f"gen_maxcut3reg_n{n}_s{s}"] = (n, terms, True)
    for n in (12, 20, 30):
        for s in (1, 2):
            rng = random.Random(f"C38-VAL|sat3|{n}|{s}")
            terms = {}
            for _ in range(3 * n):
                a, b, c = rng.sample(range(n), 3)
                for sub in ((a,), (b,), (c,), (a, b), (a, c), (b, c), (a, b, c)):
                    lb = _label(n, {q: "Z" for q in sub})
                    terms[lb] = terms.get(lb, 0.0) + rng.uniform(-1, 1)
            out[f"gen_sat3_n{n}_s{s}"] = (n, sorted(terms.items()), True)
    for n in (10, 20, 30):
        rng = random.Random(f"C38-VAL|tfim|{n}")
        terms = [(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a, b in _regular3(n, rng)]
        terms += [(_label(n, {q: "X"}), rng.uniform(-1, 1)) for q in range(n)]
        out[f"gen_tfim3reg_n{n}"] = (n, terms, False)
    for n in (8, 16):
        rng = random.Random(f"C38-VAL|heis|{n}")
        terms = [(_label(n, {q: p, (q + 1) % n: p}), rng.uniform(-1, 1)) for q in range(n) for p in "XYZ"]
        out[f"gen_heisenberg_ring_n{n}"] = (n, terms, False)
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
    tests = [(t, a) for t, a in hamlib_new(bp)] + [(g, None) for g in generated()]
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
    mods = {"REL": load(os.path.join(REPO, "psf_compile.py"), "psf_rel"), "C38": load(C38, "psf_c38")}
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
        if arm == "C38":
            rec["steiner_stats"] = dict(mods["C38"].STEINER_STATS)
            rec["same_as_rel"] = sigs.get("C38") == sigs.get("REL")
        print(json.dumps(rec), flush=True)


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    js = jobs(a.bp)
    if a.smoke:
        js = [j for j in js if j[0] in ("gen_maxcut3reg_n10_s1", "gen_tfim3reg_n10")][:4]
    js.sort(key=lambda j: hashlib.sha256(("C38-VAL|" + j[0] + j[2]).encode()).hexdigest())
    os.makedirs(a.out)
    import shutil
    shutil.copyfile(C38, os.path.join(a.out, "psf_compile_c38_as_run.py"))
    path = os.path.join(a.out, "c38_val.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), jobs=len(js),
                                           par=a.par, c38_sha256=hashlib.sha256(open(C38, "rb").read()).hexdigest(),
                                           rel_sha256=hashlib.sha256(open(os.path.join(REPO, "psf_compile.py"), "rb")
                                                                     .read()).hexdigest()))) + "\n")
    t_start = time.perf_counter()

    def go(job):
        test, arg, d = job
        base = dict(test=test, device=d)
        if time.perf_counter() - t_start > BUDGET:
            return [dict(base, arm=x, error="not started (budget)") for x in ARMS]
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--device", d,
                                "--job", json.dumps([test, arg])], capture_output=True, text=True, timeout=TIMEOUT,
                               cwd=REPO, env=dict(os.environ, C38_FILE=os.path.join(a.out,
                                                                                    "psf_compile_c38_as_run.py")))
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

def commuting_of(test, bp_cache={}):  # noqa: B006 - a cache
    if test.startswith("gen_"):
        return generated()[test][2]
    return None  # filled from C38's own record: STEINER used or "not_commuting"


def report(a):
    rows = [json.loads(x) for x in open(os.path.join(a.out, "c38_val.jsonl"), encoding="utf-8")][1:]
    by = {}
    for r in rows:
        by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
    L = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731
    out = ["# C38-VAL", "", f"jobs: {len(by)}; records with an error: {sum(1 for r in rows if 'error' in r)}", ""]
    for r in [r for r in rows if "error" in r][:10]:
        out.append(f"- {r['device']} {r['arm']} {r['test'][-50:]}: {r['error'][:150]}")

    def is_comm(x):
        c = x.get("C38", {})
        st = c.get("steiner_stats") or {}
        return st.get("used", 0) > 0 or st.get("fallback", 0) > 0

    def esp(r):
        return 0.0 if r["failed"] else 10 ** -L(r)
    res = {}
    out += ["", "| device | commuting tests (STEINER used) | V1 C38 on failed | V2 non-commuting: C38 = REL | "
            "runnable commuting | V3 ESP C38/REL | V4 ESP C38/QK3 | V5 2q C38/REL (all commuting) | "
            "V6 time C38/REL (commuting) | QK3 lost to failed (runnable) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d in DEVICES + ("all",):
        sel = [x for (dv, _), x in by.items() if (d == "all" or dv == d) and all(k in x and "error" not in x[k]
                                                                                for k in ARMS)]
        comm = [x for x in sel if is_comm(x)]
        noncomm = [x for x in sel if not is_comm(x)]
        v1 = sum(x["C38"]["failed"] > 0 for x in sel)
        v2 = (sum(1 for x in noncomm if x["C38"].get("same_as_rel")), len(noncomm))
        runn = [x for x in comm if max(esp(x[k]) for k in ARMS) >= 0.01]
        def gm(p, q, xs):
            ok = [x for x in xs if esp(x[p]) > 0 and esp(x[q]) > 0]
            return (10 ** (sum(L(x[q]) - L(x[p]) for x in ok) / len(ok)), len(ok)) if ok else (float("nan"), 0)
        v3, v4 = gm("C38", "REL", runn), gm("C38", "QK3", runn)
        v5 = math.exp(sum(math.log((x["C38"]["n2"] + 1) / (x["REL"]["n2"] + 1)) for x in comm) / len(comm)) \
            if comm else float("nan")
        tr = sum(x["REL"]["t"] for x in comm)
        v6 = sum(x["C38"]["t"] for x in comm) / tr if tr else float("nan")
        lost = sum(1 for x in runn if x["QK3"]["failed"])
        res[d] = dict(v1=v1, v2=v2, v3=v3, v4=v4, v5=v5, v6=v6, comm=len(comm), runn=len(runn))
        out.append(f"| {d} | {len(comm)} | {v1} | {v2[0]} / {v2[1]} | {len(runn)} | {v3[0]:.3f} ({v3[1]}) | "
                   f"{v4[0]:.3f} ({v4[1]}) | {v5:.3f} | {v6:.2f} | {lost} |")
    out += ["", "## Commuting runnable tests, per test (log10 ESP C38 - REL, C38 - QK3; two-qubit C38 / REL / QK3)", ""]
    for (d, t), x in sorted(by.items()):
        if all(k in x and "error" not in x[k] for k in ARMS) and is_comm(x) and max(esp(x[k]) for k in ARMS) >= 0.01:
            out.append(f"- {d[4:]} {t[-45:]}: {L(x['REL']) - L(x['C38']):+.4f}, {L(x['QK3']) - L(x['C38']):+.4f}; "
                       f"{x['C38']['n2']} / {x['REL']['n2']} / {x['QK3']['n2']}")
    txt = "\n".join(out) + "\n"
    open(os.path.join(a.out, "c38_val.md"), "w", encoding="utf-8", newline="\n").write(txt)
    json.dump(res, open(os.path.join(a.out, "c38_val_summary.json"), "w"), default=str)
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
