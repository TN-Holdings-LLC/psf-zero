"""a246_check.py -- A246-FE (2026-10-10; a check of a past record, nothing predicted): did the outputs of Addendum
246 (the layout cliff on ibm_kingston's live Target, 2026-09-28) put operations on elements that Target reports as
failed? KT (2026-10-10) found that Target keeps 4 couplers and 3 qubits with error 1 (and qubit 146's measurement
at 0.504).

The 12 circuits of Addendum 246 (spare 0, 2, 8, 16; inputs 0-2; built as benchmarks/real_target_cliff.py's build() builds
them, with loop_endurance.add_pair24 copied here) are compiled again from the pickled Target, each compile in
its own process. No service call, no job.
  Q3   qiskit.transpile(qc, target=target, optimization_level=3, seed_transpiler=0)  (as Addendum 246)
  P0   Addendum 246's PSF-Zero call, with the psf_compile.py that run loaded (2026-09-28.1, found in git history
       by its normalized hash) and the Rust core installed now
  PR   the same call with release 2026-10-10.2 (no Target: the default call)
  C35  candidate c35 given the Target (target=target; on_failed_elements="raise", its default)
Q3 and P0 are first compared with Addendum 246's record: the two-qubit count and the per-pair operator distance
(its pair_check, to 1e-6 relative); where both match, the re-run is taken as Addendum 246's output.

    cd <psf-zero repository>
    python <this file> run --pkl PICKLE --old OLD_SOURCE_DIR --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import pickle
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
C35 = os.path.join(REPO, "patches", "psf_compile_c35_2026-10-10")
RECORD = os.path.join(REPO, "data", "2026-09-28", "real_target_cliff", "real_target_cliff_2026-09-28.csv")
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
FAILED = 0.5
ARMS = ("Q3", "P0", "PR", "C35")
CASES = [(spare, k) for spare in (0, 2, 8, 16) for k in range(3)]
TIMEOUT = 600


def failed_sets(target):
    g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    edges = {tuple(sorted(q)) for q, p in target[g2].items()
             if q is not None and p is not None and p.error is not None and p.error >= FAILED}
    qubits = set()
    for name in ("sx", "x", "measure"):
        if name in target.operation_names:
            qubits |= {q[0] for q, p in target[name].items()
                       if q is not None and p is not None and p.error is not None and p.error >= FAILED}
    return edges, qubits


def audit(out, target):
    edges, qubits = failed_sets(target)
    lg, dead, n_failed_ops = 0.0, False, 0
    used_q, used_e = set(), set()
    for ins in out.data:
        if ins.operation.name in ("barrier", "delay"):
            continue
        qa = tuple(out.find_bit(q).index for q in ins.qubits)
        used_q |= set(qa) & qubits
        if len(qa) == 2 and tuple(sorted(qa)) in edges:
            used_e.add(tuple(sorted(qa)))
        try:
            p = target[ins.operation.name][qa]
        except KeyError:
            continue
        e = p.error if p is not None and p.error is not None else 0.0
        n_failed_ops += e >= FAILED
        if e >= 1.0:
            dead = True
        else:
            lg += math.log10(1.0 - e)
    return dict(failed_ops=n_failed_ops, failed_qubits_used=sorted(used_q),
                failed_couplers_used=[list(e) for e in sorted(used_e)], log10_esp=None if dead else round(lg, 4))


def add_pair24(qc, a, b, p):
    """benchmarks/loop_endurance.py's add_pair24 (copied, so that loading that module does not import a
    psf_compile): two rounds of (local ZYZ on both, canonical core), then a final local layer."""
    k = 0
    for _ in range(2):
        for q in (a, b):
            qc.rz(p[k], q)
            qc.ry(p[k + 1], q)
            qc.rz(p[k + 2], q)
            k += 3
        qc.rxx(p[k], a, b)
        qc.ryy(p[k + 1], a, b)
        qc.rzz(p[k + 2], a, b)
        k += 3
    for q in (a, b):
        qc.rz(p[k], q)
        qc.ry(p[k + 1], q)
        qc.rz(p[k + 2], q)
        k += 3


def one(a):
    warnings.simplefilter("ignore")
    spare, k = json.loads(a.case)
    with open(a.pkl, "rb") as f:
        target = pickle.load(f)
    if a.arm == "P0":
        sys.path[:0] = [a.old, os.path.join(a.old, "benchmarks")]
    elif a.arm == "C35":
        sys.path[:0] = [C35, REPO, os.path.join(REPO, "benchmarks")]
    else:
        sys.path[:0] = [REPO, os.path.join(REPO, "benchmarks")]
    import numpy as np
    n = 2 * 64 - spare  # Addendum 246: maximum matching 64 pairs; n = 2M - spare
    rng = np.random.default_rng(1000 * spare + k)  # real_target_cliff.py's build(), seed 1000 * spare + input
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
    for j in range(n // 2):
        add_pair24(qc, 2 * j, 2 * j + 1, th[j])
    native = [g for g in target.operation_names if g in NATIVE]
    cmap = target.build_coupling_map()
    rec = dict(spare=spare, input=k, n=n, arm=a.arm)
    t0 = time.perf_counter()
    try:
        if a.arm == "Q3":
            from qiskit import transpile
            out = transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
        else:
            import psf_compile as pc
            rec["loaded"] = ("old/" + os.path.relpath(pc.__file__, a.old)) if a.arm == "P0" else os.path.relpath(pc.__file__, REPO)
            rec["version"] = pc.VERSION
            rec["core"] = getattr(pc, "CORE_VERSION", None)
            with contextlib.redirect_stdout(io.StringIO()):
                if a.arm == "C35":
                    pc.WARN_WITHOUT_TARGET = False
                    out = pc.compile_for_hardware(qc, target=target, basis_gates=native, entangling_basis="cx",
                                                  layout_search=True, on_unsupported="raise", seed_transpiler=0)
                else:
                    out = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=native, entangling_basis="cx",
                                                  layout_search=True, on_unsupported="raise", seed_transpiler=0)
    except Exception as exc:  # noqa: BLE001 - recorded (C35 raises FailedElementsError when it cannot avoid them)
        rec.update(error=f"{type(exc).__name__}: {exc}"[:300], t=round(time.perf_counter() - t0, 3))
        print(json.dumps(rec), flush=True)
        return
    rec["t"] = round(time.perf_counter() - t0, 3)
    rec["q2"] = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "delay"))
    rec.update(audit(out, target))
    import importlib.util  # Addendum 246's own per-pair check (its script, hash 11d2ce81...), after psf_compile is loaded
    spec = importlib.util.spec_from_file_location("rtc_246", os.path.join(REPO, "benchmarks", "real_target_cliff.py"))
    rtc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rtc)
    try:
        rec["pair_applicable"], rec["pair_worst"] = rtc.pair_check(qc, out, n)
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["pair_applicable"], rec["pair_worst"] = None, f"{type(exc).__name__}: {exc}"[:120]
    print(json.dumps(rec), flush=True)


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    recorded, recorded_worst = {}, {}
    with open(RECORD, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] == "OK":
                recorded[(int(r["spare"]), int(r["input"]), r["arm"])] = int(r["twoq"])
                recorded_worst[(int(r["spare"]), int(r["input"]), r["arm"])] = float(r["pair_worst"])
    os.makedirs(a.out)
    jobs = [(c, arm) for c in CASES for arm in ARMS]

    def go(job):
        (spare, k), arm = job
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--pkl", a.pkl, "--old", a.old,
                                "--arm", arm, "--case", json.dumps([spare, k])], capture_output=True, text=True,
                               timeout=TIMEOUT, cwd=REPO, env=dict(os.environ, PSF_ZERO_REPO=REPO))
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            return json.loads(lines[-1]) if lines else dict(spare=spare, input=k, arm=arm,
                                                            error=(p.stderr or "")[-300:])
        except subprocess.TimeoutExpired:
            return dict(spare=spare, input=k, arm=arm, error=f"timeout {TIMEOUT} s")

    recs = []
    with ThreadPoolExecutor(4) as ex, open(os.path.join(a.out, "a246_check.jsonl"), "w", encoding="utf-8",
                                           newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in jobs]):
            r = f.result()
            old = recorded.get((r["spare"], r["input"], {"Q3": "Q3", "P0": "P"}.get(r["arm"], "-")))
            key = (r["spare"], r["input"], {"Q3": "Q3", "P0": "P"}.get(r["arm"], "-"))
            if old is not None and "q2" in r:
                w = recorded_worst[key]
                same_w = isinstance(r.get("pair_worst"), float) and math.isclose(r["pair_worst"], w, rel_tol=1e-6,
                                                                                abs_tol=1e-15)
                r.update(recorded_q2=old, recorded_pair_worst=w, matches_record=r["q2"] == old and same_w)
            recs.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"spare {r['spare']:2d} input {r['input']} {r['arm']:3s}: " + (
                r["error"][:90] if "error" in r else
                f"q2 {r['q2']}{' (recorded ' + str(old) + ')' if old is not None else ''}, failed ops "
                f"{r['failed_ops']}, failed qubits {r['failed_qubits_used']}, failed couplers "
                f"{r['failed_couplers_used']}, log10 ESP {r['log10_esp']}"), flush=True)
    report(a.out, recs)


def report(out, recs):
    L = ["# A246-FE: did Addendum 246's outputs use ibm_kingston's failed elements? (nothing predicted)", "",
         "Failed elements of the Target pickled on 2026-09-28 (KT): couplers (112,113), (130,131), (145,146), "
         "(146,147) and qubits 113, 121, 146 at error 1; qubit 146's measurement 0.504. ESP = 0 (log10 None) when an "
         "operation of error 1 is used.", "",
         "| arm | compiled | matches Addendum 246 (two-qubit count and per-pair distance) | outputs using a failed element | ESP = 0 | "
         "raised / errors |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        rs = [r for r in recs if r["arm"] == arm]
        ok = [r for r in rs if "q2" in r]
        mt = [r for r in ok if "matches_record" in r]
        L.append(f"| {arm} | {len(ok)} of {len(rs)} | "
                 + (f"{sum(r['matches_record'] for r in mt)} of {len(mt)}" if mt else "-")
                 + f" | {sum(1 for r in ok if r['failed_ops'] or r['failed_qubits_used'] or r['failed_couplers_used'])}"
                 f" | {sum(1 for r in ok if r['log10_esp'] is None)} | {len(rs) - len(ok)} |")
    L += ["", "| spare | input | n | arm | q2 | recorded | matches | failed qubits used | failed couplers used | log10 ESP | "
          "error |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda r: (r["spare"], r["input"], ARMS.index(r["arm"]))):
        L.append(f"| {r['spare']} | {r['input']} | {r.get('n', '-')} | {r['arm']} | {r.get('q2', '-')} | "
                 f"{r.get('recorded_q2', '-')} | {r.get('matches_record', '-')} | {r.get('failed_qubits_used', '-')} | "
                 f"{r.get('failed_couplers_used', '-')} | {r.get('log10_esp', '-')} | {r.get('error', '')[:80]} |")
    v = sorted({(r.get("version"), r.get("core")) for r in recs if r["arm"] in ("P0", "PR", "C35") and "version" in r})
    L += ["", f"PSF-Zero versions loaded (version, core): {v}"]
    txt = "\n".join(L) + "\n"
    with open(os.path.join(out, "a246_check.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one"))
    ap.add_argument("--pkl")
    ap.add_argument("--old")
    ap.add_argument("--out")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--case")
    a = ap.parse_args()
    {"run": run, "one": one}[a.mode](a)


if __name__ == "__main__":
    main()
