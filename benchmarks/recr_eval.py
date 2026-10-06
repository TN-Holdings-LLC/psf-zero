"""recr_eval.py -- test RECR (readout and ecr), 2026-10-06: do the workplace candidates psf_compile 2026-10-05.c14
(items 40-41: readout of measured qubits in hybrid_cost; direction-aware failed-element check) and psf_ai_compile
2026-10-05.a11 (items 15-16: readout in the state-aware estimate; gate direction kept on directional devices), on top of
release 2026-10-05.1 and front end a9, give sampled circuits more faithful output distributions, stay exact, change
nothing without measurements, and keep every gate on the target, on cx, cz and ecr devices?

Devices: FakeAuckland, FakeHanoiV2 (cx); FakeTorino, FakeKingston (cz); FakeBrussels, FakeOsaka (ecr, one direction per
coupler).
Circuits per device (the same on every device): MODEL-RO2's generator (data/2026-10-05/workplace/model_ro2/ai10_eval2.py,
_family and _model_style, unchanged), 6 per (family, n) cell = 96, seed base 74,000,000; plus 8 classifier circuits
(DEPTH's depth_eval.circuit, n 4 and 6, L 4), seed base 74,400,000. Smoke: 1 per cell for the first 8 cells and 2
classifier circuits, seed bases 74,500,000 and 74,900,000.
Arms (FULL = the recommended call; M = compiled with measure_all()):
  R51   release 2026-10-05.1, FULL, without measurements; measured afterwards at the final layout (the old practice)
  R51M  release, FULL, with measurements
  C14   c14, FULL, without measurements (compared with R51 instruction by instruction, not simulated)
  C14M  c14, FULL, with measurements
  A9    front end a9 on the release, without measurements (compared with A11, not simulated)
  A11   a11 on c14, without measurements (compared with A9; off-target counted)
  A9M   a9 on the release, with measurements
  A11M  a11 on c14, with measurements
  L3TM  Qiskit level 3 with the Target, approximation_degree 1.0, with measurements
Per simulated arm (R51, R51M, C14M, A9M, A11M, L3TM): state infidelity of the compiled circuit without its measurements
against the logical circuit (readout_eval.state_infid, the Addendum-340 probe's check); whether clbit j measures logical
j's final qubit; off-target instructions; two-qubit gates in a failed direction or on a failed qubit; the summed Target
measure error of the measured qubits; the classical infidelity 1 - (sum_x sqrt(p_x q_x))^2 between the ideal output
distribution and the noisy one (MODEL-RO2's metric: Aer density matrix with the device noise restricted to the touched
qubits, then each qubit's readout assignment); two-qubit count; compile time. Aer's noise model has no entry for an
instruction the device does not provide, so an off-target gate is simulated as error-free: off-target counts are
scored on their own (workplace ecr exploration, 2026-10-05).

    python benchmarks/recr_eval.py run --device D --out DIR [--smoke]
    python benchmarks/recr_eval.py score --out DIR
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
for _p in (os.path.join(WORK, "model_ro2"), os.path.join(WORK, "readout"), os.path.join(WORK, "depth1"), HERE, REPO):
    sys.path.insert(0, _p)

CX, CZ, ECR = ("FakeAuckland", "FakeHanoiV2"), ("FakeTorino", "FakeKingston"), ("FakeBrussels", "FakeOsaka")
DEVICES = CX + CZ + ECR
SIM_ARMS = ("R51", "R51M", "C14M", "A9M", "A11M", "L3TM")
MEASURED = ("R51", "R51M", "C14M", "A9M", "A11M", "L3TM")
FULL = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
            candidate_score="hybrid")
C14_PATH = os.path.join(REPO, "patches", "psf_compile_c14_2026-10-05", "psf_compile.py")
A11_PATH = os.path.join(REPO, "patches", "psf_ai_compile_a11_2026-10-05", "psf_ai_compile.py")
WANT = dict(release="2026-10-05.1", c14="2026-10-05.c14", a9="2026-10-05.a9", a11="2026-10-05.a11")
EXACT = 1e-6


def norm_sha(path):
    t = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in t.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load():
    """Release and a9 (a9 sees the release as psf_compile), then c14 and a11 (a11 sees c14), then the helpers."""
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_recr")
    sys.modules["psf_smart_layout"] = lay
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_recr")
    sys.modules["psf_compile"] = rel
    a9 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile_a9_recr")
    c14 = H.load_module(C14_PATH, "psf_compile_c14_recr")
    sys.modules["psf_compile"] = c14
    a11 = H.load_module(A11_PATH, "psf_ai_compile_a11_recr")
    got = dict(release=rel.VERSION, c14=c14.VERSION, a9=a9.AI_COMPILE_VERSION, a11=a11.AI_COMPILE_VERSION)
    if got != WANT or a9.pc is not rel or a11.pc is not c14:
        raise SystemExit(f"STOP: versions or module wiring not as locked: {got}")
    import depth_eval as DE
    import readout_eval as RE
    import ai10_eval2 as E2
    return rel, c14, a9, a11, lay, DE, RE, E2


def circuits(E2, DE, smoke):
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    out, k = [], 0
    cells = [(name, n) for name, ns in E2.FAMILIES for n in ns]
    base = 74_500_000 if smoke else 74_000_000
    for name, n in (cells[:8] if smoke else cells):
        for _ in range(1 if smoke else 6):
            rng = np.random.default_rng(base + k)
            k += 1
            qc = E2._model_style(E2._family(name, n, rng), rng)
            for q in range(n):
                qc.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
            out.append((f"{name}{n}", qc))
    cbase = 74_900_000 if smoke else 74_400_000
    for j, n in enumerate((4, 6) if smoke else (4, 4, 4, 4, 6, 6, 6, 6)):
        rng = np.random.default_rng(cbase + j)
        out.append((f"CLS{n}", DE.circuit(rng.uniform(-1, 1, n), rng.normal(0, 1, DE.n_params(n, 4)), n, 4)))
    return out


def measured_after(out, n):
    """The old practice: compile without measurements, then measure logical j's final qubit into clbit j."""
    from qiskit.circuit import ClassicalRegister
    m = out.copy()
    m.add_register(ClassicalRegister(n, "meas"))
    fin = list(out.layout.final_index_layout(filter_ancillas=True))
    for j in range(n):
        m.measure(fin[j], m.clbits[j])
    m._layout = out._layout
    return m


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def git_head():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def run(args):
    import qiskit
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_ibm_runtime import fake_provider
    rel, c14, a9, a11, lay, DE, RE, E2 = load()
    be = getattr(fake_provider, args.device)()
    t = be.target
    cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    base = dict(coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True, seed_transpiler=0,
                target=t)
    comp = {
        "R51": lambda qc: rel.compile_for_hardware(qc, **base, **FULL),
        "R51M": lambda qc: rel.compile_for_hardware(qc, **base, **FULL),
        "C14": lambda qc: c14.compile_for_hardware(qc, **base, **FULL),
        "C14M": lambda qc: c14.compile_for_hardware(qc, **base, **FULL),
        "A9": lambda qc: a9.compile_for_model_circuit(qc, cm, basis, target=t),
        "A11": lambda qc: a11.compile_for_model_circuit(qc, cm, basis, target=t),
        "A9M": lambda qc: a9.compile_for_model_circuit(qc, cm, basis, target=t),
        "A11M": lambda qc: a11.compile_for_model_circuit(qc, cm, basis, target=t),
        "L3TM": lambda qc: transpile(qc, target=t, optimization_level=3, seed_transpiler=0, approximation_degree=1.0),
    }
    edges, fq = rel._failed_elements(t, 0.5)
    sim = DE.Noisy(be)
    rows, t0 = [], time.time()
    for name, qc0 in circuits(E2, DE, args.smoke):
        n = qc0.num_qubits
        qcm = qc0.copy()
        qcm.measure_all()
        ideal = Statevector(qc0).probabilities()
        row = dict(name=name, n=n)
        outs = {}
        for arm, f in comp.items():
            qc = qcm if arm.endswith("M") else qc0
            t1 = time.perf_counter()
            try:
                out = quiet(f, qc)
            except Exception as e:  # recorded; P0 requires none
                row[arm] = dict(error=f"{type(e).__name__}: {e}"[:300])
                continue
            cs = time.perf_counter() - t1
            off = failed = 0
            for ins in out.data:
                nm = ins.operation.name
                if nm in ("barrier", "measure", "delay"):
                    continue
                q = tuple(out.find_bit(b).index for b in ins.qubits)
                if nm not in t.operation_names or q not in t[nm]:
                    off += 1
                if any(i in fq for i in q) or (len(q) == 2 and (q in edges or (nm == "cz" and q[::-1] in edges))):
                    failed += 1
            r = dict(compile_s=round(cs, 4), off_target=off, failed_uses=failed,
                     n2q=sum(1 for g in out.data if len(g.qubits) == 2 and g.operation.name != "barrier"))
            outs[arm] = out
            if arm in SIM_ARMS:
                m = measured_after(out, n) if arm == "R51" else out
                fin = list(m.layout.final_index_layout(filter_ancillas=True))
                p0, mq, cl = E2.reduced_probs(m, sim, False)
                pn, _, _ = E2.reduced_probs(m, sim, True)
                qd = E2.apply_readout(pn, mq, sim)
                r.update(state_infid=RE.state_infid(qc0, RE.strip_measure(m)),
                         meas_ok=bool(cl == list(range(n)) and mq == fin[:n]),
                         meas_err=float(sum(t["measure"][(x,)].error for x in mq)),
                         infid=float(1 - np.sum(np.sqrt(np.clip(ideal, 0, None) * np.clip(qd, 0, None))) ** 2))
            row[arm] = r
        if "C14" in outs and "R51" in outs:
            row["C14"]["same_as_R51"] = RE.sig(outs["C14"]) == RE.sig(outs["R51"])
        if "A11" in outs and "A9" in outs:
            row["A11"]["same_as_A9"] = RE.sig(outs["A11"]) == RE.sig(outs["A9"])
        rows.append(row)
    meta = dict(device=args.device, smoke=bool(args.smoke), git_head=git_head(), qiskit=qiskit.__version__,
                versions=dict(release=rel.VERSION, c14=c14.VERSION, a9=a9.AI_COMPILE_VERSION,
                              a11=a11.AI_COMPILE_VERSION, layout=getattr(lay, "VERSION", None)),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), release=norm_sha(rel.__file__),
                         c14=norm_sha(C14_PATH), a9=norm_sha(a9.__file__), a11=norm_sha(A11_PATH),
                         ai10_eval2=norm_sha(E2.__file__), readout_eval=norm_sha(RE.__file__),
                         depth_eval=norm_sha(DE.__file__)),
                stats=dict(exact=dict(c14.EXACT_STATS), direction=dict(getattr(a11, "DIRECTION_STATS", {})),
                           l3t_check_a11=dict(a11.L3T_CHECK_STATS)),
                wall_s=round(time.time() - t0, 1))
    os.makedirs(args.out, exist_ok=True)
    name = f"recr_{args.device}{'_smoke' if args.smoke else ''}.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, name), "w"))
    print(f"wrote {name}: {len(rows)} circuits, {meta['wall_s']} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    files = {}
    for p in sorted(glob.glob(os.path.join(args.out, "recr_*.json"))):
        r = json.load(open(p))
        files[r["meta"]["device"]] = r
    smoke = any(r["meta"]["smoke"] for r in files.values())
    want_n = 10 if smoke else 104
    lines = [f"# RECR score{' (SMOKE)' if smoke else ''}", ""]
    errors = sum(1 for r in files.values() for row in r["rows"] for a in row if isinstance(row[a], dict)
                 and "error" in row[a])
    inexact = sum(1 for r in files.values() for row in r["rows"] for a in SIM_ARMS
                  if a in row and row[a].get("state_infid", 1) > EXACT)
    badmeas = sum(1 for r in files.values() for row in r["rows"] for a in MEASURED
                  if a in row and not row[a].get("meas_ok", False))
    counts_ok = all(len(files[d]["rows"]) == want_n for d in DEVICES if d in files)
    p0 = len(files) == len(DEVICES) and errors == 0 and inexact == 0 and badmeas == 0 and counts_ok
    lines.append(f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(files)} of {len(DEVICES)}; compile errors {errors}; "
                 f"inexact outputs {inexact}; measurement mapping wrong {badmeas}; circuit counts ok {counts_ok}")

    def mean(d, arm, key):
        return float(np.mean([row[arm][key] for row in files[d]["rows"]]))

    def med(d, arm):
        return float(np.median([row[arm]["compile_s"] for row in files[d]["rows"]]))

    def frac(d, arm, key):
        return float(np.mean([bool(row[arm].get(key)) for row in files[d]["rows"]]))

    def total(d, arm, key):
        return int(sum(row[arm][key] for row in files[d]["rows"]))

    if not p0:
        lines.append("Nothing below is scored.")
    else:
        lines += ["", "| device | arm | infid (mean) | summed measure error | off-target | failed uses | 2q | "
                  "median compile s |", "|---|---|---|---|---|---|---|---|"]
        for d in DEVICES:
            for a in SIM_ARMS:
                lines.append(f"| {d} | {a} | {mean(d, a, 'infid'):.5f} | {mean(d, a, 'meas_err'):.4f} | "
                             f"{total(d, a, 'off_target')} | {total(d, a, 'failed_uses')} | {mean(d, a, 'n2q'):.2f} | "
                             f"{med(d, a):.3f} |")
        V = []
        q1 = {d: mean(d, "R51M", "meas_err") / mean(d, "R51", "meas_err") for d in CZ}
        V.append(("Q1", "compiling with the measurements moves the release's measured qubits to better readout "
                  "(mean summed measure error R51M / R51 <= 0.85 on both cz devices; refuted >= 1.00 on either)",
                  verdict(all(v <= 0.85 for v in q1.values()), any(v >= 1.0 for v in q1.values())), q1))
        q2 = {d: mean(d, "C14M", "infid") / mean(d, "R51M", "infid") for d in DEVICES}
        q2e = {d: mean(d, "C14M", "meas_err") - mean(d, "R51M", "meas_err") for d in DEVICES}
        V.append(("Q2", "c14's readout term never hurts (C14M / R51M infid <= 1.005 and measure error difference "
                  "<= +0.0005 on every device; refuted infid ratio > 1.02 on any)",
                  verdict(all(v <= 1.005 for v in q2.values()) and all(v <= 5e-4 for v in q2e.values()),
                          any(v > 1.02 for v in q2.values())), dict(infid=q2, meas_err=q2e)))
        q3 = {d: frac(d, "C14", "same_as_R51") for d in DEVICES}
        V.append(("Q3", "without measurements c14 is the release (identical in >= 99.9% on every device; refuted "
                  "< 99% on any)", verdict(all(v >= 0.999 for v in q3.values()), any(v < 0.99 for v in q3.values())),
                  q3))
        q4 = {d: total(d, "A11M", "off_target") + total(d, "A11", "off_target") for d in DEVICES}
        V.append(("Q4", "a11 keeps every instruction on the target (A11 and A11M off-target 0 on every device; "
                  "refuted any)", verdict(all(v == 0 for v in q4.values()), any(v > 0 for v in q4.values())), q4))
        q5 = {d: total(d, "A9M", "off_target") for d in ECR}
        V.append(("Q5", "the a9 defect reproduces on ecr devices (A9M off-target >= 1 on both ecr devices; refuted 0 "
                  "on both)", verdict(all(v >= 1 for v in q5.values()), all(v == 0 for v in q5.values())), q5))
        q6 = {d: frac(d, "A11", "same_as_A9") for d in CX + CZ}
        V.append(("Q6", "without measurements a11 is a9 on the cx and cz devices (identical in >= 99.9% on each; "
                  "refuted < 99% on any)", verdict(all(v >= 0.999 for v in q6.values()),
                                                   any(v < 0.99 for v in q6.values())), q6))
        q7 = {d: mean(d, "A11M", "infid") / mean(d, "A9M", "infid") for d in CZ}
        V.append(("Q7", "a11 counts readout (A11M / A9M infid <= 0.80 on both cz devices; refuted >= 1.00 on either)",
                  verdict(all(v <= 0.80 for v in q7.values()), any(v >= 1.0 for v in q7.values())), q7))
        q8 = {d: mean(d, "A11M", "infid") / mean(d, "L3TM", "infid") for d in DEVICES}
        V.append(("Q8", "a11 is level with or ahead of Qiskit level 3 (A11M / L3TM <= 1.00 on >= 5 of 6 devices; "
                  "refuted > 1.05 on any)", verdict(sum(v <= 1.0 for v in q8.values()) >= 5,
                                                    any(v > 1.05 for v in q8.values())), q8))
        q9 = {d: mean(d, "C14M", "infid") / mean(d, "L3TM", "infid") for d in DEVICES}
        V.append(("Q9", "the release candidate with measurements is level with or ahead of Qiskit level 3 "
                  "(C14M / L3TM <= 1.00 on >= 5 of 6 devices; refuted > 1.05 on any)",
                  verdict(sum(v <= 1.0 for v in q9.values()) >= 5, any(v > 1.05 for v in q9.values())), q9))
        q10 = {d: med(d, "C14M") / med(d, "R51M") for d in DEVICES}
        q10.update({d + " (A11M/A9M)": med(d, "A11M") / med(d, "A9M") for d in CX + CZ})
        V.append(("Q10", "both cost nothing (median compile time C14M / R51M on every device and A11M / A9M on the cx "
                  "and cz devices <= 1.15; refuted > 1.50 on any)",
                  verdict(all(v <= 1.15 for v in q10.values()), any(v > 1.5 for v in q10.values())), q10))
        lines += ["", "## Predictions", ""]
        for qid, text, v, num in V:
            lines.append(f"- {qid} ({text}): **{v}** -- {json.dumps(num, default=lambda x: round(x, 4))}")
        lines += ["", "Reported: A9M on the ecr devices is simulated with its off-target gates error-free; its infid "
                  "there is not comparable.",
                  "Reported: check counters by device: " + json.dumps({d: files[d]["meta"]["stats"] for d in DEVICES})]
    out = "\n".join(lines) + "\n"
    print(out)
    open(os.path.join(args.out, "score.md"), "w").write(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", required=True, choices=DEVICES)
    r.add_argument("--out", required=True)
    r.add_argument("--smoke", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    main()
