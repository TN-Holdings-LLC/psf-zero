"""region_diag.py -- exploratory diagnosis (2026-10-02, not a test): why does Qiskit's level-1 error-aware layout
stage (candidate c4, Addendum 307) choose regions that are better on FakeAuckland but worse on FakeKingston?

For every circuit of the c4 run (same generator, same seeds), the three arms C3, C4 and L3T are compiled again
(compile only, no simulation) and the placement each one chose is scored three ways, per gate actually placed:

  S_rep   sum of -log(1 - e) with e the Target's reported error of that gate on those qubits;
  S_eff   the same with e = 1 - average gate fidelity of the QuantumError that NoiseModel.from_backend attaches to
          that gate on those qubits (the error the simulation actually applies; floor-aware);
  S_avg   the same with e from a replica of Qiskit's average error map (per qargs, the mean reported error over
          every instruction on those qargs, which for one qubit includes measure, i.e. readout).

The measured infidelities are read from the c4 run (data/2026-10-02/c4/outputs); a row is used only if the
recompiled two-qubit count equals the recorded one. The question is which score orders the arms as the measured
infidelity does.

  python region_diag.py run --repo <repo> --out <dir> --device <d>
  python region_diag.py summary --out <dir>
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston")
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
ARMS = ("C3", "C4", "L3T")
SCORES = ("S_rep", "S_eff", "S_avg")
CAND = os.path.join("patches", "psf_compile_c4_2026-10-02", "psf_compile.py")
RUN = os.path.join("data", "2026-10-02", "c4", "outputs")


def error_tables(backend):
    """Per (gate name, qargs): reported error, applied (noise-model) error, and Qiskit-style average error."""
    from qiskit.quantum_info import average_gate_fidelity
    from qiskit_aer.noise import NoiseModel
    tgt = backend.target
    nm = NoiseModel.from_backend(backend)
    local = getattr(nm, "_local_quantum_errors", {})
    rep, eff = {}, {}
    for name in tgt.operation_names:
        if name in ("measure", "delay", "reset", "barrier"):
            continue
        for qargs, props in tgt[name].items():
            if qargs is None:
                continue
            q = tuple(qargs)
            rep[(name, q)] = float(props.error) if props is not None and props.error is not None else 0.0
            err = local.get(name, {}).get(q)
            eff[(name, q)] = 1.0 - float(average_gate_fidelity(err.to_quantumchannel())) if err is not None else 0.0
    avg = {}
    for qargs in tgt.qargs or []:
        if qargs is None:
            continue
        vals = [tgt[op][qargs].error for op in tgt.operation_names_for_qargs(qargs)
                if tgt[op].get(qargs) is not None and tgt[op][qargs].error is not None]
        if vals:
            avg[tuple(qargs)] = float(np.mean(vals))
    return rep, eff, avg, len(local)


def nlog(e):
    return -math.log(max(1.0 - e, 1e-300))


def score(out, rep, eff, avg):
    s = dict(S_rep=0.0, S_eff=0.0, S_avg=0.0)
    qubits, edges = set(), {}
    for ins in out.data:
        name = ins.operation.name
        if name in ("barrier", "measure", "delay"):
            continue
        q = tuple(out.find_bit(b).index for b in ins.qubits)
        qubits.update(q)
        if len(q) == 2:
            k = tuple(sorted(q))
            edges[k] = edges.get(k, 0) + 1
        s["S_rep"] += nlog(rep.get((name, q), rep.get((name, q[::-1]), 0.0)))
        s["S_eff"] += nlog(eff.get((name, q), eff.get((name, q[::-1]), 0.0)))
        if name != "rz":  # virtual; Qiskit's map would charge the qubit's average, which mixes in readout
            s["S_avg"] += nlog(avg.get(q, avg.get(q[::-1], 0.0)))
    return s, sorted(qubits), sorted([list(k) + [v] for k, v in edges.items()])


def run(args):
    from qiskit import transpile
    from qiskit_ibm_runtime import fake_provider
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    sys.path.insert(0, args.repo)
    import core_fix_c2_eval as H
    import gap_eval as G
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    mod = {"C3": H.load_module(os.path.join(args.repo, "psf_compile.py"), "psf_compile"),
           "C4": H.load_module(os.path.join(args.repo, CAND), "psf_compile_c4")}
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    rep, eff, avg, nloc = error_tables(backend)
    print("tables: rep %d, eff %d (noise-model local entries %d), avg %d" % (len(rep), sum(v > 0 for v in eff.values()),
                                                                           nloc, len(avg)), flush=True)
    if nloc == 0:
        raise SystemExit("STOP: NoiseModel._local_quantum_errors not found; this Aer version stores errors elsewhere")
    t0 = time.time()
    rows, mismatch = [], 0
    for fam in FAMILIES:
        circs = list(G.family(fam, False))
        rec = {a: json.load(open(os.path.join(args.repo, RUN, "c4_%s_%s_%s.json" % (args.device, a, fam))))["rows"]
               for a in ARMS}
        for i, (params, qc) in enumerate(circs):
            row = dict(family=fam, params=params)
            for a in ARMS:
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if a == "L3T":
                        out = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)
                    else:
                        kw = dict(target=tgt) if a == "C3" else dict(target=tgt, error_aware_layout=True)
                        out = mod[a].compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                                          layout_search=True, seed_transpiler=0, **kw)
                s, qubits, edges = score(out, rep, eff, avg)
                r = rec[a][i]
                two_q = sum(1 for ins in out.data if len(ins.qubits) == 2)
                ok = r["params"] == params and r["two_q"] == two_q and "infid" in r
                mismatch += not ok
                row[a] = dict(s, infid=r.get("infid") if ok else None, two_q=two_q, qubits=qubits, edges=edges)
            rows.append(row)
        print("  %s %s: %d circuits, %.0f s" % (args.device, fam, len(circs), time.time() - t0), flush=True)
    with open(os.path.join(args.out, "diag_%s.json" % args.device), "w") as f:
        json.dump(dict(device=args.device, mismatch=mismatch, rows=rows), f)
    print("wrote diag_%s.json: %d circuits, %d arm-rows not matching the c4 run" % (args.device, len(rows), mismatch))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def agree(rows, a, b, key):
    """Fraction of circuits where the score difference a-b has the sign of the measured infidelity difference."""
    n = k = 0
    for r in rows:
        x, y = r[a], r[b]
        if x["infid"] is None or y["infid"] is None or abs(x["infid"] - y["infid"]) <= 1e-9:
            continue
        n += 1
        k += (x[key] - y[key]) * (x["infid"] - y["infid"]) > 0
    return k / n if n else float("nan"), n


def summary(args):
    L = ["# region_diag summary (exploratory, not a test)", ""]
    for d in DEVICES:
        p = os.path.join(args.out, "diag_%s.json" % d)
        if not os.path.exists(p):
            L.append("%s: missing" % d)
            continue
        R = json.load(open(p))
        rows = R["rows"]
        L += ["## %s (%d circuits; arm-rows not matching the c4 run: %d)" % (d, len(rows), R["mismatch"]), ""]
        L += ["| score | Spearman with infidelity (all arms) | C4 vs C3 sign agreement | C4 vs L3T | C3 vs L3T |",
              "|---|---|---|---|---|"]
        for key in SCORES:
            xs = [r[a][key] for r in rows for a in ARMS if r[a]["infid"] is not None]
            ys = [r[a]["infid"] for r in rows for a in ARMS if r[a]["infid"] is not None]
            cells = []
            for a, b in (("C4", "C3"), ("C4", "L3T"), ("C3", "L3T")):
                f, n = agree(rows, a, b, key)
                cells.append("%.3f (n=%d)" % (f, n))
            L.append("| %s | %.3f | %s |" % (key, spearman(xs, ys), " | ".join(cells)))
        L += ["", "| family | arm | mean infid | S_rep | S_eff | S_avg | S_eff - S_rep |", "|---|---|---|---|---|---|---|"]
        for fam in FAMILIES:
            fr = [r for r in rows if r["family"] == fam]
            for a in ARMS:
                v = [r[a] for r in fr if r[a]["infid"] is not None]
                if not v:
                    continue
                m = lambda k: np.mean([x[k] for x in v])
                L.append("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f |" % (fam, a, m("infid"), m("S_rep"), m("S_eff"),
                                                                         m("S_avg"), m("S_eff") - m("S_rep")))
        L.append("")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("run", "summary"))
    ap.add_argument("--repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=DEVICES)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (run if args.part == "run" else summary)(args)


if __name__ == "__main__":
    main()
