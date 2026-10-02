"""a5_f3_diag.py -- exploratory diagnosis (2026-10-02, not a test): why does psf_ai_compile a5 lose to Qiskit L3T on every
F3 (Heisenberg chain) circuit on FakeAuckland (Addendum 313: A5/L3T 1.111 open, 1.227 periodic, 150 of 150 worse) with
the same two-qubit count?

For every F3 circuit of the ai6 run (same generator and seeds), A5 and L3T are compiled again (compile only). Each output
is scored by
  est_sa   a5's own state-aware estimate (psf_ai_compile.state_aware_cost), the score a5 selects with;
  S_eff    sum of -log(1 - e) with e the error Aer's NoiseModel.from_backend actually applies to each gate;
  S_rep    the same with the Target's reported error;
and the measured infidelities are read from the ai6 run (data/2026-10-02/ai6/outputs). A row is used only if the
recompiled two-qubit count equals the recorded one. If est_sa ranks A5 below L3T while the measured infidelity ranks it
above, a5's estimate is what misleads it; if est_sa also ranks L3T better, a5's candidates never reached L3T's placement.

  python a5_f3_diag.py run --repo <repo> --out <dir> --device <d>
  python a5_f3_diag.py summary --out <dir>
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
RUN = os.path.join("data", "2026-10-02", "ai6", "outputs")


def tables(backend):
    from qiskit.quantum_info import average_gate_fidelity
    from qiskit_aer.noise import NoiseModel
    tgt = backend.target
    local = getattr(NoiseModel.from_backend(backend), "_local_quantum_errors", {})
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
    if not local:
        raise SystemExit("STOP: NoiseModel._local_quantum_errors not found")
    return rep, eff


def scores(out, rep, eff):
    s_rep = s_eff = 0.0
    qubits, rev = set(), 0
    for ins in out.data:
        name = ins.operation.name
        if name in ("barrier", "measure", "delay"):
            continue
        q = tuple(out.find_bit(b).index for b in ins.qubits)
        qubits.update(q)
        s_rep += -math.log(max(1 - rep.get((name, q), 0.0), 1e-300))
        s_eff += -math.log(max(1 - eff.get((name, q), 0.0), 1e-300))
    return s_rep, s_eff, sorted(qubits)


def run(args):
    from qiskit import transpile
    from qiskit_ibm_runtime import fake_provider
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    sys.path.insert(0, args.repo)
    import core_fix_c2_eval as H
    import gap_eval as G
    H.load_module(os.path.join(args.repo, "psf_compile.py"), "psf_compile")
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    a5 = H.load_module(os.path.join(args.repo, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    rep, eff = tables(backend)
    rec = {a: json.load(open(os.path.join(args.repo, RUN, "ai6_%s_%s_F3.json" % (args.device, a))))["rows"]
           for a in ("A5", "L3T")}
    t0, rows, mismatch = time.time(), [], 0
    for i, (params, qc) in enumerate(G.family("F3", False)):
        row = dict(params=params)
        for a in ("A5", "L3T"):
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if a == "A5":
                    out = a5.compile_for_model_circuit(qc, cm, nat, target=tgt)
                else:
                    out = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)
            s_rep, s_eff, qubits = scores(out, rep, eff)
            r = rec[a][i]
            two_q = sum(1 for ins in out.data if len(ins.qubits) == 2)
            ok = r["params"] == params and r["two_q"] == two_q
            mismatch += not ok
            row[a] = dict(est_sa=float(a5.state_aware_cost(out, tgt)), S_rep=s_rep, S_eff=s_eff, qubits=qubits,
                          two_q=two_q, infid=r["infid"] if ok else None)
        rows.append(row)
    with open(os.path.join(args.out, "diag_%s.json" % args.device), "w") as f:
        json.dump(dict(device=args.device, mismatch=mismatch, rows=rows), f)
    print("wrote diag_%s.json: %d circuits, %d arm-rows not matching the ai6 run, %.0f s" % (
        args.device, len(rows), mismatch, time.time() - t0))


def summary(args):
    L = ["# a5_f3_diag summary (exploratory, not a test)", ""]
    for d in DEVICES:
        p = os.path.join(args.out, "diag_%s.json" % d)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        L += ["## %s (arm-rows not matching the ai6 run: %d)" % (d, R["mismatch"]), "",
              "| chain | n | A5/L3T measured | est_sa says A5 better | S_eff says A5 better | S_rep says A5 better | "
              "measured says A5 better | est_sa agrees with measured | S_eff agrees | same qubit set |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for bc in ("o", "p"):
            rs = [r for r in R["rows"] if r["params"]["bc"] == bc and r["A5"]["infid"] is not None
                  and r["L3T"]["infid"] is not None]
            if not rs:
                continue
            better = lambda k: sum(r["A5"][k] < r["L3T"][k] for r in rs)
            agree = lambda k: sum((r["A5"][k] < r["L3T"][k]) == (r["A5"]["infid"] < r["L3T"]["infid"]) for r in rs)
            ratio = np.mean([r["A5"]["infid"] for r in rs]) / np.mean([r["L3T"]["infid"] for r in rs])
            same = sum(r["A5"]["qubits"] == r["L3T"]["qubits"] for r in rs)
            L.append("| %s | %d | %.3f | %d | %d | %d | %d | %d | %d | %d |" % (
                bc, len(rs), ratio, better("est_sa"), better("S_eff"), better("S_rep"), better("infid"),
                agree("est_sa"), agree("S_eff"), same))
        from collections import Counter
        for a in ("A5", "L3T"):
            c = Counter(tuple(r[a]["qubits"]) for r in R["rows"])
            L.append("")
            L.append("%s most used qubit sets: %s" % (a, "; ".join("%s x%d" % (list(k), v) for k, v in c.most_common(3))))
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
