"""f3o_diag.py -- exploratory diagnosis (2026-10-03, not a test): where does the release's chain gap to Qiskit L3T on the
cx devices come from? HOLD2 (Addendum 321) put it in the open-boundary F3 cell (C6/L3T 1.09-1.16 on four cx devices),
where the floor-aware placement score changes nothing on three of them, so it is probably not a placement-score problem.

For every F3 open-boundary circuit of HOLD2 (same generator and seeds, 150 per device), five compilations:
  C5      release 2026-10-02.2, target, placement_refine=True (as in HOLD2)
  C6      candidate c6, the same plus placement_score="floor" (as in HOLD2)
  L3T     Qiskit level 3 with the Target, approximation_degree=1.0 (as in HOLD2)
  L3onC6  Qiskit level 3 as L3T, but pinned to C6's initial layout   -> Qiskit's routing and synthesis on C6's placement
  RELonL3 release compile pinned to L3T's initial layout (no layout search, no re-placement)
                                                                      -> PSF-Zero's routing and synthesis on L3T's placement
Each output is simulated as in HOLD2 (density matrix, NoiseModel.from_backend) and scored by
  S_eff   sum of -log(1 - e), e the error Aer actually applies to each gate, split into two-qubit and one-qubit parts;
  S_rep   the same with the Target's reported errors;
with gate counts, depth, the qubit set and the mean applied two-qubit error. If the HOLD2 data are in the repo, the C5,
C6 and L3T rows are checked against them (two-qubit count and infidelity).

Reading: if RELonL3 ~ C6 and L3onC6 ~ L3T, the gap is in routing or synthesis, not placement; if RELonL3 ~ L3T, it is
placement after all.

  python f3o_diag.py run --repo <repo> --out <dir> --device <d>
  python f3o_diag.py summary --out <dir>
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
from collections import Counter

import numpy as np

warnings.simplefilter("ignore")
DEVICES = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeTorino")
ARMS = ("C5", "C6", "L3T", "L3onC6", "RELonL3")
HOLD2 = os.path.join("data", "2026-10-03", "hold2", "outputs")
CAND = os.path.join("patches", "psf_compile_c6_2026-10-03", "psf_compile.py")
SKIP = ("barrier", "measure", "delay")


def tables(backend):
    """Reported and Aer-applied error per (gate, qubits); copied from a5_f3_diag.tables."""
    from qiskit.quantum_info import average_gate_fidelity
    from qiskit_aer.noise import NoiseModel
    tgt = backend.target
    local = getattr(NoiseModel.from_backend(backend), "_local_quantum_errors", {})
    if not local:
        raise SystemExit("STOP: NoiseModel._local_quantum_errors not found")
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
    return rep, eff


def describe(out, rep, eff):
    s_rep = s2 = s1 = 0.0
    e2, names, active = [], Counter(), set()
    for ins in out.data:
        name = ins.operation.name
        if name in SKIP:
            continue
        q = tuple(out.find_bit(b).index for b in ins.qubits)
        names[name] += 1
        active.update(q)
        s_rep += -math.log(max(1 - rep.get((name, q), 0.0), 1e-300))
        s = -math.log(max(1 - eff.get((name, q), 0.0), 1e-300))
        if len(q) == 2:
            s2 += s
            e2.append(eff.get((name, q), 0.0))
        else:
            s1 += s
    return dict(two_q=len(e2), one_q=sum(v for k, v in names.items() if k in ("sx", "x")), rz=names.get("rz", 0),
                depth=out.depth(), qubits=sorted(active), S_rep=s_rep, S_eff=s2 + s1, S_eff_2q=s2, S_eff_1q=s1,
                mean_e2=float(np.mean(e2)) if e2 else 0.0,
                init=[int(x) for x in out.layout.initial_index_layout(filter_ancillas=True)])


def run(args):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    sys.path.insert(0, args.repo)
    import core_fix_c2_eval as H
    rel = H.load_module(os.path.join(args.repo, "psf_compile.py"), "psf_compile")
    c6 = H.load_module(os.path.join(args.repo, CAND), "psf_compile_c6")
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    gen = H.load_module(os.path.join(args.repo, "benchmarks", "hold2_eval.py"), "hold2_eval")
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    rep, eff = tables(backend)
    opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    sims = {"infid": AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts),
            "infid_ideal": AerSimulator(**opts)}
    psf = dict(coupling_map=cm, basis_gates=nat, entangling_basis="cx", seed_transpiler=0, target=tgt)
    l3 = dict(target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)

    def quiet(f, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f(*a, **k)

    t0, rows, circs = time.time(), [], {a: [] for a in ARMS}
    circuits = [(p, qc) for p, qc in gen.family("F3", False) if p["bc"] == "o"]
    for params, qc in circuits:
        n = qc.num_qubits
        outs = {"C5": quiet(rel.compile_for_hardware, qc, layout_search=True, placement_refine=True, **psf),
                "C6": quiet(c6.compile_for_hardware, qc, layout_search=True, placement_refine=True,
                            placement_score="floor", **psf),
                "L3T": quiet(transpile, qc, **l3)}
        outs["L3onC6"] = quiet(transpile, qc, initial_layout=describe(outs["C6"], rep, eff)["init"][:n], **l3)
        outs["RELonL3"] = quiet(rel.compile_for_hardware, qc, initial_layout=describe(outs["L3T"], rep, eff)["init"][:n],
                                **psf)
        row = dict(params=params)
        for a, out in outs.items():
            row[a] = describe(out, rep, eff)
            c = out.copy()
            c.remove_final_measurements(inplace=True)
            c.save_density_matrix(qubits=list(out.layout.final_index_layout(filter_ancillas=True)[:n]))
            circs[a].append(c)
        row["psi"] = Statevector(qc).data
        rows.append(row)
    for a in ARMS:
        for key, sim in sims.items():
            res = sim.run(circs[a]).result()
            for j, row in enumerate(rows):
                rho = np.asarray(res.data(j)["density_matrix"])
                row[a][key] = float(1 - np.real(np.conj(row["psi"]) @ rho @ row["psi"]))
    for row in rows:
        del row["psi"]
    check = {}
    for a in ("C5", "C6", "L3T"):
        p = os.path.join(args.repo, HOLD2, "hold2_%s_%s_F3.json" % (args.device, a))
        if not os.path.exists(p):
            check[a] = "HOLD2 data not in repo"
            continue
        rec = [r for r in json.load(open(p))["rows"] if r["params"]["bc"] == "o"]
        check[a] = sum(1 for r, x in zip(rec, rows) if r["params"] != x["params"] or r["two_q"] != x[a]["two_q"]
                       or abs(r["infid"] - x[a]["infid"]) > 1e-9)
    with open(os.path.join(args.out, "f3o_%s.json" % args.device), "w") as f:
        json.dump(dict(device=args.device, hold2_mismatch=check, rows=rows, wall_s=time.time() - t0), f)
    print("wrote f3o_%s.json: %d circuits, HOLD2 mismatches %s, %.0f s" % (args.device, len(rows), check,
                                                                           time.time() - t0))


def summary(args):
    L = ["# f3o_diag summary (exploratory, not a test)", ""]
    for d in DEVICES:
        p = os.path.join(args.out, "f3o_%s.json" % d)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        rs = R["rows"]
        mean = lambda a, k: float(np.mean([r[a][k] for r in rs]))
        L += ["## %s (%d circuits; rows differing from HOLD2: %s; noiseless max %.1e)" % (
            d, len(rs), R["hold2_mismatch"], max(r[a]["infid_ideal"] for r in rs for a in ARMS)), "",
              "| arm | infid / L3T | two-qubit | sx+x | depth | S_eff 2q | S_eff 1q | mean 2q error | same qubits as L3T | "
              "same qubits as C6 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for a in ARMS:
            L.append("| %s | %.3f | %.1f | %.1f | %.1f | %.4f | %.4f | %.5f | %d | %d |" % (
                a, mean(a, "infid") / mean("L3T", "infid"), mean(a, "two_q"), mean(a, "one_q"), mean(a, "depth"),
                mean(a, "S_eff_2q"), mean(a, "S_eff_1q"), mean(a, "mean_e2"),
                sum(r[a]["qubits"] == r["L3T"]["qubits"] for r in rs), sum(r[a]["qubits"] == r["C6"]["qubits"] for r in rs)))
        agree = sum((r["C6"]["S_eff"] < r["L3T"]["S_eff"]) == (r["C6"]["infid"] < r["L3T"]["infid"]) for r in rs)
        L += ["", "S_eff orders C6 against L3T as the measured infidelity does in %d of %d circuits." % (agree, len(rs))]
        for a in ("C6", "L3T"):
            c = Counter(tuple(r[a]["qubits"]) for r in rs)
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
