"""f3o_diag2.py -- exploratory diagnosis, part 2 (2026-10-03, not a test). Part 1 (f3o_diag.py) found that on the F3
open-boundary circuits the release (C5) and Qiskit L3T use the same qubits and the same 60 two-qubit gates with the same
applied errors, but C5 has about 17 more sx/x gates and 19 more layers on the cx devices (not on FakeTorino), and is 9-22%
worse. Summed average gate infidelity (S_eff) differs by only 0.003-0.006, against measured differences of 0.025-0.070,
so the extra gates alone cannot explain the gap at face value. Two questions:

  1. Is the gap a state-dependent effect of thermal relaxation (amplitude damping during the long cx gates), which an
     average-infidelity sum cannot see? Each output is simulated under three noise models from the same backend:
       full      NoiseModel.from_backend(backend)                       (as in HOLD2)
       depol     NoiseModel.from_backend(backend, thermal_relaxation=False)
       thermal   NoiseModel.from_backend(backend, gate_error=False)
     and scored by its excitation exposure: the sum over gates of duration x (P(1) on each of the gate's qubits), with
     P(1) from the noiseless state just before the gate, for two-qubit and one-qubit gates separately.
  2. Does a one-qubit clean-up after the release close the gap? Arms:
       C5        release 2026-10-02.2 as in HOLD2
       L3T       Qiskit level 3 with the Target as in HOLD2
       C5+1q     C5's output followed by Optimize1qGatesDecomposition(target)
       C5+blk    C5's output followed by ConsolidateBlocks, UnitarySynthesis and Optimize1qGatesDecomposition (target)

  python f3o_diag2.py run --repo <repo> --out <dir> --device <d>
  python f3o_diag2.py summary --out <dir>
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
ARMS = ("C5", "L3T", "C5+1q", "C5+blk")
MODELS = ("full", "depol", "thermal")
HOLD2 = os.path.join("data", "2026-10-03", "hold2", "outputs")
SKIP = ("barrier", "measure", "delay")


def describe(out, tgt):
    """Gate counts, depth, off-target count and excitation exposure (ns x P(1)) on the active qubits."""
    from qiskit.quantum_info import Statevector
    idx = [(ins, tuple(out.find_bit(b).index for b in ins.qubits)) for ins in out.data if ins.operation.name not in SKIP]
    active = sorted({i for _, q in idx for i in q})
    pos = {p: k for k, p in enumerate(active)}
    sv = Statevector.from_label("0" * len(active))
    exp2 = exp1 = 0.0
    names, off = Counter(), 0
    for ins, q in idx:
        name = ins.operation.name
        names[name] += 1
        props = tgt[name][q] if name in tgt.operation_names and q in tgt[name] else None
        off += props is None
        dur = 1e9 * float(props.duration) if props is not None and props.duration else 0.0
        if dur:
            p1 = sum(float(sv.probabilities([pos[i]])[1]) for i in q)
            if len(q) == 2:
                exp2 += dur * p1
            else:
                exp1 += dur * p1
        sv = sv.evolve(ins.operation, qargs=[pos[i] for i in q])
    return dict(two_q=sum(1 for _, q in idx if len(q) == 2), sx=names.get("sx", 0), x=names.get("x", 0),
                rz=names.get("rz", 0), depth=out.depth(), exp2=exp2, exp1=exp1, off_target=off, qubits=active)


def run(args):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ConsolidateBlocks, Optimize1qGatesDecomposition, UnitarySynthesis
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    sys.path.insert(0, args.repo)
    import core_fix_c2_eval as H
    rel = H.load_module(os.path.join(args.repo, "psf_compile.py"), "psf_compile")
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    gen = H.load_module(os.path.join(args.repo, "benchmarks", "hold2_eval.py"), "hold2_eval")
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    sims = {"full": AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts),
            "depol": AerSimulator(noise_model=NoiseModel.from_backend(backend, thermal_relaxation=False), **opts),
            "thermal": AerSimulator(noise_model=NoiseModel.from_backend(backend, gate_error=False), **opts),
            "ideal": AerSimulator(**opts)}
    pm1 = PassManager([Optimize1qGatesDecomposition(target=tgt)])
    pmb = PassManager([ConsolidateBlocks(target=tgt), UnitarySynthesis(target=tgt),
                       Optimize1qGatesDecomposition(target=tgt)])

    def quiet(f, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f(*a, **k)

    t0, rows, circs = time.time(), [], {a: [] for a in ARMS}
    circuits = [(p, qc) for p, qc in gen.family("F3", False) if p["bc"] == "o"]
    for params, qc in circuits:
        n = qc.num_qubits
        c5 = quiet(rel.compile_for_hardware, qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                   layout_search=True, seed_transpiler=0, target=tgt, placement_refine=True)
        l3 = quiet(transpile, qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)
        fin = {"C5": list(c5.layout.final_index_layout(filter_ancillas=True)[:n]),
               "L3T": list(l3.layout.final_index_layout(filter_ancillas=True)[:n])}
        fin["C5+1q"] = fin["C5+blk"] = fin["C5"]
        outs = {"C5": c5, "L3T": l3, "C5+1q": quiet(pm1.run, c5), "C5+blk": quiet(pmb.run, c5)}
        row = dict(params=params, psi=Statevector(qc).data)
        for a, out in outs.items():
            row[a] = describe(out, tgt)
            c = out.copy()
            c.remove_final_measurements(inplace=True)
            c.save_density_matrix(qubits=fin[a])
            circs[a].append(c)
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
    for a in ("C5", "L3T"):
        p = os.path.join(args.repo, HOLD2, "hold2_%s_%s_F3.json" % (args.device, a))
        if not os.path.exists(p):
            check[a] = "HOLD2 data not in repo"
            continue
        rec = [r for r in json.load(open(p))["rows"] if r["params"]["bc"] == "o"]
        check[a] = sum(1 for r, x in zip(rec, rows) if r["params"] != x["params"] or r["two_q"] != x[a]["two_q"]
                       or abs(r["infid"] - x[a]["full"]) > 1e-9)
    with open(os.path.join(args.out, "f3o2_%s.json" % args.device), "w") as f:
        json.dump(dict(device=args.device, hold2_mismatch=check, rows=rows, wall_s=time.time() - t0), f)
    print("wrote f3o2_%s.json: %d circuits, HOLD2 mismatches %s, %.0f s" % (args.device, len(rows), check,
                                                                            time.time() - t0))


def summary(args):
    L = ["# f3o_diag2 summary (exploratory, not a test)", ""]
    for d in DEVICES:
        p = os.path.join(args.out, "f3o2_%s.json" % d)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        rs = R["rows"]
        mean = lambda a, k: float(np.mean([r[a][k] for r in rs]))
        L += ["## %s (%d circuits; rows differing from HOLD2: %s; noiseless max %.1e)" % (
            d, len(rs), R["hold2_mismatch"], max(r[a]["ideal"] for r in rs for a in ARMS)), "",
              "| arm | full / L3T | depol / L3T | thermal / L3T | full | sx | x | depth | 2q exposure | 1q exposure | "
              "off-target |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in ARMS:
            L.append("| %s | %.3f | %.3f | %.3f | %.4f | %.1f | %.1f | %.1f | %.0f | %.0f | %d |" % (
                a, mean(a, "full") / mean("L3T", "full"), mean(a, "depol") / mean("L3T", "depol"),
                mean(a, "thermal") / mean("L3T", "thermal"), mean(a, "full"), mean(a, "sx"), mean(a, "x"),
                mean(a, "depth"), mean(a, "exp2"), mean(a, "exp1"), sum(r[a]["off_target"] for r in rs)))
        worse = sum(r["C5"]["full"] > r["L3T"]["full"] for r in rs)
        agree = sum((r["C5"]["exp2"] > r["L3T"]["exp2"]) == (r["C5"]["full"] > r["L3T"]["full"]) for r in rs)
        dx = np.array([r["C5"]["exp2"] - r["L3T"]["exp2"] for r in rs])
        dy = np.array([r["C5"]["full"] - r["L3T"]["full"] for r in rs])
        cor = float(np.corrcoef(dx, dy)[0, 1]) if dx.std() > 0 and dy.std() > 0 else float("nan")
        L += ["", "C5 worse than L3T (full noise) in %d of %d circuits; the 2q-exposure difference has the same sign in "
              "%d; correlation of the two differences %.3f." % (worse, len(rs), agree, cor), ""]
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
