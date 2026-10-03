"""f3o_diag3.py -- exploratory diagnosis, part 3 (2026-10-03, not a test). Part 2 (f3o_diag2.py) found that the
release's F3 gap on the cx devices is a thermal-relaxation effect: under depolarizing noise alone C5 and L3T are level
(1.003-1.009), under thermal relaxation alone C5 is 1.19-1.41 x L3T, and C5 keeps qubits excited for longer during the
long cx gates (two-qubit excitation exposure 37-47% higher; it has about 11 x gates per circuit, L3T about 2). A one-qubit
clean-up, or re-synthesis of blocks only where it saves gates, changes nothing. Which part of the release puts the
excitation there, and does forcing Qiskit's two-qubit synthesis remove it? Arms, on all 300 F3 circuits of HOLD2 (open
and periodic):

  C5        release 2026-10-02.2 as in HOLD2 (entangling_basis="cx")
  L3T       Qiskit level 3 with the Target as in HOLD2
  C5force   C5's output, then ConsolidateBlocks(force_consolidate=True), UnitarySynthesis and
            Optimize1qGatesDecomposition, all with the target: every two-qubit block re-synthesised by Qiskit
  C5noc2    the release with elide_permutations=False and post_routing_resynthesis=False (items 29-30 off)
  C5can     the release with entangling_basis="canonical" (PSF-Zero emits RXX/RYY/RZZ, Qiskit translates them)

Each output is simulated under the full noise model and under thermal relaxation alone, and described by
f3o_diag2.describe (gate counts, depth, excitation exposure, off-target instructions).

  python f3o_diag3.py run --repo <repo> --out <dir> --device <d>
  python f3o_diag3.py summary --out <dir>
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
DEVICES = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeTorino")
ARMS = ("C5", "L3T", "C5force", "C5noc2", "C5can")
HERE = os.path.dirname(os.path.abspath(__file__))


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
    d2 = H.load_module(os.path.join(HERE, "f3o_diag2.py"), "f3o_diag2")
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    sims = {"full": AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts),
            "thermal": AerSimulator(noise_model=NoiseModel.from_backend(backend, gate_error=False), **opts),
            "ideal": AerSimulator(**opts)}
    pmf = PassManager([ConsolidateBlocks(force_consolidate=True, target=tgt), UnitarySynthesis(target=tgt),
                       Optimize1qGatesDecomposition(target=tgt)])
    psf = dict(coupling_map=cm, basis_gates=nat, layout_search=True, seed_transpiler=0, target=tgt, placement_refine=True)

    def quiet(f, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f(*a, **k)

    def fin(out, n):
        return list(out.layout.final_index_layout(filter_ancillas=True)[:n])

    t0, rows, circs = time.time(), [], {a: [] for a in ARMS}
    for params, qc in gen.family("F3", False):
        n = qc.num_qubits
        c5 = quiet(rel.compile_for_hardware, qc, entangling_basis="cx", **psf)
        outs = {"C5": (c5, fin(c5, n)),
                "L3T": quiet(transpile, qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0),
                "C5force": (quiet(pmf.run, c5), fin(c5, n)),
                "C5noc2": quiet(rel.compile_for_hardware, qc, entangling_basis="cx", elide_permutations=False,
                                post_routing_resynthesis=False, **psf),
                "C5can": quiet(rel.compile_for_hardware, qc, entangling_basis="canonical", **psf)}
        row = dict(params=params, psi=Statevector(qc).data)
        for a, o in outs.items():
            out, f = o if isinstance(o, tuple) else (o, fin(o, n))
            row[a] = d2.describe(out, tgt)
            c = out.copy()
            c.remove_final_measurements(inplace=True)
            c.save_density_matrix(qubits=f)
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
    with open(os.path.join(args.out, "f3o3_%s.json" % args.device), "w") as f:
        json.dump(dict(device=args.device, rows=rows, wall_s=time.time() - t0), f)
    print("wrote f3o3_%s.json: %d circuits, %.0f s" % (args.device, len(rows), time.time() - t0))


def summary(args):
    L = ["# f3o_diag3 summary (exploratory, not a test)", ""]
    for d in DEVICES:
        p = os.path.join(args.out, "f3o3_%s.json" % d)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        L += ["## %s (noiseless max %.1e)" % (d, max(r[a]["ideal"] for r in R["rows"] for a in ARMS)), "",
              "| bc | arm | full / L3T | thermal / L3T | full | two-qubit | sx | x | depth | 2q exposure | off-target |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for bc in ("o", "p"):
            rs = [r for r in R["rows"] if r["params"]["bc"] == bc]
            mean = lambda a, k: float(np.mean([r[a][k] for r in rs]))
            for a in ARMS:
                L.append("| %s | %s | %.3f | %.3f | %.4f | %.1f | %.1f | %.1f | %.1f | %.0f | %d |" % (
                    bc, a, mean(a, "full") / mean("L3T", "full"), mean(a, "thermal") / mean("L3T", "thermal"),
                    mean(a, "full"), mean(a, "two_q"), mean(a, "sx"), mean(a, "x"), mean(a, "depth"), mean(a, "exp2"),
                    sum(r[a]["off_target"] for r in rs)))
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
