"""ring_diag.py -- exploratory diagnosis (2026-10-03, not a test): where do release 2026-10-03.1's remaining gaps to Qiskit
L3T come from (Addendum 324)? On the cx devices the periodic F3 chains are 12-40% behind; on the cz devices the F1
rings are 6-8% behind; F6 (QFT) is about 5% behind on several devices. All three need routing on heavy-hex. The
release routes with Qiskit's level-1 preset over a bare coupling map (`routing_optimization_level=1`); L3T uses level 3
with the Target.

Circuits: HOLD3's generator (benchmarks/hold3_eval.py, scored seeds): all 150 F3 periodic circuits, every second F1
circuit (216) and all 120 F6 circuits, per device. Arms:
  R3      release 2026-10-03.1 as recommended: layout_search=True, target, placement_refine=True,
          final_resynthesis="select"
  L3T     Qiskit level 3 with the Target, approximation_degree=1.0
  R3r3    R3 with routing_optimization_level=3 (its routing done by Qiskit's level-3 preset, still without the Target)
  L3onR3  L3T pinned to R3's initial layout       -> Qiskit's routing and synthesis on the release's placement
  R3onL3  the release pinned to L3T's initial layout (no layout search, no re-placement; "select" kept)
                                                  -> the release's routing and synthesis on Qiskit's placement
  PICK    whichever of R3 and L3T has the lower excitation_cost (the release's own estimate): no new compile, a
          preview of letting "select" consider Qiskit's level-3 circuit as well
Each output is simulated as in HOLD3 (density matrix, NoiseModel.from_backend) and described by its two-qubit count,
depth, sx and x counts, touched qubits and excitation_cost. Rows of R3 and L3T are checked against HOLD3's data when it
is in the repository (the C8 and L3T arms; two-qubit count and infidelity).

  python ring_diag.py run --repo <repo> --out <dir> --device <d>
  python ring_diag.py summary --out <dir>
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
DEVICES = ("FakeAuckland", "FakeAlgiers", "FakeTorino", "FakeKingston", "FakeAachen")
ARMS = ("R3", "L3T", "R3r3", "L3onR3", "R3onL3")
HOLD3 = os.path.join("data", "2026-10-03", "hold3", "outputs")
SKIP = ("barrier", "measure", "delay")


def describe(out, rel, tgt):
    idx = [(ins.operation.name, tuple(out.find_bit(b).index for b in ins.qubits)) for ins in out.data
           if ins.operation.name not in SKIP]
    return dict(two_q=sum(1 for _, q in idx if len(q) == 2), sx=sum(1 for n, _ in idx if n == "sx"),
                x=sum(1 for n, _ in idx if n == "x"), depth=out.depth(), qubits=sorted({i for _, q in idx for i in q}),
                est=rel.excitation_cost(out, tgt),
                init=[int(v) for v in out.layout.initial_index_layout(filter_ancillas=True)])


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
    if rel.VERSION != "2026-10-03.1":
        raise SystemExit("STOP: psf_compile.py is %s, not release 2026-10-03.1" % rel.VERSION)
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    gen = H.load_module(os.path.join(args.repo, "benchmarks", "hold3_eval.py"), "hold3_eval")
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    sims = {"infid": AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts),
            "ideal": AerSimulator(**opts)}
    base = dict(coupling_map=cm, basis_gates=nat, entangling_basis="cx", seed_transpiler=0, target=tgt,
                final_resynthesis="select")
    l3 = dict(target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)

    def quiet(f, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f(*a, **k)

    circuits = []
    for fam in ("F3", "F1", "F6"):
        for j, (p, qc) in enumerate(gen.family(fam, False)):
            if (fam == "F3" and p["bc"] != "p") or (fam == "F1" and j % 2):
                continue
            circuits.append((fam, j, p, qc))
    t0, rows, circs = time.time(), [], {a: [] for a in ARMS}
    for fam, j, params, qc in circuits:
        n = qc.num_qubits
        outs = {"R3": quiet(rel.compile_for_hardware, qc, layout_search=True, placement_refine=True, **base),
                "L3T": quiet(transpile, qc, **l3)}
        outs["R3r3"] = quiet(rel.compile_for_hardware, qc, layout_search=True, placement_refine=True,
                             routing_optimization_level=3, **base)
        d = {a: describe(o, rel, tgt) for a, o in outs.items()}
        outs["L3onR3"] = quiet(transpile, qc, initial_layout=d["R3"]["init"][:n], **l3)
        outs["R3onL3"] = quiet(rel.compile_for_hardware, qc, initial_layout=d["L3T"]["init"][:n], **base)
        row = dict(family=fam, index=j, params=params, psi=Statevector(qc).data)
        for a in ARMS:
            out = outs[a]
            row[a] = d[a] if a in d else describe(out, rel, tgt)
            c = out.copy()
            c.remove_final_measurements(inplace=True)
            c.save_density_matrix(qubits=list(out.layout.final_index_layout(filter_ancillas=True)[:n]))
            circs[a].append(c)
        rows.append(row)
    for a in ARMS:
        for key, sim in sims.items():
            res = sim.run(circs[a]).result()
            for k, row in enumerate(rows):
                rho = np.asarray(res.data(k)["density_matrix"])
                row[a][key] = float(1 - np.real(np.conj(row["psi"]) @ rho @ row["psi"]))
    for row in rows:
        del row["psi"]
    check = {}
    for a, arm in (("R3", "C8"), ("L3T", "L3T")):
        bad = 0
        for fam in ("F3", "F1", "F6"):
            p = os.path.join(args.repo, HOLD3, "hold3_%s_%s_%s.json" % (args.device, arm, fam))
            if not os.path.exists(p):
                bad = "HOLD3 data not in repo"
                break
            rec = json.load(open(p))["rows"]
            for row in rows:
                if row["family"] == fam:
                    r = rec[row["index"]]
                    bad += (r["params"] != row["params"] or r["two_q"] != row[a]["two_q"]
                            or abs(r["infid"] - row[a]["infid"]) > 1e-9)
        check[a] = bad
    with open(os.path.join(args.out, "ring_%s.json" % args.device), "w") as f:
        json.dump(dict(device=args.device, version=rel.VERSION, hold3_mismatch=check, rows=rows,
                       wall_s=time.time() - t0), f)
    print("wrote ring_%s.json: %d circuits, HOLD3 mismatches %s, %.0f s" % (args.device, len(rows), check,
                                                                            time.time() - t0))


def summary(args):
    L = ["# ring_diag summary (exploratory, not a test)", ""]
    for d in DEVICES:
        p = os.path.join(args.out, "ring_%s.json" % d)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        L += ["## %s (rows differing from HOLD3: %s; noiseless max %.1e)" % (
            d, R["hold3_mismatch"], max(r[a]["ideal"] for r in R["rows"] for a in ARMS)), "",
              "| family | arm | infid / L3T | two-qubit | depth | sx | x | est / L3T est | same qubits as L3T |",
              "|---|---|---|---|---|---|---|---|---|"]
        for fam in ("F3", "F1", "F6"):
            rs = [r for r in R["rows"] if r["family"] == fam]
            m = lambda a, k: float(np.mean([r[a][k] for r in rs]))
            for a in ARMS:
                L.append("| %s | %s | %.3f | %.1f | %.1f | %.1f | %.1f | %.3f | %d |" % (
                    "F3p" if fam == "F3" else fam, a, m(a, "infid") / m("L3T", "infid"), m(a, "two_q"), m(a, "depth"),
                    m(a, "sx"), m(a, "x"), m(a, "est") / m("L3T", "est"),
                    sum(r[a]["qubits"] == r["L3T"]["qubits"] for r in rs)))
            pick = [r["R3"] if r["R3"]["est"] <= r["L3T"]["est"] else r["L3T"] for r in rs]
            orc = [min(r["R3"]["infid"], r["L3T"]["infid"]) for r in rs]
            right = sum((r["R3"]["est"] <= r["L3T"]["est"]) == (r["R3"]["infid"] <= r["L3T"]["infid"]) for r in rs)
            L.append("| %s | PICK (by est) | %.3f | | | | | | estimate agrees with measurement in %d of %d; oracle %.3f |" % (
                "F3p" if fam == "F3" else fam, np.mean([x["infid"] for x in pick]) / m("L3T", "infid"), right, len(rs),
                np.mean(orc) / m("L3T", "infid")))
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
