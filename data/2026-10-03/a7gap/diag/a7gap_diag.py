"""a7gap_diag.py -- exploratory diagnosis (2026-10-03, not a test): what is left between release 2026-10-03.2 (recommended
call, `compare_level3=True`) and the AI front end a7? In HOLD4 (Addendum 328) a7 led most on F5 (GHZ chains) on the cx
devices (A7/C9 0.71-0.93) with the same two-qubit gates and depth: a placement effect. HOLD2 (Addendum 321) had found
that the held candidate c6 (re-placement scored by max(reported error, T1/T2 floor)) gave almost exactly a7's F5 gain
(FakeGeneva 0.721, FakeAuckland 0.934, FakeAlgiers 0.931). a7 also chooses with a state-aware Pauli estimate that
includes dephasing; the release's `excitation_cost` sees amplitude damping only.

For HOLD4's scored circuits (same generator and seeds; F5 and F6 all, F2 every second, F4 every third, F3 open every
third), per device:
  R3   release 2026-10-03.2, final_resynthesis="select", no level-3 comparison
  R3F  the same pipeline, but re-placed on c6's floor-aware Target (item 34's score) before "select"
  L3T  Qiskit level 3 with the Target
  C9   release 2026-10-03.2 as recommended (compare_level3=True); checked against HOLD4's C9 rows
Each is simulated as in HOLD4. Choices among {R3, R3F, L3T} are then made by
  exc    the release's excitation_cost
  pauli  a state-aware first-order Pauli estimate (`pauli_cost` below): per gate, the Pauli-twirled thermal
         relaxation of each of its qubits for the gate's duration, each component p_P costing p_P (1 - <P>^2) with
         <P> on the noiseless state right after the gate, plus the rest of the reported error (state-independent)
and compared with the oracle (measured best of the three) and with a7's HOLD4 result.

  python a7gap_diag.py run --repo <repo> --out <dir> --device <d>
  python a7gap_diag.py summary --out <dir>
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
DEVICES = ("FakeAuckland", "FakeGeneva", "FakeAlgiers", "FakeHanoiV2", "FakeTorino", "FakeMarrakesh")
ARMS = ("R3", "R3F", "L3T", "C9")
HOLD4 = os.path.join("data", "2026-10-03", "hold4", "outputs")
C6 = os.path.join("patches", "psf_compile_c6_2026-10-03", "psf_compile.py")
STRIDE = dict(F5=1, F6=1, F2=2, F4=3, F3=3)
SKIP = ("barrier", "measure", "delay")


def pauli_cost(circ, target, max_qubits=16):
    """State-aware first-order Pauli estimate (see module docstring). None if too wide or a gate has no matrix."""
    ops = [(ins.operation, tuple(circ.find_bit(b).index for b in ins.qubits)) for ins in circ.data
           if ins.operation.name not in SKIP]
    active = sorted({i for _, q in ops for i in q})
    if len(active) > max_qubits:
        return None
    k = max(len(active), 1)
    pos = {p: j for j, p in enumerate(active)}
    qp = getattr(target, "qubit_properties", None) or []
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0
    cost = 0.0
    for op, q in ops:
        try:
            mat = np.asarray(op.to_matrix(), dtype=complex)
        except Exception:
            return None
        axes = [pos[i] for i in q]
        m = len(axes)
        rev = axes[::-1]
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
        props = target[op.name].get(q, None) if op.name in target.operation_names else None
        if props is None:
            continue
        e = props.error or 0.0
        t = props.duration or 0.0
        thermal_f = 1.0
        for i, ax in zip(q, axes):
            t1 = getattr(qp[i], "t1", None) if i < len(qp) and qp[i] is not None else None
            t2 = getattr(qp[i], "t2", None) if i < len(qp) and qp[i] is not None else None
            if not t or not t1:
                continue
            t2 = min(t2, 2 * t1) if t2 else 2 * t1
            px = (1.0 - math.exp(-t / t1)) / 4.0
            pz = max((1.0 - math.exp(-t / t2)) / 2.0 - px, 0.0)
            mm = np.moveaxis(psi, ax, 0).reshape(2, -1)
            rho = mm @ mm.conj().T
            ex, ey, ez = 2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real
            cost += px * (1 - ex * ex) + px * (1 - ey * ey) + pz * (1 - ez * ez)
            thermal_f *= (1.0 + 2.0 * math.exp(-t / t2) + math.exp(-t / t1)) / 4.0
        d = 2 ** m
        floor = 1.0 - (d * thermal_f + 1.0) / (d + 1.0)
        cost += max(e - floor, 0.0) * (d + 1) / d
    return cost


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
    if rel.VERSION != "2026-10-03.2":
        raise SystemExit("STOP: psf_compile.py is %s, not release 2026-10-03.2" % rel.VERSION)
    c6 = H.load_module(os.path.join(args.repo, C6), "psf_compile_c6")
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    gen = H.load_module(os.path.join(args.repo, "benchmarks", "hold4_eval.py"), "hold4_eval")
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    ftgt = c6.floor_aware_target(tgt)
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    edges, fq = rel._failed_elements(tgt, 0.5)
    opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    sims = {"infid": AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts),
            "ideal": AerSimulator(**opts)}
    base = dict(coupling_map=cm, basis_gates=nat, entangling_basis="cx", seed_transpiler=0, layout_search=True)

    def quiet(f, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f(*a, **k)

    def r3f(qc):
        out = quiet(rel.compile_for_hardware, qc, _refine_target=ftgt, **base)  # target=None: the inner pipeline
        if rel._uses_failed(out, edges, fq):
            return None
        out = rel._select_resynthesis(out, tgt, 0.5)
        return out if rel._acceptable(out, tgt, 0.5) else None

    circuits = []
    for fam in ("F5", "F6", "F2", "F4", "F3"):
        for j, (p, qc) in enumerate(gen.family(fam, False)):
            if (fam == "F3" and p.get("bc") != "o") or j % STRIDE[fam]:
                continue
            circuits.append((fam, j, p, qc))
    t0, rows, circs = time.time(), [], {a: [] for a in ARMS}
    for fam, j, params, qc in circuits:
        n = qc.num_qubits
        outs = {"R3": quiet(rel.compile_for_hardware, qc, target=tgt, placement_refine=True,
                            final_resynthesis="select", **base),
                "R3F": quiet(r3f, qc),
                "L3T": quiet(transpile, qc, target=tgt, optimization_level=3, seed_transpiler=0,
                             approximation_degree=1.0),
                "C9": quiet(rel.compile_for_hardware, qc, target=tgt, placement_refine=True,
                            final_resynthesis="select", compare_level3=True, **base)}
        if outs["R3F"] is None:
            outs["R3F"] = outs["R3"]
        row = dict(family=fam, index=j, params=params, psi=Statevector(qc).data)
        for a in ARMS:
            out = outs[a]
            row[a] = dict(two_q=sum(1 for i in out.data if len(i.qubits) == 2), depth=out.depth(),
                          exc=rel.excitation_cost(out, tgt), pauli=pauli_cost(out, tgt),
                          qubits=sorted({out.find_bit(b).index for i in out.data for b in i.qubits
                                         if i.operation.name not in SKIP}))
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
    check = 0
    a7 = {}
    for fam in ("F5", "F6", "F2", "F4", "F3"):
        p9 = os.path.join(args.repo, HOLD4, "hold4_%s_C9_%s.json" % (args.device, fam))
        pa = os.path.join(args.repo, HOLD4, "hold4_%s_A7_%s.json" % (args.device, fam))
        if not (os.path.exists(p9) and os.path.exists(pa)):
            check = "HOLD4 data not in repo"
            break
        r9, ra = json.load(open(p9))["rows"], json.load(open(pa))["rows"]
        for row in rows:
            if row["family"] == fam:
                r = r9[row["index"]]
                check += (r["params"] != row["params"] or r["two_q"] != row["C9"]["two_q"]
                          or abs(r["infid"] - row["C9"]["infid"]) > 1e-9)
                row["A7"] = dict(infid=ra[row["index"]]["infid"], two_q=ra[row["index"]]["two_q"])
    with open(os.path.join(args.out, "a7gap_%s.json" % args.device), "w") as f:
        json.dump(dict(device=args.device, version=rel.VERSION, hold4_mismatch=check, rows=rows,
                       wall_s=time.time() - t0), f)
    print("wrote a7gap_%s.json: %d circuits, HOLD4 C9 mismatches %s, %.0f s" % (args.device, len(rows), check,
                                                                                time.time() - t0))


def summary(args):
    L = ["# a7gap_diag summary (exploratory, not a test)", "",
         "Ratios are mean infidelity relative to L3T. PICKe / PICKp choose among {R3, R3F, L3T} by excitation_cost /",
         "pauli_cost; PICKp2 chooses among {R3, L3T} by pauli_cost (C9 does the same by excitation_cost); ORC is the",
         "measured best of the three. 'rank' counts circuits where the estimate's choice among the three is the",
         "measured best.", ""]
    for d in DEVICES:
        p = os.path.join(args.out, "a7gap_%s.json" % d)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        L += ["## %s (rows differing from HOLD4's C9: %s; noiseless max %.1e)" % (
            d, R["hold4_mismatch"], max(r[a]["ideal"] for r in R["rows"] for a in ARMS)), "",
              "| family | n | R3 | R3F | C9 | A7 | PICKe | PICKp | PICKp2 | ORC | rank e | rank p | R3F same qubits as R3 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for fam in ("F5", "F6", "F2", "F4", "F3", "all"):
            rs = [r for r in R["rows"] if fam == "all" or r["family"] == fam]
            if not rs:
                continue
            l3 = np.mean([r["L3T"]["infid"] for r in rs])
            mean = lambda xs: float(np.mean(xs)) / l3

            def pick(r, key, arms=("R3", "R3F", "L3T")):
                vals = [(r[a][key], a) for a in arms if r[a][key] is not None]
                return r[min(vals)[1]] if len(vals) == len(arms) else r["R3"]

            orc = [min(r[a]["infid"] for a in ("R3", "R3F", "L3T")) for r in rs]
            re_ = sum(abs(pick(r, "exc")["infid"] - o) <= 1e-12 for r, o in zip(rs, orc))
            rp = sum(abs(pick(r, "pauli")["infid"] - o) <= 1e-12 for r, o in zip(rs, orc))
            L.append("| %s | %d | %.3f | %.3f | %.3f | %s | %.3f | %.3f | %.3f | %.3f | %d | %d | %d |" % (
                fam, len(rs), mean([r["R3"]["infid"] for r in rs]), mean([r["R3F"]["infid"] for r in rs]),
                mean([r["C9"]["infid"] for r in rs]),
                "%.3f" % mean([r["A7"]["infid"] for r in rs]) if all("A7" in r for r in rs) else "-",
                mean([pick(r, "exc")["infid"] for r in rs]), mean([pick(r, "pauli")["infid"] for r in rs]),
                mean([pick(r, "pauli", ("R3", "L3T"))["infid"] for r in rs]), mean(orc), re_, rp,
                sum(r["R3F"]["qubits"] == r["R3"]["qubits"] for r in rs)))
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
