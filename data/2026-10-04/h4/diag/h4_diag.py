"""h4_diag.py -- exploratory diagnosis (2026-10-04, not a test): why does release 2026-10-04.1's `hybrid_cost` keep the
release's own circuit for 4-qubit GHZ chains on FakeAlgiers, where `pauli_cost` takes the floor-placed one, which is
6.7% better (HOLD6, Addendum 337, H4)?

Hypothesis (written before this script was run): `hybrid_cost` counts amplitude damping with P(1) on the noiseless
state just BEFORE each gate (as `excitation_cost` does), while Aer, like the device during the gate, applies thermal
relaxation to the state AFTER it. The target of each CX in a GHZ chain is |0> before the gate and half excited after
it, so its damping is not counted at all. If the release's placement puts the chain's targets on qubits with short T1,
`hybrid_cost` under-counts exactly that placement.

Two parts:
  detail   FakeAlgiers 4-qubit GHZ chains (the case), FakeAlgiers 6-qubit (control: same choice) and FakeMarrakesh
           4-qubit (control: hybrid chose well), HOLD6's circuits: every candidate's per-gate, per-qubit terms, and its
           simulated infidelity with the full noise model and with thermal relaxation only.
  rescore  every HOLD5 circuit on the nine devices (in-sample, as in Addendum 335's HYBRID): the release's candidates
           re-built, matched to HYBRID's simulated infidelities (no new simulation), and chosen by these estimates:
             exc, pauli, hyb          the release's
             hyb_after                hyb with damping on P(1) just after the gate
             hyb_mid                  hyb with damping on the mean of P(1) before and after
             kraus                    thermal relaxation exactly to first order on the post-gate state: per qubit
                                      1 - sum_k |<psi|K_k|psi>|^2 with the amplitude- and phase-damping Kraus operators
                                      of (t, T1, T2), plus the rest as hyb
             kraus_pur                kraus, with the rest as a depolarizing channel on the gate's qubits at their
                                      reduced purity: lambda (1 - Tr rho_S^2 / d), lambda = (e - floor) d / (d - 1)
  python h4_diag.py detail --repo <repo> --out <dir>
  python h4_diag.py rescore --repo <repo> --out <dir> --device <d>
  python h4_diag.py summary --out <dir>
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
DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
           "FakeMarrakesh", "FakeAachen")
CX = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva")
FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6")
SKIP = ("barrier", "measure", "delay")
SCORES = ("exc", "pauli", "hyb", "hyb_after", "hyb_mid", "kraus", "kraus_pur")
HYBRID = os.path.join("data", "2026-10-04", "hybrid", "diag", "outputs")


def thermal(qp, i, t):
    p = qp[i] if i < len(qp) else None
    t1 = getattr(p, "t1", None) if p is not None else None
    if not t or not t1:
        return None
    t2 = getattr(p, "t2", None)
    return t1, (min(t2, 2 * t1) if t2 else 2 * t1)


def terms(circ, target, detail=False):
    """Per-gate terms on the noiseless state. Returns dict of estimate totals (None if a gate has no matrix or more
    than 16 qubits are touched) and, if detail, a list of per-gate rows."""
    ops = [(ins.operation, tuple(circ.find_bit(b).index for b in ins.qubits)) for ins in circ.data
           if ins.operation.name not in SKIP]
    active = sorted({i for _, q in ops for i in q})
    if len(active) > 16:
        return None, []
    k = max(len(active), 1)
    pos = {p: j for j, p in enumerate(active)}
    qp = getattr(target, "qubit_properties", None) or []
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0

    def rho(axes):
        mm = np.moveaxis(psi, list(axes), list(range(len(axes)))).reshape(2 ** len(axes), -1)
        return mm @ mm.conj().T

    tot = dict(damp_b=0.0, damp_a=0.0, deph=0.0, pauli_th=0.0, kraus_th=0.0, rest=0.0, rest_pur=0.0)
    rows = []
    for op, q in ops:
        try:
            mat = np.asarray(op.to_matrix(), dtype=complex)
        except Exception:
            return None, []
        props = target[op.name].get(q, None) if op.name in target.operation_names else None
        axes = [pos[i] for i in q]
        m = len(axes)
        t = (props.duration or 0.0) if props is not None else 0.0
        p1b = [float(rho([ax])[1, 1].real) for ax in axes]
        rev = axes[::-1]
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
        if props is None:
            continue
        e = props.error or 0.0
        row = dict(gate=op.name, qubits=list(q), t=t, e=e, q=[])
        thermal_f = 1.0
        for j, (i, ax) in enumerate(zip(q, axes)):
            th = thermal(qp, i, t)
            r = rho([ax])
            p1a = float(r[1, 1].real)
            qd = dict(qubit=i, p1_before=p1b[j], p1_after=p1a)
            if th:
                t1, t2 = th
                g = 1.0 - math.exp(-t / t1)
                rate = max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0)
                pphi = (1.0 - math.exp(-t * rate)) / 2.0
                ex, ey, ez = 2 * r[0, 1].real, -2 * r[0, 1].imag, float((r[0, 0] - r[1, 1]).real)
                px = g / 4.0
                pz = max((1.0 - math.exp(-t / t2)) / 2.0 - px, 0.0)
                k0 = np.array([[1, 0], [0, math.sqrt(1 - g)]])
                k1 = np.array([[0, math.sqrt(g)], [0, 0]])
                zz = np.diag([1.0, -1.0])
                tr = lambda K: np.trace(r @ K)
                f = (1 - pphi) * (abs(tr(k0)) ** 2 + abs(tr(k1)) ** 2) + pphi * (abs(tr(zz @ k0)) ** 2 + abs(tr(zz @ k1)) ** 2)
                qd.update(t1=t1, t2=t2, damp_b=t / t1 * p1b[j], damp_a=t / t1 * p1a, deph=pphi * (1 - ez * ez),
                          pauli_th=px * (1 - ex * ex) + px * (1 - ey * ey) + pz * (1 - ez * ez), kraus_th=1.0 - f)
                for key in ("damp_b", "damp_a", "deph", "pauli_th", "kraus_th"):
                    tot[key] += qd[key]
                thermal_f *= (1.0 + 2.0 * math.exp(-t / t2) + math.exp(-t / t1)) / 4.0
            row["q"].append(qd)
        d = 2 ** m
        excess = max(e - (1.0 - (d * thermal_f + 1.0) / (d + 1.0)), 0.0)
        purity = float(np.real(np.trace(np.linalg.matrix_power(rho(axes), 2))))
        row.update(rest=excess * (d + 1) / d, rest_pur=excess * d / (d - 1) * (1 - purity / d), purity=purity)
        tot["rest"] += row["rest"]
        tot["rest_pur"] += row["rest_pur"]
        rows.append(row)
    est = dict(hyb_check=tot["damp_b"] + tot["deph"] + tot["rest"],
               hyb_after=tot["damp_a"] + tot["deph"] + tot["rest"],
               hyb_mid=(tot["damp_b"] + tot["damp_a"]) / 2 + tot["deph"] + tot["rest"],
               kraus=tot["kraus_th"] + tot["rest"], kraus_pur=tot["kraus_th"] + tot["rest_pur"],
               pauli_check=tot["pauli_th"] + tot["rest"], **tot)
    return est, rows


def choose(costs):
    if any(c is None for c in costs):
        return 0
    return min(range(len(costs)), key=lambda j: (costs[j], j))


def setup(args, device):
    from qiskit_ibm_runtime import fake_provider
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    sys.path.insert(0, args.repo)
    import core_fix_c2_eval as H
    rel = H.load_module(os.path.join(args.repo, "psf_compile.py"), "psf_compile")
    if rel.VERSION != "2026-10-04.1":
        raise SystemExit("STOP: psf_compile.py is %s, not release 2026-10-04.1" % rel.VERSION)
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    backend = getattr(fake_provider, device)()
    return H, rel, backend


def candidates(rel, qc, tgt):
    captured = {}
    orig = rel._choose

    def spy(cands, target, score):
        captured["cands"] = list(cands)
        return orig(cands, target, score)

    rel._choose = spy
    try:
        cm = tgt.build_coupling_map()
        nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                           layout_search=True, seed_transpiler=0, target=tgt, placement_refine=True,
                                           final_resynthesis="select", compare_level3=True, compare_floor=True,
                                           candidate_score="hybrid")
    finally:
        rel._choose = orig
    return captured.get("cands") or [("psf", out)], out


def scores_for(rel, c, tgt):
    est, _ = terms(c, tgt)
    if est is None:
        return {s: None for s in SCORES}
    return dict(exc=rel.excitation_cost(c, tgt), pauli=rel.pauli_cost(c, tgt), hyb=rel.hybrid_cost(c, tgt),
                hyb_after=est["hyb_after"], hyb_mid=est["hyb_mid"], kraus=est["kraus"], kraus_pur=est["kraus_pur"],
                hyb_check=est["hyb_check"], pauli_check=est["pauli_check"])


def detail(args):
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    out_rows = []
    for device, n, count in (("FakeAlgiers", 4, 6), ("FakeAlgiers", 6, 3), ("FakeMarrakesh", 4, 3)):
        H, rel, backend = setup(args, device)
        gen = H.load_module(os.path.join(args.repo, "benchmarks", "hold6_eval.py"), "hold6_eval")
        tgt = backend.target
        opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
        sims = dict(full=AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts),
                    thermal=AerSimulator(noise_model=NoiseModel.from_backend(backend, gate_error=False,
                                                                             readout_error=False), **opts))
        done = 0
        for j, (params, qc) in enumerate(gen.family("F5", False)):
            if qc.num_qubits != n or done >= count:
                continue
            done += 1
            cands, chosen = candidates(rel, qc, tgt)
            psi = Statevector(qc).data
            for nm, c in cands:
                est, rows = terms(c, tgt, detail=True)
                cc = c.copy()
                cc.remove_final_measurements(inplace=True)
                cc.save_density_matrix(qubits=list(c.layout.final_index_layout(filter_ancillas=True)[:n]))
                inf = {}
                for key, sim in sims.items():
                    rho = np.asarray(sim.run(cc).result().data(0)["density_matrix"])
                    inf[key] = float(1 - np.real(np.conj(psi) @ rho @ psi))
                out_rows.append(dict(device=device, n=n, index=j, cand=nm, chosen=c is chosen,
                                     two_q=sum(1 for i in c.data if len(i.qubits) == 2), infid=inf,
                                     scores=scores_for(rel, c, tgt), est=est, gates=rows))
        print("detail %s n=%d: %d circuits" % (device, n, done), flush=True)
    with open(os.path.join(args.out, "h4_detail.json"), "w") as f:
        json.dump(out_rows, f)
    L = ["# h4 detail (exploratory)", ""]
    for r in out_rows:
        if r["index"] != min(x["index"] for x in out_rows if x["device"] == r["device"] and x["n"] == r["n"]):
            continue
        s, e = r["scores"], r["est"]
        L.append("## %s n=%d circuit %d, candidate %s%s: infid full %.5f, thermal only %.5f, 2q %d" % (
            r["device"], r["n"], r["index"], r["cand"], " (chosen)" if r["chosen"] else "", r["infid"]["full"],
            r["infid"]["thermal"], r["two_q"]))
        L.append("estimates: " + ", ".join("%s %.5f" % (k, s[k]) for k in SCORES if s.get(k) is not None))
        L.append("thermal parts: damp_before %.5f, damp_after %.5f, deph %.5f, pauli_th %.5f, kraus_th %.5f; rest %.5f, "
                 "rest_pur %.5f" % (e["damp_b"], e["damp_a"], e["deph"], e["pauli_th"], e["kraus_th"], e["rest"],
                                    e["rest_pur"]))
        L += ["", "| gate | qubits | t (ns) | error | qubit | T1 (us) | T2 (us) | P1 before | P1 after | damp_b | damp_a | "
              "deph | pauli_th | kraus_th | rest |", "|" + "---|" * 15]
        for g in r["gates"]:
            for k, qd in enumerate(g["q"]):
                if "t1" not in qd:
                    continue
                L.append("| %s | %s | %.0f | %.5f | %d | %.1f | %.1f | %.3f | %.3f | %.5f | %.5f | %.5f | %.5f | %.5f | %s |" % (
                    g["gate"], g["qubits"], g["t"] * 1e9, g["e"], qd["qubit"], qd["t1"] * 1e6, qd["t2"] * 1e6,
                    qd["p1_before"], qd["p1_after"], qd["damp_b"], qd["damp_a"], qd["deph"], qd["pauli_th"],
                    qd["kraus_th"], "%.5f" % g["rest"] if k == 0 else ""))
        L.append("")
    L += ["## all detail circuits", "", "| device | n | index | cand | chosen | infid full | infid thermal | " +
          " | ".join(SCORES) + " | kraus_th | damp_b | damp_a | deph | rest |", "|" + "---|" * (len(SCORES) + 12)]
    for r in out_rows:
        s, e = r["scores"], r["est"]
        L.append("| %s | %d | %d | %s | %s | %.5f | %.5f | " % (r["device"], r["n"], r["index"], r["cand"],
                                                              "yes" if r["chosen"] else "", r["infid"]["full"],
                                                              r["infid"]["thermal"]) +
                 " | ".join("%.5f" % s[k] for k in SCORES) + " | %.5f | %.5f | %.5f | %.5f | %.5f |" % (
                     e["kraus_th"], e["damp_b"], e["damp_a"], e["deph"], e["rest"]))
    open(os.path.join(args.out, "h4_detail.md"), "w").write("\n".join(L) + "\n")
    print("wrote h4_detail.md", flush=True)


def rescore(args):
    H, rel, backend = setup(args, args.device)
    gen = H.load_module(os.path.join(args.repo, "benchmarks", "hold5_eval.py"), "hold5_eval")
    tgt = backend.target
    hy = json.load(open(os.path.join(args.repo, HYBRID, "hybrid_%s.json" % args.device)))["rows"]
    hy = {(r["family"], r["index"]): r for r in hy}
    t0, rows, mism = time.time(), [], 0
    for fam in FAMILIES:
        for j, (params, qc) in enumerate(gen.family(fam, False)):
            ref = hy[(fam, j)]
            cands, _ = candidates(rel, qc, tgt)
            names = [nm for nm, _ in cands]
            if names != ref["names"]:
                mism += 1
                continue
            row = dict(family=fam, index=j, n=qc.num_qubits, bc=params.get("bc"), cands=[])
            for (nm, c), rc in zip(cands, ref["cands"]):
                s = scores_for(rel, c, tgt)
                same = (sum(1 for i in c.data if len(i.qubits) == 2) == rc["two_q"] and s["hyb"] is not None and
                        rc["hyb"] is not None and abs(s["hyb"] - rc["hyb"]) <= 1e-12 * max(1.0, rc["hyb"]))
                if not same or "infid" not in rc:
                    mism += 1
                    row = None
                    break
                row["cands"].append(dict(name=nm, infid=rc["infid"], **s))
            if row is not None:
                rows.append(row)
        print("  %s %s: %d rows, %d mismatches, %.0f s" % (args.device, fam, len(rows), mism, time.time() - t0),
              flush=True)
    with open(os.path.join(args.out, "h4_rescore_%s.json" % args.device), "w") as f:
        json.dump(dict(device=args.device, version=rel.VERSION, mismatches=mism, rows=rows, wall_s=time.time() - t0), f)
    print("wrote h4_rescore_%s.json: %d rows, %d mismatches" % (args.device, len(rows), mism), flush=True)


def summary(args):
    L = ["# h4 rescore summary (exploratory, in-sample on HOLD5's circuits)", "",
         "PICK_x: the candidate chosen by estimate x among the release's candidates (as `_choose`), mean infidelity",
         "relative to the choice by hyb (the release). ORC: the measured best. rank: circuits where x picks the best.", ""]
    head = "| device | rows | mismatches | " + " | ".join(SCORES) + " | ORC | " + " | ".join("rank " + s for s in SCORES) + " |"
    L += [head, "|" + "---|" * (head.count("|") - 1)]
    fam = {}
    checks = []
    for d in DEVICES:
        p = os.path.join(args.out, "h4_rescore_%s.json" % d)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        rows = R["rows"]
        pick = lambda r, s: r["cands"][choose([c[s] for c in r["cands"]])]["infid"]
        orc = lambda r: min(c["infid"] for c in r["cands"])
        base = np.mean([pick(r, "hyb") for r in rows])
        rk = {s: sum(abs(pick(r, s) - orc(r)) <= 1e-12 for r in rows) for s in SCORES}
        L.append("| %s | %d | %d | " % (d, len(rows), R["mismatches"]) +
                 " | ".join("%.4f" % (np.mean([pick(r, s) for r in rows]) / base) for s in SCORES) +
                 " | %.4f | " % (np.mean([orc(r) for r in rows]) / base) + " | ".join(str(rk[s]) for s in SCORES) + " |")
        checks.append((d, max(abs(c["hyb_check"] - c["hyb"]) for r in rows for c in r["cands"] if c["hyb"] is not None),
                       max(abs(c["pauli_check"] - c["pauli"]) for r in rows for c in r["cands"] if c["pauli"] is not None)))
        for f in FAMILIES + ("F3o", "F5n4"):
            rs = [r for r in rows if r["family"] == f or (f == "F3o" and r["family"] == "F3" and r["bc"] == "o")
                  or (f == "F5n4" and r["family"] == "F5" and r["n"] == 4)]
            if rs:
                b = np.mean([pick(r, "hyb") for r in rs])
                fam[(d, f)] = {s: np.mean([pick(r, s) for r in rs]) / b for s in SCORES}
    for s in ("pauli", "hyb_after", "hyb_mid", "kraus", "kraus_pur"):
        L += ["", "PICK_%s / PICK_hyb by family:" % s, "", "| family | " + " | ".join(x.replace("Fake", "") for x in DEVICES) + " |",
              "|---" * (len(DEVICES) + 1) + "|"]
        for f in FAMILIES + ("F3o", "F5n4"):
            L.append("| %s | " % f + " | ".join("%.3f" % fam[(d, f)][s] if (d, f) in fam else "-" for d in DEVICES) + " |")
    L += ["", "Checks (largest |re-computed - release|): " + "; ".join("%s hyb %.1e pauli %.1e" % c for c in checks)]
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("detail", "rescore", "summary"))
    ap.add_argument("--repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=DEVICES)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dict(detail=detail, rescore=rescore, summary=summary)[args.part](args)


if __name__ == "__main__":
    main()
