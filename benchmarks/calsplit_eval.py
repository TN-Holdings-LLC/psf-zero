"""calsplit_eval.py -- CALSPLIT (2026-10-07): does the recommended call's task-level advantage survive when the
calibration it compiles with is not the device's? The first neutrality test of the "Task-Oriented Quantum Benchmark"
draft: the compiler reads one calibration, the score uses another. Pre-registration: Addendum 402.

DEPTH-R's classifiers (Addenda 399-400) are reused as they are: its data (split seed 4), its trained parameters
(data/2026-10-07/depth_r/theta_*.npy), its circuits, its exact-z noisy simulation with the device's TRUE noise model,
its readout and shots. What changes is what the calibration-aware compilers see: a stale Target made by STALE's
stale_target() (benchmarks/stale_eval.py, Addendum 334; instruction errors x exp(N(0, 0.3)), T1 and T2 x
exp(N(0, 0.2)), failed entries unchanged), drawn three times per device with new seeds (stale_eval.BASE set to
91,000,000 + 1,000,000 k for draw k = 0, 1, 2; STALE itself used 80,000,000).

Arms:
  REC   release 2026-10-07.1, recommended call, stale Target                  (calibration-aware)
  RPSF  release 2026-10-07.1, target + placement_refine, stale Target         (calibration-aware)
  L3T   Qiskit level 3 with the stale Target, approximation_degree 1.0         (calibration-aware)
  DEF   release 2026-10-07.1, default call (coupling map and basis only)       (calibration-blind)
  L3B   Qiskit level 3 with coupling map and basis only, approximation 1.0     (calibration-blind)
The blind arms do not depend on the draw and run once (draw "-"). K1 and K5 compare REC with the BETTER blind arm per
device (larger pooled margin; fewer flips): in the dry run the default call placed gates on FakeTorino's failed
couplers, which would make "calibration helps" true for that reason alone (Addendum 402, section 4).

  python benchmarks/calsplit_eval.py deploy --dataset BC --n 6 --device FakeAuckland --arm REC --draw 0 --out DIR [--dry]
  python benchmarks/calsplit_eval.py score --out DIR
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys
import time

import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, REPO]
import depth_r_eval as R  # noqa: E402  (DEPTH's depth_eval.py through DEPTH-R, both unchanged)
import stale_eval as S    # noqa: E402  (STALE's stale_target, unchanged)

E = R.E
THETA = {False: os.path.join(REPO, "data", "2026-10-07", "depth_r"),
         True: os.path.join(REPO, "data", "2026-10-07", "depth_r_dry")}
AWARE, BLIND = ("REC", "RPSF", "L3T"), ("DEF", "L3B")
ARMS = AWARE + BLIND
DRAWS = (0, 1, 2)
DEVICES = ("FakeAuckland", "FakeTorino")
BUDGETS = (15, 63, 255, 1023)
TOL = 1e-12


class _Backend:
    """What depth_eval.compiler reads from a backend: its target."""
    def __init__(self, target):
        self.target = target


def stale(be, device, k):
    S.BASE = 91_000_000 + 1_000_000 * k
    t, changed = S.stale_target(be.target, device)
    return t, changed


def compiler(arm, be, P, stale_t):
    from qiskit import transpile
    t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    cm = t.build_coupling_map()
    if arm in AWARE:
        return E.compiler(arm, _Backend(stale_t), P)  # REC is read as C12 by depth_r_eval
    if arm == "DEF":
        f = lambda qc: P.compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx",  # noqa
                                              layout_search=True, seed_transpiler=0)
    else:
        f = lambda qc: transpile(qc, coupling_map=cm, basis_gates=basis, optimization_level=3,  # noqa: E731
                                 seed_transpiler=0, approximation_degree=1.0)

    def run(qc):
        with contextlib.redirect_stdout(io.StringIO()):
            return f(qc)
    return run


def do_deploy(args):
    from qiskit_ibm_runtime import fake_provider as fp
    if (args.arm in AWARE) != (args.draw != "-"):
        raise SystemExit("aware arms take a draw 0-2, blind arms the draw '-'")
    split, _ = E.seeds(args.dry)
    Ls = (1, 4, 12) if args.dry else E.LS
    _, _, Xte, yte = E.data(args.dataset, args.n, split)
    if args.dry:
        Xte, yte = Xte[:12], yte[:12]
    be = getattr(fp, args.device)()
    P = E.load_psf(R.RELEASE)
    st, changed = (stale(be, args.device, int(args.draw)) if args.draw != "-" else (None, 0))
    comp = compiler(args.arm, be, P, st)
    sim = E.Noisy(be)                                     # the device's TRUE noise model
    edges, qubits = P._failed_elements(be.target, 0.5)
    rows = []
    for L in Ls:
        th = np.load(E.theta_path(THETA[args.dry], args.dataset, args.n, L))
        zid = E.forward(th, Xte, args.n, L)
        for i, (x, yv) in enumerate(zip(Xte, yte)):
            qc = E.circuit(x, th, args.n, L)
            t0 = time.perf_counter()
            out = comp(qc)
            tc = time.perf_counter() - t0
            z0, p, touched = sim.z(out, noisy=False)
            zn, _, _ = sim.z(out, noisy=True)
            e01, e10 = sim.readout(p)
            on_failed = sum(1 for g in out.data if len(g.qubits) == 2 and g.operation.name != "barrier"
                            and tuple(out.find_bit(q).index for q in g.qubits) in edges)
            row = dict(L=L, i=i, y=float(yv), z_ideal=float(zid[i]), z_compiled_noiseless=z0, z_noisy=zn,
                       infid=R.infidelity(qc, out), e01=e01, e10=e10, on_failed=on_failed,
                       n2q=sum(1 for g in out.data if len(g.qubits) == 2), touched=touched, compile_s=round(tc, 4))
            if i < 2:
                row["z_fullwidth"] = sim.z_fullwidth(out)
            rows.append(row)
        sel = [r for r in rows if r["L"] == L]
        print("DEPLOY", json.dumps(dict(L=L, margin=float(np.mean([r["y"] * r["z_noisy"] for r in sel])),
                                         max_infid=max(r["infid"] for r in sel),
                                         n2q=float(np.mean([r["n2q"] for r in sel])))), flush=True)
    import qiskit
    name = f"deploy_{args.dataset}_n{args.n}_{args.device}_{args.arm}_d{args.draw}.json"
    meta = dict(R.meta(args, mode="deploy", dataset=args.dataset, n=args.n, device=args.device, arm=args.arm,
                       draw=args.draw, stale_base=(S.BASE if args.draw != "-" else None), stale_t1_changed=changed,
                       psf_version=P.VERSION, qiskit_version=qiskit.__version__),
                calsplit_sha=R.norm_sha(os.path.abspath(__file__)), stale_eval_sha=R.norm_sha(S.__file__))
    with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(meta=meta, rows=rows), f)


def acc_at(rows, shots):
    q = np.array([min(1.0, max(0.0, ((1 + r["z_noisy"]) / 2) * (1 - r["e01"]) + ((1 - r["z_noisy"]) / 2) * r["e10"]))
                  for r in rows])
    y = np.array([r["y"] for r in rows])
    p_pos = binom.sf((shots - 1) // 2, shots, q)
    return float(np.mean(np.where(y > 0, p_pos, 1 - p_pos)))


def verdict(ok, bad):
    return "CONFIRMED" if ok else ("REFUTED" if bad else "AMBIGUOUS")


def do_score(args):
    D, metas = {}, []
    for p in glob.glob(os.path.join(args.out, "deploy_*.json")):
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        metas.append(m)
        D[(m["dataset"], m["n"], m["device"], m["arm"], m["draw"])] = d["rows"]
    dry = any(m["dry"] for m in metas)
    Ls = sorted({r["L"] for rows in D.values() for r in rows})
    out = [f"# CALSPLIT score{' (DRY RUN -- not a result)' if dry else ''}", ""]
    exp_files = len(E.DATASETS) * len(E.NS) * len(DEVICES) * (len(AWARE) * len(DRAWS) + len(BLIND))
    infid = max(r["infid"] for rows in D.values() for r in rows)
    fw = [abs(r["z_fullwidth"] - r["z_noisy"]) for rows in D.values() for r in rows if "z_fullwidth" in r]
    vers = {m["psf_version"] for m in metas}
    rel = {m.get("c12_sha") for m in metas}
    stale_ok = all(m["stale_t1_changed"] > 0 for m in metas if m["draw"] != "-")
    p0 = bool(len(D) == exp_files and infid <= 1e-6 and fw and max(fw) <= 1e-9 and vers == {R.RELEASE_VERSION}
              and rel == {R.norm_sha(R.RELEASE)} and stale_ok)
    out += [f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(D)}/{exp_files}; max state infidelity {infid:.2e} "
            f"(<= 1e-06); max |reduced - whole-device| {max(fw) if fw else float('nan'):.2e} over {len(fw)} circuits; "
            f"versions {sorted(vers)}; release as locked {rel == {R.norm_sha(R.RELEASE)}}; every stale Target "
            f"differs from the true one {stale_ok}", ""]
    draws = {a: (DRAWS if a in AWARE else ("-",)) for a in ARMS}

    def rows_of(ds, n, dev, a, d, L):
        return [r for r in D[(ds, n, dev, a, str(d))] if r["L"] == L]

    def pool(dev, a, key):
        """Mean over draws of the mean over the 24 cells (dataset, n, L)."""
        per = []
        for d in draws[a]:
            cells = []
            for ds in E.DATASETS:
                for n in E.NS:
                    for L in Ls:
                        rr = rows_of(ds, n, dev, a, d, L)
                        if key == "margin":
                            cells.append(np.mean([r["y"] * r["z_noisy"] for r in rr]))
                        elif key == "flips":
                            cells.append(sum(int(np.sign(r["z_noisy"]) != np.sign(r["z_ideal"])) for r in rr))
                        else:
                            cells.append(acc_at(rr, key))
            per.append(float(np.sum(cells)) if key == "flips" else float(np.mean(cells)))
        return float(np.mean(per)), per

    # DEPTH-R's own same-calibration REC - RPSF (Addendum 400), recomputed from its committed output
    ref = {}
    for dev in DEVICES:
        mm = {}
        for a in ("REC", "RPSF"):
            cells = []
            for ds in E.DATASETS:
                for n in E.NS:
                    p = os.path.join(THETA[dry], f"deploy_{ds}_n{n}_{dev}_{a}.json")
                    rows = json.load(open(p, encoding="utf-8"))["rows"]
                    for L in Ls:
                        rr = [r for r in rows if r["L"] == L]
                        cells.append(np.mean([r["y"] * r["z_noisy"] for r in rr]))
            mm[a] = float(np.mean(cells))
        ref[dev] = mm["REC"] - mm["RPSF"]
    npts = sum(len(rows_of(ds, n, DEVICES[0], "DEF", "-", L)) for ds in E.DATASETS for n in E.NS for L in Ls)
    res, det = {}, {k: [] for k in ("K1", "K2", "K3", "K4", "K5")}
    ok = {k: True for k in det}
    bad = {k: False for k in det}
    for dev in DEVICES:
        m = {a: pool(dev, a, "margin") for a in ARMS}
        fl = {a: pool(dev, a, "flips")[0] for a in ARMS}
        bm = max(BLIND, key=lambda a: m[a][0])
        bf = min(BLIND, key=lambda a: fl[a])
        d1 = m["REC"][0] - m[bm][0]
        d2 = m["REC"][0] - m["RPSF"][0]
        d3 = ref[dev] - d2
        d4 = m["REC"][0] - m["L3T"][0]
        d5 = (fl["REC"] - fl[bf]) / npts
        det["K1"].append(f"{dev} REC-{bm} {d1:+.4f} (DEF {m['DEF'][0]:.4f}, L3B {m['L3B'][0]:.4f})")
        det["K2"].append(f"{dev} REC-RPSF {d2:+.4f} (per draw {[round(a - b, 4) for a, b in zip(m['REC'][1], m['RPSF'][1])]})")
        det["K3"].append(f"{dev} same-calibration {ref[dev]:+.4f} minus stale {d2:+.4f} = {d3:+.4f}")
        det["K4"].append(f"{dev} REC-L3T {d4:+.4f}")
        det["K5"].append(f"{dev} flips REC {fl['REC']:.1f}, {bf} {fl[bf]:.0f} (DEF {fl['DEF']:.0f}, L3B "
                         f"{fl['L3B']:.0f}) of {npts} (rate difference {d5:+.4f})")
        ok["K1"] &= d1 >= 0.002 - TOL
        bad["K1"] |= d1 < -0.002 - TOL
        ok["K2"] &= d2 >= 0.003 - TOL
        bad["K2"] |= d2 < -TOL
        ok["K3"] &= d3 <= 0.005 + TOL
        bad["K3"] |= d3 > 0.01 + TOL
        ok["K4"] &= abs(d4) <= 0.01 + TOL
        bad["K4"] |= d4 < -0.02 - TOL
        ok["K5"] &= d5 <= 0.002 + TOL
        bad["K5"] |= d5 > 0.01 + TOL
    labels = {"K1": "calibration still helps: REC - better blind arm pooled margin >= +0.002 on both devices",
              "K2": "the advantage over the guarded call survives: REC - RPSF >= +0.003 on both devices",
              "K3": "it shrinks by at most 0.005 against DEPTH-R's same-calibration value, on both devices",
              "K4": "REC level with L3T (both stale): |REC - L3T| <= 0.01 on both devices",
              "K5": "no more flipped answers than the better blind arm: flip-rate difference <= +0.002"}
    for k in det:
        res[k] = verdict(ok[k], bad[k])
        out.append(f"- {k} ({labels[k]}): **{res[k]}** ({'; '.join(det[k])})")
    out += ["", "Reported without prediction -- pooled margin per arm, accuracy at fixed shot budgets (exact binomial, "
            "as Addendum 401), two-qubit gates on failed couplers of the true device:", ""]
    for dev in DEVICES:
        out.append(f"## {dev}")
        out.append("")
        out.append("| arm | margin | " + " | ".join(f"acc at {b} shots" for b in BUDGETS) + " | gates on failed couplers |")
        out.append("|" + "---|" * (3 + len(BUDGETS)))
        for a in ARMS:
            mg = pool(dev, a, "margin")[0]
            ab = [pool(dev, a, b)[0] for b in BUDGETS]
            nf = sum(r["on_failed"] for (ds, n, dv, aa, d), rows in D.items() if dv == dev and aa == a for r in rows)
            out.append(f"| {a} | {mg:.4f} | " + " | ".join(f"{v:.4f}" for v in ab) + f" | {nf} |")
        out.append("")
    out += ["SUMMARY " + json.dumps(res)]
    txt = "\n".join(out)
    with open(os.path.join(args.out, "score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")
    print(txt)


def main():
    if R.norm_sha(R.DEPTH) != R.DEPTH_SHA:
        raise SystemExit("DEPTH's depth_eval.py is not the locked file")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["deploy", "score"])
    ap.add_argument("--dataset")
    ap.add_argument("--n", type=int)
    ap.add_argument("--device")
    ap.add_argument("--arm")
    ap.add_argument("--draw", default="-")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    a.c12 = R.RELEASE
    os.makedirs(a.out, exist_ok=True)
    {"deploy": do_deploy, "score": do_score}[a.mode](a)


if __name__ == "__main__":
    main()
