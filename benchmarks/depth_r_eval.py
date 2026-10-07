"""depth_r_eval.py -- DEPTH-R (2026-10-07): QML-2 "DEPTH" stage 1 (Addenda 345-346) again, with its exactness check
stated as a state infidelity, new data, and release 2026-10-07.1. Pre-registration: Addendum 399.

Everything that is not listed here is depth_eval.py's, imported unchanged from data/2026-10-05/workplace/depth1/
(normalized SHA-256 82488d96...): data, model, training, noise simulation, shots and readout, fine-tuning.
Changes:
  - seeds: split seed 4 and init seed 4 (0: pilot, 1: DEPTH, 2: development and the dry run, 3: READOUT);
  - arms: RPSF = release 2026-10-07.1 with target + placement_refine; REC = release 2026-10-07.1's recommended call
    (DEPTH's C12 call); L3T = Qiskit level 3 with the Target, approximation_degree 1.0 (unchanged);
  - P0's exactness: every compiled circuit's noiseless state, on the qubits it touches, against the logical circuit's
    output placed at the circuit's final layout (other touched qubits in |0>): state infidelity <= 1e-6
    (Addendum 346, section 1). |z_compiled - z_ideal| is still recorded and reported, not gated;
  - deploy records the infidelity and the compiler's version; the score also reports, without prediction, how much
    of the accuracy change with depth is noise (ideal accuracy minus shot accuracy).

  python benchmarks/depth_r_eval.py train    --dataset BC --n 6 --out DIR [--dry]
  python benchmarks/depth_r_eval.py deploy   --dataset BC --n 6 --device FakeAuckland --arm REC --out DIR [--dry]
  python benchmarks/depth_r_eval.py finetune --seed 1 --L 12 --out DIR [--dry]
  python benchmarks/depth_r_eval.py score    --out DIR
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DEPTH = os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1", "depth_eval.py")
DEPTH_SHA = "82488d96144bb1c88f69676a0e6a642c22d4756a47d8a1ad8b07be5327c796ac"
RELEASE = os.path.join(REPO, "psf_compile.py")
RELEASE_VERSION = "2026-10-07.1"
sys.path[:0] = [HERE, REPO]  # psf_smart_layout.py (benchmarks/) for the release

spec = importlib.util.spec_from_file_location("depth_eval", DEPTH)
E = importlib.util.module_from_spec(spec)
sys.modules["depth_eval"] = E
spec.loader.exec_module(E)

E.SPLIT_SEED, E.INIT_SEED = 4, 4
E.FT["arm"] = "REC"
ARMS = ("RPSF", "REC", "L3T")
INFID_TOL = 1e-6
_compiler = E.compiler
E.compiler = lambda arm, be, P: _compiler({"REC": "C12"}.get(arm, arm), be, P)
_meta = E.meta


def norm_sha(path):
    return E.norm_sha(path)


def meta(args, **kw):
    m = _meta(args, **kw)
    m.update(harness_sha=norm_sha(os.path.abspath(__file__)), depth_eval_sha=norm_sha(DEPTH),
             seeds=list(E.seeds(args.dry)))
    return m


E.meta = meta


def infidelity(qc, out):
    """1 - |<logical output placed at out's final layout | compiled output>|^2 on the touched qubits (|0> start)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    red, active, idx, fin = E.Noisy.reduce(out)
    ref = QuantumCircuit(len(active))
    ref.compose(qc, qubits=[idx[fin[j]] for j in range(qc.num_qubits)], inplace=True)
    return float(max(0.0, 1.0 - abs(Statevector(ref).inner(Statevector(red))) ** 2))


def do_deploy(args):
    """depth_eval.do_deploy with two added fields per row (infid, and the compiler's version in the meta)."""
    from qiskit_ibm_runtime import fake_provider as fp
    split, _ = E.seeds(args.dry)
    Ls = (1, 4, 12) if args.dry else E.LS
    Xtr, ytr, Xte, yte = E.data(args.dataset, args.n, split)
    if args.dry:
        Xte, yte = Xte[:12], yte[:12]
    be = getattr(fp, args.device)()
    P = E.load_psf(RELEASE)
    comp = E.compiler(args.arm, be, P)
    sim = E.Noisy(be)
    rows = []
    for L in Ls:
        th = np.load(E.theta_path(args.out, args.dataset, args.n, L))
        zid = E.forward(th, Xte, args.n, L)
        for i, (x, yv) in enumerate(zip(Xte, yte)):
            qc = E.circuit(x, th, args.n, L)
            t0 = time.perf_counter()
            out = comp(qc)
            tc = time.perf_counter() - t0
            z0, p, touched = sim.z(out, noisy=False)
            zn, _, _ = sim.z(out, noisy=True)
            e01, e10 = sim.readout(p)
            zs = E.shot_z(zn, e01, e10, f"{args.dataset}|{args.n}|{args.device}|{L}|{i}")
            row = dict(L=L, i=i, y=float(yv), z_ideal=float(zid[i]), z_compiled_noiseless=z0, z_noisy=zn,
                       infid=infidelity(qc, out), shot_acc=float(np.mean(np.sign(zs) == yv)),
                       shot_flip=float(np.mean(np.sign(zs) != np.sign(zid[i]))), e01=e01, e10=e10,
                       n2q=sum(1 for g in out.data if len(g.qubits) == 2), touched=touched, compile_s=round(tc, 4))
            if i < 2:  # P0: reduced simulation equals the whole-device simulation
                row["z_fullwidth"] = sim.z_fullwidth(out)
            rows.append(row)
        sel = [r for r in rows if r["L"] == L]
        print("DEPLOY", json.dumps(dict(L=L, margin=float(np.mean([r["y"] * r["z_noisy"] for r in sel])),
                                         shot_acc=float(np.mean([r["shot_acc"] for r in sel])),
                                         max_infid=max(r["infid"] for r in sel),
                                         n2q=float(np.mean([r["n2q"] for r in sel])))), flush=True)
    import qiskit
    name = f"deploy_{args.dataset}_n{args.n}_{args.device}_{args.arm}.json"
    with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(meta=meta(args, mode="deploy", dataset=args.dataset, n=args.n, device=args.device,
                                 arm=args.arm, psf_version=P.VERSION, release_sha=norm_sha(RELEASE),
                                 qiskit_version=qiskit.__version__), rows=rows), f)


def verdict(ok, bad):
    return "CONFIRMED" if ok else ("REFUTED" if bad else "AMBIGUOUS")


def do_score(args):
    """depth_eval.do_score's predictions H1-H6 and gate, word for word, with C12 read as REC; P0 as in Addendum 399."""
    D, T = {}, {}
    for p in glob.glob(os.path.join(args.out, "deploy_*.json")):
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        D[(m["dataset"], m["n"], m["device"], m["arm"])] = (d["rows"], m)
    for p in glob.glob(os.path.join(args.out, "train_*.json")):
        d = json.load(open(p, encoding="utf-8"))
        T[(d["meta"]["dataset"], d["meta"]["n"])] = ({r["L"]: r for r in d["rows"]}, d["meta"])
    F = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(args.out, "finetune_*.json")))]
    metas = [m for _, m in D.values()] + [m for _, m in T.values()] + [f["meta"] for f in F]
    dry = any(m["dry"] for m in metas)
    Ls = sorted({r["L"] for rows, _ in D.values() for r in rows})
    out = [f"# DEPTH-R score{' (DRY RUN -- not a result)' if dry else ''}", ""]

    def rows_of(ds, n, dev, arm, Lv):
        return [r for r in D[(ds, n, dev, arm)][0] if r["L"] == Lv]

    def cell(ds, n, dev, arm, Lv, key):
        rows = rows_of(ds, n, dev, arm, Lv)
        if key == "acc":
            return float(np.mean([np.sign(r["z_noisy"]) == r["y"] for r in rows]))
        if key == "margin":
            return float(np.mean([r["y"] * r["z_noisy"] for r in rows]))
        if key == "shot_acc":
            return float(np.mean([r["shot_acc"] for r in rows]))
        if key == "flip":
            return float(np.mean([np.sign(r["z_noisy"]) != np.sign(r["z_ideal"]) for r in rows]))
        if key == "n2q":
            return float(np.mean([r["n2q"] for r in rows]))

    exp_files = len(E.DATASETS) * len(E.NS) * len(E.DEVICES) * len(ARMS)
    infid = max(r["infid"] for rows, _ in D.values() for r in rows)
    zdiff = max(abs(r["z_compiled_noiseless"] - r["z_ideal"]) for rows, _ in D.values() for r in rows)
    fw = [abs(r["z_fullwidth"] - r["z_noisy"]) for rows, _ in D.values() for r in rows if "z_fullwidth" in r]
    vers = {m.get("psf_version") for _, m in D.values()} | {m.get("psf_version") for m in metas if "psf_version" in m}
    seeds = {tuple(m.get("seeds", ())) for m in metas}
    want_seeds = {(2, 2)} if dry else {(4, 4)}
    ft_ok = len(F) == 4 or dry
    rel_shas = {m.get("c12_sha") for m in metas}
    p0 = bool(len(D) == exp_files and infid <= INFID_TOL and fw and max(fw) <= 1e-9 and ft_ok
              and vers == {RELEASE_VERSION} and seeds == want_seeds and rel_shas == {norm_sha(RELEASE)})
    out += [f"P0: {'PASS' if p0 else 'FAIL'} -- deploy files {len(D)}/{exp_files}; max state infidelity {infid:.2e} "
            f"(<= {INFID_TOL:.0e}); max |reduced - whole-device| {max(fw) if fw else float('nan'):.2e} over {len(fw)} "
            f"circuits (<= 1e-9); finetune files {len(F)}; versions {sorted(map(str, vers))}; seeds {sorted(seeds)}; "
            f"release file as locked in every output {rel_shas == {norm_sha(RELEASE)}}",
            f"(recorded, not gated: max |z_compiled_noiseless - z_ideal| {zdiff:.2e})", ""]
    for dev in E.DEVICES:
        out += [f"## {dev}", "", "| dataset | n | L | ideal acc / margin | " + " | ".join(
            f"{a} acc / shot acc / margin / flip / 2q" for a in ARMS) + " |", "|" + "---|" * (4 + len(ARMS))]
        for ds in E.DATASETS:
            for n in E.NS:
                for Lv in Ls:
                    t = T[(ds, n)][0][Lv]
                    out.append(f"| {ds} | {n} | {Lv} | {t['ideal_acc']:.3f} / {t['ideal_margin']:.3f} | " + " | ".join(
                        f"{cell(ds, n, dev, a, Lv, 'acc'):.3f} / {cell(ds, n, dev, a, Lv, 'shot_acc'):.3f} / "
                        f"{cell(ds, n, dev, a, Lv, 'margin'):.3f} / {cell(ds, n, dev, a, Lv, 'flip'):.3f} / "
                        f"{cell(ds, n, dev, a, Lv, 'n2q'):.0f}" for a in ARMS) + " |")
        out.append("")
    res, A = {}, "FakeAuckland"
    ok1, bad1, det = True, False, []
    for ds in E.DATASETS:
        for a in ARMS:
            ms = [cell(ds, 6, A, a, Lv, "margin") for Lv in Ls]
            det.append(f"{ds}/{a}: deepest {ms[-1]:.3f}, best {max(ms):.3f} at L={Ls[int(np.argmax(ms))]}")
            ok1 &= ms[-1] < 0.8 * max(ms)
            bad1 |= int(np.argmax(ms)) == len(Ls) - 1
    res["H1"] = verdict(ok1, bad1)
    out.append(f"- H1 (noise-limited depth, margin; FakeAuckland, n=6): **{res['H1']}** ({'; '.join(det)})")
    det, okd, badd = [], [], True
    for ds in E.DATASETS:
        nt = len(rows_of(ds, 6, A, ARMS[0], Ls[0]))
        oks = []
        for a in ARMS:
            sa = [cell(ds, 6, A, a, Lv, "shot_acc") for Lv in Ls]
            oks.append(max(sa) - sa[-1] >= 2.0 / nt - 1e-12)
            badd &= sa[-1] >= max(sa) - 1e-12
            det.append(f"{ds}/{a}: deepest {sa[-1]:.3f}, best {max(sa):.3f}")
        okd.append(all(oks))
    res["H2"] = verdict(any(okd), badd)
    out.append(f"- H2 (noise-limited depth, shot accuracy; FakeAuckland, n=6): **{res['H2']}** ({'; '.join(det)})")
    det3, det4, det5 = [], [], []
    ok3 = ok4 = ok5 = True
    bad3 = bad4 = bad5 = False
    for dev in E.DEVICES:
        def pool(a, key):
            return float(np.mean([cell(ds, n, dev, a, Lv, key) for ds in E.DATASETS for n in E.NS for Lv in Ls]))
        mR, mC, mL = pool("RPSF", "margin"), pool("REC", "margin"), pool("L3T", "margin")
        fR, fC = pool("RPSF", "flip"), pool("REC", "flip")
        det3.append(f"{dev} REC-RPSF {mC - mR:+.4f}")
        det4.append(f"{dev} REC-L3T {mC - mL:+.4f}")
        det5.append(f"{dev} REC {fC:.4f} vs RPSF {fR:.4f}")
        ok3 &= mC - mR >= 0.005
        bad3 |= mC - mR < 0
        ok4 &= abs(mC - mL) <= 0.01
        bad4 |= mC < mL - 0.02
        ok5 &= fC <= fR
        bad5 |= fC > fR + 0.01
    res["H3"], res["H4"], res["H5"] = verdict(ok3, bad3), verdict(ok4, bad4), verdict(ok5, bad5)
    out.append(f"- H3 (REC keeps more margin than RPSF, pooled, every device): **{res['H3']}** ({'; '.join(det3)})")
    out.append(f"- H4 (REC level with L3T, pooled margin, every device): **{res['H4']}** ({'; '.join(det4)})")
    out.append(f"- H5 (REC flips no more predictions than RPSF, pooled, every device): **{res['H5']}** "
               f"({'; '.join(det5)})")
    f12 = [f["result"] for f in F if f["meta"]["L"] == 12]
    f4 = [f["result"] for f in F if f["meta"]["L"] == 4]
    if f12:
        dN = float(np.mean([r["FTN"]["margin"] - r["DEP"]["margin"] for r in f12]))
        d0 = float(np.mean([r["FT0"]["margin"] - r["DEP"]["margin"] for r in f12]))
        res["H6"] = verdict(dN >= 0.02 and dN > d0, dN < -0.01)
        out.append(f"- H6 (fine-tuning through the noise helps at L=12): **{res['H6']}** (FTN-DEP {dN:+.4f}, "
                   f"FT0-DEP {d0:+.4f})")
    for tag, fs in (("L=4", f4), ("L=12", f12)):
        for r in fs:
            out.append(f"  - {tag}: DEP {r['DEP']}, FT0 {r['FT0']}, FTN {r['FTN']}")
    gate_peak = res["H1"] == "CONFIRMED" or res["H2"] == "CONFIRMED"
    big = []
    for dev in E.DEVICES:
        for ds in E.DATASETS:
            for n in E.NS:
                nt = len(rows_of(ds, n, dev, "REC", Ls[0]))
                for Lv in Ls:
                    if abs(cell(ds, n, dev, "REC", Lv, "shot_acc") - cell(ds, n, dev, "RPSF", Lv, "shot_acc")) \
                            >= 2.0 / nt - 1e-12:
                        big.append(f"{dev}/{ds}/n{n}/L{Lv}")
    gate_diff = res["H3"] == "CONFIRMED" or len(big) > 0
    out += ["", f"GATE (stage 2): {'GO' if (gate_peak and gate_diff) else 'NO-GO'} -- peak {gate_peak}, compiler "
            f"difference {gate_diff} (cells with |shot acc REC - RPSF| >= 2 test points: {', '.join(big) or 'none'})"]
    out += ["", "Reported without prediction -- the noise's share of the accuracy (FakeAuckland, n = 6): ideal "
            "accuracy minus shot accuracy, per L (points of the test set):"]
    for ds in E.DATASETS:
        nt = len(rows_of(ds, 6, A, "REC", Ls[0]))
        for a in ARMS:
            out.append(f"  - {ds}/{a}: " + ", ".join(
                f"L={Lv} {(T[(ds, 6)][0][Lv]['ideal_acc'] - cell(ds, 6, A, a, Lv, 'shot_acc')) * nt:+.1f}"
                for Lv in Ls))
    out += ["", "SUMMARY " + json.dumps(res)]
    txt = "\n".join(out)
    with open(os.path.join(args.out, "score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")
    print(txt)


def main():
    if norm_sha(DEPTH) != DEPTH_SHA:
        raise SystemExit(f"{DEPTH} is not DEPTH's locked file")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "deploy", "finetune", "score"])
    ap.add_argument("--dataset")
    ap.add_argument("--n", type=int)
    ap.add_argument("--device")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--L", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    a.c12 = RELEASE
    os.makedirs(a.out, exist_ok=True)
    {"train": E.do_train, "deploy": do_deploy, "finetune": E.do_finetune, "score": do_score}[a.mode](a)


if __name__ == "__main__":
    main()
