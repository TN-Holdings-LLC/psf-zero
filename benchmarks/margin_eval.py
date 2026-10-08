"""margin_eval.py -- MARGIN (2026-10-07): does candidate 2026-10-07.c24 (changelog item 51: the recommended call
switches to an alternative only if its estimate is lower by more than SWITCH_MARGIN = 0.05) keep the recommended
call's gain with an up-to-date calibration and lose less with a stale one? Pre-registration: Addendum 405.

DEPTH-R's machinery (Addenda 399-400) and CALSPLIT's stale calibrations (Addenda 402-403) are reused: DEPTH's
depth_eval.py (data, model, training, circuits, exact-z noisy simulation, readout) through depth_r_eval.py, and
STALE's stale_target() (benchmarks/stale_eval.py; instruction errors x exp(N(0, 0.3)), T1 and T2 x exp(N(0, 0.2)),
failed entries unchanged), all imported unchanged. What is new:
  - data: a new split and init seed 5 (0: pilot, 1: DEPTH, 2: development and the dry run, 3: READOUT, 4: DEPTH-R
    and CALSPLIT); the classifiers are trained here (4 jobs), because item 51 was proposed after seeing seed 4;
  - calibrations the compilers read: "t", the device's true Target, and stale draws 0-2 (stale_eval.BASE set to
    97,000,000 + 1,000,000 k; STALE used 80,000,000, CALSPLIT 91,000,000-93,000,000). The score always uses the
    device's TRUE noise model;
  - arms, all given the same Target:
      REC   release 2026-10-07.1 (psf_compile.py), recommended call
      C24   candidate 2026-10-07.c24 (patches/psf_compile_c24_2026-10-07/psf_compile.py), recommended call
      RPSF  release 2026-10-07.1, target + placement_refine (the guarded call)
  - every row records a signature of the compiled circuit (to count where C24 and REC differ) and every file the
    compiler's choice counters (COMPARE_STATS, RESYNTH_STATS).

  python benchmarks/margin_eval.py train  --dataset BC --n 6 --out DIR [--dry]
  python benchmarks/margin_eval.py deploy --dataset BC --n 6 --device FakeAuckland --arm C24 --cal 0 --out DIR [--dry]
  python benchmarks/margin_eval.py score  --out DIR
"""
import argparse
import contextlib
import glob
import hashlib
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
E.SPLIT_SEED, E.INIT_SEED = 5, 5
C24 = os.path.join(REPO, "patches", "psf_compile_c24_2026-10-07", "psf_compile.py")
C24_VERSION = "2026-10-07.c24"
FILES = {"REC": R.RELEASE, "C24": C24, "RPSF": R.RELEASE}
ARMS = ("REC", "C24", "RPSF")
CALS = ("t", "0", "1", "2")
STALE = ("0", "1", "2")
DEVICES = ("FakeAuckland", "FakeTorino")
BUDGETS = (15, 63, 255, 1023)
TOL = 1e-12
_meta = R.meta


def meta(args, **kw):
    m = _meta(args, **kw)
    m.update(margin_sha=R.norm_sha(os.path.abspath(__file__)), c24_sha=R.norm_sha(C24),
             stale_eval_sha=R.norm_sha(S.__file__))
    return m


E.meta = meta  # do_train (unchanged) records this


class _Backend:
    """What depth_eval.compiler reads from a backend: its target."""
    def __init__(self, target):
        self.target = target


def calibration(be, device, cal):
    if cal == "t":
        return be.target, None
    S.BASE = 97_000_000 + 1_000_000 * int(cal)
    return S.stale_target(be.target, device)


def sig_hash(c):
    lay = c.layout
    s = [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [repr(p) for p in i.operation.params]]
         for i in c.data] + [repr(c.global_phase)]
    if lay is not None:
        s += [list(lay.initial_index_layout(filter_ancillas=True)), list(lay.final_index_layout(filter_ancillas=True))]
    return hashlib.sha256(json.dumps(s).encode("utf-8")).hexdigest()[:16]


def do_deploy(args):
    from qiskit_ibm_runtime import fake_provider as fp
    if args.arm not in ARMS or args.cal not in CALS:
        raise SystemExit(f"--arm one of {ARMS}, --cal one of {CALS}")
    split, _ = E.seeds(args.dry)
    Ls = (1, 4, 12) if args.dry else E.LS
    _, _, Xte, yte = E.data(args.dataset, args.n, split)
    if args.dry:
        Xte, yte = Xte[:12], yte[:12]
    be = getattr(fp, args.device)()
    P = E.load_psf(FILES[args.arm])
    t, changed = calibration(be, args.device, args.cal)
    with contextlib.redirect_stdout(io.StringIO()):
        comp = E.compiler(args.arm if args.arm == "RPSF" else "REC", _Backend(t), P)  # REC is read as C12
    sim = E.Noisy(be)                                     # the device's TRUE noise model
    edges, qubits = P._failed_elements(be.target, 0.5)    # failed elements of the TRUE device
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
            on_failed = sum(1 for g in out.data if len(g.qubits) == 2 and g.operation.name != "barrier"
                            and tuple(out.find_bit(q).index for q in g.qubits) in edges)
            row = dict(L=L, i=i, y=float(yv), z_ideal=float(zid[i]), z_compiled_noiseless=z0, z_noisy=zn,
                       infid=R.infidelity(qc, out), e01=e01, e10=e10, on_failed=on_failed, sig=sig_hash(out),
                       n2q=sum(1 for g in out.data if len(g.qubits) == 2), touched=touched, compile_s=round(tc, 4))
            if i < 2:
                row["z_fullwidth"] = sim.z_fullwidth(out)
            rows.append(row)
        sel = [r for r in rows if r["L"] == L]
        print("DEPLOY", json.dumps(dict(L=L, margin=float(np.mean([r["y"] * r["z_noisy"] for r in sel])),
                                         max_infid=max(r["infid"] for r in sel),
                                         n2q=float(np.mean([r["n2q"] for r in sel])))), flush=True)
    import qiskit
    name = f"deploy_{args.dataset}_n{args.n}_{args.device}_{args.arm}_c{args.cal}.json"
    m = meta(args, mode="deploy", dataset=args.dataset, n=args.n, device=args.device, arm=args.arm, cal=args.cal,
             stale_base=(S.BASE if args.cal != "t" else None), stale_t1_changed=changed, psf_version=P.VERSION,
             arm_file_sha=R.norm_sha(FILES[args.arm]), qiskit_version=qiskit.__version__,
             switch_margin=getattr(P, "SWITCH_MARGIN", None), compare_stats=dict(P.COMPARE_STATS),
             resynth_stats=dict(P.RESYNTH_STATS))
    with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(meta=m, rows=rows), f)


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
        D[(m["dataset"], m["n"], m["device"], m["arm"], m["cal"])] = d["rows"]
    T = [json.load(open(p, encoding="utf-8"))["meta"] for p in glob.glob(os.path.join(args.out, "train_*.json"))]
    dry = any(m["dry"] for m in metas + T)
    Ls = sorted({r["L"] for rows in D.values() for r in rows})
    out = [f"# MARGIN score{' (DRY RUN -- not a result)' if dry else ''}", ""]
    exp_files = len(E.DATASETS) * len(E.NS) * len(DEVICES) * len(ARMS) * len(CALS)
    infid = max(r["infid"] for rows in D.values() for r in rows)
    fw = [abs(r["z_fullwidth"] - r["z_noisy"]) for rows in D.values() for r in rows if "z_fullwidth" in r]
    want_v = {"REC": R.RELEASE_VERSION, "RPSF": R.RELEASE_VERSION, "C24": C24_VERSION}
    vers_ok = all(m["psf_version"] == want_v[m["arm"]] for m in metas)
    sha_ok = all(m["c12_sha"] == R.norm_sha(R.RELEASE) and m["c24_sha"] == R.norm_sha(C24)
                 and m["arm_file_sha"] == R.norm_sha(FILES[m["arm"]]) for m in metas)
    margin_ok = all(m["switch_margin"] == (0.05 if m["arm"] == "C24" else None) for m in metas)
    stale_ok = all((m["stale_t1_changed"] or 0) > 0 for m in metas if m["cal"] != "t")
    seeds = {tuple(m["seeds"]) for m in metas + T}
    want_seeds = {(2, 2)} if dry else {(5, 5)}
    p0 = bool(len(D) == exp_files and len(T) == len(E.DATASETS) * len(E.NS) and infid <= 1e-6 and fw
              and max(fw) <= 1e-9 and vers_ok and sha_ok and margin_ok and stale_ok and seeds == want_seeds)
    out += [f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(D)}/{exp_files}, training files {len(T)}/4; max state "
            f"infidelity {infid:.2e} (<= 1e-06); max |reduced - whole-device| "
            f"{max(fw) if fw else float('nan'):.2e} over {len(fw)} circuits; versions per arm {vers_ok}; files as "
            f"locked {sha_ok}; SWITCH_MARGIN 0.05 in C24 only {margin_ok}; every stale Target differs from the true "
            f"one {stale_ok}; seeds {sorted(seeds)}", ""]

    def rows_of(ds, n, dev, a, c, L):
        return [r for r in D[(ds, n, dev, a, c)] if r["L"] == L]

    def pool(dev, a, cals, key):
        """Mean over the calibrations `cals` of the mean over the 24 cells (dataset, n, L); per calibration too."""
        per = []
        for c in cals:
            cells = []
            for ds in E.DATASETS:
                for n in E.NS:
                    for L in Ls:
                        rr = rows_of(ds, n, dev, a, c, L)
                        if key == "margin":
                            cells.append(np.mean([r["y"] * r["z_noisy"] for r in rr]))
                        elif key == "flips":
                            cells.append(sum(int(np.sign(r["z_noisy"]) != np.sign(r["z_ideal"])) for r in rr))
                        else:
                            cells.append(acc_at(rr, key))
            per.append(float(np.sum(cells)) if key == "flips" else float(np.mean(cells)))
        return float(np.mean(per)), per

    def same_share(dev, cals):
        same = tot = 0
        for ds in E.DATASETS:
            for n in E.NS:
                for c in cals:
                    for ra, rb in zip(D[(ds, n, dev, "C24", c)], D[(ds, n, dev, "REC", c)]):
                        same += ra["sig"] == rb["sig"]
                        tot += 1
        return same / tot

    det = {k: [] for k in ("M1", "M2", "M3", "M4", "M5")}
    ok = {k: True for k in det}
    bad = {k: False for k in det}
    bit = {}
    for dev in DEVICES:
        m = {(a, w): pool(dev, a, cs, "margin") for a in ARMS for w, cs in (("t", ("t",)), ("s", STALE))}
        bit[dev] = 1 - same_share(dev, STALE)
        d1 = m[("C24", "s")][0] - m[("RPSF", "s")][0]
        d2 = m[("C24", "s")][0] - m[("REC", "s")][0]
        d3 = m[("REC", "t")][0] - m[("C24", "t")][0]
        per = [a - b for a, b in zip(m[("C24", "s")][1], m[("RPSF", "s")][1])]
        det["M1"].append(f"{dev} C24-RPSF stale {d1:+.4f} (per draw {[round(v, 4) for v in per]}; REC-RPSF "
                         f"{m[('REC', 's')][0] - m[('RPSF', 's')][0]:+.4f})")
        det["M2"].append(f"{dev} C24-REC stale {d2:+.4f} (per draw "
                         f"{[round(a - b, 4) for a, b in zip(m[('C24', 's')][1], m[('REC', 's')][1])]})")
        det["M3"].append(f"{dev} REC-C24 true {d3:+.4f} (REC-RPSF true {m[('REC', 't')][0] - m[('RPSF', 't')][0]:+.4f})")
        ok["M1"] &= d1 >= -TOL
        bad["M1"] |= d1 < -0.002 - TOL
        ok["M2"] &= d2 >= -0.001 - TOL
        bad["M2"] |= d2 < -0.005 - TOL
        ok["M3"] &= d3 <= 0.002 + TOL
        bad["M3"] |= d3 > 0.005 + TOL
        if dev == "FakeTorino":
            d4 = [m[("C24", w)][0] - m[("RPSF", w)][0] for w in ("t", "s")]
            det["M4"].append(f"{dev} C24-RPSF true {d4[0]:+.4f}, stale {d4[1]:+.4f}")
            ok["M4"] &= min(d4) >= 0.01 - TOL
            bad["M4"] |= min(d4) < 0.005 - TOL
        if dev == "FakeAuckland":
            det["M5"].append(f"{dev} worst draw C24-RPSF {min(per):+.4f}")
            ok["M5"] &= min(per) >= -0.005 - TOL
            bad["M5"] |= min(per) < -0.01 - TOL
    labels = {"M1": "with stale calibrations C24 keeps at least the guarded call's margin: C24 - RPSF >= 0 on both "
                    "devices (REFUTED if < -0.002 on either)",
              "M2": "with stale calibrations the margin costs nothing: C24 - REC >= -0.001 on both devices (REFUTED if "
                    "< -0.005 on either)",
              "M3": "with the true calibration it costs little: REC - C24 <= +0.002 on both devices (REFUTED if "
                    "> +0.005 on either)",
              "M4": "FakeTorino's structural lead is kept: C24 - RPSF >= +0.01 with the true and with stale "
                    "calibrations (REFUTED if < +0.005 in either)",
              "M5": "no bad draw: FakeAuckland's worst stale draw C24 - RPSF >= -0.005 (REFUTED if < -0.01)"}
    res = {}
    for k in det:
        res[k] = verdict(ok[k], bad[k])
        out.append(f"- {k} ({labels[k]}): **{res[k]}** ({'; '.join(det[k])})")
    rare = [dev for dev in DEVICES if bit[dev] < 0.05]
    out += ["", "Reading rule (Addendum 405): where C24 returns REC's circuit on more than 95% of the stale rows of a "
            "device, M1, M2 and M5 on that device say that the margin rarely bit, not that it works. Share of stale "
            "rows where C24's circuit differs from REC's: " + ", ".join(f"{dev} {bit[dev]:.3f}" for dev in DEVICES)
            + (f" -- RULE APPLIES to {', '.join(rare)}" if rare else ""), ""]
    accept = (p0 and res["M1"] == "CONFIRMED" and res["M2"] != "REFUTED" and res["M3"] != "REFUTED"
              and len(rare) < len(DEVICES))
    out.append(f"ITEM 51 (decision rule of Addendum 405: P0, M1 CONFIRMED, M2 and M3 not REFUTED, the margin bit on "
               f"at least one device): "
               f"{'PROPOSE ACCEPTANCE' if accept else 'DO NOT PROPOSE'}")
    out += ["", "Reported without prediction -- pooled margin, accuracy at fixed shot budgets (exact binomial, as "
            "Addendum 401), flips, two-qubit gates on failed couplers of the true device, compile time, and how often "
            "the margin turned an alternative away (C24's counters, summed over its files):", ""]
    for dev in DEVICES:
        for w, cs in (("true", ("t",)), ("stale (mean of 3 draws)", STALE)):
            out += [f"## {dev}, {w} calibration", "",
                    "| arm | margin | " + " | ".join(f"acc at {b} shots" for b in BUDGETS)
                    + " | flips | gates on failed couplers | compile s (median) |", "|" + "---|" * (5 + len(BUDGETS))]
            for a in ARMS:
                ab = [pool(dev, a, cs, b)[0] for b in BUDGETS]
                nf = sum(r["on_failed"] for ds in E.DATASETS for n in E.NS for c in cs for r in D[(ds, n, dev, a, c)])
                ct = float(np.median([r["compile_s"] for ds in E.DATASETS for n in E.NS for c in cs
                                      for r in D[(ds, n, dev, a, c)]]))
                out.append(f"| {a} | {pool(dev, a, cs, 'margin')[0]:.4f} | " + " | ".join(f"{v:.4f}" for v in ab)
                           + f" | {pool(dev, a, cs, 'flips')[0]:.1f} | {nf} | {ct:.3f} |")
            out.append("")
        kept = {k: sum(m["compare_stats"].get(k, 0) for m in metas if m["device"] == dev and m["arm"] == "C24")
                for k in ("margin_kept", "psf", "level3", "floor")}
        rk = sum(m["resynth_stats"].get("margin_kept", 0) for m in metas if m["device"] == dev and m["arm"] == "C24")
        out += [f"{dev}: C24's circuit differs from REC's on {1 - same_share(dev, ('t',)):.3f} of the true-calibration "
                f"rows and {bit[dev]:.3f} of the stale rows; the margin turned away {kept['margin_kept']} candidates "
                f"of item 37 and {rk} re-syntheses (C24's choices: psf {kept['psf']}, floor {kept['floor']}, level3 "
                f"{kept['level3']})", ""]
    out += ["SUMMARY " + json.dumps(dict(res, P0="PASS" if p0 else "FAIL", ITEM51="PROPOSE" if accept else "NO"))]
    txt = "\n".join(out)
    with open(os.path.join(args.out, "score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")
    print(txt)


def main():
    if R.norm_sha(R.DEPTH) != R.DEPTH_SHA:
        raise SystemExit("DEPTH's depth_eval.py is not the locked file")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "deploy", "score"])
    ap.add_argument("--dataset")
    ap.add_argument("--n", type=int)
    ap.add_argument("--device")
    ap.add_argument("--arm")
    ap.add_argument("--cal", default="t")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    a.c12 = R.RELEASE
    os.makedirs(a.out, exist_ok=True)
    {"train": E.do_train, "deploy": do_deploy, "score": do_score}[a.mode](a)


if __name__ == "__main__":
    main()
