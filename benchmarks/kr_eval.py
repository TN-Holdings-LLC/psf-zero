"""kr_eval.py -- test KRAUS (2026-10-06): does candidate psf_compile 2026-10-06.c15 (changelog item 42:
candidate_score="kraus", Addendum 339's estimate exact to first order) choose better among the recommended call's
candidates than release 2026-10-05.1's `hybrid_cost`, on fresh circuits, with HOLD6's H4 case repaired?

Circuits: HOLD6's six families F1-F6 (hold6_eval.family, unchanged, with its module constant BASE set to 110,000,000:
seeds between 111 and 117 million, none used before; HOLD6's W families used 96-101.5 million), HOLD6's per-cell sizes,
1,506 per device. Smoke: hold6_eval's smoke sizes (1 per cell, its own offset of 500,000).
Devices: HOLD6's nine.
Per circuit:
  R    the release's recommended call (placement_refine, final_resynthesis "select", compare_level3, compare_floor,
       candidate_score "hybrid")
  K    c15 with the same call and candidate_score "kraus"; its candidates (the release's circuit, the floor-placed one,
       level 3's) are captured from c15._choose. R and K run in alternating order, circuit by circuit, so that neither
       is always the warm second call
  Every candidate is scored by hybrid_cost, kraus_cost and pauli_cost and simulated once (Aer density matrix, the
  device's noise model, HOLD6's metric: 1 - <psi|rho|psi> on the final-layout qubits; at most 11 touched qubits;
  a noiseless statevector run for the P0 exactness check). The choices HYB, KRA and PAU are the candidates with the
  lowest estimate (ties and any estimate that cannot be made: the release's circuit, as in _choose); BEST is the
  candidate with the lowest simulated infidelity; L3 is level 3's candidate.
  Consistency: R must equal HYB's candidate and K must equal KRA's, instruction by instruction. Every 10th circuit,
  c15 with candidate_score "hybrid" must equal R.

    python benchmarks/kr_eval.py run --device D --family F --out DIR [--smoke]
    python benchmarks/kr_eval.py score --out DIR
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for _p in (HERE, REPO):
    sys.path.insert(0, _p)

DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
           "FakeMarrakesh", "FakeAachen")
FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6")
C15_PATH = os.path.join(REPO, "patches", "psf_compile_c15_2026-10-06", "psf_compile.py")
BASE = 110_000_000
MAX_ACTIVE = 11
FULL = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True)
SCORES = ("hyb", "kra", "pau")


def norm_sha(path):
    t = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in t.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_kr")
    sys.modules["psf_smart_layout"] = lay
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_kr")
    c15 = H.load_module(C15_PATH, "psf_compile_c15_kr")
    hold6 = H.load_module(os.path.join(REPO, "benchmarks", "hold6_eval.py"), "hold6_eval_kr")
    if rel.VERSION != "2026-10-05.1" or c15.VERSION != "2026-10-06.c15":
        raise SystemExit(f"STOP: versions {rel.VERSION}, {c15.VERSION}")
    hold6.BASE = BASE  # new seeds for HOLD6's generator
    return rel, c15, hold6, lay


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [round(float(p), 12) for p in i.operation.params])
            for i in c.data]


def pick(costs):
    """_choose's rule: the lowest estimate, the first candidate on ties or when any estimate is None."""
    if any(c is None for c in costs):
        return 0
    return min(range(len(costs)), key=lambda j: (costs[j], j))


def git_head():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def run(args):
    import qiskit
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    rel, c15, hold6, lay = load()
    be = getattr(fake_provider, args.device)()
    tgt = be.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    kw = dict(coupling_map=cm, basis_gates=nat, entangling_basis="cx", layout_search=True, seed_transpiler=0,
              target=tgt, **FULL)
    opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    noisy = AerSimulator(noise_model=NoiseModel.from_backend(be), **opts)
    ideal = AerSimulator(**dict(opts, method="statevector"))
    captured = {}
    orig = c15._choose

    def spy(cands, target, score):
        captured["cands"] = list(cands)
        return orig(cands, target, score)

    c15._choose = spy
    rows, sims, t0 = [], [], time.time()
    for k, (params, qc) in enumerate(hold6.family(args.family, args.smoke)):
        n = qc.num_qubits
        psi = Statevector(qc).data
        row = dict(params=params, n=n)
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # The two calls alternate in order (the first call on a circuit pays for warm-up that the second
                # does not; the smoke run showed K at 0.68-0.88 x R with R always first).
                def run_r():
                    t = time.perf_counter()
                    o = rel.compile_for_hardware(qc, candidate_score="hybrid", **kw)
                    return o, time.perf_counter() - t

                def run_k():
                    captured.clear()
                    t = time.perf_counter()
                    o = c15.compile_for_hardware(qc, candidate_score="kraus", **kw)
                    dt = time.perf_counter() - t
                    captured["k"] = captured.get("cands")  # K's own candidates, before any later c15 call
                    return o, dt

                if k % 2 == 0:
                    (out_r, dt_r), (out_k, dt_k) = run_r(), run_k()
                else:
                    (out_k, dt_k), (out_r, dt_r) = run_k(), run_r()
                check_h = sig(c15.compile_for_hardware(qc, candidate_score="hybrid", **kw)) == sig(out_r) \
                    if k % 10 == 0 else None
        except Exception as e:  # recorded; P0 requires none
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
            continue
        cands = captured.get("k") or [("psf", out_k)]
        est = {s: [] for s in SCORES}
        est_s = dict(hyb=0.0, kra=0.0)
        for _, c in cands:
            t = time.perf_counter()
            est["hyb"].append(c15.hybrid_cost(c, tgt))
            est_s["hyb"] += time.perf_counter() - t
            t = time.perf_counter()
            est["kra"].append(c15.kraus_cost(c, tgt))
            est_s["kra"] += time.perf_counter() - t
            est["pau"].append(c15.pauli_cost(c, tgt))
        ch = {s: pick(est[s]) for s in SCORES}
        # K's cost: the release's call plus the extra time of kraus_cost over hybrid_cost on the same candidates. The
        # measured K and R times are recorded too, but a second call on a circuit runs warm (smoke run: 0.43 x when
        # second, 1.14 x when first), so they do not measure the change itself.
        row.update(compile_r=round(dt_r, 4), compile_k=round(dt_k, 4), est_hyb_s=round(est_s["hyb"], 5),
                   est_kra_s=round(est_s["kra"], 5), compile_k_est=round(dt_r + est_s["kra"] - est_s["hyb"], 4),
                   cands=[nm for nm, _ in cands],
                   est=est, choice=ch, r_is_hyb=sig(out_r) == sig(cands[ch["hyb"]][1]),
                   k_is_kra=sig(out_k) == sig(cands[ch["kra"]][1]), c15_hybrid_is_r=check_h,
                   two_q=[sum(1 for i in c.data if len(i.qubits) == 2) for _, c in cands])
        uniq = {}
        for j, (_, c) in enumerate(cands):
            uniq.setdefault(json.dumps(sig(c)), []).append(j)
        same = [0] * len(cands)
        for js in uniq.values():
            for j in js:
                same[j] = min(js)
        row["same_as"] = same
        widest = 0
        for j, (_, c) in enumerate(cands):
            if same[j] != j:
                continue
            fin = list(c.layout.final_index_layout(filter_ancillas=True)[:n])
            active = {c.find_bit(q).index for i in c.data for q in i.qubits}
            widest = max(widest, len(active))
            if len(active) > MAX_ACTIVE:
                continue
            s = c.copy()
            s.remove_final_measurements(inplace=True)
            s.save_density_matrix(qubits=fin)
            sims.append((len(rows), j, s, psi))
        row["widest"] = widest
        rows.append(row)
    for which, key in ((noisy, "infid"), (ideal, "infid_ideal")):
        if not sims:
            break
        res = which.run([s for _, _, s, _ in sims]).result()
        for m, (r, j, _, psi) in enumerate(sims):
            rho = np.asarray(res.data(m)["density_matrix"])
            rows[r].setdefault(key, {})[str(j)] = float(1 - np.real(np.conj(psi) @ rho @ psi))
    for row in rows:  # copy the simulated values to duplicate candidates
        for key in ("infid", "infid_ideal"):
            if key in row:
                row[key] = [row[key].get(str(row["same_as"][j])) for j in range(len(row["cands"]))]
    meta = dict(device=args.device, family=args.family, smoke=bool(args.smoke), git_head=git_head(),
                qiskit=qiskit.__version__, release=rel.VERSION, c15=c15.VERSION,
                layout=getattr(lay, "LAYOUT_VERSION", None),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), release=norm_sha(rel.__file__),
                         c15=norm_sha(C15_PATH), hold6=norm_sha(hold6.__file__)),
                stats=dict(compare=dict(c15.COMPARE_STATS), exact=dict(c15.EXACT_STATS)),
                wall_s=round(time.time() - t0, 1))
    os.makedirs(args.out, exist_ok=True)
    name = f"kr_{args.device}_{args.family}{'_smoke' if args.smoke else ''}.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, name), "w"))
    print(f"wrote {name}: {len(rows)} circuits, {len(sims)} simulations, {meta['wall_s']} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    R = {}
    for p in sorted(glob.glob(os.path.join(args.out, "kr_*.json"))):
        r = json.load(open(p))
        R[(r["meta"]["device"], r["meta"]["family"])] = r
    smoke = any(r["meta"]["smoke"] for r in R.values())
    want = None if smoke else dict(F1=432, F2=240, F3=300, F4=270, F5=144, F6=120)
    lines = [f"# KRAUS score{' (SMOKE)' if smoke else ''}", ""]
    for (d, f), r in R.items():
        for row in r["rows"]:
            row["_fam"] = f
    rows = {d: [row for f in FAMILIES if (d, f) in R for row in R[(d, f)]["rows"]] for d in DEVICES}
    errors = sum(1 for d in DEVICES for row in rows[d] if "error" in row)
    inexact = sum(1 for d in DEVICES for row in rows[d] for v in (row.get("infid_ideal") or []) if v is not None and v > 1e-6)
    incons = sum(1 for d in DEVICES for row in rows[d] if "error" not in row and
                 (not row["r_is_hyb"] or not row["k_is_kra"] or row["c15_hybrid_is_r"] is False))
    counts = all(len(R[(d, f)]["rows"]) == want[f] for d in DEVICES for f in FAMILIES if (d, f) in R) if want else True
    p0 = len(R) == len(DEVICES) * len(FAMILIES) and not errors and not inexact and not incons and counts
    lines.append(f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(R)} of {len(DEVICES) * len(FAMILIES)}; errors {errors}; "
                 f"inexact candidates {inexact}; inconsistent choices {incons}; counts ok {counts}")
    if not p0:
        lines.append("Nothing below is scored.")
        out = "\n".join(lines) + "\n"
        print(out)
        open(os.path.join(args.out, "score.md"), "w").write(out)
        return

    def sim_rows(d, fam=None, n=None):
        return [row for row in rows[d] if "infid" in row and all(v is not None for v in row["infid"])
                and (fam is None or row["_fam"] == fam) and (n is None or row["n"] == n)]

    def mean(rs, how):
        if not rs:
            return float("nan")
        if how == "best":
            return float(np.mean([min(row["infid"]) for row in rs]))
        if how == "l3":
            return float(np.mean([row["infid"][row["cands"].index("level3")] for row in rs]))
        return float(np.mean([row["infid"][row["choice"][how]] for row in rs]))

    lines += ["", "| device | HYB | KRA | PAU | BEST | KRA/HYB | KRA/L3 | KRA=BEST | HYB=BEST | changed | KRA better |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    K = {}
    for d in DEVICES:
        rs = sim_rows(d)
        l3 = [row for row in rs if "level3" in row["cands"]]
        ch = [row for row in rs if row["infid"][row["choice"]["kra"]] != row["infid"][row["choice"]["hyb"]]]
        kb = sum(1 for row in ch if row["infid"][row["choice"]["kra"]] < row["infid"][row["choice"]["hyb"]])
        best_k = sum(1 for row in rs if row["infid"][row["choice"]["kra"]] == min(row["infid"]))
        best_h = sum(1 for row in rs if row["infid"][row["choice"]["hyb"]] == min(row["infid"]))
        K[d] = dict(hyb=mean(rs, "hyb"), kra=mean(rs, "kra"), pau=mean(rs, "pau"), best=mean(rs, "best"),
                    l3_ratio=mean(l3, "kra") / mean(l3, "l3") if l3 else float("nan"), n=len(rs),
                    changed=len(ch), kra_better=kb, best_k=best_k, best_h=best_h,
                    med_r=float(np.median([row["compile_r"] for row in rows[d]])),
                    med_k=float(np.median([row["compile_k_est"] for row in rows[d]])),
                    med_k_meas=float(np.median([row["compile_k"] for row in rows[d]])))
        x = K[d]
        lines.append(f"| {d} | {x['hyb']:.5f} | {x['kra']:.5f} | {x['pau']:.5f} | {x['best']:.5f} | "
                     f"{x['kra'] / x['hyb']:.4f} | {x['l3_ratio']:.4f} | {best_k}/{x['n']} | {best_h}/{x['n']} | "
                     f"{len(ch)} | {kb} |")
    cell = {}
    for d in DEVICES:
        for f in FAMILIES:
            rs = sim_rows(d, f)
            cell[(d, f)] = mean(rs, "kra") / mean(rs, "hyb") if rs else float("nan")
    h4 = sim_rows("FakeAlgiers", "F5", 4)
    V = []
    k1 = {d: K[d]["kra"] / K[d]["hyb"] for d in DEVICES}
    V.append(("K1", "kraus chooses at least as well as hybrid overall (KRA/HYB <= 1.000 on >= 8 of 9 devices; "
              "refuted > 1.002 on any)", verdict(sum(v <= 1.0 for v in k1.values()) >= 8,
                                                 any(v > 1.002 for v in k1.values())), k1))
    k2 = mean(h4, "kra") / mean(h4, "hyb") if h4 else float("nan")
    V.append(("K2", "the H4 case is repaired (FakeAlgiers F5 n = 4: KRA/HYB <= 0.96; refuted >= 1.00)",
              verdict(k2 <= 0.96, k2 >= 1.0), dict(ratio=k2, circuits=len(h4))))
    gap = {d: ((K[d]["kra"] / K[d]["best"] - 1), (K[d]["hyb"] / K[d]["best"] - 1)) for d in DEVICES}
    V.append(("K3", "kraus is closer to the measured best (gap KRA <= 0.5 x gap HYB on >= 7 of 9 devices; refuted "
              "gap KRA > gap HYB on >= 3)", verdict(sum(g[0] <= 0.5 * g[1] for g in gap.values()) >= 7,
                                                    sum(g[0] > g[1] for g in gap.values()) >= 3), gap))
    k4 = {d: K[d]["kra_better"] / K[d]["changed"] for d in DEVICES if K[d]["changed"] >= 20}
    V.append(("K4", "where the choice changes, kraus is better more often (>= 60% on every device with >= 20 changes; "
              "refuted < 50% on any)", verdict(bool(k4) and all(v >= 0.6 for v in k4.values()),
                                               any(v < 0.5 for v in k4.values())), k4))
    V.append(("K5", "no family loses (KRA/HYB <= 1.01 in every family-device cell; refuted > 1.03 in any)",
              verdict(all(v <= 1.01 for v in cell.values()), any(v > 1.03 for v in cell.values())),
              {f"{d}/{f}": v for (d, f), v in cell.items() if v > 1.005}))
    k6 = {d: K[d]["l3_ratio"] for d in DEVICES}
    V.append(("K6", "kraus stays ahead of Qiskit level 3 (KRA/L3 <= 1.00 on 9 of 9; refuted > 1.02 on any)",
              verdict(all(v <= 1.0 for v in k6.values()), any(v > 1.02 for v in k6.values())), k6))
    k7 = {d: K[d]["med_k"] / K[d]["med_r"] for d in DEVICES}
    V.append(("K7", "it costs little (median compile time of K, as the release's call plus kraus_cost's extra time "
              "over hybrid_cost on the same candidates, / R <= 1.15 on every device; refuted > 1.50 on any)",
              verdict(all(v <= 1.15 for v in k7.values()), any(v > 1.5 for v in k7.values())), k7))
    lines += ["", "## Predictions", ""]
    for qid, text, v, num in V:
        lines.append(f"- {qid} ({text}): **{v}** -- {json.dumps(num, default=lambda x: round(x, 5))}")
    lines += ["", "Reported: family-device KRA/HYB cells: " +
              json.dumps({f"{d}/{f}": round(v, 4) for (d, f), v in cell.items()}),
              "Reported: measured K / R median compile time (order alternates; a second call runs warm): " +
              json.dumps({d: round(K[d]["med_k_meas"] / K[d]["med_r"], 3) for d in DEVICES}),
              "Reported: candidates too wide to simulate are excluded from every mean: " +
              json.dumps({d: len(rows[d]) - len(sim_rows(d)) for d in DEVICES})]
    out = "\n".join(lines) + "\n"
    print(out)
    open(os.path.join(args.out, "score.md"), "w").write(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", required=True, choices=DEVICES)
    r.add_argument("--family", required=True, choices=FAMILIES)
    r.add_argument("--out", required=True)
    r.add_argument("--smoke", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    main()
