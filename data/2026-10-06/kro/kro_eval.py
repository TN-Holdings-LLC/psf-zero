"""kro_eval.py -- test KRO (2026-10-06): with readout added to it (candidate psf_compile 2026-10-06.c16, changelog item
43), does `candidate_score="kraus"` choose better than release 2026-10-06.1's recommended `hybrid` on circuits compiled
with their final measurements, so that it can become the recommended call?

Circuits: HOLD6's six families F1-F6 (hold6_eval.family, unchanged, with its module constant BASE set to 120,000,000:
seeds between 121 and 127 million, none used before), each with `measure_all()` appended; HOLD6's per-cell sizes,
1,506 per device. Smoke: hold6_eval's smoke sizes (1 per cell, its own offset of 500,000). Devices: HOLD6's nine.
Per circuit (all calls with measurements unless stated):
  R    the release's recommended call (placement_refine, final_resynthesis "select", compare_level3, compare_floor,
       candidate_score "hybrid")
  K    c16 with the same call and candidate_score "kraus"; its candidates (the release's circuit, the floor-placed one,
       level 3's) are captured from c16._choose. R and K run in alternating order, circuit by circuit.
  Every candidate is scored by hybrid_cost (the release's, with readout), c16's kraus_cost (with readout) and the
  release's kraus_cost (without readout). Every distinct candidate is simulated once: Aer density matrix on its
  touched qubits with the device's noise model restricted to them, then each measured qubit's readout assignment
  (Aer's own readout probabilities); the metric is MODEL-RO2's classical infidelity 1 - (sum_x sqrt(p_x q_x))^2
  against the ideal output distribution. At most 11 touched qubits. Exactness: the workplace probe's state infidelity
  of the candidate without its measurements (Addendum 340).
  Choices: HYB, KRA and KRA0 are the candidates with the lowest hybrid, c16-kraus and release-kraus estimates (ties
  and any estimate that cannot be made: the release's circuit, as in _choose); BEST has the lowest simulated
  infidelity; L3 is level 3's candidate.
  Consistency: R = HYB's candidate and K = KRA's; every 10th circuit, c16 with "hybrid" = R, and c16 with "kraus" on
  the circuit WITHOUT measurements = the release with "kraus" on it (item 43 changes nothing without measurements).
The helpers reduced_probs, apply_readout, the restricted noise model and state_infid are copied from the workplace
harnesses (ai10_eval2.py, depth_eval.py, readout_eval.py; Addenda 348-352), unchanged in what they compute.

    python benchmarks/kro_eval.py run --device D --family F --out DIR [--smoke]
    python benchmarks/kro_eval.py score --out DIR
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
C16_PATH = os.path.join(REPO, "patches", "psf_compile_c16_2026-10-06", "psf_compile.py")
BASE = 120_000_000
MAX_ACTIVE = 11
FULL = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True)
SCORES = ("hyb", "kra", "kra0")


def norm_sha(path):
    t = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in t.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_kro")
    sys.modules["psf_smart_layout"] = lay
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_kro")
    c16 = H.load_module(C16_PATH, "psf_compile_c16_kro")
    hold6 = H.load_module(os.path.join(REPO, "benchmarks", "hold6_eval.py"), "hold6_eval_kro")
    if rel.VERSION != "2026-10-06.1" or c16.VERSION != "2026-10-06.c16":
        raise SystemExit(f"STOP: versions {rel.VERSION}, {c16.VERSION}")
    hold6.BASE = BASE  # new seeds for HOLD6's generator
    return rel, c16, hold6, lay


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


class Noisy:
    """depth_eval.Noisy (DEPTH stage 1), copied: the device's noise model restricted to given qubits, and Aer's
    readout probabilities."""

    def __init__(self, be):
        from qiskit_aer.noise import NoiseModel
        self.nm = NoiseModel.from_backend(be)
        self.cache = {}

    def reduced_model(self, active):
        key = tuple(active)
        if key in self.cache:
            return self.cache[key]
        from qiskit_aer.noise import NoiseModel
        idx = {p: i for i, p in enumerate(active)}
        m = NoiseModel(basis_gates=self.nm.basis_gates)
        for gate, d in self.nm._local_quantum_errors.items():
            for qs, err in d.items():
                if all(q in idx for q in qs):
                    m.add_quantum_error(err, gate, [idx[q] for q in qs])
        if len(self.cache) > 2000:
            self.cache.clear()
        self.cache[key] = m
        return m

    def readout(self, p):
        e = self.nm._local_readout_errors.get((p,))
        if e is None:
            return 0.0, 0.0
        pr = np.asarray(e.probabilities)
        return float(pr[0][1]), float(pr[1][0])  # P(1|0), P(0|1)


def reduced_probs(out, sim):
    """ai10_eval2.reduced_probs (MODEL-RO2), copied, noisy: distribution over the measured physical qubits, in clbit
    order, before readout; with the measured qubits and their clbits."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    meas = sorted(((out.find_bit(i.clbits[0]).index, out.find_bit(i.qubits[0]).index)
                   for i in out.data if i.operation.name == "measure"))
    mq = [q for _, q in meas]
    act = sorted({out.find_bit(b).index for i in out.data for b in i.qubits} | set(mq))
    idx = {p: k for k, p in enumerate(act)}
    red = QuantumCircuit(len(act))
    for i in out.data:
        if i.operation.name in ("measure", "barrier", "delay"):
            continue
        red.append(i.operation, [idx[out.find_bit(b).index] for b in i.qubits])
    red.save_probabilities(qubits=[idx[q] for q in mq])
    s = AerSimulator(method="density_matrix", noise_model=sim.reduced_model(act), max_parallel_threads=1)
    return np.asarray(s.run(red).result().data()["probabilities"]), mq, [c for c, _ in meas]


def apply_readout(p, mq, sim):
    """ai10_eval2.apply_readout (MODEL-RO2), copied."""
    k = len(mq)
    t = p.reshape([2] * k)  # axis 0 = last clbit (little-endian)
    for j, q in enumerate(mq):
        e01, e10 = sim.readout(q)
        A = np.array([[1 - e01, e10], [e01, 1 - e10]])  # A[measured, true]
        ax = k - 1 - j
        t = np.moveaxis(np.tensordot(A, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def state_infid(qc, out):
    """readout_eval.state_infid (the workplace probe's check, Addendum 340), copied; measurements removed from `out`,
    `qc` without measurements."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
    ideal = Statevector(qc)
    fin = list(out.layout.final_index_layout(filter_ancillas=True))
    active = sorted({out.find_bit(b).index for ins in out.data for b in ins.qubits} | set(fin))
    idx = {p: i for i, p in enumerate(active)}
    red = QuantumCircuit(len(active))
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        red.append(ins.operation, [idx[out.find_bit(b).index] for b in ins.qubits])
    sv = Statevector(red)
    keep = [idx[p] for p in fin]
    trace_out = [i for i in range(len(active)) if i not in keep]
    rho = DensityMatrix(partial_trace(sv, trace_out) if trace_out else sv)
    order = sorted(keep)
    perm = [order.index(k) for k in keep]
    n = len(keep)
    t = rho.data.reshape([2] * (2 * n))
    t = np.transpose(t, [n - 1 - perm[v] for v in range(n)][::-1] + [2 * n - 1 - perm[v] for v in range(n)][::-1])
    return float(1 - state_fidelity(DensityMatrix(t.reshape(2 ** n, 2 ** n)), ideal))


def run(args):
    import qiskit
    from qiskit.quantum_info import Statevector
    from qiskit_ibm_runtime import fake_provider
    rel, c16, hold6, lay = load()
    be = getattr(fake_provider, args.device)()
    tgt = be.target
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    kw = dict(coupling_map=tgt.build_coupling_map(), basis_gates=nat, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=tgt, **FULL)
    sim = Noisy(be)
    captured = {}
    orig = c16._choose

    def spy(cands, target, score):
        captured["cands"] = list(cands)
        return orig(cands, target, score)

    c16._choose = spy
    rows, t0 = [], time.time()
    for k, (params, qc) in enumerate(hold6.family(args.family, args.smoke)):
        n = qc.num_qubits
        qcm = qc.copy()
        qcm.measure_all()
        ideal = Statevector(qc).probabilities()
        row = dict(params=params, n=n)
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")

                def run_r():
                    t = time.perf_counter()
                    o = rel.compile_for_hardware(qcm, candidate_score="hybrid", **kw)
                    return o, time.perf_counter() - t

                def run_k():
                    captured.clear()
                    t = time.perf_counter()
                    o = c16.compile_for_hardware(qcm, candidate_score="kraus", **kw)
                    dt = time.perf_counter() - t
                    captured["k"] = captured.get("cands")
                    return o, dt

                if k % 2 == 0:
                    (out_r, dt_r), (out_k, dt_k) = run_r(), run_k()
                else:
                    (out_k, dt_k), (out_r, dt_r) = run_k(), run_r()
                cands = captured.get("k") or [("psf", out_k)]
                check_h = check_u = None
                if k % 10 == 0:
                    check_h = sig(c16.compile_for_hardware(qcm, candidate_score="hybrid", **kw)) == sig(out_r)
                    check_u = sig(c16.compile_for_hardware(qc, candidate_score="kraus", **kw)) == \
                        sig(rel.compile_for_hardware(qc, candidate_score="kraus", **kw))
        except Exception as e:  # recorded; P0 requires none
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
            continue
        est = {s: [] for s in SCORES}
        est_s = dict(hyb=0.0, kra=0.0)
        for _, c in cands:
            t = time.perf_counter()
            est["hyb"].append(rel.hybrid_cost(c, tgt))
            est_s["hyb"] += time.perf_counter() - t
            t = time.perf_counter()
            est["kra"].append(c16.kraus_cost(c, tgt))
            est_s["kra"] += time.perf_counter() - t
            est["kra0"].append(rel.kraus_cost(c, tgt))
        ch = {s: pick(est[s]) for s in SCORES}
        uniq = {}
        for j, (_, c) in enumerate(cands):
            uniq.setdefault(json.dumps(sig(c)), []).append(j)
        same = [0] * len(cands)
        for js in uniq.values():
            for j in js:
                same[j] = min(js)
        infid, exact, meas_ok, meas_err, widest = [None] * len(cands), [None] * len(cands), [None] * len(cands), \
            [None] * len(cands), 0
        for j, (_, c) in enumerate(cands):
            if same[j] != j:
                continue
            fin = list(c.layout.final_index_layout(filter_ancillas=True)[:n])
            active = {c.find_bit(q).index for i in c.data for q in i.qubits if i.operation.name != "barrier"}
            widest = max(widest, len(active))
            exact[j] = state_infid(qc, c)
            meas_err[j] = rel.readout_cost(c, tgt)
            if len(active) > MAX_ACTIVE:
                continue
            p, mq, cl = reduced_probs(c, sim)
            meas_ok[j] = bool(cl == list(range(n)) and mq == fin)
            q = apply_readout(p, mq, sim)
            infid[j] = float(1 - np.sum(np.sqrt(np.clip(ideal, 0, None) * np.clip(q, 0, None))) ** 2)
        for lst in (infid, exact, meas_ok, meas_err):
            for j in range(len(cands)):
                lst[j] = lst[same[j]]
        row.update(compile_r=round(dt_r, 4), compile_k=round(dt_k, 4), est_hyb_s=round(est_s["hyb"], 5),
                   est_kra_s=round(est_s["kra"], 5), compile_k_est=round(dt_r + est_s["kra"] - est_s["hyb"], 4),
                   cands=[nm for nm, _ in cands], est=est, choice=ch,
                   r_is_hyb=sig(out_r) == sig(cands[ch["hyb"]][1]), k_is_kra=sig(out_k) == sig(cands[ch["kra"]][1]),
                   c16_hybrid_is_r=check_h, c16_kraus_unmeasured_is_rel=check_u, same_as=same, widest=widest,
                   exact=exact, meas_ok=meas_ok, meas_err=meas_err, infid=infid,
                   two_q=[sum(1 for i in c.data if len(i.qubits) == 2 and i.operation.name != "barrier") for _, c in cands])
        rows.append(row)
    meta = dict(device=args.device, family=args.family, smoke=bool(args.smoke), git_head=git_head(),
                qiskit=qiskit.__version__, release=rel.VERSION, c16=c16.VERSION,
                layout=getattr(lay, "LAYOUT_VERSION", None),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), release=norm_sha(rel.__file__),
                         c16=norm_sha(C16_PATH), hold6=norm_sha(hold6.__file__)),
                stats=dict(compare=dict(c16.COMPARE_STATS), exact=dict(c16.EXACT_STATS)),
                wall_s=round(time.time() - t0, 1))
    os.makedirs(args.out, exist_ok=True)
    name = f"kro_{args.device}_{args.family}{'_smoke' if args.smoke else ''}.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, name), "w"))
    print(f"wrote {name}: {len(rows)} circuits, {meta['wall_s']} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    R = {}
    for p in sorted(glob.glob(os.path.join(args.out, "kro_*.json"))):
        r = json.load(open(p))
        R[(r["meta"]["device"], r["meta"]["family"])] = r
    smoke = any(r["meta"]["smoke"] for r in R.values())
    want = None if smoke else dict(F1=432, F2=240, F3=300, F4=270, F5=144, F6=120)
    lines = [f"# KRO score{' (SMOKE)' if smoke else ''}", ""]
    for (d, f), r in R.items():
        for row in r["rows"]:
            row["_fam"] = f
    rows = {d: [row for f in FAMILIES if (d, f) in R for row in R[(d, f)]["rows"]] for d in DEVICES}
    ok_rows = [row for d in DEVICES for row in rows[d] if "error" not in row]
    errors = sum(1 for d in DEVICES for row in rows[d] if "error" in row)
    inexact = sum(1 for row in ok_rows for v in row["exact"] if v is not None and v > 1e-6)
    badmeas = sum(1 for row in ok_rows for v in row["meas_ok"] if v is False)
    incons = sum(1 for row in ok_rows if not row["r_is_hyb"] or not row["k_is_kra"] or row["c16_hybrid_is_r"] is False
                 or row["c16_kraus_unmeasured_is_rel"] is False)
    counts = all(len(R[(d, f)]["rows"]) == want[f] for d in DEVICES for f in FAMILIES if (d, f) in R) if want else True
    p0 = len(R) == len(DEVICES) * len(FAMILIES) and not errors and not inexact and not badmeas and not incons and counts
    lines.append(f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(R)} of {len(DEVICES) * len(FAMILIES)}; errors {errors}; "
                 f"inexact candidates {inexact}; wrong measurement mapping {badmeas}; inconsistent choices {incons}; "
                 f"counts ok {counts}")
    if not p0:
        lines.append("Nothing below is scored.")
        out = "\n".join(lines) + "\n"
        print(out)
        open(os.path.join(args.out, "score.md"), "w").write(out)
        return

    def sim_rows(d, fam=None, n=None):
        return [row for row in rows[d] if "error" not in row and all(v is not None for v in row["infid"])
                and (fam is None or row["_fam"] == fam) and (n is None or row["n"] == n)]

    def mean(rs, how):
        if not rs:
            return float("nan")
        if how == "best":
            return float(np.mean([min(row["infid"]) for row in rs]))
        if how == "l3":
            return float(np.mean([row["infid"][row["cands"].index("level3")] for row in rs]))
        return float(np.mean([row["infid"][row["choice"][how]] for row in rs]))

    def mean_err(rs, how):
        return float(np.mean([row["meas_err"][row["choice"][how]] for row in rs])) if rs else float("nan")

    lines += ["", "| device | HYB | KRA | KRA0 | BEST | KRA/HYB | KRA/KRA0 | KRA/L3 | meas err HYB / KRA | changed | KRA better |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    K = {}
    for d in DEVICES:
        rs = sim_rows(d)
        l3 = [row for row in rs if "level3" in row["cands"]]
        ch = [row for row in rs if row["infid"][row["choice"]["kra"]] != row["infid"][row["choice"]["hyb"]]]
        kb = sum(1 for row in ch if row["infid"][row["choice"]["kra"]] < row["infid"][row["choice"]["hyb"]])
        K[d] = dict(hyb=mean(rs, "hyb"), kra=mean(rs, "kra"), kra0=mean(rs, "kra0"), best=mean(rs, "best"),
                    l3_ratio=mean(l3, "kra") / mean(l3, "l3") if l3 else float("nan"), n=len(rs), changed=len(ch),
                    kra_better=kb, err_h=mean_err(rs, "hyb"), err_k=mean_err(rs, "kra"),
                    med_r=float(np.median([row["compile_r"] for row in rows[d]])),
                    med_k=float(np.median([row["compile_k_est"] for row in rows[d]])),
                    med_k_meas=float(np.median([row["compile_k"] for row in rows[d]])))
        x = K[d]
        lines.append(f"| {d} | {x['hyb']:.5f} | {x['kra']:.5f} | {x['kra0']:.5f} | {x['best']:.5f} | "
                     f"{x['kra'] / x['hyb']:.4f} | {x['kra'] / x['kra0']:.4f} | {x['l3_ratio']:.4f} | "
                     f"{x['err_h']:.4f} / {x['err_k']:.4f} | {len(ch)} | {kb} |")
    cell = {}
    for d in DEVICES:
        for f in FAMILIES:
            rs = sim_rows(d, f)
            cell[(d, f)] = mean(rs, "kra") / mean(rs, "hyb") if rs else float("nan")
    h4 = sim_rows("FakeAlgiers", "F5", 4)
    V = []
    o1 = {d: K[d]["kra"] / K[d]["hyb"] for d in DEVICES}
    V.append(("O1", "kraus with readout chooses at least as well as hybrid on measured circuits (KRA/HYB <= 1.000 on "
              ">= 8 of 9 devices; refuted > 1.002 on any)",
              verdict(sum(v <= 1.0 for v in o1.values()) >= 8, any(v > 1.002 for v in o1.values())), o1))
    o2 = {d: K[d]["kra"] / K[d]["kra0"] for d in DEVICES}
    V.append(("O2", "the readout term helps kraus on measured circuits (KRA/KRA0 <= 1.000 on >= 8 of 9 devices; refuted "
              "> 1.002 on any)", verdict(sum(v <= 1.0 for v in o2.values()) >= 8, any(v > 1.002 for v in o2.values())), o2))
    o3 = mean(h4, "kra") / mean(h4, "hyb") if h4 else float("nan")
    V.append(("O3", "the H4 case stays repaired with measurements (FakeAlgiers F5 n = 4: KRA/HYB <= 0.98; refuted >= 1.00)",
              verdict(o3 <= 0.98, o3 >= 1.0), dict(ratio=o3, circuits=len(h4))))
    gap = {d: ((K[d]["kra"] / K[d]["best"] - 1), (K[d]["hyb"] / K[d]["best"] - 1)) for d in DEVICES}
    V.append(("O4", "kraus is closer to the measured best (gap KRA <= 0.5 x gap HYB on >= 7 of 9 devices; refuted gap "
              "KRA > gap HYB on >= 3)", verdict(sum(g[0] <= 0.5 * g[1] for g in gap.values()) >= 7,
                                                sum(g[0] > g[1] for g in gap.values()) >= 3), gap))
    o5 = {d: K[d]["kra_better"] / K[d]["changed"] for d in DEVICES if K[d]["changed"] >= 20}
    V.append(("O5", "where the choice changes, kraus is better more often (>= 60% on every device with >= 20 changes; "
              "refuted < 50% on any)", verdict(bool(o5) and all(v >= 0.6 for v in o5.values()),
                                               any(v < 0.5 for v in o5.values())), o5))
    V.append(("O6", "no family loses (KRA/HYB <= 1.01 in every family-device cell; refuted > 1.03 in any)",
              verdict(all(v <= 1.01 for v in cell.values()), any(v > 1.03 for v in cell.values())),
              {f"{d}/{f}": v for (d, f), v in cell.items() if v > 1.005}))
    o7 = {d: K[d]["l3_ratio"] for d in DEVICES}
    V.append(("O7", "kraus stays ahead of Qiskit level 3 with measurements (KRA/L3 <= 1.00 on 9 of 9; refuted > 1.02 on "
              "any)", verdict(all(v <= 1.0 for v in o7.values()), any(v > 1.02 for v in o7.values())), o7))
    o8 = {d: K[d]["med_k"] / K[d]["med_r"] for d in DEVICES}
    V.append(("O8", "it costs little (median compile time of K, as the release's call plus kraus_cost's extra time over "
              "hybrid_cost on the same candidates, / R <= 1.15 on every device; refuted > 1.50 on any)",
              verdict(all(v <= 1.15 for v in o8.values()), any(v > 1.5 for v in o8.values())), o8))
    lines += ["", "## Predictions", ""]
    for qid, text, v, num in V:
        lines.append(f"- {qid} ({text}): **{v}** -- {json.dumps(num, default=lambda x: round(x, 5))}")
    lines += ["", "Reported: family-device KRA/HYB cells: " +
              json.dumps({f"{d}/{f}": round(v, 4) for (d, f), v in cell.items()}),
              "Reported: measured K / R median compile time (order alternates): " +
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
