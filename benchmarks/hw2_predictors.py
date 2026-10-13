"""hw2_predictors.py -- HW2's additional predictors (Addendum 466), fixed before HW2 ran and computed afterwards from
HW2's circuits and the calibration snapshot of its device. Run from the repository root:

    python benchmarks/hw2_predictors.py <out dir> [hw2 dir] [--only P1,P2,P3,P4]

  P1  ns_sim2: as soon as possible, no preparation error on measured qubits, reported T2 while waiting
  P2  ns_sim2: as P1, with T2 / 2 while waiting (an unechoed waiting qubit dephases faster than the echo T2)
  P3  a14's estimate (gate + readout + state-aware idle term), exact reduced states (workplace est_lc.py, Aer)
  P4  exposure: sum over the circuit's used qubits of (end of circuit - its first gate) / T2, as soon as possible

Each predictor gives a number per circuit (P1, P2: simulated F; P3: exp(-(gate + idle)); P4: exp(-exposure)) and a
ratio PSF / QK3 per test (part R) or S1 / S0 per pair (part S). Scored against HW2's measured F (hw2_score.json)."""
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ns_sim2 as N  # noqa: E402

READING = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12")
A14_DIR = os.path.join("patches", "psf_ai_compile_a14_2026-10-13")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


def exposure(circ, target):
    """sum over used qubits of (end - first gate) / T2, gates as soon as possible on the Target's durations."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    clock, first = {}, {}
    for ins in circ.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure", "delay"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = target[nm].get(qs) if nm in target.operation_names else None
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            first.setdefault(q, s)
            clock[q] = s + d
    T = max(clock.values()) if clock else 0.0
    tot = 0.0
    for q in clock:
        t2 = getattr(target.qubit_properties[q], "t2", None) or N.MISSING_T_S
        tot += (T - first[q]) / t2
    return tot


def main(out, w, only):
    from qiskit import qpy
    sys.path[:0] = ["benchmarks", "."]
    import snapshot_target as ST
    H = _load(os.path.join("data", "2026-10-13", "hw2_kit", "hw2_real.py"), "hw2")
    os.makedirs(out, exist_ok=False)
    man = json.load(open(os.path.join(w, "hw2_manifest.json")))
    sc = json.load(open(os.path.join(w, "hw2_score.json")))
    circs = qpy.load(open(os.path.join(w, "hw2_circuits.qpy"), "rb"))
    snap = os.path.join(w, f"ns_snapshot_{man['device']}.json")
    cal = N.calibration_from_snapshot(snap)
    tgt = ST.target_from_snapshot(snap)
    a14 = est = None
    if "P3" in only:
        import core_fix_c2_eval as CF
        CF.load_module(os.path.join("benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
        CF.load_module("psf_compile.py", "psf_compile")
        a14 = CF.load_module(os.path.join(A14_DIR, "psf_ai_compile.py"), "psf_ai_compile_a14")
        sys.path.insert(0, A14_DIR)
        import est_lc as est
    todo = [(man["entries"][p["entry"]], c) for p, c in zip(man["pubs"], circs)
            if p["repeat"] == 1 and man["entries"][p["entry"]]["submitted"]
            and (man["entries"][p["entry"]]["part"] == "S" or man["entries"][p["entry"]]["test"] in READING)]
    todo.sort(key=lambda x: x[0]["active"])
    dists, rows = {}, []
    f = open(os.path.join(out, "hw2_predictors.jsonl"), "w")
    for e, c in todo:
        t = e["test"]
        if t not in dists:
            qc, _ = H.build_test(t)
            dists[t] = (H.measured_distribution(H._unrolled(qc))[0], qc.num_qubits)
        d, n = dists[t]
        r = dict(part=e["part"], test=t, arm=e["arm"], active=e["active"],
                 f_real=sc["F"][f"{e['part']}|{t}|{e['arm']}"]["f"])
        t0 = time.time()
        if "P1" in only:
            r["P1"] = N.simulate_xeb(c, d, n, cal, method="asap", init_on_measured=False)[:2]
        if "P2" in only:
            r["P2"] = N.simulate_xeb(c, d, n, cal, method="asap", init_on_measured=False, idle_t2_factor=0.5)[:2]
        if "P3" in only:
            g, idle = est.estimate(c, tgt, est.states_aer(c), a14)[:2]
            r["P3"] = (math.exp(-(g + idle)), None)
        if "P4" in only:
            r["P4"] = (math.exp(-exposure(c, tgt)), None)
        r["secs"] = round(time.time() - t0, 1)
        rows.append(r)
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(json.dumps(r), flush=True)
    f.close()
    score(out, rows, only)


def score(out, rows, only):
    R = {(r["test"], r["arm"]): r for r in rows if r["part"] == "R"}
    S = {(r["test"], r["arm"]): r for r in rows if r["part"] == "S"}
    lines = ["# HW2's additional predictors (Addendum 466)", ""]
    res = {}
    for P in only:
        pr = []
        for t in READING:
            a, b = R.get((t, "PSF")), R.get((t, "QK3"))
            if a and b and P in a and P in b and min(a["f_real"], b["f_real"], a[P][0], b[P][0]) > 0:
                pr.append((t, a[P][0] / b[P][0], a["f_real"] / b["f_real"]))
        agree = sum(1 for _, x, y in pr if (x > 1) == (y > 1))
        rho = float(np.corrcoef(ranks([x for _, x, _ in pr]), ranks([y for _, _, y in pr]))[0, 1]) if len(pr) >= 3 else float("nan")
        useful = len(pr) >= 3 and agree >= math.ceil(5 * len(pr) / 7) and rho >= 0.5
        sp = []
        for t in sorted({k[0] for k in S}):
            a, b = S.get((t, "S1")), S.get((t, "S0"))
            if a and b and P in a and P in b and min(a["f_real"], b["f_real"], a[P][0], b[P][0]) > 0:
                sp.append((t, a[P][0] / b[P][0], a["f_real"] / b["f_real"]))
        res[P] = dict(pairs=pr, agree=agree, rho=rho, useful=useful, part_s=sp)
        lines += [f"## {P}", "", "| test | predicted ratio PSF / QK3 | device |", "|---|---|---|"]
        lines += [f"| {t} | {x:.3f} | {y:.3f} |" for t, x, y in pr]
        lines += ["", f"- same winner {agree} of {len(pr)}; rank correlation {rho:+.3f}; useful (>= 5 of 7 and >= 0.5): "
                  f"{'YES' if useful else 'NO'}",
                  "- part S (S1 / S0, predicted vs device): " + "; ".join(f"{t} {x:.3f} vs {y:.3f}" for t, x, y in sp), ""]
    if "P1" in res and "P2" in res:
        r1, r2 = res["P1"]["rho"], res["P2"]["rho"]
        verdict = "NOT TESTABLE" if (math.isnan(r1) or math.isnan(r2)) else ("CONFIRMED" if r2 >= r1 else "REFUTED")
        lines.append(f"- H-T2* (P2 ranks at least as well as P1): rho P2 {r2:+.3f} vs P1 {r1:+.3f} -> {verdict}")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(out, "hw2_predictors_score.md"), "w").write(txt)
    json.dump(res, open(os.path.join(out, "hw2_predictors_score.json"), "w"), indent=1, default=str)
    print(txt)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = ("P1", "P2", "P3", "P4")
    if "--only" in sys.argv:
        only = tuple(sys.argv[sys.argv.index("--only") + 1].split(","))
        args = [a for a in args if a != sys.argv[sys.argv.index("--only") + 1]]
    main(args[0], args[1] if len(args) > 1 else os.path.join("data", "2026-10-13", "hw2"), only)
