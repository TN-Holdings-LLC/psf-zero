"""NS1: ns_sim.py on HW1's submitted circuits (ibm_kingston, 2026-10-12) with the calibration exported from the
dashboard that day, against HW1's measured F. Run from the repository root:
    python ns_hw1.py <out dir>        (writes ns_hw1.jsonl as it goes, then ns_hw1_score.md; never overwrites)"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ns_sim as N  # noqa: E402

W = os.path.join("data", "2026-10-12", "hw1")
CAL = os.path.join("data", "2026-10-12", "hw_analysis", "ibm_kingston_calibrations_2026-10-12T11_15_24Z.csv")
READING = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12")


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


def main(out):
    from qiskit import qpy
    sys.path[:0] = ["benchmarks", "."]
    import importlib.util
    spec = importlib.util.spec_from_file_location("hw1", os.path.join("benchmarks", "hw1_real.py"))
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    H.KIT = os.path.join("data", "2026-10-12", "hw_kit")
    os.makedirs(out, exist_ok=False)
    man = json.load(open(os.path.join(W, "hw1_manifest.json")))
    sc = json.load(open(os.path.join(W, "hw1_score.json")))
    circs = qpy.load(open(os.path.join(W, "hw1_circuits.qpy"), "rb"))
    cal = N.calibration_from_csv(CAL)
    rows = []
    f = open(os.path.join(out, "ns_hw1.jsonl"), "w")
    todo = [(man["entries"][p["entry"]], c) for p, c in zip(man["pubs"], circs)
            if man["entries"][p["entry"]]["part"] == "HW1" and p["repeat"] == 1
            and man["entries"][p["entry"]]["arm"] in ("PSF", "QK3")
            and man["entries"][p["entry"]]["test"] in READING + ("ghz_n12",)]  # C47 and tfim_n16: not in NS1
    todo.sort(key=lambda x: x[0]["active"])  # small first
    dists = {}
    for e, c in todo:
        t = e["test"]
        if t not in dists:
            qc, _ = H.build_test(t)
            dists[t] = (H.measured_distribution(H._unrolled(qc))[0], qc.num_qubits)
        d, n = dists[t]
        t0 = time.time()
        fs, se, dur, nact, ntraj = N.simulate_xeb(c, d, n, cal)
        r = dict(test=t, arm=e["arm"], f_sim=fs, se_sim=se, dur_us=dur * 1e6, active=nact, esp=e["esp"],
                 f_real=sc["F"][f"{t}|{e['arm']}"]["f"], traj=ntraj, secs=round(time.time() - t0, 1))
        rows.append(r)
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(f"{t:12s} {e['arm']:4s} active {nact:2d} traj {ntraj:4d} F_sim {fs:.4f} ({se:.4f}) F_real {r['f_real']:.4f} ESP "
              f"{e['esp']:.4f} dur {dur * 1e6:.1f} us  {r['secs']} s", flush=True)
    f.close()
    score(out, rows)


def score(out, rows):
    R = {(r["test"], r["arm"]): r for r in rows}
    lines = ["# NS1 scores", "", "| test | arm | F_sim (SE) | F_real | ESP | F_sim / F_real |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['test']} | {r['arm']} | {r['f_sim']:.4f} ({r['se_sim']:.4f}) | {r['f_real']:.4f} | "
                     f"{r['esp']:.4f} | {r['f_sim'] / r['f_real'] if r['f_real'] > 0 else float('nan'):.3f} |")
    pr = []
    for t in READING:
        p, q = R.get((t, "PSF")), R.get((t, "QK3"))
        if p and q and p["f_real"] > 0 and q["f_real"] > 0 and p["f_sim"] > 0 and q["f_sim"] > 0:
            pr.append(dict(test=t, sim=p["f_sim"] / q["f_sim"], real=p["f_real"] / q["f_real"], esp=p["esp"] / q["esp"]))
    lines += ["", "| test | ESP ratio | F_sim ratio | F_real ratio |", "|---|---|---|---|"]
    for x in pr:
        lines.append(f"| {x['test']} | {x['esp']:.3f} | {x['sim']:.3f} | {x['real']:.3f} |")
    agree = sum(1 for x in pr if (x["sim"] > 1) == (x["real"] > 1))
    rho = float(np.corrcoef(ranks([x["sim"] for x in pr]), ranks([x["real"] for x in pr]))[0, 1]) if len(pr) >= 3 else float("nan")
    rho_e = float(np.corrcoef(ranks([x["esp"] for x in pr]), ranks([x["real"] for x in pr]))[0, 1]) if len(pr) >= 3 else float("nan")
    allc = [r for r in rows if r["test"] != "tfim_n16" and r["f_real"] > 0]
    within = sum(1 for r in allc if 0.5 <= r["f_sim"] / r["f_real"] <= 2.0)
    gm = {}
    for r in allc:
        if r["f_sim"] > 0:
            gm.setdefault(r["arm"], []).append(math.log(r["f_real"] / r["f_sim"]))
    gmv = {k: math.exp(np.mean(v)) for k, v in gm.items()}
    gap = abs(math.log(gmv["PSF"] / gmv["QK3"])) if "PSF" in gmv and "QK3" in gmv else float("nan")
    v = dict(
        NS1a=("CONFIRMED" if agree >= 6 else "REFUTED"),
        NS1b=("CONFIRMED" if rho >= 0.6 else "REFUTED"),
        NS1c=("CONFIRMED" if within >= 0.8 * len(allc) else "REFUTED"),
        NS1d=("CONFIRMED" if gap < 0.153 else "REFUTED"))
    lines += ["", "## Predictions (fixed in NS1_preregistration.md)", "",
              f"- NS1a (same winner as the device on >= 6 of 7 reading tests): {agree} of {len(pr)} -> {v['NS1a']}",
              f"- NS1b (rank correlation of F_sim ratio and F_real ratio >= 0.6): {rho:+.3f} (ESP's {rho_e:+.3f}) -> {v['NS1b']}",
              f"- NS1c (F_sim within 0.5-2 x F_real on >= 80% of circuits, tfim_n16 left out): {within} of {len(allc)} -> {v['NS1c']}",
              "- NS1d (F_real / F_sim per arm: |ln(PSF / QK3)| < 0.153, ESP's gap): "
              + ", ".join(f"{k} {x:.3f}" for k, x in gmv.items()) + f"; gap {gap:.3f} -> {v['NS1d']}"]
    txt = "\n".join(lines) + "\n"
    open(os.path.join(out, "ns_hw1_score.md"), "w").write(txt)
    json.dump(dict(pairs=pr, verdict=v, rho=rho, rho_esp=rho_e, gm=gmv), open(os.path.join(out, "ns_hw1_score.json"), "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1])
