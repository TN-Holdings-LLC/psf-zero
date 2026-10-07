"""depth_r_qpu_time.py -- exploratory (Addendum 401, 2026-10-07; not pre-registered): from DEPTH-R's recorded
outputs (Addendum 400), how many shots each compiler's circuits need to reach a common target accuracy, and how much
device time that is.

Shots: for every test point the recorded noisy z and the readout error of the qubit carrying logical 0 give the
probability q of reading 0; with S shots (S odd) the prediction is the majority outcome, so the chance it is right
is a binomial tail, exactly (no sampling). A cell's expected accuracy at S shots is the mean over its points. Target:
the noiseless model's accuracy on that cell minus 1 test point, the same for every arm. Reported: the smallest S on a
fixed grid that reaches it, or "not reached" (up to 131,071 shots).

Device time per shot: the compiled circuit's duration (as-soon-as-possible schedule with the fake device's own gate
durations; rz is virtual) plus the duration of the final measurement of the qubit carrying logical 0. Measured on
the first 3 test points of each cell, recompiled exactly as DEPTH-R compiled them (the two-qubit count is checked
against the record); the median is used. A real device adds a per-shot reset or repetition delay that is the same for
every compiler; with --overhead-us device the fake device's own default repetition delay (`default_rep_delay`,
recorded by `durations` when the backend reports one) is added to every shot, otherwise the given number of us.

    python benchmarks/depth_r_qpu_time.py durations --out data/2026-10-07/depth_r_qpu_time   (needs Qiskit; minutes)
    python benchmarks/depth_r_qpu_time.py score --out data/2026-10-07/depth_r_qpu_time [--overhead-us 0|device|<us>]
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RUN = os.path.join(REPO, "data", "2026-10-07", "depth_r")
ARMS = ("RPSF", "REC", "L3T")
GRID = sorted({s for s in range(1, 200, 2)} | {int(round(201 * 1.08 ** k)) | 1 for k in range(0, 80)
                                                if 201 * 1.08 ** k < 131072})
K_DUR = 3
BUDGETS = (15, 63, 255, 1023, 4095)


def cells():
    out = {}
    for p in sorted(glob.glob(os.path.join(RUN, "deploy_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        for r in d["rows"]:
            out.setdefault((m["device"], m["dataset"], m["n"], r["L"], m["arm"]), []).append(r)
    return out


def q0(r):
    p0 = (1 + r["z_noisy"]) / 2
    return min(1.0, max(0.0, p0 * (1 - r["e01"]) + (1 - p0) * r["e10"]))


def expected_acc(rows, S):
    """Mean over points of P(majority of S shots gives the right sign), S odd."""
    q = np.array([q0(r) for r in rows])
    y = np.array([r["y"] for r in rows])
    half = (S - 1) // 2
    p_pos = binom.sf(half, S, q)          # P(#zeros > S/2): predicts +1
    return float(np.mean(np.where(y > 0, p_pos, 1 - p_pos)))


def shots_to(rows, target):
    for S in GRID:
        if expected_acc(rows, S) >= target - 1e-12:
            return S
    return None


def asap_duration(out, t, p_meas):
    avail = {}
    for ins in out.data:
        name = ins.operation.name
        qs = tuple(out.find_bit(q).index for q in ins.qubits)
        start = max((avail.get(q, 0.0) for q in qs), default=0.0)
        d = 0.0
        if name not in ("barrier",):
            try:
                props = t[name][qs]
                d = float(props.duration or 0.0) if props is not None else 0.0
            except KeyError:
                d = 0.0
        for q in qs:
            avail[q] = start + d
    body = max(avail.values(), default=0.0)
    meas = float(t["measure"][(p_meas,)].duration or 0.0)
    return body, meas


def do_durations(args):
    sys.path[:0] = [HERE, REPO]
    import depth_r_eval as R
    from qiskit_ibm_runtime import fake_provider as fp
    E = R.E
    P = E.load_psf(R.RELEASE)
    C = cells()
    split, _ = E.seeds(False)
    res, data, bes, comps = {}, {}, {}, {}
    for (dev, ds, n, L, arm) in sorted(C):
        if (ds, n) not in data:
            data[(ds, n)] = E.data(ds, n, split)
        if dev not in bes:
            bes[dev] = getattr(fp, dev)()
        if (dev, arm) not in comps:
            comps[(dev, arm)] = E.compiler(arm, bes[dev], P)
        _, _, Xte, _ = data[(ds, n)]
        th = np.load(E.theta_path(RUN, ds, n, L))
        rows = sorted(C[(dev, ds, n, L, arm)], key=lambda r: r["i"])[:K_DUR]
        vals = []
        for r in rows:
            out = comps[(dev, arm)](E.circuit(Xte[r["i"]], th, n, L))
            fin = list(out.layout.final_index_layout(filter_ancillas=True))
            body, meas = asap_duration(out, bes[dev].target, fin[0])
            n2q = sum(1 for g in out.data if len(g.qubits) == 2)
            vals.append(dict(i=r["i"], body_s=body, meas_s=meas, n2q=n2q, n2q_recorded=r["n2q"]))
        key = f"{dev}|{ds}|{n}|{L}|{arm}"
        res[key] = vals
        print(key, [round(v["body_s"] * 1e6, 2) for v in vals], "us; n2q", [v["n2q"] for v in vals],
              "recorded", [v["n2q_recorded"] for v in vals], flush=True)
    rep = {}
    for dev, be in bes.items():
        v = getattr(be, "default_rep_delay", None)
        if v is None:
            try:
                v = be.configuration().default_rep_delay
            except Exception:  # noqa: BLE001 - BackendV2 without a configuration
                v = None
        rep[dev] = None if v is None else float(v)
    print("default_rep_delay (s):", rep)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "durations.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(psf_version=P.VERSION, k=K_DUR, rep_delay_s=rep, cells=res), f, indent=1)


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def do_score(args):
    C = cells()
    durf = os.path.join(args.out, "durations.json")
    dj = json.load(open(durf, encoding="utf-8")) if os.path.exists(durf) else {}
    dur, repd = dj.get("cells", {}), dj.get("rep_delay_s", {})

    def overhead(dev):
        if args.overhead_us == "device":
            v = repd.get(dev)
            return None if v is None else v * 1e6
        return float(args.overhead_us)
    mism = [(k, v["i"]) for k, vs in dur.items() for v in vs if v["n2q"] != v["n2q_recorded"]]
    lines = ["# DEPTH-R: shots and device time to a common target (exploratory, Addendum 401)", "",
             f"target = the noiseless model's accuracy on the cell minus 1 test point; S odd, majority vote; "
             f"per-shot overhead {args.overhead_us} us ({ {d: overhead(d) for d in ('FakeAuckland', 'FakeTorino')} }); durations from {K_DUR} recompiled points per cell "
             f"({'none' if not dur else len(dur)} cells; two-qubit count differing from the record: {len(mism)})", ""]
    rows_out, summary = [], {}
    for dev in ("FakeAuckland", "FakeTorino"):
        lines += [f"## {dev}", "", "| dataset | n | L | target | " + " | ".join(
            f"{a}: acc at inf / shots / us per shot / ms per prediction" for a in ARMS) + " |",
                  "|" + "---|" * (4 + len(ARMS))]
        for ds in ("BC", "D38"):
            for n in (4, 6):
                for L in (1, 2, 4, 8, 12, 16):
                    base = C[(dev, ds, n, L, ARMS[0])]
                    nt = len(base)
                    ideal = float(np.mean([np.sign(r["z_ideal"]) == r["y"] for r in base]))
                    target = ideal - 1.0 / nt
                    cellrow = dict(device=dev, dataset=ds, n=n, L=L, target=target)
                    parts = []
                    for a in ARMS:
                        rows = C[(dev, ds, n, L, a)]
                        inf = float(np.mean([(q0(r) > 0.5) == (r["y"] > 0) for r in rows]))
                        S = shots_to(rows, target)
                        v = dur.get(f"{dev}|{ds}|{n}|{L}|{a}")
                        ov = overhead(dev)
                        per = statistics.median([x["body_s"] + x["meas_s"] for x in v]) * 1e6 + ov \
                            if (v and ov is not None) else None
                        ms = S * per / 1000 if (S and per is not None) else None
                        cellrow[a] = dict(acc_inf=inf, shots=S, us_per_shot=per, ms_per_prediction=ms)
                        parts.append(f"{inf:.3f} / {S if S else 'not reached'} / "
                                     f"{'' if per is None else f'{per:.1f}'} / {'' if ms is None else f'{ms:.2f}'}")
                    rows_out.append(cellrow)
                    lines.append(f"| {ds} | {n} | {L} | {target:.3f} | " + " | ".join(parts) + " |")
        lines.append("")
        both = [r for r in rows_out if r["device"] == dev and all(r[a]["shots"] for a in ARMS)]
        s = {}
        for a, b in (("REC", "RPSF"), ("REC", "L3T")):
            s[f"shots {a}/{b}"] = gmean([r[a]["shots"] / r[b]["shots"] for r in both])
            tt = [r for r in both if r[a]["ms_per_prediction"] and r[b]["ms_per_prediction"]]
            s[f"time {a}/{b}"] = gmean([r[a]["ms_per_prediction"] / r[b]["ms_per_prediction"] for r in tt])
        for a, b in (("REC", "RPSF"), ("REC", "L3T")):
            rs = [r[a]["shots"] / r[b]["shots"] for r in both]
            s[f"shots {a}/{b} median"] = statistics.median(rs) if rs else float("nan")
            s[f"shots {a}<{b} / = / > cells"] = f"{sum(x < 1 for x in rs)}/{sum(x == 1 for x in rs)}/{sum(x > 1 for x in rs)}"
        nr = {a: sum(1 for r in rows_out if r["device"] == dev and not r[a]["shots"]) for a in ARMS}
        budget = {}
        for S in BUDGETS:
            budget[S] = {a: float(np.mean([expected_acc(C[(dev, ds, n, L, a)], S) for ds in ("BC", "D38")
                                           for n in (4, 6) for L in (1, 2, 4, 8, 12, 16)])) for a in ARMS}
        summary[dev] = dict(cells_all_reached=len(both), not_reached=nr, accuracy_at_budget=budget, **s)
        lines += [f"{dev}: cells where every arm reached the target {len(both)} of 24; not reached {nr}; "
                  + "; ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()), "",
                  f"{dev}, pooled expected accuracy at a fixed shot budget (24 cells; percentage points REC - RPSF, "
                  f"REC - L3T):", ""]
        lines += [f"- {S} shots: " + ", ".join(f"{a} {budget[S][a]:.4f}" for a in ARMS)
                  + f" ({100 * (budget[S]['REC'] - budget[S]['RPSF']):+.2f}, {100 * (budget[S]['REC'] - budget[S]['L3T']):+.2f})"
                  for S in BUDGETS]
        lines.append("")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"score_overhead_{args.overhead_us}.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, f"qpu_time_overhead_{args.overhead_us}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(overhead_us=args.overhead_us, rep_delay_s=repd, summary=summary, cells=rows_out, n2q_mismatch=mism), f,
                  indent=1)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("durations", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--overhead-us", default="0")
    a = ap.parse_args()
    {"durations": do_durations, "score": do_score}[a.mode](a)


if __name__ == "__main__":
    main()
