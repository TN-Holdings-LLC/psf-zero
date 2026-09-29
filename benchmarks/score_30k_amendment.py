"""score_30k_amendment.py -- scoring for Amendment 1 (2026-09-29) of the pl-heavyhex-100k
pre-registration: the pod run is stopped once all three parts have passed lap 30,000
(for time), and the predictions are scored on laps 1-30,000 with the thresholds scaled
as written in the amendment. Reads the CSVs and logs of pl_heavyhex_100k.py; rows after
lap N (including a possibly half-written last row) are ignored.

    python -u score_30k_amendment.py [--n 30000] [--tag 2026-09-29]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st

import numpy as np

PARTS = {"A": ("PN", 0), "B": ("PN", 4), "C": ("P", 4)}


def load(tag, part, n):
    rows = []
    with open(f"pl_100k_{part}_{tag}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lap = int(r["lap"])
                if lap > n:
                    break
                r["lap"], r["twoq"], r["swapfree_twoq"] = lap, int(r["twoq"]), int(r["swapfree_twoq"])
                r["mapped"], r["fallbacks"], r["compile_s"] = int(r["mapped"]), int(r["fallbacks"]), float(r["compile_s"])
                for k in ("block_distance", "whole_circuit", "rss_mb"):
                    r[k] = float(r[k]) if r.get(k) not in ("", None) else None
            except (TypeError, ValueError, KeyError):
                break  # a half-written row at the point the process was stopped
            rows.append(r)
    return rows


def v(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--tag", default="2026-09-29")
    a = ap.parse_args()
    n = a.n
    R = {p: load(a.tag, p, n) for p in PARTS}
    logs = {p: open(f"pl_100k_{p}_{a.tag}.txt", encoding="utf-8").read() for p in PARTS}
    print("=" * 78)
    print(f"SCORING, Amendment 1: laps 1-{n:,} (thresholds as written in the amendment)")
    print("=" * 78)
    full = all(len(R[p]) == n and R[p][-1]["lap"] == n for p in PARTS)
    ok_v = ("CORE_VERSION 2026-09-29.1 | LAYOUT 2026-09-29.c1" in logs["A"] and
            "CORE_VERSION 2026-09-29.1 | LAYOUT 2026-09-29.c1" in logs["B"] and
            "CORE_VERSION 2026-09-28.1 | LAYOUT 2026-09-26.m1" in logs["C"])
    dev = all("device lightning.gpu" in logs[p] for p in PARTS)
    c1 = [float(re.search(r"changes the values by ([0-9.e+-]+)", logs[p]).group(1)) for p in PARTS]
    print(f"C0 all parts have laps 1-{n:,}: {full} ({[len(R[p]) for p in PARTS]}); versions: {ok_v}; lightning.gpu: {dev}; "
          f"C1 smallest {min(c1):.2e} -> {'passed' if full and ok_v and dev and min(c1) >= 1e-9 else 'FAILED'}")
    for p in PARTS:
        r = R[p]
        t = [x["compile_s"] for x in r]
        tenth = len(t) // 10
        bd = [x for x in r if x["block_distance"] is not None]
        wc = [(x["lap"], x["whole_circuit"]) for x in r if x["whole_circuit"] is not None]
        rs = [(x["lap"], x["rss_mb"]) for x in r if x["rss_mb"] is not None]
        print(f"   {p} ({PARTS[p][0]}, spare {PARTS[p][1]}): compile median {st.median(t) * 1000:.2f} ms, p99 "
              f"{np.percentile(t, 99) * 1000:.2f} ms, max {max(t) * 1000:.0f} ms; within 1 s {sum(x <= 1.0 for x in t)}; "
              f"mapped {sum(x['mapped'] for x in r)}; 2q {sorted({x['twoq'] for x in r})}; fallbacks "
              f"{sum(x['fallbacks'] for x in r)}; block lap 1 / 1,000 / {bd[-1]['lap']:,}: {bd[0]['block_distance']:.2e} / "
              f"{next(x['block_distance'] for x in bd if x['lap'] >= 1000):.2e} / {bd[-1]['block_distance']:.2e}; "
              f"whole-circuit checkpoints {[(l, f'{w:.1e}') for l, w in wc]}; RSS {rs[0][1]:.0f} -> {rs[-1][1]:.0f} MB; "
              f"first/last 10% median {st.median(t[:tenth]) * 1000:.2f} / {st.median(t[-tenth:]) * 1000:.2f} ms")
    A = R["A"]
    n1 = sum(x["compile_s"] <= 1.0 for x in A)
    c_h1, r_h1 = n - n // 10_000, n - n // 100
    print(f"H1 A within 1 s in {n1}/{n} (confirmed >= {c_h1}, refuted < {r_h1}) -> {v(n1 >= c_h1, n1 < r_h1)}")
    mp = sum(x["mapped"] and x["twoq"] == x["swapfree_twoq"] for x in A)
    r_h2 = n - n // 1_000
    print(f"H2 A swap-free and mapped back in {mp}/{n} (confirmed {n}, refuted < {r_h2}) -> {v(mp == n, mp < r_h2)}")
    fab = sum(x["fallbacks"] for x in R["A"] + R["B"])
    print(f"H3 candidate core fallbacks in A + B: {fab} (confirmed 0, refuted >= 3) -> {v(fab == 0, fab >= 3)}")
    d = [(x["lap"], x["block_distance"]) for x in A if x["block_distance"] is not None and x["lap"] <= 1000]
    k, c = np.polyfit(np.array([l for l, _ in d], float), np.array([y for _, y in d]), 1)
    last = [x for x in A if x["block_distance"] is not None][-1]
    pred = c + k * last["lap"]
    ratio = last["block_distance"] / pred
    print(f"H4 A drift: slope laps 1-1,000 {k:.3e}/lap, extrapolated to lap {last['lap']:,} {pred:.3e}, measured "
          f"{last['block_distance']:.3e}, ratio {ratio:.2f} (confirmed <= 2, refuted > 10) -> {v(ratio <= 2, ratio > 10)}")
    wmax = max(x["whole_circuit"] for p in ("A", "B") for x in R[p] if x["whole_circuit"] is not None)
    print(f"H5 whole-circuit check, every checkpoint of A and B up to lap {n:,}: max {wmax:.2e} (confirmed <= 1e-8, "
          f"refuted > 1e-6) -> {v(wmax <= 1e-8, wmax > 1e-6)}")
    grow = []
    for p in PARTS:
        rs = [(x["lap"], x["rss_mb"]) for x in R[p] if x["rss_mb"] is not None]
        base = next(r for lap, r in rs if lap >= 1000)
        grow.append(rs[-1][1] - base)
    print(f"H6 RSS growth from lap 1,000 to lap {n:,}: {[round(x) for x in grow]} MB (confirmed all <= 100, refuted any "
          f"> 500) -> {v(max(grow) <= 100, max(grow) > 500)}")
    bB = [x for x in R["B"] if x["block_distance"] is not None][-1]["block_distance"]
    bC = [x for x in R["C"] if x["block_distance"] is not None][-1]["block_distance"]
    rr = bB / bC
    print(f"H7 spare 4, candidate / release block distance at lap {n:,}: {bB:.3e} / {bC:.3e} = {rr:.2f} (confirmed "
          f"0.5-2, refuted < 0.2 or > 5) -> {v(0.5 <= rr <= 2, rr < 0.2 or rr > 5)}")
    t = [x["compile_s"] for x in A]
    tenth = len(t) // 10
    q = st.median(t[-tenth:]) / st.median(t[:tenth])
    print(f"H8 A timing: last-10% / first-10% median {q:.2f} (confirmed <= 1.5, refuted > 3) -> {v(q <= 1.5, q > 3)}")
    print(f"Reported without prediction: release core fallbacks in C {sum(x['fallbacks'] for x in R['C'])}; "
          f"rows present beyond lap {n:,} were ignored")


if __name__ == "__main__":
    main()
