"""compare_100k_home_pod.py -- R1-R3 of Addendum 261: the home 100,000-lap run against the
pod's run (Addendum 259), lap by lap over laps 1-30,000.

R1 (part A) and R2 (parts B and C): block distance identical as printed at all 30,000 laps
(confirmed); any lap differing by more than 1% (refuted); otherwise ambiguous.
R3: two-qubit count, mapped flag and fallbacks equal at every lap of A, B and C.

    python compare_100k_home_pod.py <home dir> <pod dir>
home dir: pl_100k_{A,B,C}_home.csv; pod dir: pl_100k_{A,B,C}_2026-09-29.csv
"""
import csv
import os
import sys

N = 30_000


def load(path):
    with open(path, encoding="utf-8") as f:
        return {int(r["lap"]): r for r in csv.DictReader(f) if int(r["lap"]) <= N}


def main(home, pod):
    res = {}
    for part in "ABC":
        h = load(os.path.join(home, f"pl_100k_{part}_home.csv"))
        p = load(os.path.join(pod, f"pl_100k_{part}_2026-09-29.csv"))
        laps = sorted(set(h) & set(p))
        same = worst = 0
        first_diff = None
        other = 0
        for lap in laps:
            a, b = h[lap]["block_distance"], p[lap]["block_distance"]
            if a == b:
                same += 1
            else:
                rel = abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)
                worst = max(worst, rel)
                first_diff = first_diff or (lap, a, b)
            if any(h[lap][k] != p[lap][k] for k in ("twoq", "mapped", "fallbacks")):
                other += 1
        res[part] = (len(laps), same, worst, first_diff, other)
        print(f"{part}: laps compared {len(laps)} (home {len(h)}, pod {len(p)}); block distance identical as printed "
              f"{same}; largest relative difference {worst:.3e}; first difference {first_diff}; laps with a different "
              f"two-qubit count / mapped / fallbacks {other}")

    def verdict(parts):
        ok = all(res[x][0] == N and res[x][1] == N for x in parts)
        bad = any(res[x][2] > 0.01 for x in parts)
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
    print(f"R1 part A identical at all {N} laps -> {verdict('A')}")
    print(f"R2 parts B and C identical at all {N} laps -> {verdict('BC')}")
    r3 = all(res[x][0] == N and res[x][4] == 0 for x in "ABC")
    print(f"R3 two-qubit count, mapped and fallbacks equal at every lap -> {'CONFIRMED' if r3 else 'REFUTED'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
