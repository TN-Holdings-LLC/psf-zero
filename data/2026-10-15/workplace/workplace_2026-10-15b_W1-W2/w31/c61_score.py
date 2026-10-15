"""C61-VAL scoring (the predictions fixed in C61-VAL_preregistration.md).  python c61_score.py <run dir>"""
import collections
import glob
import json
import re
import statistics
import sys

rows = [json.loads(l) for f in sorted(glob.glob(sys.argv[1] + "/**/c61_val_rows_*.jsonl", recursive=True))
        for l in open(f)]
R = [r for r in rows if r["part"] == "R"]
D = [r for r in rows if r["part"] == "D"]
K = [r for r in rows if r["part"] == "K"]
out = []
p = out.append


def nq(name):
    return int(re.search(r"(\d+)", name).group(1))


def ok_common(x):
    return (x["exact"] is None or x["exact"] <= 1e-6) and x["isa"] and not x["failed"] and x["rzz_ok"]


def verdict(bad, total):
    return "CONFIRMED" if total and not bad else ("NOT TESTED" if not total else "REFUTED")


T1 = [(r, x) for r in R if r["kind"] == "native" for x in r["rows"]]
bad1 = collections.Counter(r["test"] + " " + r["target"] for r, x in T1 if not x["same"])
p(f"T1 (native identical to c60): {len(T1)} outputs over {len({(r['test'], r['target']) for r, _ in T1})} cells; "
  f"not identical {sum(bad1.values())} {dict(bad1)} -> {verdict(bad1, len(T1))}")
T2 = [(r, x) for r in R if r["kind"] != "native" and r["target"] == "F" for x in r["rows"]]
bad2 = collections.Counter(r["test"] for r, x in T2 if not (ok_common(x) and abs(x["dur"] - x["dur60"]) <= 1e-9
                                                              and x["twoq"] == x["twoq60"]))
nchk2 = sum(1 for _, x in T2 if x["exact"] is not None)
p(f"T2 (frame path: exact, native, duration and 2q equal to c60): {len(T2)} outputs (exactness checked on {nchk2}); "
  f"failing {sum(bad2.values())} {dict(bad2)} -> {verdict(bad2, len(T2))}")
T3 = [(r, x) for r in R if r["kind"] != "native" and r["target"] == "T" for x in r["rows"]]
fails3 = collections.defaultdict(collections.Counter)
for r, x in T3:
    if not ok_common(x):
        fails3[r["test"]]["exact/native"] += 1
    if x["twoq"] > x["twoq60"]:
        fails3[r["test"]]["2q more"] += 1
    if x["dur"] > x["dur60"] + 1e-9:
        fails3[r["test"]]["longer"] += 1
nchk3 = sum(1 for _, x in T3 if x["exact"] is not None)
p(f"T3 (CZ path: exact, native, 2q <= c60, duration <= c60): {len(T3)} outputs (exactness checked on {nchk3}); "
  f"failing {dict((k, dict(v)) for k, v in fails3.items())} -> {verdict(fails3, len(T3))}")
for t, v in fails3.items():
    xs = [x["dur"] / x["dur60"] for r, x in T3 if r["test"] == t]
    p(f"    {t}: template / c60 duration over its {len(xs)} outputs: median {statistics.median(xs):.4f}, "
      f"max {max(xs):.4f}, min {min(xs):.4f}")
bad4 = []
for r in R:
    lim = 5.0 if nq(r["test"]) > 20 else 2.0
    if r["bind_median_ms"] is None or r["bind_median_ms"] > lim:
        bad4.append((r["test"], r["target"], None if r["bind_median_ms"] is None else round(r["bind_median_ms"], 3)))
p(f"T4 (bind median <= 2 ms, n = 40: <= 5 ms): {len(R) - len(bad4)} of {len(R)} cells; missing {bad4} -> "
  f"{verdict(bad4, len(R))}")
bad5 = [(r["test"], r["target"], round(r["build_s"] / r["c60_s"], 2)) for r in R if r["build_s"] > 2 * r["c60_s"]]
p(f"T5 (build <= 2 x c60 compile): {len(R) - len(bad5)} of {len(R)}; over {bad5} -> {verdict(bad5, len(R))}")
bad6 = collections.Counter()
for r in D:
    for x in r["rows"]:
        good = x["same"] if r["kind"] == "native" else ok_common(x)
        if not good:
            bad6[(r["device"], r["test"], r["target"])] += 1
nd = sum(len(r["rows"]) for r in D)
p(f"T6 (other 4 devices, correctness): {nd} outputs over {len(D)} cells; failing {dict(bad6)} -> {verdict(bad6, nd)}")
for r in K:
    rss = dict((a, b) for a, b in r["rss"])
    grow = rss.get(r["laps"], max(rss.values())) - rss.get(10000, min(rss.values()))
    ok = r["worst_check"] is not None and r["worst_check"] <= 1e-6 and grow <= 20
    p(f"T7 (100k, {r['target']}): {r['laps']} binds; worst check {r['worst_check']:.1e}; RSS growth lap 10k -> "
      f"{r['laps'] // 1000}k {grow:.1f} MB; median {r['median_ms']:.3f} ms, p99 {r['p99_ms']:.3f}, max "
      f"{r['max_ms']:.1f}; rebuilt {r['rebuilt']}, switched {r.get('switched')} -> {'CONFIRMED' if ok else 'REFUTED'}")
p("")
p("Per cell (FakeFez): kind | bind median ms (values_of ms) | build ms / c60 ms | Qiskit assign ms | rebuilt, switched |"
  " template/c60 duration (median, max) | 2q template/c60")
for r in R:
    xs = [x["dur"] / x["dur60"] for x in r["rows"]]
    qs = [x["twoq"] / x["twoq60"] for x in r["rows"]]
    qa = r.get("qiskit_assign_median_ms")
    p(f"  {r['test']:16s} {r['target']} {r['kind']:6s} | {r['bind_median_ms'] or float('nan'):6.3f} "
      f"({r['values_of_median_ms']:6.3f}) | {r['build_s']*1000:6.1f} / {r['c60_s']*1000:6.1f} | "
      f"{'-' if qa is None else f'{qa:.3f}'} | {r['rebuilt']}, {r['switched']} | {statistics.median(xs):.4f}, "
      f"{max(xs):.4f} | {statistics.median(qs):.3f}")
print("\n".join(out))
open(sys.argv[1] + "/c61_val_score.txt", "w").write("\n".join(out) + "\n")
