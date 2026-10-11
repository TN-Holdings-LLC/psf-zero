"""analyze.py -- compares arms across C37-X runs (exploratory). Rows are joined by (device, test); an arm name may be
prefixed by its run's label, e.g. r2:C37 and r3:C37.
    python analyze.py r2=<run2 jsonl> r3=<run3 jsonl> -- r3:C37/r2:QK3 r3:C37/r3:QK3x4 ...
Two numbers per pair and device:
  display    geometric-mean ESP ratio on the runnable tests where both outputs use no failed element
  essential  on the runnable tests (best ESP of the pair >= 0.01) with an output on a failed element counted as ESP 0:
             how many of each are 0, and the geometric-mean ratio where both > 0"""
import json, math, sys
import os
DEVICES = tuple(os.environ.get("C37X_DEVICES", "FakeTorino,FakeKingston,FakeMarrakesh,FakeFez").split(","))
args = sys.argv[1:]
sep = args.index("--")
runs, pairs = args[:sep], args[sep + 1:]
data = {}
for spec in runs:
    label, path = spec.split("=", 1)
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        if "arm" not in r or "s2" not in r:
            continue
        data.setdefault((r["device"], r["test"]), {})[f"{label}:{r['arm']}"] = r
L = lambda r: r["s2"] + r["s1"] + r["sm"]
print("| pair | device | display: tests, ESP ratio, 2q ratio | essential: runnable, A=0, B=0, ratio both>0 | time A/B s |")
print("|---|---|---|---|---|")
for pair in pairs:
    A, B = pair.split("/")
    for d in DEVICES:
        rows = [x for (dv, _), x in data.items() if dv == d and A in x and B in x]
        clean = [x for x in rows if not x[A]["failed"] and not x[B]["failed"] and min(L(x[A]), L(x[B])) <= 2]
        disp = 10 ** (sum(L(x[B]) - L(x[A]) for x in clean) / len(clean)) if clean else float("nan")
        r2 = math.exp(sum(math.log(max(x[A]["n2"], 1) / max(x[B]["n2"], 1)) for x in clean) / len(clean)) if clean else float("nan")
        esp = [{k: (0.0 if x[k]["failed"] else 10 ** -L(x[k])) for k in (A, B)} for x in rows]
        run_ = [e for e in esp if max(e.values()) >= 0.01]
        both = [e for e in run_ if e[A] > 0 and e[B] > 0]
        er = 10 ** (sum(math.log10(e[A] / e[B]) for e in both) / len(both)) if both else float("nan")
        ta, tb = sum(x[A]["t"] for x in rows), sum(x[B]["t"] for x in rows)
        print(f"| {A}/{B} | {d} | {len(clean)}, {disp:.3f}, {r2:.3f} | {len(run_)}, {sum(1 for e in run_ if e[A] == 0)}, "
              f"{sum(1 for e in run_ if e[B] == 0)}, {er:.3f} | {ta:.0f}/{tb:.0f} |")
