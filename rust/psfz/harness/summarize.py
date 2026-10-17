"""Summary of compare.py's record: python rust/psfz/harness/summarize.py <compare.json>"""
import json
import math
import statistics
import sys

rows = json.load(open(sys.argv[1]))["rows"]
ok = [r for r in rows if "refused" not in r["psfz"]]
ref = [r for r in rows if "refused" in r["psfz"]]
print(f"inputs {len(rows)}; psfz compiled {len(ok)}, refused {len(ref)} {sorted({(r['device'], r['test'], r['psfz']['refused'][:50]) for r in ref})[:6]}")
print(f"native: {sum(r['psfz']['native'] for r in ok)} of {len(ok)}; exact (1e-6): "
      f"{sum(1 for r in ok if r['psfz']['tvd'] is not None and r['psfz']['tvd'] < 1e-6)} of "
      f"{sum(1 for r in ok if r['psfz']['tvd'] is not None)} checked; largest {max((r['psfz']['tvd'] for r in ok if r['psfz']['tvd'] is not None), default=None)}")
both = [r for r in ok if "refused" not in r["rel"]]
rd = [r["psfz"]["dur_us"] / r["rel"]["dur_us"] for r in both]
r2 = [r["psfz"]["q2"] / r["rel"]["q2"] for r in both if r["rel"]["q2"]]
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))  # noqa: E731
print("duration over the release's: geometric mean %.4f, median %.4f, longer in %d of %d, largest %.4f, smallest %.4f"
      % (gm(rd), statistics.median(rd), sum(1 for a in rd if a > 1 + 1e-9), len(rd), max(rd), min(rd)))
print("two-qubit gates over the release's: geometric mean %.4f, more in %d, fewer in %d" %
      (gm(r2), sum(1 for a in r2 if a > 1 + 1e-9), sum(1 for a in r2 if a < 1 - 1e-9)))
print("compile time, median ms: psfz %.3f, release %.2f" % (1000 * statistics.median(r["psfz_s"] for r in ok),
                                                             1000 * statistics.median(r["rel_s"] for r in both)))
print("per test: duration over the release's (geometric mean / largest), two-qubit gates psfz / release (first input)")
for name in dict.fromkeys(r["test"] for r in both):
    v = [r["psfz"]["dur_us"] / r["rel"]["dur_us"] for r in both if r["test"] == name]
    f = next(r for r in both if r["test"] == name)
    print(f"  {name:14s} {gm(v):.4f} / {max(v):.4f}   q2 {f['psfz']['q2']} / {f['rel']['q2']}")
