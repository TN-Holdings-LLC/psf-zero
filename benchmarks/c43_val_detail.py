"""c43_val_detail.py -- C43-VAL in detail (Addendum 452): the tests only C43 compiled (the others raised
FailedElementsError), the tests where C43 is below 0.95 x C41, and ESP by family.
    cd <psf-zero repository>; python benchmarks/c43_val_detail.py data/2026-10-11/c43_val"""
import collections
import json
import math
import os
import sys

d = sys.argv[1]
rows = [json.loads(x) for x in open(os.path.join(d, "c43_val.jsonl"))][1:]
fams = json.load(open(os.path.join(d, "c43_tests.json")))
by = collections.defaultdict(dict)
for r in rows:
    by[(r["device"], r["test"])][r["arm"]] = r
L = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731
esp = lambda r: 0 if r["failed"] else 10 ** -L(r)  # noqa: E731
ALL = ("QK3", "C41", "C42", "C43", "C41r", "C43r")
print("errors by arm:", dict(collections.Counter(r["arm"] for r in rows if "error" in r)))
print("compiled by C43 where C41 raised FailedElementsError:")
for k, x in sorted(by.items()):
    if "error" in x.get("C41", {}) and "error" not in x.get("C43", {}):
        c = x["C43"]
        print(f"  {k[1]:16s} {k[0][4:]:10s} C43 on failed {c['failed']}, ESP {esp(c):.3g}, repeats {x['C43r'].get('repeats')}; "
              f"QK3 on failed {x['QK3'].get('failed')}, ESP {esp(x['QK3']):.3g}")
ok = [x for x in by.values() if all(a in x and "error" not in x[a] for a in ALL)]
run = [x for x in ok if max(esp(x[a]) for a in ("QK3", "C41", "C42", "C43")) >= 0.01]
print("runnable tests with ESP C43 below 0.95 x C41:")
for x in run:
    if L(x["C43"]) - L(x["C41"]) > -math.log10(0.95):
        c = x["C43"]
        print(f"  {c['test']} {c['device'][4:]}: C43 {esp(c):.3g}, C41 {esp(x['C41']):.3g}; item 74 {c['near_stats']}")
print("by family, runnable (ESP C43 / C41, C43 / QK3):")
for f in sorted({v[0] for v in fams.values()}):
    xs = [x for x in run if fams[x["C43"]["test"]][0] == f]
    if xs:
        print(f"  {f:12s} {len(xs):3d} {10 ** (sum(L(x['C41']) - L(x['C43']) for x in xs) / len(xs)):.3f} "
              f"{10 ** (sum(L(x['QK3']) - L(x['C43']) for x in xs) / len(xs)):.3f}")
print("C41 repeated its output on", sum(1 for x in ok if x["C41r"].get("repeats")), "of", len(ok), "; C43 on",
      sum(1 for x in ok if x["C43r"].get("repeats")))
print("largest C43 time %.1f s" % max(x["C43"]["t"] for x in ok))
