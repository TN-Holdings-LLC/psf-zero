"""c42_val_detail.py -- C42-VAL in detail (Addendum 450): what item 74 did (tried, kept, no alternative), ESP ratios
where it tried, by family, the largest gains and the time.
    cd <psf-zero repository>; python benchmarks/c42_val_detail.py data/2026-10-11/c42_val"""
import collections
import json
import math
import os
import sys

d = sys.argv[1]
rows = [json.loads(x) for x in open(os.path.join(d, "c42_val.jsonl"))][1:]
fams = json.load(open(os.path.join(d, "c42_tests.json")))
by = collections.defaultdict(dict)
for r in rows:
    by[(r["device"], r["test"])][r["arm"]] = r
L = lambda r: r["s2"] + r["s1"] + r["sm"]  # noqa: E731
esp = lambda r: 0.0 if r["failed"] else 10 ** -L(r)  # noqa: E731
ARMS = ("QK3", "R1", "C41", "C42")
ok = [x for x in by.values() if all(a in x and "error" not in x[a] for a in ARMS)]
tried = [x for x in ok if (x["C42"].get("near_stats") or {}).get("tried", 0) > 0]
run = lambda xs: [x for x in xs if max(esp(x[a]) for a in ARMS) >= 0.01]  # noqa: E731


def gm(p, q, xs):
    xs = [x for x in xs if esp(x[p]) > 0 and esp(x[q]) > 0]
    return (10 ** (sum(L(x[q]) - L(x[p]) for x in xs) / len(xs)), len(xs)) if xs else (float('nan'), 0)


print("errors: tests", len({(r['device'], r['test']) for r in rows if 'error' in r}), "QK3 on failed in those:",
      sum(1 for x in by.values() if 'error' in x.get('R1', {}) and 'error' not in x['QK3'] and x['QK3']['failed'] > 0))
kept = [x for x in tried if x["C42"]["near_stats"].get("kept_alternative")]
print("tried", len(tried), "kept alt", len(kept), "kept first",
      sum(1 for x in tried if x["C42"]["near_stats"].get("kept_first")), "no alt",
      sum(1 for x in tried if x["C42"]["near_stats"].get("no_alternative")))
print("W4 all tried runnable", gm("C42", "C41", run(tried)))
print("kept-alternative runnable", gm("C42", "C41", run(kept)))
print("W4 without Boston, Aachen", gm("C42", "C41", [x for x in run(tried)
                                                     if x["C42"]["device"] not in ("FakeBoston", "FakeAachen")]))
print("better >=10% among runnable tried", sum(1 for x in run(tried) if L(x["C41"]) - L(x["C42"]) > math.log10(1.1)),
      "of", len(run(tried)))
for f in sorted({v[0] for v in fams.values()}):
    s = [x for x in ok if fams[x["C42"]["test"]][0] == f]
    t = [x for x in s if x in tried]
    print(f"{f:16s} tests {len(s):3d} tried {len(t):3d} ESP C42/C41 runnable tried {gm('C42', 'C41', run(t))[0]:.3f} "
          f"all-run C42/R1 {gm('C42', 'R1', run(s))[0]:.3f} C42/QK3 {gm('C42', 'QK3', run(s))[0]:.3f}")
print("largest gains:")
for x in sorted(run(tried), key=lambda x: L(x["C42"]) - L(x["C41"]))[:10]:
    print(f"  {x['C42']['test']:30s} {x['C42']['device'][4:]:10s} {esp(x['C41']):.3g} -> {esp(x['C42']):.3g} "
          f"QK3 {esp(x['QK3']):.3g} 2q {x['C41']['n2']}->{x['C42']['n2']}")
ts = sorted(x["C42"]["t"] / max(x["C41"]["t"], 1e-3) for x in tried)
print("time ratio tried median %.2f p90 %.2f max %.1f" % (ts[len(ts) // 2], ts[int(.9 * len(ts))], ts[-1]))
print("max C42 t", max(x["C42"]["t"] for x in ok), "on", max(ok, key=lambda x: x["C42"]["t"])["C42"]["test"])
