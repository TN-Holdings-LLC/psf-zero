"""c40_val_families.py -- C40-VAL by family (Addendum 448): ESP and two-qubit ratios, how often STEINER6 was kept.
    cd <psf-zero repository>; python benchmarks/c40_val_families.py data/2026-10-11/c40_val/c40_val.jsonl"""
import json, math, sys
sys.path.insert(0, "benchmarks")
from c40_val import generated
rows = [json.loads(x) for x in open(sys.argv[1])][1:]
by = {}
for r in rows:
    by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
L = lambda r: r["s2"] + r["s1"] + r["sm"]
esp = lambda r: 0.0 if r["failed"] else 10 ** -L(r)
def gm(p, q, xs):
    ok = [x for x in xs if esp(x[p]) > 0 and esp(x[q]) > 0]
    return (10 ** (sum(L(x[q]) - L(x[p]) for x in ok) / len(ok)), len(ok)) if ok else (float("nan"), 0)
def g2(p, q, xs):
    return math.exp(sum(math.log((x[p]["n2"] + 1) / (x[q]["n2"] + 1)) for x in xs) / len(xs))
fam = lambda x: generated()[x["C40"]["test"]][1]
xs = list(by.values())
run = lambda s: [x for x in s if max(esp(x[k]) for k in ("QK3", "R1", "C39", "C40")) >= 0.01]
groups = {f: [x for x in xs if fam(x) == f] for f in sorted({fam(x) for x in xs})}
groups["Y6 noncommuting (trotter+single)"] = [x for x in xs if fam(x) in ("trotter_noncommuting", "single_noncommuting")]
for g, s in groups.items():
    r = run(s)
    ch = sum(1 for x in s if (x["C40"].get("steiner_stats") or {}).get("chosen"))
    better = sum(1 for x in r if L(x["R1"]) - L(x["C40"]) > math.log10(1.1))
    print(f"{g:34s} n={len(s):3d} run={len(r):3d} ESP C40/R1 {gm('C40','R1',r)[0]:.3f} C40/QK3 {gm('C40','QK3',r)[0]:.3f} "
          f"R1/QK3 {gm('R1','QK3',r)[0]:.3f} C40/C39 {gm('C40','C39',r)[0]:.3f} 2q C40/R1 {g2('C40','R1',s):.3f} "
          f"C40/QK3 {g2('C40','QK3',s):.3f} steiner chosen {ch}/{len(s)} >=1.1x better {better}/{len(r)}")
# per-test qaoa by n,p
print()
for t in sorted({x["C40"]["test"] for x in xs if fam(x) in ("qaoa", "trotter_commuting", "rotation_gates")}):
    s = [x for x in xs if x["C40"]["test"] == t]
    print(t, "ESP C40/R1 %.3f C40/QK3 %.3f  ESP C40 median %.3g  2q C40/R1/QK3 %d/%d/%d" % (
        gm("C40", "R1", s)[0], gm("C40", "QK3", s)[0], sorted(esp(x["C40"]) for x in s)[3],
        sorted(x["C40"]["n2"] for x in s)[3], sorted(x["R1"]["n2"] for x in s)[3], sorted(x["QK3"]["n2"] for x in s)[3]))
# time
ts = sorted(x["C40"]["t"] / max(x["R1"]["t"], 1e-3) for x in xs if x["C40"].get("parsed_terms"))
print("time ratio median %.1f max %.1f; max C40 t %.2f" % (ts[len(ts)//2], ts[-1], max(x["C40"]["t"] for x in xs)))
