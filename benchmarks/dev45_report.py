import json, sys, math, collections
old = [json.loads(x) for x in open("<c44_val run dir>/c44_val.jsonl")][1:]
new = [json.loads(x) for x in open(sys.argv[1])]
by = collections.defaultdict(dict)
for r in old + new:
    by[(r["test"], r["device"])][r["arm"]] = r
L = lambda r: r["s2"] + r["s1"] + r["sm"]
def ok(r): return r is not None and "error" not in r and r["failed"] == 0
def esp(r): return 10 ** -L(r) if ok(r) else 0.0
arms = ("QK3", "C44", "C45n", "C45")
rows = [(k, x) for k, x in by.items() if all(ok(x.get(a)) for a in arms)]
run = [(k, x) for k, x in rows if max(esp(x[a]) for a in arms) >= 0.01]
def gm(p, q, sel):
    return 10 ** (sum(L(x[q]) - L(x[p]) for _, x in sel) / len(sel))
print("complete", len(rows), "runnable", len(run), "errors", sum(1 for r in new if "error" in r))
for d in sorted({k[1] for k, _ in run}) + ["all"]:
    sel = [(k, x) for k, x in run if d == "all" or k[1] == d]
    print(f"{d:14s} n={len(sel):3d} C45n/C44 {gm('C45n','C44',sel):.3f} C45n/QK3 {gm('C45n','QK3',sel):.3f} C45/C44 {gm('C45','C44',sel):.3f} C45/QK3 {gm('C45','QK3',sel):.3f}"
          f"  C45<0.95QK3 {sum(1 for _, x in sel if L(x['C45'])-L(x['QK3']) > -math.log10(0.95))}  C45n<0.95C44 {sum(1 for _, x in sel if L(x['C45n'])-L(x['C44']) > -math.log10(0.95))}")
fam = lambda t: t.split("_")[0] if not t.startswith("gen_") else t.split("_")[1]
print()
for f in sorted({fam(k[0]) for k, _ in run}):
    sel = [(k, x) for k, x in run if fam(k[0]) == f]
    print(f"{f:8s} n={len(sel):3d} C45n/C44 {gm('C45n','C44',sel):.3f} C45n/QK3 {gm('C45n','QK3',sel):.3f} C45/C44 {gm('C45','C44',sel):.3f} C45/QK3 {gm('C45','QK3',sel):.3f}")
med = lambda v: sorted(v)[len(v)//2]
print("median time C45/C44", round(med([x["C45"]["t"]/max(x["C44"]["t"],1e-3) for _, x in rows]),2), "C45n/C44", round(med([x["C45n"]["t"]/max(x["C44"]["t"],1e-3) for _, x in rows]),2), "max C45", max(x["C45"]["t"] for _, x in rows))
l3 = collections.Counter()
for r in new:
    if r.get("arm") == "C45": l3.update(r.get("l3c", {}))
print("L3C", dict(l3))
worst = sorted(run, key=lambda kx: L(kx[1]["QK3"]) - L(kx[1]["C45"]))[:6]
for k, x in worst: print("worst vs QK3", k, f"{esp(x['C45'])/esp(x['QK3']):.3f}", x["C45"].get("l3c"))
