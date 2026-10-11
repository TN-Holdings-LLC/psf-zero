"""c41_val_families.py -- C41-VAL by family (Addendum 449): ESP ratios, how often STEINER6 was kept, the largest
gains and the slowest calls.
    cd <psf-zero repository>; python benchmarks/c41_val_families.py data/2026-10-11/c41_val"""
import json, math, sys, collections
out_dir = sys.argv[1]
rows = [json.loads(x) for x in open(out_dir + "/c41_val.jsonl")][1:]
fams = json.load(open(out_dir + "/c41_tests.json"))
by = {}
for r in rows:
    by.setdefault((r["device"], r["test"]), {})[r["arm"]] = r
L = lambda r: r["s2"] + r["s1"] + r["sm"]
esp = lambda r: 0.0 if r["failed"] else 10 ** -L(r)
def gm(p, q, xs):
    ok = [x for x in xs if esp(x[p]) > 0 and esp(x[q]) > 0]
    return (10 ** (sum(L(x[q]) - L(x[p]) for x in ok) / len(ok)) if ok else float("nan"))
xs = list(by.values())
fam = lambda x: fams[x["C41"]["test"]][0]
run = lambda s: [x for x in s if max(esp(x[k]) for k in ("QK3", "R1", "C40", "C41")) >= 0.01]
groups = collections.OrderedDict((f, [x for x in xs if fam(x) == f]) for f in
    ("qaoa_qasm", "heisenberg_qasm", "qasmbench", "feynman", "gen_export", "gen_qaoa_gates", "control"))
groups["Z8 gen_export+gen_qaoa_gates"] = groups["gen_export"] + groups["gen_qaoa_gates"]
groups["all read (non-control)"] = [x for x in xs if fam(x) != "control"]
groups["all read, without Boston"] = [x for x in xs if fam(x) != "control" and x["C41"]["device"] != "FakeBoston"]
groups["heisenberg without Boston"] = [x for x in groups["heisenberg_qasm"] if x["C41"]["device"] != "FakeBoston"]
groups["qasmbench without Boston"] = [x for x in groups["qasmbench"] if x["C41"]["device"] != "FakeBoston"]
for g, s in groups.items():
    r = run(s)
    ch = sum(1 for x in s if (x["C41"].get("steiner_stats") or {}).get("chosen"))
    better = sum(1 for x in r if L(x["R1"]) - L(x["C41"]) > math.log10(1.1))
    print(f"{g:30s} tests={len(s):3d} run={len(r):3d} C41/R1 {gm('C41','R1',r):.3f} C41/QK3 {gm('C41','QK3',r):.3f} "
          f"R1/QK3 {gm('R1','QK3',r):.3f} chosen {ch}/{len(s)} >=1.1x {better}/{len(r)}")
print("\nBiggest gains C41/R1 (runnable):")
r = sorted(run([x for x in xs if fam(x) != "control"]), key=lambda x: L(x["C41"]) - L(x["R1"]))
for x in r[:15]:
    print(f"  {x['C41']['test']:28s} {x['C41']['device'][4:]:10s} C41/R1 {10**(L(x['R1'])-L(x['C41'])):.2f} "
          f"C41/QK3 {10**(L(x['QK3'])-L(x['C41'])):.2f} ESP C41 {esp(x['C41']):.3g} R1 {esp(x['R1']):.3g} QK3 {esp(x['QK3']):.3g} "
          f"2q {x['C41']['n2']}/{x['R1']['n2']}/{x['QK3']['n2']}")
print("\nslowest C41:")
for x in sorted(xs, key=lambda x: -x["C41"]["t"])[:8]:
    print(f"  {x['C41']['test']:28s} {x['C41']['device'][4:]:10s} C41 {x['C41']['t']:.1f}s R1 {x['R1']['t']:.1f}s QK3 {x['QK3']['t']:.1f}s terms {x['C41'].get('parsed_terms')} chosen {(x['C41'].get('steiner_stats') or {}).get('chosen')}")
rd = [x for x in xs if x["C41"].get("parsed_terms")]
ts = sorted(x["C41"]["t"] / max(x["R1"]["t"], 1e-3) for x in rd)
print("time ratio median %.1f p90 %.1f" % (ts[len(ts)//2], ts[int(len(ts)*0.9)]))
