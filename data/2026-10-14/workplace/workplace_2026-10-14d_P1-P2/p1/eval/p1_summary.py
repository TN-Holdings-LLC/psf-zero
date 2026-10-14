import json, sys, math
rows = [json.loads(l) for l in open(sys.argv[1])]
def gm(xs): return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
print("| device | family | cells | new <= c51 | new / c51 (gm, range) | new / c49 | new / QK3F | new / QK3F-BW | 2q new / c51 | max err (new) | rzz in range (new) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for dev in ("FakeFez", "FakeMarrakesh"):
    for fam in ("tfim", "heis", "tfim_signs"):
        rs = [r for r in rows if r["device"] == dev and r["family"] == fam]
        if not rs: continue
        def ratio(a):
            v = [r["new"]["dur_us"] / r[a]["dur_us"] for r in rs if a in r and "dur_us" in r.get(a, {})]
            return f"{gm(v):.2f} ({min(v):.2f}-{max(v):.2f})" if v else "-"
        le = sum(1 for r in rs if r["new"]["dur_us"] <= r["c51"]["dur_us"] + 1e-9)
        q2 = [r["new"]["two_q"] / r["c51"]["two_q"] for r in rs]
        errs = [r["new"]["err"] for r in rs if isinstance(r["new"].get("err"), float)]
        print(f"| {dev} | {fam} | {len(rs)} | {le}/{len(rs)} | {ratio('c51')} | {ratio('c49')} | {ratio('QK3F')} | {ratio('QK3F-BW')} | {min(q2):.2f}-{max(q2):.2f} | {max(errs):.1e} ({len(errs)} checked) | {all(r['new']['in_range'] for r in rs)} |")
print()
print("kept_default cells:", [(r["device"], r["family"], r["n"], r["steps"]) for r in rows if r["new_path"].get("kept_default")])
print("c51 in range all:", all(r["c51"]["in_range"] for r in rows), " isa new all:", all(r["new"]["isa"] for r in rows))
le_bw = [(r["device"], r["n"], r["steps"], r["new"]["dur_us"], r["QK3F-BW"]["dur_us"]) for r in rows if "QK3F-BW" in r and r["new"]["dur_us"] > r["QK3F-BW"]["dur_us"] + 1e-9]
print("new longer than QK3F-BW:", le_bw)
