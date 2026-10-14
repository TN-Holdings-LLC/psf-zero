import json, glob, math, os, sys
d = sys.argv[1]
R = {}
for f in glob.glob(os.path.join(d, "p2_*.jsonl")):
    for l in open(f):
        r = json.loads(l); R[(r["arm"], r["target"], r["device"], r["n"], r["L"])] = r
arms = [("a15", "T"), ("a16", "T"), ("QK3", "T"), ("a15", "F"), ("a16", "F"), ("QK3F", "F")]
def gm(v): return math.exp(sum(map(math.log, v)) / len(v))
print("| device, n | L | " + " | ".join(f"{a} ({t})" for a, t in arms) + " |")
print("|---|---|" + "---|" * len(arms))
for dev in ("FakeFez", "FakeTorino"):
    for n in (4, 6):
        for L in (1, 2, 4, 8, 12, 16):
            cells = []
            for a, t in arms:
                r = R.get((a, t, dev, n, L))
                cells.append(f"{r['dur_us']:.2f} ({r['two_q']})" if r else "-")
            print(f"| {dev}, {n} | {L} | " + " | ".join(cells) + " |")
print()
for t in ("T", "F"):
    a15 = "a15"; q = "QK3" if t == "T" else "QK3F"
    keys = [(dev, n, L) for dev in ("FakeFez", "FakeTorino") for n in (4, 6) for L in range(1, 17)]
    have = [k for k in keys if ("a16", t) + k in R and (a15, t) + k in R]
    le = sum(1 for k in have if R[("a16", t) + k]["dur_us"] <= R[(a15, t) + k]["dur_us"] + 1e-9)
    lt = sum(1 for k in have if R[("a16", t) + k]["dur_us"] < R[(a15, t) + k]["dur_us"] - 1e-9)
    rat = [R[("a16", t) + k]["dur_us"] / R[(a15, t) + k]["dur_us"] for k in have]
    print(f"target {t}: a16 <= a15 in {le}/{len(have)} (shorter in {lt}); a16/a15 gm {gm(rat):.3f}, min {min(rat):.2f}, max {max(rat):.2f}")
    hq = [k for k in have if (q, t) + k in R]
    if hq:
        rq = [R[("a16", t) + k]["dur_us"] / R[(q, t) + k]["dur_us"] for k in hq]
        print(f"   a16/{q} gm {gm(rq):.3f} ({min(rq):.2f}-{max(rq):.2f}), a16 <= {q} in {sum(1 for x in rq if x <= 1 + 1e-9)}/{len(hq)}")
    ch = {}
    for k in have:
        c = R[("a16", t) + k]["chosen"]; ch[c] = ch.get(c, 0) + 1
    print("   a16 chosen:", ch, " a15 chosen search:", sum(1 for k in have if R[(a15, t) + k]["chosen"] == "search"))
    for a in ("a16", a15):
        print(f"   {a}: max err {max(R[(a, t) + k]['err'] for k in have):.1e}, rzz in range in "
              f"{sum(1 for k in have if R[(a, t) + k]['in_range'])}/{len(have)}")
    longer = [(k, R[("a16", t) + k]["dur_us"], R[(a15, t) + k]["dur_us"]) for k in have if R[("a16", t) + k]["dur_us"] > R[(a15, t) + k]["dur_us"] + 1e-9]
    print("   a16 longer than a15:", longer)
if any(k[0] == "a16" and k[1] == "F" for k in R):
    keys = [(dev, n, L) for dev in ("FakeFez", "FakeTorino") for n in (4, 6) for L in range(1, 17)]
    rr = [R[("a16", "F") + k]["dur_us"] / R[("a16", "T") + k]["dur_us"] for k in keys]
    print(f"a16 F/T (rx one pulse): gm {gm(rr):.3f} ({min(rr):.2f}-{max(rr):.2f})")
