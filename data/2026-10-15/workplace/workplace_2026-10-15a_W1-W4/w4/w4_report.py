"""W4 / item 97: summarise w4_orient.py's rows.  python w4_report.py <run dir>"""
import collections
import glob
import json
import sys

rows = [json.loads(l) for f in sorted(glob.glob(sys.argv[1] + "/w4_rows_*.jsonl")) for l in open(f)]
oos = [r for r in rows if not r["in_scope"]]
R = {(r["device"], r["tk"], r["test"], r["arm"], r["orient"]): r for r in rows if r["in_scope"]}
cells = sorted({k[:3] for k in R})
print(f"in-scope cells {len(cells)}; out-of-scope (not measured) {len(oos)}: "
      f"{sorted({(r['test'], r['tk']) for r in oos})}")
bad = [k for k, r in R.items() if r["uses_failed"] or not r["isa"] or (r["exact"] is not None and r["exact"] > 1e-6)]
print("outputs failing a check:", bad)
print("largest exactness diff:", max((r["exact"] or 0) for r in R.values()))
spread_before, spread_after, longer, shorter, same_sig, n_sig = [], [], [], [], 0, 0
per_test = collections.defaultdict(list)
for c in cells:
    d = [R.get(c + ("C57", o)) for o in ("as_written", "reversed", "random")]
    rule = R.get(c + ("C57+97", "reversed"))
    if None in d or rule is None:
        continue
    du = [x["dur_us"] for x in d]
    sp = max(du) / min(du)
    spread_before.append(sp)
    per_test[(c[2], c[1])].append(sp)
    n_sig += 1
    same_sig += rule["sig"] == d[0]["sig"]
    for o, x in zip(("as_written", "reversed", "random"), d):
        if rule["dur_us"] > x["dur_us"] * (1 + 1e-9):
            longer.append((c, o, rule["dur_us"] / x["dur_us"]))
        elif rule["dur_us"] < x["dur_us"] * (1 - 1e-9):
            shorter.append((c, o, rule["dur_us"] / x["dur_us"]))
    ff = [R.get(c + ("C57+97", o)) for o in ("as_written", "reversed", "random")]
    if c[0] == "FakeFez" and None not in ff:
        spread_after.append(len({x["sig"] for x in ff}) == 1)
k = len(spread_before)
print(f"cells with all four outputs: {k}")
print(f"C57 spread over the three orientations: >1 in {sum(s > 1 + 1e-9 for s in spread_before)} cells; "
      f"largest {max(spread_before):.4f}")
print(f"C57+97 on 'reversed' identical to C57 on 'as written' (the canonical form): {same_sig} of {n_sig}")
print(f"FakeFez: C57+97 gives one circuit for all three orientations in {sum(spread_after)} of {len(spread_after)} cells")
print(f"(cell, orientation) where the rule is longer: {len(longer)} of {3 * k}; shorter: {len(shorter)}")
for c, o, r in sorted(longer, key=lambda x: -x[2]):
    print(f"  longer {c} vs {o}: x{r:.4f}")
for c, o, r in sorted(shorter, key=lambda x: x[2]):
    print(f"  shorter {c} vs {o}: x{r:.4f}")
print("per test (T/F): cells with spread > 1, largest spread")
for (t, tk), v in sorted(per_test.items()):
    print(f"  {t:16s} {tk}: {sum(s > 1 + 1e-9 for s in v)} of {len(v)}; largest {max(v):.4f}")
secs = collections.defaultdict(list)
for r in R.values():
    secs[r["arm"]].append(r["secs"])
print("compile time, mean s:", {a: round(sum(v) / len(v), 2) for a, v in secs.items()})
