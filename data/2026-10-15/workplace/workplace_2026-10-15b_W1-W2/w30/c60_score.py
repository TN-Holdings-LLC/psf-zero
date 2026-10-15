"""C60-VAL scoring (the predictions fixed in C60-VAL_preregistration.md).  python c60_score.py <run dir>"""
import collections
import json
import math
import sys

rows = [json.loads(l) for l in open(sys.argv[1] + "/c60_val_rows.jsonl")]
A = [r for r in rows if r["part"] == "A"]
B = [r for r in rows if r["part"] == "B"]
C = [r for r in rows if r["part"] == "C"]
out = []
p = out.append


def gm(v):
    v = [x for x in v if x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


bad1 = [(r["device"], r["test"], r["target"]) for r in A
        if r["C60"].get("error") or r["C60"].get("exact") is None or r["C60"]["exact"] > 1e-6
        or not r["C60"]["isa"] or r["C60"]["uses_failed"] or not r["C60"]["rzz_in_range"]]
p(f"V1: {len(A)} cells; failing {len(bad1)} {bad1[:5]}; largest difference "
  f"{max((r['C60'].get('exact') or 0) for r in A):.2e} -> {'CONFIRMED' if not bad1 and len(A) == 210 else 'REFUTED'}")
bad2 = [(r["device"], r["test"], r["target"]) for r in A if r["C60"].get("sig_repeat") != r["C60"].get("sig")]
p(f"V2: second call identical in {len(A) - len(bad2)} of {len(A)} -> {'CONFIRMED' if not bad2 else 'REFUTED'}")
rat = [(r["C60"]["dur_us"] / r["C57"]["dur_us"], r) for r in A if "dur_us" in r["C57"] and "dur_us" in r["C60"]]
nq = sum(1 for r in A if "dur_us" not in r["C57"])
g = gm([x for x, _ in rat])
worst = max(rat, key=lambda t: t[0])
over = [(r["device"], r["test"], r["target"], round(x, 4)) for x, r in rat if x > 1.05]
p(f"V3: C60 / C57 over {len(rat)} cells (C57 without output: {nq}): geometric mean {g:.4f}; largest {worst[0]:.4f} "
  f"({worst[1]['device']}, {worst[1]['test']}, {worst[1]['target']}); cells > 1.05: {len(over)} {over[:6]} -> "
  f"{'CONFIRMED' if g <= 1.0 and not over else 'REFUTED'}")
# V4 / V5
iden = [r for r in B if r["c59_sig"]]
diff = [(r["source"], r["device"], r["test"], r["target"]) for r in iden if r["c60_sig"] != r["c59_sig"]]
oos = [r for r in B if r["c59_error"] and "neither formula" not in r["c59_error"]]
oos_bad = [(r["source"], r["device"], r["test"], r["target"]) for r in oos if not r["c60_error"]]
p(f"V4: c59 builds {len(iden)} cells: c60 identical in {len(iden) - len(diff)} {diff[:5]}; c59 refuses as out of "
  f"scope in {len(oos)}: c60 refuses too in {len(oos) - len(oos_bad)} {oos_bad[:5]} -> "
  f"{'CONFIRMED' if not diff and not oos_bad else 'REFUTED'}")
new = [r for r in B if r["c59_error"] and "neither formula" in r["c59_error"]]
newb = [r for r in new if r.get("c60_new")]
bad5 = [(r["source"], r["device"], r["test"], r["target"]) for r in newb
        if (r["c60_new"]["exact"] is not None and r["c60_new"]["exact"] > 1e-6) or not r["c60_new"]["isa"]
        or r["c60_new"]["uses_failed"]]
unchk = sum(1 for r in newb if r["c60_new"]["exact"] is None)
p(f"V5: c59 'neither formula reads' in {len(new)} cells; c60 builds {len(newb)} (not checkable: {unchk}); failing "
  f"{len(bad5)} {bad5[:5]} -> {'CONFIRMED' if not bad5 else 'REFUTED'}")
p("")
p("Reported:")
rq = [(r["C60"]["dur_us"] / r["QK3"]["dur_us"], r) for r in A if "dur_us" in r["QK3"] and "dur_us" in r["C60"]]
wq = max(rq, key=lambda t: t[0])
p(f"- C60 / QK3: geometric mean {gm([x for x, _ in rq]):.4f}; largest {wq[0]:.4f} ({wq[1]['device']}, {wq[1]['test']}, "
  f"{wq[1]['target']}); cells > 1: {sum(1 for x, _ in rq if x > 1 + 1e-9)}")
per = collections.defaultdict(list)
for x, r in rat:
    per[r["test"]].append(x)
p("- C60 / C57 per test (10 cells each; geometric mean, range):")
for t, v in per.items():
    p(f"    {t:14s} {gm(v):.3f}  [{min(v):.3f}, {max(v):.3f}]")
for arm in ("C60", "C57", "QK3"):
    secs = sorted(r[arm]["secs"] for r in A if "secs" in r[arm])
    tq = [r[arm]["twoq"] for r in A if "twoq" in r[arm]]
    p(f"- {arm}: compile time median {secs[len(secs)//2]*1000:.1f} ms, max {secs[-1]*1000:.1f} ms; two-qubit gates "
      f"total {sum(tq)}")
tq_cmp = collections.Counter("fewer" if r["C60"]["twoq"] < r["C57"]["twoq"] else "equal" if r["C60"]["twoq"] ==
                             r["C57"]["twoq"] else "more" for r in A if "twoq" in r["C57"])
p(f"- two-qubit count C60 against C57: {dict(tq_cmp)}")
s59 = sorted(r["c59_secs"] for r in iden)
s60 = sorted(r["c60_secs"] for r in iden)
p(f"- part B, cells c59 builds: compile median c59 {s59[len(s59)//2]*1000:.1f} ms, c60 {s60[len(s60)//2]*1000:.1f} ms")
p("- part C (FakeFez, c60): test target: as written / reversed / random / runs reversed (us); distinct outputs")
for r in C:
    d = r["dur"]
    vals = [v for v in d.values() if v]
    sp = max(vals) / min(vals) if vals else float("nan")
    p(f"    {r['test']:16s} {r['target']} " + " / ".join(f"{d[k]:.3f}" if d[k] else "-" for k in
                                                     ("as_written", "reversed", "random", "runs_reversed"))
      + f"  spread {sp:.4f}; distinct {r['distinct_outputs']}")
print("\n".join(out))
open(sys.argv[1] + "/c60_val_score.txt", "w").write("\n".join(out) + "\n")
