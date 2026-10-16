"""C63A-VAL home part: A-V1 (part D) and A-V4, by c63_score.py's V1 and V4 rules on the part D rows; reports the F
cells' duration over c62's.   python score_d.py <D dir>"""
import json
import os
import statistics
import sys

d = sys.argv[1]
D = []
for f in sorted(os.listdir(d)):
    if f.startswith("c63_val_rows_"):
        D += [json.loads(line) for line in open(os.path.join(d, f))]


def bad_exact(r):
    return r["exact"] is not None and r["exact"] > 1e-6


v1 = [(c["device"], c["test"], c["target"]) for c in D if c["kind"] == "native" and not all(r["same"] for r in c["rows"])]
n1 = sum(len(c["rows"]) for c in D if c["kind"] == "native")
print(f"A-V1 (part D): native outputs {n1}; cells with a difference {v1} -> {'CONFIRMED' if not v1 else 'REFUTED'}")
v4, n4, u4, worst = [], 0, 0, 0.0
for c in D:
    for r in c["rows"]:
        n4 += 1
        if c["kind"] == "native":
            if not r["same"]:
                v4.append((c["device"], c["test"], c["target"], "not identical"))
            continue
        u4 += r["exact"] is None
        worst = max(worst, r["exact"] or 0.0)
        if bad_exact(r) or not r["isa"] or r["failed"] or not r["rzz_ok"]:
            v4.append((c["device"], c["test"], c["target"]))
print(f"A-V4: part D outputs {n4} (chain/graph exactness not checked: {u4}; largest difference {worst:.1e}); failing "
      f"{len(v4)} {v4[:10]} -> {'CONFIRMED' if not v4 else 'REFUTED'}")
print("Reported: F cells (chain/graph), duration over c62's (median, max), two-qubit count equal, shape changes")
for c in D:
    if c["kind"] != "native" and c["target"] == "F":
        rat = [r["dur"] / r["dur62"] for r in c["rows"]]
        print(f"  {c['device']:18s} {c['test']:16s} dur/c62 {statistics.median(rat):.4f} max {max(rat):.4f}  "
              f"2q= {sum(r['twoq'] == r['twoq62'] for r in c['rows'])}/{len(c['rows'])}  "
              f"shape changes {c['rebuilt'] + c['switched']}")
