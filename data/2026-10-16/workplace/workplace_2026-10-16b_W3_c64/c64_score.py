"""C64 scoring of D1 and D5 (C64-PREREG.md).   python c64_score.py <run_d1 dir>  (every finished segment; unfinished
segments moved aside as *.killed* are not read)"""
import json
import os
import statistics
import sys

d = sys.argv[1]
segs = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)) and ".killed" not in x)
rows = [json.loads(line) for x in segs for line in open(os.path.join(d, x, "c64_check_rows.jsonl"))]
X = [r for r in rows if r["part"] == "X"]
C = [r for r in rows if r["part"] in ("R", "D")]
out = []
p = out.append
xbad = [(r["kind"], r.get("test") or (r["device"], r["target"]), k) for r in X for k, v in r["res"].items()
        if not v["eq"]]
n = sum(r["inputs"] for r in C)
bad = [(r["part"], r["device"], r["test"], r["target"], r["unequal"]) for r in C if r["unequal"]]
p(f"D1: cells {len(C)} (part R {sum(r['part'] == 'R' for r in C)}, part D {sum(r['part'] == 'D' for r in C)}); inputs {n}; "
  f"cells with an unequal output {bad}; refusal checks {2 * len(X)} (no Target {sum(r['kind'] == 'no_target' for r in X)} "
  f"tests, ccx {sum(r['kind'] == 'ccx' for r in X)} device-targets), unequal {xbad} -> "
  f"{'CONFIRMED' if not bad and not xbad and n == 49590 and len(C) == 190 else 'REFUTED'}")
p(f"   value_sig (repr-exact) also equal in every input: {all(r['sig_unequal'] == 0 for r in C)} "
  f"(inputs with a difference: {sum(r['sig_unequal'] for r in C)})")
rat = [r["t_c64"] / r["t_c63a"] for r in C]
med = statistics.median(rat)
p(f"D5: {len(rat)} cells; median of c64 / c63a {med:.4f} (largest {max(rat):.4f}, smallest {min(rat):.4f}) -> "
  f"{'CONFIRMED' if med <= 1.00 else 'REFUTED'}")
print("\n".join(out))
open(os.path.join(d, "c64_score.txt"), "w").write("\n".join(out) + "\n")
