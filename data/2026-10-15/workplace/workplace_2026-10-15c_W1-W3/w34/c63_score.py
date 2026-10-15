"""C63-VAL scoring (C63-VAL_preregistration.md).  python c63_score.py <run dir with R/ and D/>"""
import json
import os
import statistics
import sys

base = sys.argv[1]


def load(part):
    d = os.path.join(base, part)
    out = []
    for f in sorted(os.listdir(d)):
        if f.startswith("c63_val_rows_"):
            out += [json.loads(line) for line in open(os.path.join(d, f))]
    return out


R, D = load("R"), load("D")
out = []
p = out.append


def bad_exact(r):
    return r["exact"] is not None and r["exact"] > 1e-6


def verdict(ok):
    return "CONFIRMED" if ok else "REFUTED"


# V1
v1 = [(c["part"], c["device"], c["test"], c["target"], sum(not r["same"] for r in c["rows"]))
      for c in R + D if c["kind"] == "native" and not all(r["same"] for r in c["rows"])]
n1 = sum(len(c["rows"]) for c in R + D if c["kind"] == "native")
p(f"V1: native outputs {n1} (parts R and D); cells with a difference {v1} -> {verdict(not v1)}")
# V2
F = [c for c in R if c["kind"] != "native" and c["target"] == "F"]
v2, n2, u2, worst2 = [], 0, 0, 0.0
for c in F:
    for r in c["rows"]:
        n2 += 1
        u2 += r["exact"] is None
        worst2 = max(worst2, r["exact"] or 0.0)
        if bad_exact(r) or not r["isa"] or r["failed"] or not r["rzz_ok"] or abs(r["dur"] - r["dur62"]) > 1e-9 \
                or r["twoq"] != r["twoq62"]:
            v2.append((c["test"], r["kind"]))
p(f"V2: F outputs {n2} (exactness not checked: {u2}; largest difference {worst2:.1e}); failing {len(v2)} "
  f"{v2[:10]} -> {verdict(not v2)}")
# V3
T = [c for c in R if c["kind"] != "native" and c["target"] == "T"]
v3, n3, u3, worst3 = [], 0, 0, 0.0
for c in T:
    for r in c["rows"]:
        n3 += 1
        u3 += r["exact"] is None
        worst3 = max(worst3, r["exact"] or 0.0)
        if bad_exact(r) or not r["isa"] or r["failed"]:
            v3.append((c["test"], r["kind"]))
p(f"V3: T outputs {n3} (exactness not checked: {u3}; largest difference {worst3:.1e}); failing {len(v3)} "
  f"{v3[:10]} -> {verdict(not v3)}")
# V4
v4, n4, u4, worst4 = [], 0, 0, 0.0
for c in D:
    for r in c["rows"]:
        n4 += 1
        if c["kind"] == "native":
            if not r["same"]:
                v4.append((c["device"], c["test"], c["target"], "not identical"))
            continue
        u4 += r["exact"] is None
        worst4 = max(worst4, r["exact"] or 0.0)
        if bad_exact(r) or not r["isa"] or r["failed"] or not r["rzz_ok"]:
            v4.append((c["device"], c["test"], c["target"]))
p(f"V4: part D outputs {n4} (chain/graph exactness not checked: {u4}; largest difference {worst4:.1e}); failing "
  f"{len(v4)} {v4[:10]} -> {verdict(not v4)}")


def limit(c):
    return 5.0 if c["test"] == "tfim_n40_r2" else 2.0


for sid, key in (("S1", "bind_median_ms"), ("S2", "bind_tight_median_ms")):
    miss = [(c["test"], c["target"], round(c[key], 3)) for c in R if c[key] is None or c[key] > limit(c)]
    p(f"{sid} ({key}): {len(R) - len(miss)} of {len(R)} cells within the limit; misses {miss} -> "
      f"{verdict(not miss)}")
p("")
p("Reported, per cell (part R, FakeFez): bind interleaved / back to back, values_of, own assign_parameters, Qiskit's "
  "alternative assign (ms); rebuilt + switched; build / c62 compile (ms); T cells: duration vs c62 (median ratio, "
  "largest, share <= c62), 2q equal share; non-native: outputs identical to c62")
for c in R:
    rows = c["rows"]
    line = (f"  {c['test']:16s} {c['target']} {c['kind']:6s} bind {c['bind_median_ms']:6.3f} / "
            f"{c['bind_tight_median_ms']:6.3f}  values_of {c['values_of_median_ms']:6.3f}  assign "
            f"{c['own_assign_median_ms']:6.3f}  qiskit "
            f"{'  -   ' if c['qiskit_assign_median_ms'] is None else format(c['qiskit_assign_median_ms'], '6.3f')}  "
            f"shape changes {c['rebuilt'] + c['switched']:4d}  build {c['build_s'] * 1000:6.1f} c62 "
            f"{c['c62_s'] * 1000:6.1f}")
    if c["kind"] != "native":
        rat = [r["dur"] / r["dur62"] for r in rows]
        line += (f"  dur/c62 {statistics.median(rat):.4f} max {max(rat):.4f} <= {sum(x <= 1 + 1e-9 for x in rat)}/"
                 f"{len(rat)} 2q= {sum(r['twoq'] == r['twoq62'] for r in rows)}/{len(rows)} identical "
                 f"{sum(r['same'] for r in rows)}")
    p(line)
print("\n".join(out))
open(os.path.join(base, "c63_val_score.txt"), "w").write("\n".join(out) + "\n")
