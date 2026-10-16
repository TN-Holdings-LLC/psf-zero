"""C63A-VAL scoring (C63A-VAL-R_preregistration.md; part D by home's pre-registration if given).

    python c63a_score.py <part R dir: run_r with R_<test>/ segments> [<part D dir> ...]

Reads every finished segment (a sub-directory with c63_val_rows_*.jsonl; '*.killed*' directories are not read).
A-V1, A-V2 (two-qubit count equal to c62; the duration ratio is reported only), A-V3, A-S1, A-S2 on part R; with part
D directories also A-V1's identity and C63-VAL's V4 conditions there. Tests not finished are listed as not run."""
import json
import os
import statistics
import sys

ALL = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
       "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12", "chain_cz_n12", "chain_cz_n16", "ring_cz_n12",
       "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")


def load(d):
    out = []
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if ".killed" not in x]
        for f in sorted(files):
            if f.startswith("c63_val_rows_"):
                out += [json.loads(line) for line in open(os.path.join(root, f))]
    return out


R = [c for c in load(sys.argv[1]) if c["part"] == "R"]
D = [c for d in sys.argv[2:] for c in load(d) if c["part"] == "D"]
out = []
p = out.append
done = sorted({c["test"] for c in R}, key=ALL.index)
p(f"part R: {len(R)} cells finished ({len(done)} of 19 tests); not run: {[t for t in ALL if t not in done]}")


def verdict(ok):
    return "CONFIRMED (on the cells run)" if ok else "REFUTED"


def bad_exact(r):
    return r["exact"] is not None and r["exact"] > 1e-6


nat = [c for c in R + D if c["kind"] == "native"]
v1 = [(c["part"], c.get("device"), c["test"], c["target"]) for c in nat if not all(r["same"] for r in c["rows"])]
p(f"A-V1: native outputs {sum(len(c['rows']) for c in nat)}; cells with a difference {v1} -> {verdict(not v1)}")
F = [c for c in R if c["kind"] != "native" and c["target"] == "F"]
v2, n2, u2, w2 = [], 0, 0, 0.0
for c in F:
    for r in c["rows"]:
        n2 += 1
        u2 += r["exact"] is None
        w2 = max(w2, r["exact"] or 0.0)
        if bad_exact(r) or not r["isa"] or r["failed"] or not r["rzz_ok"] or r["twoq"] != r["twoq62"]:
            v2.append((c["test"], r["kind"]))
p(f"A-V2: F outputs {n2} (exactness not checked: {u2}; largest difference {w2:.1e}); failing {len(v2)} {v2[:10]} -> "
  f"{verdict(not v2)}")
T = [c for c in R if c["kind"] != "native" and c["target"] == "T"]
v3, n3, u3, w3 = [], 0, 0, 0.0
for c in T:
    for r in c["rows"]:
        n3 += 1
        u3 += r["exact"] is None
        w3 = max(w3, r["exact"] or 0.0)
        if bad_exact(r) or not r["isa"] or r["failed"]:
            v3.append((c["test"], r["kind"]))
p(f"A-V3: T outputs {n3} (exactness not checked: {u3}; largest difference {w3:.1e}); failing {len(v3)} {v3[:10]} -> "
  f"{verdict(not v3)}")
if D:
    v4 = []
    for c in D:
        if c["kind"] == "native":
            continue
        for r in c["rows"]:
            if bad_exact(r) or not r["isa"] or r["failed"] or not r["rzz_ok"] or r["twoq"] != r["twoq62"]:
                v4.append((c["device"], c["test"], c["target"]))
    p(f"A-V4 (part D, chain/graph): {sum(len(c['rows']) for c in D if c['kind'] != 'native')} outputs; failing "
      f"{len(v4)} {v4[:10]} -> {verdict(not v4)}")


def limit(c):
    return 5.0 if c["test"] == "tfim_n40_r2" else 2.0


for sid, key in (("A-S1", "bind_median_ms"), ("A-S2", "bind_tight_median_ms")):
    miss = [(c["test"], c["target"], round(c[key], 3)) for c in R if c[key] is None or c[key] > limit(c)]
    p(f"{sid} ({key}): {len(R) - len(miss)} of {len(R)} cells within the limit; misses {miss} -> {verdict(not miss)}")
p("")
p("Reported (part R): per cell bind interleaved / back to back (ms); shape changes on same-sign / free draws; F and T "
  "cells: duration vs c62 (median ratio, largest, share above c62), 2q equal share")
for c in sorted(R, key=lambda c: (ALL.index(c["test"]), c["target"])):
    rows = c["rows"]
    same = sum(r["rebuilt"] for r in rows if r["kind"] == "same")
    free = sum(r["rebuilt"] for r in rows if r["kind"] == "free")
    line = (f"  {c['test']:16s} {c['target']} {c['kind']:6s} bind {c['bind_median_ms']:6.3f} / "
            f"{c['bind_tight_median_ms']:6.3f}  shape changes same {same:4d} free {free:3d}")
    if c["kind"] != "native":
        rat = [r["dur"] / r["dur62"] for r in rows]
        line += (f"  dur/c62 median {statistics.median(rat):.4f} max {max(rat):.4f} above "
                 f"{sum(x > 1 + 1e-9 for x in rat)}/{len(rat)}  2q= {sum(r['twoq'] == r['twoq62'] for r in rows)}/"
                 f"{len(rows)}")
    p(line)
print("\n".join(out))
open(os.path.join(sys.argv[1], "c63a_score.txt"), "w").write("\n".join(out) + "\n")
