"""C62-VAL scoring (C62-VAL_preregistration.md).  python c62_score.py <run dir>"""
import json, math, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1] + "/c62_val_rows.jsonl")]
ISING = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "tfim_n40_r2", "hhI_n16", "hhI_n20", "ring12_tfim")
HEIS = ("heis_n12_r2", "hhtree_heis_n16")
out = []; p = out.append
def gm(v): return math.exp(sum(math.log(x) for x in v) / len(v))
built = [r for r in rows if not r["C62"]["error"]]
b1 = [(r["device"], r["test"], r["target"]) for r in built if (r["C62"]["exact"] is not None and r["C62"]["exact"] > 1e-6)
      or not r["C62"]["isa"] or not r["C62"]["rzz_in_range"] or r["C62"]["uses_failed"]]
unchk = sum(1 for r in built if r["C62"]["exact"] is None)
p(f"P1: {len(built)} outputs (not checkable, > 20 active qubits: {unchk}); failing {b1} -> {'CONFIRMED' if not b1 else 'REFUTED'}")
I = [r for r in rows if r["test"] in ISING and r["target"] == "T"]
ri = [r["C62"]["dur_us"] / r["C60"]["dur_us"] for r in I]
over = [(r["device"], r["test"], round(x, 4)) for r, x in zip(I, ri) if x > 1 + 1e-9]
p(f"P2: Ising T cells {len(I)}; geometric mean {gm(ri):.4f}; range [{min(ri):.3f}, {max(ri):.3f}]; cells > c60: {over} -> "
  f"{'CONFIRMED' if gm(ri) <= 0.85 and not over else 'REFUTED'}")
H = [r for r in rows if r["test"] in HEIS and r["target"] == "T"]
rh = [r["C62"]["dur_us"] / r["C60"]["dur_us"] for r in H]
overh = [(r["device"], r["test"], round(x, 4)) for r, x in zip(H, rh) if x > 1.01 + 1e-12]
p(f"P3: Heisenberg T cells {len(H)}; ratios {[round(x, 4) for x in rh]}; over 1.01: {overh} -> {'CONFIRMED' if not overh else 'REFUTED'}")
same_need = [r for r in rows if (r["target"] == "F" or (r["test"] not in ISING + HEIS)) and not (r["C62"]["error"] and r["C60"]["error"])]
b4 = [(r["device"], r["test"], r["target"]) for r in same_need if r["C62"].get("sig") != r["C60"].get("sig")]
ref = [(r["device"], r["test"], r["target"]) for r in rows if bool(r["C62"]["error"]) != bool(r["C60"]["error"])]
p(f"P4: cells required identical {len(same_need)}: different {b4}; refusal mismatches {ref} -> {'CONFIRMED' if not b4 and not ref else 'REFUTED'}")
b5 = [(r["device"], r["test"], r["target"]) for r in built if r["C62"].get("sig_repeat") != r["C62"]["sig"]]
p(f"P5: second call identical in {len(built) - len(b5)} of {len(built)} -> {'CONFIRMED' if not b5 else 'REFUTED'}")
p("")
p("Reported: per device, Ising T geometric mean; two-qubit counts; compile times")
for d in sorted({r["device"] for r in I}):
    v = [x for r, x in zip(I, ri) if r["device"] == d]
    p(f"  {d:20s} Ising T geomean {gm(v):.4f}  range [{min(v):.3f}, {max(v):.3f}]")
per = collections.defaultdict(list)
for r, x in zip(I, ri): per[r["test"]].append(x)
for t, v in per.items(): p(f"  {t:16s} {gm(v):.3f} [{min(v):.3f}, {max(v):.3f}]")
q = collections.Counter("fewer" if r["C62"]["twoq"] < r["C60"]["twoq"] else "equal" if r["C62"]["twoq"] == r["C60"]["twoq"] else "more" for r in I + H)
p(f"  two-qubit count (Ising + Heisenberg T): {dict(q)}")
for arm in ("C62", "C60"):
    s = sorted(r[arm]["secs"] for r in I + H)
    p(f"  {arm} compile on the CZ-path cells: median {s[len(s)//2]*1000:.1f} ms, max {s[-1]*1000:.1f} ms")
print("\n".join(out)); open(sys.argv[1] + "/c62_val_score.txt", "w").write("\n".join(out) + "\n")
