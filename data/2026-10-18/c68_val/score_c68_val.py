"""Scores C68-VAL against its pre-registration (C68-VAL_preregistration.md next to this file).

    python data/2026-10-18/c68_val/score_c68_val.py <rows.jsonl> [<rows.jsonl> ...]   (prints)
"""
import json
import math
import statistics
import sys

DEVICES = 16
N_TESTS, PER = 25, 11
rows = [json.loads(line) for f in sys.argv[1:] for line in open(f, encoding="utf-8")]
want = DEVICES * N_TESTS * PER
complete = len(rows) == want


def verdict(ok):
    return ("CONFIRMED" if ok else "REFUTED") if complete else "INCOMPLETE"


def where(bad):
    return sorted({(r["device"], r["test"]) for r in bad})[:6]


print(f"rows {len(rows)} of {want}")
psfz = [r for r in rows if r["psfz_ok"]]
g1 = [r for r in rows if r["psfz_ok"] != r["c68_used_psfz"] or (r["psfz_ok"] and not r.get("same_as_psfz"))]
print(f"G1 psfz's inputs -> psfz's own output: {len(psfz)} inputs psfz takes; failing {len(g1)} {where(g1)} "
      f"-> {verdict(not g1)}")
other = [r for r in rows if not r["c68_used_psfz"]]
g2 = [r for r in other if not (r["same_as_release"] if r["c68_refused"] is None else r["same_refusal"])]
print(f"G2 every other input -> the release's output or refusal, bit for bit: {len(other)} inputs "
      f"({sum(1 for r in other if r['c68_refused'])} refused by both); failing {len(g2)} {where(g2)} -> {verdict(not g2)}")
used = [r for r in rows if r["c68_used_psfz"]]
chk = [r for r in used if r["c68"]["tvd"] is not None]
g3 = [r for r in used if not r["c68"]["native"] or r["c68"]["failed"] or
      (r["c68"]["tvd"] is not None and r["c68"]["tvd"] > 1e-6)]
print(f"G3 psfz's outputs native, no failed element, exact (1e-6) where checked: {len(used)} outputs, {len(chk)} "
      f"checked, largest {max((r['c68']['tvd'] for r in chk), default=None)}; failing {len(g3)} {where(g3)} "
      f"-> {verdict(not g3)}")
both = [r for r in rows if "c68" in r and "rel" in r]
ratio = [(r["c68"]["dur_us"] / r["rel"]["dur_us"], r) for r in both]
g4 = [r for a, r in ratio if a > 1 + 1e-9] + [r for r in rows if "rel" in r and "c68" not in r]
print(f"G4 never longer than the release: {len(both)} compared (c68 compiled where the release refused: "
      f"{sum(1 for r in rows if 'c68' in r and 'rel' not in r)}); longer or refused {len(g4)}; largest "
      f"{max(a for a, _ in ratio):.4f} -> {verdict(not g4)}")
g5 = [r for r in rows if not r["no_program_same"]]
print(f"G5 without the psfz program, the release bit for bit: failing {len(g5)} {where(g5)} -> {verdict(not g5)}")
g6 = [r for r in rows if not r["deterministic"]]
print(f"G6 the same output twice: failing {len(g6)} {where(g6)} -> {verdict(not g6)}")
print()
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))  # noqa: E731
v = [a for a, r in ratio if r["c68_used_psfz"]]
print(f"Reported: psfz used on {len(used)} of {len(rows)} inputs; there, duration over the release's: geometric mean "
      f"{gm(v):.4f}, shorter {sum(1 for a in v if a < 1 - 1e-9)}, equal {sum(1 for a in v if abs(a - 1) <= 1e-9)}")
for label, f in (("CZ devices", lambda r: r["device"] in ("FakeAachen", "FakeBoston", "FakeKingston", "FakePittsburgh",
                                                          "FakeFez", "FakeMarrakesh", "FakeTorino")),
                 ("ECR devices", lambda r: r["device"] not in ("FakeAachen", "FakeBoston", "FakeKingston",
                                                               "FakePittsburgh", "FakeFez", "FakeMarrakesh",
                                                               "FakeTorino"))):
    w = [a for a, r in ratio if f(r)]
    print(f"  {label}: all compared inputs {gm(w):.4f} ({len(w)})")
print(f"Compile time, median: c68 {1000 * statistics.median(r['c68_s'] for r in rows):.2f} ms "
      f"(where psfz was used {1000 * statistics.median(r['c68_s'] for r in used):.2f} ms), release "
      f"{1000 * statistics.median(r['rel_s'] for r in rows):.2f} ms")
print("Per test: inputs psfz took / geometric mean duration over the release's")
for name in dict.fromkeys(r["test"] for r in rows):
    w = [a for a, r in ratio if r["test"] == name]
    print(f"  {name:16s} {sum(1 for r in rows if r['test'] == name and r['c68_used_psfz']):4d} / "
          f"{gm(w) if w else float('nan'):.4f}")
