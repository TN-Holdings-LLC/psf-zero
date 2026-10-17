"""Scores PSFZ-VAL against its pre-registration (PSFZ-VAL_preregistration.md next to this file).

    python data/2026-10-17/psfz_val/score_psfz_val.py <rows.jsonl> [<rows.jsonl> ...]   (prints)
"""
import json
import math
import statistics
import sys

UNSEEN = ("FakeAachen", "FakeBoston", "FakeKingston", "FakePittsburgh", "FakeKyiv", "FakeOsaka", "FakeKawasaki",
          "FakeQuebec")
SEEN = ("FakeFez", "FakeMarrakesh", "FakeTorino", "FakeSherbrooke", "FakeBrisbane")
N_TESTS, PER = 20, 11
rows = [json.loads(line) for f in sys.argv[1:] for line in open(f, encoding="utf-8")]
want = (len(UNSEEN) + len(SEEN)) * N_TESTS * PER


def verdict(ok, complete=True):
    return ("CONFIRMED" if ok else "REFUTED") if complete else "INCOMPLETE"


complete = len(rows) == want
print(f"rows {len(rows)} of {want}")
ok = [r for r in rows if "refused" not in r["psfz"]]
ref = [r for r in rows if "refused" in r["psfz"]]
print(f"V1 every input compiled: {len(ok)} of {len(rows)}; refused {[(r['device'], r['test'], r['k']) for r in ref[:5]]} "
      f"-> {verdict(not ref, complete)}")
bad2 = [r for r in ok if not r["psfz"]["native"] or r["psfz"]["failed"]]
print(f"V2 native, no failed element: failing {len(bad2)} -> {verdict(not bad2, complete)}")
chk = [r for r in ok if r["psfz"]["tvd"] is not None]
bad3 = [r for r in chk if r["psfz"]["tvd"] > 1e-6]
print(f"V3 exact (1e-6) where checked: {len(chk)} checked, largest {max((r['psfz']['tvd'] for r in chk), default=None)}, "
      f"failing {len(bad3)} -> {verdict(not bad3, complete)}")
bad4 = [r for r in ok if not r["psfz"]["deterministic"]]
print(f"V4 the same output twice: failing {len(bad4)} -> {verdict(not bad4, complete)}")
both = [r for r in ok if "refused" not in r["rel"]]
relref = [r for r in ok if "refused" in r["rel"]]
ratio = [(r["psfz"]["dur_us"] / r["rel"]["dur_us"], r) for r in both]
longer = [(a, r) for a, r in ratio if a > 1 + 1e-9]
print(f"V5 never longer than release 2026-10-17.1: longer in {len(longer)} of {len(ratio)} (the release refused "
      f"{len(relref)}); largest {max(a for a, _ in ratio):.4f}; "
      f"{sorted([(r['device'], r['test'], r['k'], round(a, 4)) for a, r in longer], key=lambda x: -x[3])[:8]} "
      f"-> {verdict(not longer, complete)}")
mp, mr = statistics.median(r["psfz_s"] for r in ok), statistics.median(r["rel_s"] for r in both)
print(f"V6 compile time, median: psfz {1000 * mp:.3f} ms, release {1000 * mr:.2f} ms -> {verdict(mp < mr, complete)}")
print()
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))  # noqa: E731
for label, devs in (("unseen devices", UNSEEN), ("seen devices, new draws", SEEN)):
    v = [a for a, r in ratio if r["device"] in devs]
    q = [r["psfz"]["q2"] / r["rel"]["q2"] for _, r in ratio if r["device"] in devs and r["rel"]["q2"]]
    if v:
        print(f"Reported ({label}): duration over the release's -- geometric mean {gm(v):.4f}, longer in "
              f"{sum(1 for a in v if a > 1 + 1e-9)} of {len(v)}, largest {max(v):.4f}; two-qubit gates {gm(q):.4f}")
print("Per test (all devices): duration over the release's, geometric mean / largest / inputs longer")
for name in dict.fromkeys(r["test"] for _, r in ratio):
    v = [a for a, r in ratio if r["test"] == name]
    print(f"  {name:15s} {gm(v):.4f} / {max(v):.4f} / {sum(1 for a in v if a > 1 + 1e-9)}")
