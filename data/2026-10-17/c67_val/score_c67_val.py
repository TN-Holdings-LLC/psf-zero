"""Scores C67-VAL against its pre-registration (C67-VAL_preregistration.md next to this file).

    python data/2026-10-17/c67_val/score_c67_val.py <rows.jsonl> [<rows.jsonl> ...]   (writes nothing; prints)
"""
import json
import math
import statistics
import sys

DEVICES = ("FakeSherbrooke", "FakeBrisbane", "FakeKyiv", "FakeOsaka")
N_TESTS, PER = 19, 51
rows = [json.loads(line) for f in sys.argv[1:] for line in open(f, encoding="utf-8")]
want = len(DEVICES) * N_TESTS * PER


def verdict(ok):
    return "CONFIRMED" if ok else "REFUTED"


print(f"rows {len(rows)} of {want}; devices {sorted({r['device'] for r in rows})}")
built = [r for r in rows if "refused" not in r]
ref = [r for r in rows if "refused" in r]
print(f"E1 every input compiled: {len(built)} of {len(rows)}; refused {len(ref)} {[(r['device'], r['test'], r['k'], r['refused'][:60]) for r in ref[:5]]} -> {verdict(not ref and len(rows) == want)}")
bad2 = [r for r in built if not r["native"] or r["failed"]]
print(f"E2 native, no failed element: failing {len(bad2)} {[(r['device'], r['test'], r['k']) for r in bad2[:5]]} -> {verdict(not bad2 and len(rows) == want)}")
chk = [r for r in built if r["tvd"] is not None]
bad3 = [r for r in chk if r["tvd"] > 1e-6]
print(f"E3 exact within 1e-6 (up to 20 active qubits): checked {len(chk)}, not checked {len(built) - len(chk)}; largest "
      f"{max((r['tvd'] for r in chk), default=None)}; failing {len(bad3)} -> {verdict(not bad3 and len(rows) == want)}")
bad4 = [r for r in built if not r["deterministic"]]
print(f"E4 the same output twice: failing {len(bad4)} -> {verdict(not bad4 and len(rows) == want)}")
ratio = [(r["dur_us"] / r["qk3"]["dur_us"], r) for r in built if r["qk3"]["dur_us"] > 0]
bad5 = [(a, r) for a, r in ratio if a > 1.10 + 1e-12]
print(f"E5 duration at most 1.10 x Qiskit level 3's on every input: largest {max(a for a, _ in ratio):.4f}; above "
      f"{len(bad5)} {[(r['device'], r['test'], r['k'], round(a, 3)) for a, r in sorted(bad5, key=lambda x: -x[0])[:5]]} -> "
      f"{verdict(not bad5 and len(rows) == want)}")
print()
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))  # noqa: E731
rd = [a for a, _ in ratio]
r2 = [r["q2"] / r["qk3"]["q2"] for r in built if r["qk3"]["q2"] > 0]
print("Reported: duration over Qiskit level 3's -- geometric mean %.3f, median %.3f, longer than level 3 in %d of %d, "
      "largest %.3f" % (gm(rd), statistics.median(rd), sum(1 for a in rd if a > 1 + 1e-12), len(rd), max(rd)))
print("Reported: two-qubit gates over level 3's -- geometric mean %.3f, more than level 3 in %d of %d" %
      (gm(r2), sum(1 for a in r2 if a > 1 + 1e-12), len(r2)))
print("Reported: compile time (median ms) -- c67 %.1f, Qiskit level 3 %.1f" %
      (1000 * statistics.median(r["psf_s"] for r in built), 1000 * statistics.median(r["qk3_s"] for r in rows)))
print("Per test (all devices): duration over level 3, geometric mean / largest")
for name in sorted({r["test"] for r in built}):
    v = [a for a, r in ratio if r["test"] == name]
    print(f"  {name:17s} {gm(v):.3f} / {max(v):.3f}")
