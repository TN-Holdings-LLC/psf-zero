"""cancel_verify.py -- independent re-computation of the CANCEL run (Addendum 394), written before the scored run and
before any of its output exists (2026-10-07). Reads the raw json only, imports none of the test's scripts, takes the
expected tests from BP-MOCK's and BP-MOCK2's committed output, recomputes P0 and K1-K5, and compares its verdicts with
score.md.
    python cancel_verify.py <run dir>"""
import json
import os
import re
import statistics
import sys

RUN = sys.argv[1]
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
want = set()
for p in ("data/2026-10-07/bp_mock/bp_mock.json", "data/2026-10-07/bp_mock2/bp_mock2.json"):
    want |= {x["test"] for x in json.load(open(os.path.join(REPO, p)))["rows"]}
r = json.load(open(os.path.join(RUN, "cancel.json")))
m, rows = r["meta"], r["rows"]
by = {}
for x in rows:
    by.setdefault(x["test"], {})[x["arm"]] = x
print(f"expected tests (BP-MOCK + BP-MOCK2 output) {len(want)}; in the run {len(by)}; same set {set(by) == want}")


def ok(x):
    return "error" not in x


bad = [t for t in by if (not ok(by[t]["C23"]) and ok(by[t]["C22"])) or (ok(by[t]["C23"]) and by[t]["C23"]["valid"] is not True)]
vers = all(x.get("version") == ("2026-10-07.c23" if x["arm"] == "C23" else "2026-10-07.c22") for x in rows if ok(x))
p0 = set(by) == want and len(want) == 140 and not bad and vers and not m["smoke"] and not m["dirty_tracked"]
print(f"P0 {'PASS' if p0 else 'FAIL'}: C23 failed or invalid {len(bad)}, versions ok {vers}, git_head {m['git_head']}")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def V(good, badv):
    return "REFUTED" if badv else ("CONFIRMED" if good else "AMBIGUOUS")


done = [t for t in by if all(ok(by[t][a]) for a in ("C22", "C22B", "C23"))]
tried = [t for t in done if by[t]["C23"]["cancel"]["tried"] > 0]
quiet = [t for t in done if by[t]["C23"]["cancel"]["tried"] == 0]
stable = [t for t in quiet if by[t]["C22"]["sig"] == by[t]["C22B"]["sig"]]
k1 = [t for t in stable if by[t]["C23"]["sig"] != by[t]["C22"]["sig"]]
k2 = [t for t in tried if by[t]["C23"]["q2"] > by[t]["C22"]["q2"]]
chk = [t for t in done if by[t]["C23"].get("checkable") is True and isinstance(by[t]["C23"].get("implements"), bool)]
k4 = [t for t in chk if not by[t]["C23"]["implements"]]
tr = [max(by[t]["C23"]["t"], 1e-4) / max(by[t]["C22"]["t"], 1e-4) for t in quiet]
k5 = statistics.median(tr)
mine = {"K1": V(not k1, len(k1) >= 2), "K2": V(bool(tried) and not k2, bool(k2)),
        "K4": V(len(chk) >= 5 and not k4, bool(k4)), "K5": V(k5 <= 1.15, k5 > 1.5)}
print(f"done {len(done)}; tried {len(tried)}: {[t[:40] for t in tried]}; K1 {len(k1)} of {len(stable)}; "
      f"K2 {len(k2)}; K4 {len(k4)} of {len(chk)}; K5 {k5:.3f}")
for k, v in mine.items():
    print(k, v)
sc = os.path.join(RUN, "score.md")
theirs = dict(re.findall(r"\| (K\d) \| [^|]+ \| [^|]+ \| \*\*(\w+)\*\* \|", open(sc).read())) if os.path.exists(sc) else {}
print(f"verdicts identical to score.md: {theirs == mine}")
