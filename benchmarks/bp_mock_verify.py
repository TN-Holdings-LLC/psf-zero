"""bp_mock_verify.py -- independent re-computation of the BP-MOCK run (Addendum 388), written before the scored run and
before any of its output exists (2026-10-07). Reads the raw json only and does not import bp_mock.py: it re-draws the
sample from the Benchpress clone by the rule of Addendum 388, recomputes P0 and M1-M5, and compares its verdicts with
score.md.
    python bp_mock_verify.py <run dir> <benchpress clone>"""
import hashlib
import json
import math
import os
import re
import sys

RUN, BP = sys.argv[1], sys.argv[2]
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REF = json.load(open(os.path.join(REPO, "data", "2026-10-06", "bp_probe", "published_ref.json")))
TOPOS = ("all-to-all", "square", "heavy-hex", "linear")

# -- the sample, re-drawn
root = os.path.join(BP, "benchpress")
probed = set()
for d in ("run1_small", "run2_large", "run3_c18"):
    p = os.path.join(REPO, "data", "2026-10-06", "bp_probe", d, "bp_probe.json")
    if os.path.exists(p):
        probed |= {r["test"] for r in json.load(open(p))}


def first(ids, k):
    ids = [i for i in ids if i in REF and i not in probed]
    return sorted(ids, key=lambda i: hashlib.sha256(("BP-MOCK|" + i).encode()).hexdigest())[:k]


want = []
for size, k in (("small", 5), ("medium", 4), ("large", 4)):
    names = sorted({f.split(".")[0] for r, _, fs in os.walk(os.path.join(root, "qasm", "qasmbench-" + size))
                    for f in fs if f.endswith(".qasm") and "transpiled" not in f})
    for t in TOPOS:
        want += first([f"test_QASMBench_{size}[{n}-{t}]" for n in names], k)
hams = json.load(open(os.path.join(root, "hamiltonian", "hamlib", "100_representative.json")))
for t in TOPOS:
    want += first([f"test_hamiltonians[ham_{h['ham_instance'][1:-1]}-{t}]" for h in hams], 5)
want += first([f"test_hamlib_hamiltonians_transpile[ham_{h['ham_instance'][1:-1]}]" for h in hams], 8)
want += first([f"test_feynman_transpile[{f}]" for f in os.listdir(os.path.join(root, "qasm", "feynman"))
               if f.endswith(".qasm")], 6)
want += first([k for k in REF if "[" not in k], 6)

r = json.load(open(os.path.join(RUN, "bp_mock.json")))
m, rows = r["meta"], r["rows"]
by = {}
for x in rows:
    by.setdefault(x["test"], {})[x["arm"]] = x
print(f"probed ids found {len(probed)}; sample re-drawn {len(want)}; tests in the run {len(by)}; "
      f"same set {set(want) == set(by)}")


def ok(x):
    return "error" not in x


torino = [t for t in by if "RELR" in by[t]]
bad = [t for t in by if (not ok(by[t]["C22"]) and ok(by[t]["REL"])) or (ok(by[t]["C22"]) and by[t]["C22"]["valid"] is not True)]
bad_r = [t for t in torino if (not ok(by[t]["C22R"]) and ok(by[t]["RELR"])) or
         (ok(by[t]["C22R"]) and by[t]["C22R"]["valid"] is not True)]
versions_ok = all(x.get("version") == ("2026-10-06.4" if x["arm"].startswith("REL") else "2026-10-07.c22")
                  for x in rows if x["arm"] != "QK" and ok(x))
p0 = set(want) == set(by) and len(want) == 92 and not bad and not bad_r and versions_ok and not m["smoke"] and \
    not m["dirty_tracked"] and m["benchpress"] == "b695f30"
print(f"P0 {'PASS' if p0 else 'FAIL'}: C22 failed or invalid {len(bad)}, C22R failed or invalid {len(bad_r)}, "
      f"versions ok {versions_ok}, git_head {m['git_head']}, dirty {bool(m['dirty_tracked'])}")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def g(xs):
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def q(t, a, b, key="q2"):
    return (by[t][a][key] + 1) / (by[t][b][key] + 1)


done = [t for t in by if all(ok(by[t][a]) for a in ("QK", "REL", "C22"))]
flat = [t for t in done if by[t]["C22"]["wide"] == 0]
wides = [t for t in done if by[t]["C22"]["wide"] > 0]
diff = sum(1 for t in flat if by[t]["C22"]["sig"] != by[t]["REL"]["sig"])
gw, gq = g([q(t, "C22", "REL") for t in wides]), g([q(t, "C22", "QK") for t in done])
rr = [t for t in torino if ok(by[t]["C22R"]) and ok(by[t]["RELR"])]
gr = g([q(t, "C22R", "RELR") for t in rr])
eq = [t for t in done if isinstance(by[t]["C22"].get("equivalent"), bool)]
ne = sum(1 for t in eq if not by[t]["C22"]["equivalent"])


def V(good, badv):
    return "REFUTED" if badv else ("CONFIRMED" if good else "AMBIGUOUS")


mine = {"M1": V(diff == 0, diff > 0), "M2": V(gw <= 0.85, gw > 1.00), "M3": V(gq <= 1.15, gq > 1.30),
        "M4": V(gr <= 1.00, gr > 1.05), "M5": V(bool(eq) and ne == 0, ne > 0)}
print(f"done {len(done)} (flat {len(flat)}, wide {len(wides)}); M1 differ {diff}; M2 {gw:.3f}; M3 {gq:.3f}; "
      f"M4 {gr:.3f} on {len(rr)}; M5 {ne} of {len(eq)} not equivalent")
for k, v in mine.items():
    print(k, v)
sc = os.path.join(RUN, "score.md")
theirs = dict(re.findall(r"\| (M\d) \| [^|]+ \| [^|]+ \| \*\*(\w+)\*\* \|", open(sc).read())) if os.path.exists(sc) else {}
print(f"verdicts identical to score.md: {theirs == mine}")
