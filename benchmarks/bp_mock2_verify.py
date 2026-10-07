"""bp_mock2_verify.py -- independent re-computation of the BP-MOCK2 run (Addendum 391), written before the scored run
and before any of its output exists (2026-10-07). Reads the raw json only and imports neither bp_mock.py nor
bp_mock2.py: it re-draws BP-MOCK's and BP-MOCK2's samples from the Benchpress clone by the rules of Addenda 389 and
391, recomputes P0, M1-M3 and M5, and compares its verdicts with score.md.
    python bp_mock2_verify.py <run dir> <benchpress clone>"""
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
root = os.path.join(BP, "benchpress")
probed = set()
for d in ("run1_small", "run2_large", "run3_c18"):
    p = os.path.join(REPO, "data", "2026-10-06", "bp_probe", d, "bp_probe.json")
    if os.path.exists(p):
        probed |= {r["test"] for r in json.load(open(p))}
hams = json.load(open(os.path.join(root, "hamiltonian", "hamlib", "100_representative.json")))
qnames = {size: sorted({f.split(".")[0] for r, _, fs in os.walk(os.path.join(root, "qasm", "qasmbench-" + size))
                        for f in fs if f.endswith(".qasm") and "transpiled" not in f})
          for size in ("small", "medium", "large")}
feyn = [f for f in os.listdir(os.path.join(root, "qasm", "feynman")) if f.endswith(".qasm")]


def strata():
    S = []
    for size in ("small", "medium", "large"):
        for t in TOPOS:
            S.append((f"qasm-{size}", [f"test_QASMBench_{size}[{n}-{t}]" for n in qnames[size]]))
    for t in TOPOS:
        S.append(("ham", [f"test_hamiltonians[ham_{h['ham_instance'][1:-1]}-{t}]" for h in hams]))
    S.append(("ham-dev", [f"test_hamlib_hamiltonians_transpile[ham_{h['ham_instance'][1:-1]}]" for h in hams]))
    S.append(("feyn", [f"test_feynman_transpile[{f}]" for f in feyn]))
    return S


def draw(salt, k, exclude):
    out = []
    for kind, ids in strata():
        ids = [i for i in ids if i in REF and i not in exclude]
        out += sorted(ids, key=lambda i: hashlib.sha256((salt + i).encode()).hexdigest())[:k[kind]]
    return out


mock = draw("BP-MOCK|", {"qasm-small": 5, "qasm-medium": 4, "qasm-large": 4, "ham": 5, "ham-dev": 8, "feyn": 6}, probed)
want = draw("BP-MOCK2|", {"qasm-small": 3, "qasm-medium": 2, "qasm-large": 2, "ham": 3, "ham-dev": 4, "feyn": 4},
            probed | set(mock))
r = json.load(open(os.path.join(RUN, "bp_mock2.json")))
m, rows = r["meta"], r["rows"]
by = {}
for x in rows:
    by.setdefault(x["test"], {})[x["arm"]] = x
print(f"BP-MOCK's sample re-drawn {len(mock)} (without its 6 100-qubit tests); BP-MOCK2's {len(want)}; "
      f"in the run {len(by)}; same set {set(want) == set(by)}")


def ok(x):
    return "error" not in x


bad = [t for t in by if (not ok(by[t]["C22"]) and ok(by[t]["REL"])) or (ok(by[t]["C22"]) and by[t]["C22"]["valid"] is not True)]
vers = all(x.get("version") == ("2026-10-06.4" if x["arm"].startswith("REL") else "2026-10-07.c22")
           for x in rows if x["arm"] != "QK" and ok(x))
p0 = set(want) == set(by) and len(want) == 48 and not bad and vers and not m["smoke"] and not m["dirty_tracked"] and \
    m["benchpress"] == "b695f30"
print(f"P0 {'PASS' if p0 else 'FAIL'}: C22 failed or invalid {len(bad)}, versions ok {vers}, git_head {m['git_head']}")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def g(xs):
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def q(t, a, b):
    return (by[t][a]["q2"] + 1) / (by[t][b]["q2"] + 1)


def V(good, badv):
    return "REFUTED" if badv else ("CONFIRMED" if good else "AMBIGUOUS")


done = [t for t in by if all(ok(by[t][a]) for a in ("QK", "REL", "REL2", "C22"))]
flat = [t for t in done if by[t]["C22"]["wide"] == 0]
wide = [t for t in done if by[t]["C22"]["wide"] > 0]
stable = [t for t in flat if by[t]["REL"]["sig"] == by[t]["REL2"]["sig"]]
m1 = [t for t in stable if by[t]["C22"]["sig"] != by[t]["REL"]["sig"]]
gw, gq = g([q(t, "C22", "REL") for t in wide]), g([q(t, "C22", "QK") for t in done])
chk = [t for t in done if by[t]["C22"].get("checkable") is True and isinstance(by[t]["C22"].get("implements"), bool)]
m5 = [t for t in chk if not by[t]["C22"]["implements"]]
mine = {"M1": V(not m1, len(m1) >= 2), "M2": V(gw <= 0.85, gw > 1.00), "M3": V(gq <= 1.15, gq > 1.30),
        "M5": V(len(chk) >= 5 and not m5, bool(m5))}
print(f"done {len(done)} (flat {len(flat)}, REL reproducible on {len(stable)}; wide {len(wide)}); M1 differ {len(m1)}; "
      f"M2 {gw:.3f}; M3 {gq:.3f}; M5 {len(m5)} of {len(chk)} do not implement")
for k, v in mine.items():
    print(k, v)
sc = os.path.join(RUN, "score.md")
theirs = dict(re.findall(r"\| (M\d) \| [^|]+ \| [^|]+ \| \*\*(\w+)\*\* \|", open(sc).read())) if os.path.exists(sc) else {}
print(f"verdicts identical to score.md: {theirs == mine}")
