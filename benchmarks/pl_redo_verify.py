"""pl_redo_verify.py -- independent re-computation of the PL-REDO run (Addendum 372), written before the scored run.
Reads the raw json only and does not import pl_redo.py: validity, the per-arm table and D1-D10, then compares the
verdicts with score.md.
    python pl_redo_verify.py <run dir>"""
import json
import os
import statistics
import sys

RUN = sys.argv[1]
ARMS = ("R", "RR", "A12", "RRC", "A12C", "Q3")
CAND = ("RRC", "A12C")
N = 20
VER = {"core": "2026-09-29.1", "layout": "2026-10-01.1", "a12": "2026-10-06.a12"}
_COMMON = dict(script="4150e16e903c", chain="84aafa1f5668", layout="624e8f8a00e1", a12="2227cca2b067")
SHA = {"rel": dict(_COMMON, release="bf4630d6356d"),  # normalized SHA-256 prefixes (Addendum 372, section 6)
       "cand": dict(_COMMON, release="8864c546a420")}

R = {a: json.load(open(os.path.join(RUN, f"pl_redo_{a}.json"), encoding="utf-8")) for a in ARMS}
flags = []
for a in ARMS:
    m = R[a]["meta"]
    if m["smoke"] or m["git_dirty"] or not m["qiskit_17057_present"] or m["laps"] != N:
        flags.append(f"{a}: run flags")
    want = dict(VER, release="2026-10-06.c16" if a in CAND else "2026-10-06.1")
    if any(m["versions"][k] != v for k, v in want.items()):
        flags.append(f"{a}: versions {m['versions']}")
    for k, v in SHA.get("cand" if a in CAND else "rel", {}).items():
        if not m["sha"].get(k, "").startswith(v):
            flags.append(f"{a}: sha {k}")
heads = sorted({R[a]["meta"]["git_head"] for a in ARMS})


def rows(a, s):
    return [r for r in R[a]["rows"] if r["spare"] == s]


def is_err(r):
    return r["status"].startswith("error")


counts = {(a, s): len(rows(a, s)) for a in ARMS for s in (0, 16)}
p0 = not flags and len(heads) == 1 and all(c == N for c in counts.values())
print(f"git_head {heads}, flags {flags}, laps per cell {sorted(set(counts.values()))}")
print(f"P0 {'PASS' if p0 else 'FAIL'}")
for s in (0, 16):
    for a in ARMS:
        c = rows(a, s)
        print(f"spare {s:2d} {a:4s} median {statistics.median(r['compile_s'] for r in c):8.3f} s  "
              f"errors {sum(map(is_err, c)):2d}  within 1 s {sum(r['compile_s'] <= 1.0 and not is_err(r) for r in c):2d}"
              f"  back {sum(r['status'] == 'OK' and r['twoq'] == r['swapfree_twoq'] for r in c):2d}")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def V(good, bad):
    return "REFUTED" if bad else ("CONFIRMED" if good else "AMBIGUOUS")


def win(a, s):
    return sum(r["compile_s"] <= 1.0 and not is_err(r) for r in rows(a, s))


def back(a, s):
    return sum(r["status"] == "OK" and r["twoq"] == r["swapfree_twoq"] for r in rows(a, s))


def nerr(a, s):
    return sum(map(is_err, rows(a, s)))


def dist(*arms):
    return [r["max_block_distance"] for a in arms for r in R[a]["rows"] if r["max_block_distance"] is not None]


same = total = 0
for rel, cand in (("RR", "RRC"), ("A12", "A12C")):
    for s in (0, 16):
        for x, y in zip(rows(rel, s), rows(cand, s)):
            if is_err(x):
                break
            total += 1
            same += x["digest"] == y["digest"]
dr, dd = dist("R"), dist(*CAND)
e6 = sum(nerr(a, s) for a in CAND for s in (0, 16))
out = [("D1", V(win("R", 0) == N, win("R", 0) <= N // 2)),
       ("D2", V(back("R", 0) + back("R", 16) == 2 * N, back("R", 0) + back("R", 16) <= N)),
       ("D3", V(len(dr) == 2 * N and max(dr) <= 1e-12, bool(dr) and max(dr) > 1e-10)),
       ("D4", V(win("Q3", 0) == 0, win("Q3", 0) >= N // 2)),
       ("D5", V(nerr("RR", 0) == N and nerr("A12", 0) == N, min(nerr("RR", 0), nerr("A12", 0)) <= N // 2)),
       ("D6", V(e6 == 0, e6 >= 1)),
       ("D7", V(max(win("RRC", 0), win("A12C", 0)) <= 2, max(win("RRC", 0), win("A12C", 0)) >= N // 2)),
       ("D8", V(back("RRC", 0) == N and back("A12C", 0) == N, min(back("RRC", 0), back("A12C", 0)) <= N // 2)),
       ("D9", V(bool(dd) and max(dd) <= 1e-12, bool(dd) and max(dd) > 1e-10)),
       ("D10", V(same == total and total >= N, same < total)),
       ("D11", V(min(win("RRC", 16), win("A12C", 16)) >= N - 2, min(win("RRC", 16), win("A12C", 16)) <= N // 2))]
print(f"identical outputs {same}/{total}")
for q, v in out:
    print(q, v)
sc = os.path.join(RUN, "score.md")
if os.path.exists(sc):
    txt = open(sc, encoding="utf-8").read()
    same_v = all(f"**{v}**" in txt.split(f"- {q} (", 1)[1].split("\n", 1)[0] for q, v in out)
    print("verdicts identical to score.md:", same_v)
