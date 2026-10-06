"""pl_gpu_redo_verify.py -- independent re-computation of the PL-GPU-REDO run (Addendum 375), written before the
scored run. Reads the raw json only and does not import pl_gpu_redo.py: validity, the per-arm table and G1-G8, then
compares the verdicts with score.md.
    python pl_gpu_redo_verify.py <run dir>"""
import json
import math
import os
import statistics
import sys

RUN = sys.argv[1]
ARMS = ("R", "RR", "A12", "Q3")
SP = (0, 4)
N = 30
VER = {"release": "2026-10-06.2", "core": "2026-09-29.1", "layout": "2026-10-01.1", "a12": "2026-10-06.a12"}
SHA = dict(script="132739b23ca4", gpu_helpers="a0081c291759", release="1c3dfb0853c2", layout="624e8f8a00e1",
           a12="2227cca2b067")  # normalized SHA-256 prefixes (Addendum 375, section 5)

R = {a: json.load(open(os.path.join(RUN, f"pl_gpu_redo_{a}.json"), encoding="utf-8")) for a in ARMS}
flags = []
for a in ARMS:
    m = R[a]["meta"]
    if m["smoke"] or m["git_dirty"] or m["laps"] != N or m["device"] != "lightning.gpu":
        flags.append(f"{a}: run flags")
    if any(m["versions"][k] != v for k, v in VER.items()):
        flags.append(f"{a}: versions {m['versions']}")
    for k, v in SHA.items():
        if not m["sha"].get(k, "").startswith(v):
            flags.append(f"{a}: sha {k}")
    for s, c in m["checks"].items():
        if not (c["c0_gpu_vs_cpu"] <= 1e-12 and c["c1_control"] >= 1e-9):
            flags.append(f"{a}: C0/C1 at spare {s}")
heads = sorted({R[a]["meta"]["git_head"] for a in ARMS})


def rows(a, s):
    return [r for r in R[a]["rows"] if r["spare"] == s]


def is_err(r):
    return r["status"].startswith("error")


counts = {(a, s): len(rows(a, s)) for a in ARMS for s in SP}
p0 = not flags and len(heads) == 1 and all(c == N for c in counts.values())
print(f"git_head {heads}, flags {flags}, laps per cell {sorted(set(counts.values()))}")
print(f"P0 {'PASS' if p0 else 'FAIL'}")
for s in SP:
    for a in ARMS:
        c = rows(a, s)
        w = [r["whole_circuit_max_diff"] for r in c if r["whole_circuit_max_diff"] is not None]
        print(f"spare {s} {a:3s} median {statistics.median(r['compile_s'] for r in c):7.3f} s  errors "
              f"{sum(map(is_err, c)):2d}  within 1 s {sum(r['compile_s'] <= 1.0 and not is_err(r) for r in c):2d}  "
              f"back {sum(r['status'] == 'OK' and r['twoq'] == r['swapfree_twoq'] for r in c):2d}  "
              f"whole max {max(w) if w else float('nan'):.2e}")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def V(good, bad):
    return "REFUTED" if bad else ("CONFIRMED" if good else "AMBIGUOUS")


def win(a, s):
    return sum(r["compile_s"] <= 1.0 and not is_err(r) for r in rows(a, s))


def back(a, s):
    return sum(r["status"] == "OK" and r["twoq"] == r["swapfree_twoq"] for r in rows(a, s))


def whole(*arms):
    return [r["whole_circuit_max_diff"] for a in arms for r in R[a]["rows"] if r["whole_circuit_max_diff"] is not None]


wr, wt = whole("R"), whole("RR", "A12", "Q3")
mr = max(wr) if wr else math.nan
mt = max(wt) if wt else math.nan
e = sum(is_err(r) for a in ARMS for r in R[a]["rows"])
out = [("G1", V(win("R", 0) == N, win("R", 0) <= N // 2)),
       ("G2", V(back("R", 0) + back("R", 4) == 2 * N, back("R", 0) + back("R", 4) <= N)),
       ("G3", V(len(wr) == 2 * N and mr <= 1e-12, bool(wr) and mr > 1e-10)),
       ("G4", V(e == 0, e >= 1)),
       ("G5", V(len(wt) == 6 * N and mt <= 1e-10, bool(wt) and mt > 1e-6)),
       ("G6", V(win("Q3", 0) == 0, win("Q3", 0) >= N // 2)),
       ("G7", V(max(win("RR", 0), win("A12", 0)) <= 2, max(win("RR", 0), win("A12", 0)) >= N // 2)),
       ("G8", V(min(win("RR", 4), win("A12", 4), win("Q3", 4)) >= N - 2,
                min(win("RR", 4), win("A12", 4), win("Q3", 4)) <= N // 2))]


def same_as_r(a):
    return sum(x["digest"] == y["digest"] for x, y in zip(rows("R", 0), rows(a, 0)) if y["digest"])


g9 = min(same_as_r("RR"), same_as_r("A12"))
out.append(("G9", V(g9 >= N - 2, g9 <= N // 2)))
for q, v in out:
    print(q, v)
sc = os.path.join(RUN, "score.md")
if os.path.exists(sc):
    txt = open(sc, encoding="utf-8").read()
    same = all(f"**{v}**" in txt.split(f"- {q} (", 1)[1].split("\n", 1)[0] for q, v in out)
    print("verdicts identical to score.md:", same)
