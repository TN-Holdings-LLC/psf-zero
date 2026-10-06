"""kr_verify.py -- independent re-computation of the KRAUS run (Addendum 359), written before the scored run and before
any of its output exists (2026-10-06). Reads the raw json only and does not import kr_eval.py: provenance, counts, P0
(including a re-derivation of every recorded choice from the recorded estimates), K1-K7 and the per-device table.
    python kr_verify.py <run dir>"""
import glob
import json
import os
import statistics
import sys

RUN = sys.argv[1]
DEV = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
       "FakeMarrakesh", "FakeAachen")
FAM = ("F1", "F2", "F3", "F4", "F5", "F6")
WANT = dict(F1=432, F2=240, F3=300, F4=270, F5=144, F6=120)
SHA = dict(script="dc55a9db28a2", c15="24affde8f5d6", release="33853989e0bf", hold6="8740a33225f2")
VER = dict(release="2026-10-05.1", c15="2026-10-06.c15")

F, flags, heads = {}, [], set()
for p in glob.glob(os.path.join(RUN, "kr_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m.get("git_head"))
    bad = m.get("smoke") or any(not m["sha"][k].startswith(v) for k, v in SHA.items()) or \
        any(m[k] != v for k, v in VER.items()) or len(r["rows"]) != WANT[m["family"]]
    if bad:
        flags.append(os.path.basename(p))
    for row in r["rows"]:
        row["_f"] = m["family"]
    F[(m["device"], m["family"])] = r["rows"]


def pick(costs):
    """Lowest estimate; the first candidate on ties or when any estimate is missing (psf_compile._choose's rule)."""
    if any(c is None for c in costs):
        return 0
    best = 0
    for j in range(1, len(costs)):
        if costs[j] < costs[best]:
            best = j
    return best


rows = {d: [row for f in FAM for row in F.get((d, f), [])] for d in DEV}
allrows = [row for d in DEV for row in rows[d]]
err = sum(1 for row in allrows if "error" in row)
ok_rows = [row for row in allrows if "error" not in row]
inexact = sum(1 for row in ok_rows for v in (row.get("infid_ideal") or []) if v is not None and v > 1e-6)
incons = sum(1 for row in ok_rows if not row["r_is_hyb"] or not row["k_is_kra"] or row["c15_hybrid_is_r"] is False)
rederived = sum(1 for row in ok_rows for s in ("hyb", "kra", "pau") if pick(row["est"][s]) != row["choice"][s])
k_est = sum(1 for row in ok_rows
            if abs(row["compile_k_est"] - (row["compile_r"] + row["est_kra_s"] - row["est_hyb_s"])) > 2e-4)
hyb_checks = sum(1 for row in ok_rows if row["c15_hybrid_is_r"] is not None)
p0 = len(F) == 54 and not flags and not err and not inexact and not incons and not rederived and len(heads) == 1
print(f"files {len(F)} of 54, git_head {sorted(heads)}, flags {flags}")
print(f"P0 {'PASS' if p0 else 'FAIL'}: errors {err}, inexact {inexact}, inconsistent {incons}, "
      f"choices not re-derived from the estimates {rederived}; c15-hybrid checks made {hyb_checks}")
print(f"(not part of P0) rows whose compile_k_est is not compile_r + est_kra_s - est_hyb_s: {k_est}")
if not p0:
    print("Nothing below is scored.")
    sys.exit(0)


def sim(d, f=None, n=None):
    return [row for row in rows[d] if "error" not in row and row.get("infid") and all(v is not None for v in row["infid"])
            and (f is None or row["_f"] == f) and (n is None or row["n"] == n)]


def chosen(row, s):
    return row["infid"][row["choice"][s]]


def avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def V(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


T = {}
print("\ndevice         n_sim  HYB      KRA      PAU      BEST     KRA/HYB  KRA/L3   changed  KRA better  K/R time")
for d in DEV:
    rs = sim(d)
    l3 = [row for row in rs if "level3" in row["cands"]]
    ch = [row for row in rs if chosen(row, "kra") != chosen(row, "hyb")]
    T[d] = dict(hyb=avg(chosen(r, "hyb") for r in rs), kra=avg(chosen(r, "kra") for r in rs),
                pau=avg(chosen(r, "pau") for r in rs), best=avg(min(r["infid"]) for r in rs),
                l3=avg(r["infid"][r["cands"].index("level3")] for r in l3), kra_l3=avg(chosen(r, "kra") for r in l3),
                changed=len(ch), better=sum(1 for r in ch if chosen(r, "kra") < chosen(r, "hyb")),
                t=statistics.median(r["compile_k_est"] for r in rows[d] if "error" not in r) /
                statistics.median(r["compile_r"] for r in rows[d] if "error" not in r), n=len(rs))
    x = T[d]
    print(f"{d:14s} {x['n']:5d}  {x['hyb']:.5f}  {x['kra']:.5f}  {x['pau']:.5f}  {x['best']:.5f}  "
          f"{x['kra'] / x['hyb']:.4f}   {x['kra_l3'] / x['l3']:.4f}   {x['changed']:5d}    {x['better']:5d}     {x['t']:.3f}")

k1 = {d: T[d]["kra"] / T[d]["hyb"] for d in DEV}
h4 = sim("FakeAlgiers", "F5", 4)
k2 = avg(chosen(r, "kra") for r in h4) / avg(chosen(r, "hyb") for r in h4) if h4 else float("nan")
gap = {d: (T[d]["kra"] / T[d]["best"] - 1, T[d]["hyb"] / T[d]["best"] - 1) for d in DEV}
k4 = {d: T[d]["better"] / T[d]["changed"] for d in DEV if T[d]["changed"] >= 20}
cell = {}
for d in DEV:
    for f in FAM:
        rs = sim(d, f)
        if rs:
            cell[f"{d}/{f}"] = avg(chosen(r, "kra") for r in rs) / avg(chosen(r, "hyb") for r in rs)
k6 = {d: T[d]["kra_l3"] / T[d]["l3"] for d in DEV}
k7 = {d: T[d]["t"] for d in DEV}
out = [
    ("K1", V(sum(v <= 1.0 for v in k1.values()) >= 8, any(v > 1.002 for v in k1.values())), k1),
    ("K2", V(k2 <= 0.96, k2 >= 1.0), dict(ratio=k2, circuits=len(h4))),
    ("K3", V(sum(a <= 0.5 * b for a, b in gap.values()) >= 7, sum(a > b for a, b in gap.values()) >= 3), gap),
    ("K4", V(bool(k4) and all(v >= 0.6 for v in k4.values()), any(v < 0.5 for v in k4.values())), k4),
    ("K5", V(all(v <= 1.01 for v in cell.values()), any(v > 1.03 for v in cell.values())),
     {k: v for k, v in cell.items() if v > 1.005}),
    ("K6", V(all(v <= 1.0 for v in k6.values()), any(v > 1.02 for v in k6.values())), k6),
    ("K7", V(all(v <= 1.15 for v in k7.values()), any(v > 1.5 for v in k7.values())), k7),
]
print()
for q, v, num in out:
    print(q, v, json.dumps(num, default=lambda x: round(x, 4)))
print("\nexcluded as too wide to simulate, by device:", {d: len(rows[d]) - len(sim(d)) for d in DEV})
print("per-circuit KRA vs HYB (worse, better) by device:",
      {d: (sum(chosen(r, "kra") > chosen(r, "hyb") for r in sim(d)), sum(chosen(r, "kra") < chosen(r, "hyb") for r in sim(d)))
       for d in DEV})
sc = os.path.join(RUN, "score.md")
if os.path.exists(sc):
    txt = open(sc).read()
    same = all(f"- {q} (" in txt and f"**{v}**" in txt.split(f"- {q} (", 1)[1].split("\n", 1)[0] for q, v, _ in out)
    print("verdicts identical to score.md:", same)
