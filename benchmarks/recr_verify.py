"""recr_verify.py -- independent re-computation of the RECR run (Addendum 358), written before the scored run and before
any of its output was seen (2026-10-06, morning). Reads the raw json only and does not import recr_eval.py: provenance,
counts, P0, Q1-Q10, and the per-device table.
    python recr_verify.py <run dir>"""
import glob
import json
import os
import statistics
import sys

RUN = sys.argv[1]
CX, CZ, ECR = ("FakeAuckland", "FakeHanoiV2"), ("FakeTorino", "FakeKingston"), ("FakeBrussels", "FakeOsaka")
DEV = CX + CZ + ECR
SIM = ("R51", "R51M", "C14M", "A9M", "A11M", "L3TM")
SHA = dict(script="ed9f5dbe8e95", c14="8cdeaa30c8ae", a11="380102b2856f")
VER = dict(release="2026-10-05.1", c14="2026-10-05.c14", a9="2026-10-05.a9", a11="2026-10-05.a11")

F, flags, heads = {}, [], set()
for p in glob.glob(os.path.join(RUN, "recr_*.json")):
    r = json.load(open(p))
    m = r["meta"]
    heads.add(m["git_head"])
    if m["smoke"] or any(not m["sha"][k].startswith(v) for k, v in SHA.items()) or \
            any(m["versions"][k] != v for k, v in VER.items()) or len(r["rows"]) != 104:
        flags.append(os.path.basename(p))
    F[m["device"]] = r["rows"]

err = sum(1 for rows in F.values() for row in rows for a, v in row.items() if isinstance(v, dict) and "error" in v)
inexact = sum(1 for rows in F.values() for row in rows for a in SIM if row[a]["state_infid"] > 1e-6)
badm = sum(1 for rows in F.values() for row in rows for a in SIM if not row[a]["meas_ok"])
p0 = sorted(F) == sorted(DEV) and not flags and not err and not inexact and not badm and len(heads) == 1
print(f"files {len(F)}, git_head {sorted(heads)}, flags {flags}")
print(f"P0 {'PASS' if p0 else 'FAIL'}: compile errors {err}, inexact {inexact}, wrong measurement mapping {badm}")


def M(d, a, k):
    v = [row[a][k] for row in F[d]]
    return sum(v) / len(v)


def MED(d, a):
    return statistics.median(row[a]["compile_s"] for row in F[d])


def FR(d, a, k):
    return sum(1 for row in F[d] if row[a].get(k)) / len(F[d])


def S(d, a, k):
    return sum(row[a][k] for row in F[d])


def V(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


print("\ndevice        arm   infid     meas_err  off  failed  2q     med_s")
for d in DEV:
    for a in SIM:
        print(f"{d:13s} {a:5s} {M(d, a, 'infid'):.5f}  {M(d, a, 'meas_err'):.4f}   {S(d, a, 'off_target'):3d}  "
              f"{S(d, a, 'failed_uses'):3d}     {M(d, a, 'n2q'):5.2f}  {MED(d, a):.3f}")

q1 = {d: M(d, "R51M", "meas_err") / M(d, "R51", "meas_err") for d in CZ}
q2 = {d: M(d, "C14M", "infid") / M(d, "R51M", "infid") for d in DEV}
q2e = {d: M(d, "C14M", "meas_err") - M(d, "R51M", "meas_err") for d in DEV}
q3 = {d: FR(d, "C14", "same_as_R51") for d in DEV}
q4 = {d: S(d, "A11M", "off_target") + S(d, "A11", "off_target") for d in DEV}
q5 = {d: S(d, "A9M", "off_target") for d in ECR}
q6 = {d: FR(d, "A11", "same_as_A9") for d in CX + CZ}
q7 = {d: M(d, "A11M", "infid") / M(d, "A9M", "infid") for d in CZ}
q8 = {d: M(d, "A11M", "infid") / M(d, "L3TM", "infid") for d in DEV}
q9 = {d: M(d, "C14M", "infid") / M(d, "L3TM", "infid") for d in DEV}
q10 = {d: MED(d, "C14M") / MED(d, "R51M") for d in DEV}
q10.update({d + "/A": MED(d, "A11M") / MED(d, "A9M") for d in CX + CZ})
out = [
    ("Q1", V(all(v <= 0.85 for v in q1.values()), any(v >= 1 for v in q1.values())), q1),
    ("Q2", V(all(v <= 1.005 for v in q2.values()) and all(v <= 5e-4 for v in q2e.values()),
             any(v > 1.02 for v in q2.values())), (q2, q2e)),
    ("Q3", V(all(v >= 0.999 for v in q3.values()), any(v < 0.99 for v in q3.values())), q3),
    ("Q4", V(all(v == 0 for v in q4.values()), any(v > 0 for v in q4.values())), q4),
    ("Q5", V(all(v >= 1 for v in q5.values()), all(v == 0 for v in q5.values())), q5),
    ("Q6", V(all(v >= 0.999 for v in q6.values()), any(v < 0.99 for v in q6.values())), q6),
    ("Q7", V(all(v <= 0.8 for v in q7.values()), any(v >= 1 for v in q7.values())), q7),
    ("Q8", V(sum(v <= 1 for v in q8.values()) >= 5, any(v > 1.05 for v in q8.values())), q8),
    ("Q9", V(sum(v <= 1 for v in q9.values()) >= 5, any(v > 1.05 for v in q9.values())), q9),
    ("Q10", V(all(v <= 1.15 for v in q10.values()), any(v > 1.5 for v in q10.values())), q10),
]
print()
for q, v, num in out:
    print(q, v, json.dumps(num, default=lambda x: round(x, 4)))
print("\nper-circuit C14M vs R51M infid (worse, better) by device:",
      {d: (sum(row["C14M"]["infid"] > row["R51M"]["infid"] + 1e-12 for row in F[d]),
           sum(row["C14M"]["infid"] < row["R51M"]["infid"] - 1e-12 for row in F[d])) for d in DEV})
