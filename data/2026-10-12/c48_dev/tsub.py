"""Time breakdown of the circuits submitted in HW1 (2026-10-12, ibm_kingston): per test and arm, two-qubit gate
count, two-qubit depth, ASAP duration with the day's calibration, the most loaded qubit's two-qubit count (a lower
bound on the two-qubit depth) and the summed active windows (qubit-microseconds)."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import csv, json, sys
from qiskit import qpy
W, CAL = sys.argv[1], sys.argv[2]
rows = {int(r["Qubit"]): r for r in csv.DictReader(open(CAL))}
fl = lambda x, d: float(x) if x not in ("", None) else d
L1 = {q: fl(r["Single-qubit gate length (ns)"], 32.0) for q, r in rows.items()}
CZL = {}
for q, r in rows.items():
    for part in r["Gate length (ns)"].split(";"):
        if part:
            o, d = part.split(":"); CZL[(q, int(o))] = float(d)
man = json.load(open(W + "/hw1_manifest.json")); sc = json.load(open(W + "/hw1_score.json"))
circs = qpy.load(open(W + "/hw1_circuits.qpy", "rb"))

def stats(c):
    clock, first, d2, load = {}, {}, {}, {}
    n2 = 0
    for ins in c.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure"):
            continue
        qs = tuple(c.find_bit(q).index for q in ins.qubits)
        d = 0.0 if nm == "rz" else (L1[qs[0]] if len(qs) == 1 else CZL.get(qs, CZL.get(qs[::-1], 68.0)))
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            first.setdefault(q, s); clock[q] = s + d
        if len(qs) == 2:
            n2 += 1
            k = max(d2.get(q, 0) for q in qs) + 1
            for q in qs:
                d2[q] = k; load[q] = load.get(q, 0) + 1
    T = max(clock.values())
    win = sum(T - first[q] for q in clock)
    hub = max(load, key=load.get)
    return n2, max(d2.values()), max(load.values()), T / 1000, win / 1000, len(clock)

R = {}
for p, c in zip(man["pubs"], circs):
    e = man["entries"][p["entry"]]
    if e["part"] != "HW1" or p["repeat"] != 1:
        continue
    R[(e["test"], e["arm"])] = stats(c) + (sc["F"][f"{e['test']}|{e['arm']}"]["f"],)
print(f"{'test':12s} {'arm':4s} {'2q':>5s} {'2q-depth':>8s} {'maxload':>7s} {'slack':>6s} {'dur us':>7s} {'win q.us':>8s} {'act':>4s} {'F':>6s}")
for (t, a), v in R.items():
    n2, dep, ml, T, win, act, F = v
    print(f"{t:12s} {a:4s} {n2:5d} {dep:8d} {ml:7d} {dep/ml:6.2f} {T:7.1f} {win:8.0f} {act:4d} {F:6.3f}")
