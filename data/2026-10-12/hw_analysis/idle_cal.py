"""Exploratory (after HW1's scoring): does an estimate with idling error, from the device's own calibration of the day,
rank PSF-Zero against Qiskit level 3 as the real device did? Uses the dashboard's calibration CSV (13:15 CEST)."""
import csv, json, math, sys
import numpy as np
from qiskit import qpy
W, CAL = sys.argv[1], sys.argv[2]
rows = {int(r["Qubit"]): r for r in csv.DictReader(open(CAL))}
def fl(x, default):
    try:
        return float(x)
    except ValueError:
        return default
T1 = {q: fl(r["T1 (us)"], 1.0) * 1e-6 for q, r in rows.items()}  # missing: 1 us (a failed qubit)
T2 = {q: fl(r["T2 (us)"], 1.0) * 1e-6 for q, r in rows.items()}
SX = {q: fl(r["√x (sx) error"], 1.0) for q, r in rows.items()}
RO = {q: fl(r["MEASURE error"], 1.0) for q, r in rows.items()}
L1 = {q: fl(r["Single-qubit gate length (ns)"], 32.0) * 1e-9 for q, r in rows.items()}
print("qubits with a missing T1 or T2:", [q for q, r in rows.items() if not r["T1 (us)"] or not r["T2 (us)"]])
CZ, CZL = {}, {}
for q, r in rows.items():
    for part in r["CZ error"].split(";"):
        if part:
            o, e = part.split(":"); CZ[(q, int(o))] = float(e)
    for part in r["Gate length (ns)"].split(";"):
        if part:
            o, d = part.split(":"); CZL[(q, int(o))] = float(d) * 1e-9
man = json.load(open(W + "/hw1_manifest.json"))
sc = json.load(open(W + "/hw1_score.json"))
circs = qpy.load(open(W + "/hw1_circuits.qpy", "rb"))

def estimates(c):
    clock, busy, first = {}, {}, {}
    ln_gate = 0.0
    for ins in c.data:
        nm = ins.operation.name
        if nm == "barrier":
            continue
        qs = tuple(c.find_bit(q).index for q in ins.qubits)
        if nm == "measure":
            ln_gate += -math.log1p(-RO[qs[0]]); continue
        if nm == "rz":
            d, e = 0.0, 0.0
        elif len(qs) == 1:
            d, e = L1[qs[0]], SX[qs[0]]
        else:
            d, e = CZL.get(qs, CZL.get(qs[::-1], 68e-9)), CZ.get(qs, CZ.get(qs[::-1], 0.01))
        ln_gate += -math.log1p(-min(e, 0.999))
        start = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            first.setdefault(q, start); clock[q] = start + d; busy[q] = busy.get(q, 0.0) + d
    total = max(clock.values()) if clock else 0.0
    ln_pauli = ln_t2 = 0.0
    for q in clock:
        idle = total - first[q] - busy[q] + busy[q]  # coherence is lost during gates too: the whole window counts
        ln_pauli += idle * (1 / (4 * T1[q]) + 1 / (2 * T2[q]))
        ln_t2 += idle / T2[q]
    return math.exp(-ln_gate), math.exp(-ln_gate - ln_pauli), math.exp(-ln_gate - ln_t2), total * 1e6

R = {}
for p, c in zip(man["pubs"], circs):
    e = man["entries"][p["entry"]]
    if e["part"] != "HW1" or p["repeat"] != 1:
        continue
    esp, idle_p, idle_t2, us = estimates(c)
    R[(e["test"], e["arm"])] = dict(esp=esp, pauli=idle_p, t2=idle_t2, us=us, F=sc["F"][f"{e['test']}|{e['arm']}"]["f"])
print(f"{'test':12s} {'arm':4s} {'ESP(day)':>8s} {'+idle(Pauli)':>12s} {'+idle(T2)':>10s} {'dur us':>7s} {'F real':>7s}")
for (t, a), v in R.items():
    print(f"{t:12s} {a:4s} {v['esp']:8.3f} {v['pauli']:12.3f} {v['t2']:10.3f} {v['us']:7.1f} {v['F']:7.3f}")
def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x)
    for k, i in enumerate(o): r[i] = k
    return r
tests = [t for t in dict.fromkeys(t for t, _ in R) if (t, "PSF") in R and (t, "QK3") in R and not t.startswith(("ghz", "tfim_n16"))]
print("\nPSF/QK3 ratios (tfim_n16 left out: F > 1, the measure does not work there; GHZ: identical circuits)")
cols = {"esp": [], "pauli": [], "t2": [], "F": []}
for t in tests:
    a, b = R[(t, "PSF")], R[(t, "QK3")]
    for k in cols: cols[k].append(a[k] / b[k])
    print(f"{t:12s} ESP {a['esp']/b['esp']:.3f}  +idle(Pauli) {a['pauli']/b['pauli']:.3f}  +idle(T2) {a['t2']/b['t2']:.3f}  real F {a['F']/b['F']:.3f}")
for k in ("esp", "pauli", "t2"):
    rho = np.corrcoef(ranks(cols[k]), ranks(cols["F"]))[0, 1]
    agree = sum(1 for x, y in zip(cols[k], cols["F"]) if (x > 1) == (y > 1))
    print(f"rank correlation with the real F ratio: {k:6s} {rho:+.3f}; same winner in {agree}/{len(tests)}")
# all arms: does the estimate rank all circuits' F?
allk = [(k, v) for k, v in R.items() if not k[0].startswith("tfim_n16")]
for key in ("esp", "pauli", "t2"):
    rho = np.corrcoef(ranks([v[key] for _, v in allk]), ranks([v["F"] for _, v in allk]))[0, 1]
    print(f"over all {len(allk)} circuits, rank correlation of {key:6s} with F: {rho:+.3f}")
