"""Request 4 (exploratory): the physical qubits and couplers PSF, QK3 (and C47) used on HW1's BA circuits, weighted by use,
against the day's calibration CSV."""
import csv, json, os, sys
from collections import Counter
import numpy as np
from qiskit import qpy
REPO = sys.argv[1]
W = os.path.join(REPO, "data/2026-10-12/hw1")
CAL = os.path.join(REPO, "data/2026-10-12/hw_analysis/ibm_kingston_calibrations_2026-10-12T11_15_24Z.csv")
rows = {int(r["Qubit"]): r for r in csv.DictReader(open(CAL, encoding="utf-8"))}
def fl(x, d=float("nan")):
    try: return float(x)
    except ValueError: return d
T1 = {q: fl(r["T1 (us)"]) for q, r in rows.items()}
T2 = {q: fl(r["T2 (us)"]) for q, r in rows.items()}
RO = {q: fl(r["MEASURE error"]) for q, r in rows.items()}
SX = {q: fl(r["√x (sx) error"]) for q, r in rows.items()}
CZ = {}
for q, r in rows.items():
    for part in r["CZ error"].split(";"):
        if part:
            o, e = part.split(":"); CZ[frozenset((q, int(o)))] = float(e)
okq = [q for q in rows if not np.isnan(T2[q]) and T2[q] > 0]
t2_p20 = np.percentile([T2[q] for q in okq], 20); t1_p20 = np.percentile([T1[q] for q in okq], 20)
cz_vals = [v for v in CZ.values() if v < 0.5]; cz_p80 = np.percentile(cz_vals, 80)
man = json.load(open(W + "/hw1_manifest.json")); sc = json.load(open(W + "/hw1_score.json"))
circs = qpy.load(open(W + "/hw1_circuits.qpy", "rb"))
print(f"device reference: T2 20th pct {t2_p20:.0f} us, T1 20th pct {t1_p20:.0f} us, CZ error 80th pct {cz_p80:.4f}\n")
hdr = ("test", "arm", "F", "2q", "uses-w T1", "uses-w T2", "min T2", "q<T2p20", "w CZ err", "CZ>p80 share",
       "max uses/coupler", "couplers", "meas RO err")
print(" | ".join(hdr)); print("|".join(["---"] * len(hdr)))
out = []
for p, c in zip(man["pubs"], circs):
    e = man["entries"][p["entry"]]
    if e["part"] != "HW1" or p["repeat"] != 1 or not e["test"].startswith("ba_"):
        continue
    pos = {q: i for i, q in enumerate(c.qubits)}
    quse, cuse, meas = Counter(), Counter(), []
    for ins in c.data:
        nm = ins.operation.name; qs = [pos[q] for q in ins.qubits]
        if nm in ("barrier", "rz"): continue
        if nm == "measure": meas.append(qs[0]); continue
        for q in qs: quse[q] += 1
        if len(qs) == 2: cuse[frozenset(qs)] += 1
    w = np.array([quse[q] for q in quse], float)
    wt1 = float(np.average([T1[q] for q in quse], weights=w)); wt2 = float(np.average([T2[q] for q in quse], weights=w))
    n2 = sum(cuse.values())
    wcz = sum(CZ.get(k, np.nan) * v for k, v in cuse.items()) / n2
    bad = sum(v for k, v in cuse.items() if CZ.get(k, 1) > cz_p80) / n2
    r = dict(test=e["test"], arm=e["arm"], F=sc["F"][f"{e['test']}|{e['arm']}"]["f"], twoq=n2, wT1=wt1, wT2=wt2,
             minT2=min(T2[q] for q in quse), lowT2=sum(1 for q in quse if T2[q] < t2_p20), wCZ=wcz, czbad=bad,
             maxc=max(cuse.values()), ncoup=len(cuse), ro=float(np.mean([RO[q] for q in meas])))
    out.append(r)
    print(f"{r['test']} | {r['arm']} | {r['F']:.3f} | {n2} | {wt1:.0f} | {wt2:.0f} | {r['minT2']:.0f} | {r['lowT2']}/{len(quse)} | "
          f"{wcz:.4f} | {bad:.2f} | {r['maxc']} | {r['ncoup']} | {r['ro']:.4f}")
json.dump(out, open(sys.argv[2], "w"), indent=1)

# addition: low-T2 qubits used and their exposure (window from first gate to measurement, Heron durations 32/68 ns)
print("\nlow-T2 qubits (T2 below the device's 20th percentile) used, and the arm's total exposure sum(window/T2)")
for p, c in zip(man["pubs"], circs):
    e = man["entries"][p["entry"]]
    if e["part"] != "HW1" or p["repeat"] != 1 or not e["test"].startswith("ba_"):
        continue
    pos = {q: i for i, q in enumerate(c.qubits)}
    clock, first, uses = {}, {}, Counter()
    for ins in c.data:
        nm = ins.operation.name; qs = [pos[q] for q in ins.qubits]
        if nm in ("barrier", "measure"): continue
        d = 0.0 if nm == "rz" else (68e-3 if len(qs) == 2 else 32e-3)  # us
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            first.setdefault(q, s); clock[q] = s + d; uses[q] += nm != "rz"
    T = max(clock.values())
    expo = sum((T - first[q]) / T2[q] for q in first)
    low = [(q, round(T2[q]), uses[q], round(T - first[q], 1)) for q in first if T2[q] < t2_p20]
    print(f"{e['test']:7s} {e['arm']:4s} duration {T:5.1f} us  exposure {expo:.3f}  low-T2 (qubit, T2 us, uses, window us): {low}")
