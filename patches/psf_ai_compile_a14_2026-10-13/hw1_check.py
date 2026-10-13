"""Exploratory (after the a14 design note was saved): a14's estimate (a4 gate terms + a10 readout + item 19 idle) on
HW1's committed circuits, with a Target built from FakeKingston's structure and the day's calibration CSV."""
import copy, csv, json, math, os, sys, time, warnings
warnings.simplefilter("ignore")
REPO, A14 = sys.argv[1], sys.argv[2]
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
import numpy as np
import core_fix_c2_eval as H
H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
a14 = H.load_module(A14, "psf_ai_compile_a14")
from qiskit import qpy
from qiskit.transpiler import InstructionProperties, QubitProperties
from qiskit_ibm_runtime import fake_provider
W = os.path.join(REPO, "data/2026-10-12/hw1")
CAL = os.path.join(REPO, "data/2026-10-12/hw_analysis/ibm_kingston_calibrations_2026-10-12T11_15_24Z.csv")
rows = {int(r["Qubit"]): r for r in csv.DictReader(open(CAL, encoding="utf-8"))}
def fl(x, d):
    try: return float(x)
    except ValueError: return d
t = copy.deepcopy(fake_provider.FakeKingston().target)
qp = list(t.qubit_properties)
for q, r in rows.items():
    qp[q] = QubitProperties(t1=fl(r["T1 (us)"], 1.0) * 1e-6, t2=fl(r["T2 (us)"], 1.0) * 1e-6,
                            frequency=getattr(qp[q], "frequency", None))
t.qubit_properties = qp
L1 = {q: fl(r["Single-qubit gate length (ns)"], 32.0) * 1e-9 for q, r in rows.items()}
for q, r in rows.items():
    for nm, col in (("sx", "√x (sx) error"), ("x", "Pauli-X error")):
        if nm in t.operation_names and (q,) in t[nm]:
            t.update_instruction_properties(nm, (q,), InstructionProperties(duration=L1[q], error=fl(r[col], 1.0)))
    if (q,) in t["measure"]:
        t.update_instruction_properties("measure", (q,), InstructionProperties(
            duration=fl(r["Readout length (ns)"], 2000) * 1e-9, error=fl(r["MEASURE error"], 1.0)))
    for part_e, part_d in zip(r["CZ error"].split(";"), r["Gate length (ns)"].split(";")):
        if not part_e:
            continue
        o, e = part_e.split(":"); o2, d = part_d.split(":")
        for qa in ((q, int(o)), (int(o), q)):
            if qa in t["cz"]:
                t.update_instruction_properties("cz", qa, InstructionProperties(duration=float(d) * 1e-9, error=float(e)))
man = json.load(open(W + "/hw1_manifest.json"))
sc = json.load(open(W + "/hw1_score.json"))
circs = qpy.load(open(W + "/hw1_circuits.qpy", "rb"))
R = {}
for p, c in zip(man["pubs"], circs):
    e = man["entries"][p["entry"]]
    if e["part"] != "HW1" or p["repeat"] != 1 or e["active"] > 22:
        continue
    t0 = time.time()
    pos = {q: i for i, q in enumerate(c.qubits)}
    used = {pos[q] for i in c.data for q in i.qubits}
    ident = {q: q for q in used}
    gate = a14._score_weights(a14._state_weights(c), t, ident)
    iw, T = a14.idle_weights(c, t)
    idle = a14.idle_cost(iw, t, ident)
    R[(e["test"], e["arm"])] = dict(gate=gate, idle=idle, tot=gate + idle, us=T * 1e6, esp=e["esp"],
                                    F=sc["F"][f"{e['test']}|{e['arm']}"]["f"])
    print(f"{e['test']:12s} {e['arm']:4s} gate {gate:.4f} idle {idle:.4f} dur {T*1e6:6.2f}us F {R[(e['test'], e['arm'])]['F']:.3f} ({time.time()-t0:.0f}s)", flush=True)
def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x)
    for k, i in enumerate(o): r[i] = k
    return r
tests = [x for x in dict.fromkeys(t_ for t_, _ in R) if (x, "PSF") in R and (x, "QK3") in R and not x.startswith(("ghz", "tfim_n16"))]
cols = {"esp": [], "gate": [], "tot": [], "F": []}
print("\nPSF / QK3 (predicted fidelity ratio exp(-cost_PSF)/exp(-cost_QK3); ESP ratio; real F ratio)")
for x in tests:
    a, b = R[(x, "PSF")], R[(x, "QK3")]
    cols["esp"].append(a["esp"] / b["esp"]); cols["gate"].append(math.exp(b["gate"] - a["gate"]))
    cols["tot"].append(math.exp(b["tot"] - a["tot"])); cols["F"].append(a["F"] / b["F"])
    print(f"{x:12s} ESP {cols['esp'][-1]:.3f}  a14 gate-only {cols['gate'][-1]:.3f}  a14 with idle {cols['tot'][-1]:.3f}  real F {cols['F'][-1]:.3f}")
for k in ("esp", "gate", "tot"):
    rho = np.corrcoef(ranks(cols[k]), ranks(cols["F"]))[0, 1]
    agree = sum(1 for u, v in zip(cols[k], cols["F"]) if (u > 1) == (v > 1))
    print(f"{k:5s}: same winner {agree}/{len(tests)}, rank correlation {rho:+.2f}")
allk = [(k, v) for k, v in R.items() if not k[0].startswith("tfim_n16")]
for k in ("esp", "gate", "tot"):
    xs = [v["esp"] if k == "esp" else math.exp(-v[k]) for _, v in allk]
    rho = np.corrcoef(ranks(xs), ranks([v["F"] for _, v in allk]))[0, 1]
    print(f"all {len(allk)} circuits: rank correlation of {k} with F {rho:+.2f}")
