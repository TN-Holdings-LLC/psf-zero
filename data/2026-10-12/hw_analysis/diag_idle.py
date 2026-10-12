import sys, json, math, warnings
warnings.simplefilter("ignore")
from qiskit import qpy
from qiskit_ibm_runtime.fake_provider import FakeKingston
W = "<hw1 results>"
man = json.load(open(W + "/hw1_manifest.json"))
sc = json.load(open(W + "/hw1_score.json"))
circs = qpy.load(open(W + "/hw1_circuits.qpy", "rb"))
t = FakeKingston().target
props = FakeKingston().properties()
def _g(f, q):
    try:
        return f(q)
    except Exception:
        return None
T2 = {q: _g(props.t2, q) for q in range(156)}
med = sorted(v for v in T2.values() if v)[78]
T2 = {q: (v or med) for q, v in T2.items()}
def dur(name, qs):
    try:
        d = t[name][qs].duration
        return d or 0.0
    except Exception:
        return 0.0
def analyse(c):
    # ASAP schedule: per-qubit clock
    clock = {}
    busy = {}
    first = {}
    for ins in c.data:
        nm = ins.operation.name
        if nm == "barrier": continue
        qs = tuple(c.find_bit(q).index for q in ins.qubits)
        start = max(clock.get(q, 0.0) for q in qs)
        d = dur(nm, qs)
        for q in qs:
            first.setdefault(q, start)
            clock[q] = start + d
            busy[q] = busy.get(q, 0.0) + d
    total = max(clock.values())
    idle_lnF = 0.0
    for q in clock:
        idle = total - first[q] - busy[q]  # idle between first op and end of circuit (measurement at the end)
        idle_lnF += idle / T2[q] if T2[q] else 0.0
    return total, idle_lnF
seen = set()
rows = {}
for p, c in zip(man["pubs"], circs):
    e = man["entries"][p["entry"]]
    if e["part"] != "HW1" or p["repeat"] != 1: continue
    total, il = analyse(c)
    key = f"{e['test']}|{e['arm']}"
    f = sc["F"][key]["f"]
    rows[key] = (e["test"], e["arm"], e["esp"], total * 1e6, il, e["esp"] * math.exp(-il), f, c.depth(lambda i: len(i.qubits) == 2))
for k, (te, ar, esp, us, il, esp_idle, f, d2) in rows.items():
    print(f"{te:12s} {ar:4s} ESP {esp:.3f} duration {us:6.1f} us 2q-depth {d2:4d} idle -ln {il:.3f} ESP*idle {esp_idle:.3f}  F {f:.3f}")
print()
for te in sorted({r[0] for r in rows.values()}):
    a, b = rows.get(f"{te}|PSF"), rows.get(f"{te}|QK3")
    if a and b:
        print(f"{te:12s} ESP ratio {a[2]/b[2]:.3f}  idle-aware ratio {a[5]/b[5]:.3f}  F ratio {a[6]/b[6]:.3f}  duration ratio {a[3]/b[3]:.2f}")
