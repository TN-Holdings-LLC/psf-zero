import json, math, sys
import numpy as np
R = json.load(open(sys.argv[1]))
def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0] * len(x)
    for k, i in enumerate(o): r[i] = k
    return r
tests = [t for t in dict.fromkeys(k.split("|")[0] for k in R) if f"{t}|PSF" in R and f"{t}|QK3" in R
         and not t.startswith(("ghz", "tfim_n16"))]
print("rule vs exact per circuit (<= 25 active; exact = Aer)")
for k, v in R.items():
    ex, ru = sum(v["aer"]), sum(v["rule"])
    print(f"  {k:18s} exact {ex:.4f} rule {ru:.4f} ({(ru-ex)/ex*100:+.1f}%)  gate {v['aer'][0]:.4f}/{v['rule'][0]:.4f} idle {v['aer'][1]:.4f}/{v['rule'][1]:.4f}")
for src in ("aer", "rule"):
    cols = {"esp": [], "gate": [], "tot": [], "F": []}
    print(f"\nPSF/QK3, {src}")
    for t in tests:
        a, b = R[f"{t}|PSF"], R[f"{t}|QK3"]
        cols["esp"].append(a["esp"] / b["esp"]); cols["gate"].append(math.exp(b[src][0] - a[src][0]))
        cols["tot"].append(math.exp(sum(b[src]) - sum(a[src]))); cols["F"].append(a["F"] / b["F"])
        print(f"  {t:12s} ESP {cols['esp'][-1]:.3f}  gate+readout {cols['gate'][-1]:.3f}  with idle {cols['tot'][-1]:.3f}  real F {cols['F'][-1]:.3f}")
    for k in ("esp", "gate", "tot"):
        rho = np.corrcoef(ranks(cols[k]), ranks(cols["F"]))[0, 1]
        agree = sum(1 for u, w in zip(cols[k], cols["F"]) if (u > 1) == (w > 1))
        print(f"  {k:5s}: same winner {agree}/{len(tests)}, rank correlation {rho:+.2f}")
