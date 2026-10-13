# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).
import sys, os
import ev48 as E
from qiskit_ibm_runtime import fake_provider
M, H = E.M, E.H
CAP = []
orig_t = M.transpile
def tr(c, *a, **k):
    if k.get("routing_method") == "none":
        CAP.append(c.copy())
    return orig_t(c, *a, **k)
M.transpile = tr
name, dev = sys.argv[1], sys.argv[2]
b = getattr(fake_provider, dev)()
qc = E.gens.build(name) if name not in H.TESTS else H.build_test(name)[0]
p0, _ = H.measured_distribution(H._unrolled(qc))
for rule in (None, "start"):
    E.RULE["rule"] = rule
    CAP.clear()
    out = E.psf(qc, b)
    for k, c in enumerate(CAP):
        p1, _ = H.measured_distribution(c)
        print(rule, "phys", k, f"{max(abs(p0.get(x,0)-p1.get(x,0)) for x in set(p0)|set(p1)):.1e}", sum(1 for i in c.data if len(i.qubits)==2))
    p1, _ = H.measured_distribution(out)
    print(rule, "final", f"{max(abs(p0.get(x,0)-p1.get(x,0)) for x in set(p0)|set(p1)):.1e}")
