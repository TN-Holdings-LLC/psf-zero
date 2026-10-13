"""Per logical qubit (the state, followed through SWAPs): its gadget CX units and SWAP units in the chosen mover; the
lower bound 2 x (number of two-qubit terms on it) CX units; and the mover's makespan."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import sys
import ev48 as E
S, M = E.S, E.M
from qiskit_ibm_runtime import fake_provider
orig = M._st_construct
LOG = []
def wrap(*a, **k):
    got = orig(*a, **k)
    if got is not None:
        LOG.append(got)
    return got
M._st_construct = wrap
tests, dev = sys.argv[1].split(","), sys.argv[2]
for rule in (None, "start"):
    E.RULE["rule"] = rule
    for name in tests:
        b = getattr(fake_provider, dev)()
        qc, _ = E.H.build_test(name)
        LOG.clear()
        E.psf(qc, b)
        m, pi0 = LOG[-1]
        inv = {p: q for q, p in enumerate(pi0)}
        g, s = {}, {}
        for nm, q, _ in m.ops:
            if nm == "swap":
                a, c = inv.get(q[0]), inv.get(q[1])
                for x in (a, c):
                    if x is not None:
                        s[x] = s.get(x, 0) + 3
                if a is not None: inv[q[1]] = a
                else: inv.pop(q[1], None)
                if c is not None: inv[q[0]] = c
                else: inv.pop(q[0], None)
            elif nm == "cx":
                for x in q:
                    l = inv.get(x)
                    if l is not None:
                        g[l] = g.get(l, 0) + 1
        terms2 = {}
        for lb, _, _ in m.triples:
            sup = [j for j, ch in enumerate(reversed(lb)) if ch != "I"]
            if len(sup) >= 2:
                for j in sup:
                    terms2[j] = terms2.get(j, 0) + 1
        top = sorted(range(m.n), key=lambda x: -(g.get(x, 0) + s.get(x, 0)))[:3]
        desc = "  ".join(f"L{x}: terms {terms2.get(x,0)} LB {2*terms2.get(x,0)} gadget-cx {g.get(x,0)} swap {s.get(x,0)}" for x in top)
        print(f"{name:12s} rule={rule!s:6s} makespan {S.asap(m.ops):6.1f} | {desc}")
