"""Per logical qubit: gadget CX and SWAP units in the mover, with and without holders (BA)."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import sys
import hubrun as R
import hub
from qiskit_ibm_runtime import fake_provider
name, dev = sys.argv[1], sys.argv[2]
b = getattr(fake_provider, dev)(); t = b.target
qc = R.H.build_test(name)[0]
for hubs, k in ((0, 0), (1, 1), (1, 2)):
    fn = (lambda tr, n: (tr, n, "none")) if hubs == 0 else (lambda tr, n, hubs=hubs, k=k: hub.transform(tr, n, hubs, k))
    out, desc, m = R.construct(qc, t, fn)
    pi0 = m.__dict__.get("_pi0")
    # follow states from the final pi backwards is awkward: recount from the blocks (each gadget's Pauli is on physical
    # qubits at its time; SWAPs move states) using the mover's own initial placement, recovered by undoing the SWAPs
    pi = list(m.pi)
    for nm, q, _ in reversed(m.ops):
        if nm == "swap":
            a, c = q
            for i, v in enumerate(pi):
                if v == a: pi[i] = c
                elif v == c: pi[i] = a
    inv = {p: i for i, p in enumerate(pi)}
    g, s = {}, {}
    for nm, q, _ in m.ops:
        if nm == "swap":
            a, c = inv.get(q[0]), inv.get(q[1])
            for x in (a, c):
                if x is not None: s[x] = s.get(x, 0) + 3
            inv.pop(q[0], None); inv.pop(q[1], None)
            if a is not None: inv[q[1]] = a
            if c is not None: inv[q[0]] = c
        elif nm == "cx":
            for x in q:
                l = inv.get(x)
                if l is not None: g[l] = g.get(l, 0) + 1
    top = sorted(set(g) | set(s), key=lambda x: -(g.get(x, 0) + s.get(x, 0)))[:5]
    print(f"hubs {hubs} k {k} makespan {R.S.asap(m.ops):.0f} swaps {sum(1 for o in m.ops if o[0]=='swap')} cx {sum(1 for o in m.ops if o[0]=='cx')} | " +
          "  ".join(f"L{x}: cx {g.get(x,0)} swap {s.get(x,0)}" for x in top))
