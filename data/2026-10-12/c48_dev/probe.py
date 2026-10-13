"""Chosen mover's makespan vs its busiest qubit's load (CX units), and what that qubit carries."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import sys
sys.argv = [sys.argv[0]] + sys.argv[1:]
import ev48 as E
S, M = E.S, E.M
from qiskit_ibm_runtime import fake_provider
orig = M._st_construct
LOG = []
def wrap(*a, **k):
    got = orig(*a, **k)
    if got is not None:
        LOG.append(got[0])
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
        m = LOG[-1]
        load, kinds = {}, {}
        for nm, q, _ in m.ops:
            for x in q:
                load[x] = load.get(x, 0) + S._dur(nm)
                kinds.setdefault(x, {}); kinds[x][nm] = kinds[x].get(nm, 0) + 1
        hub = max(load, key=load.get)
        cx = sum(1 for nm, q, _ in m.ops if nm == "cx"); sw = sum(1 for nm, q, _ in m.ops if nm == "swap")
        print(f"{name:12s} rule={rule!s:6s} makespan {S.asap(m.ops):6.1f}  busiest q{hub} load {load[hub]:6.1f} {kinds[hub]}  | cx {cx} swaps {sw} blocks {len(m.blocks)}")
