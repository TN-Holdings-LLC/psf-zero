import sys, collections
from qiskit_ibm_runtime import fake_provider
d = sys.argv[1]; t = getattr(fake_provider, d)().target
op = "cz" if "cz" in t.operation_names else "ecr"
dur = {}; err = {}
for qa, p in t[op].items():
    e = tuple(sorted(qa)); dur[e] = max(dur.get(e, 0), round(p.duration * 1e9)); err[e] = max(err.get(e, 0), p.error or 0)
print(op, collections.Counter(dur.values()).most_common())
bad = [e for e in err if err[e] >= 0.5]; print("failed couplers", bad)
fq = [q for q in range(t.num_qubits) if (t["sx"][(q,)].error or 0) >= 0.5 or (t["measure"][(q,)].error or 0) >= 0.5]
print("failed qubits", fq)
for path in [list(map(int, a.split(","))) for a in sys.argv[2:]]:
    print([dur.get(tuple(sorted((a, b)))) for a, b in zip(path, path[1:])])
