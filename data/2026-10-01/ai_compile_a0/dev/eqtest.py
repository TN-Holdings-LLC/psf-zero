"""Exploratory: c2 compile_for_hardware -- equivalence (incl. final layout) and 2q vs release/c1/L3 on small circuits."""
import sys, io, contextlib, warnings, importlib.util, random
import numpy as np
sys.path.insert(0, '.')
from physeq import phys_equiv
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime import fake_provider
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
mods = [load(p, n) for p, n in zip(sys.argv[1:4], ("rel", "c1", "c2"))]
with warnings.catch_warnings():
    warnings.simplefilter("ignore"); tgt = fake_provider.FakeKingston().target if "--kingston" in sys.argv else fake_provider.FakeAuckland().target
cm = tgt.build_coupling_map(); nat = [g for g in tgt.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
tq = lambda c: sum(1 for i in c.data if len(i.qubits) == 2)
def rand_dense(n, ngates, rng):
    qc = QuantumCircuit(n)
    for _ in range(ngates):
        r = rng.random()
        if r < 0.3:
            q = rng.randrange(n); getattr(qc, rng.choice(["h", "x", "s", "t"]))(q) if rng.random() < .5 else qc.ry(rng.uniform(0, 6.28), q)
        else:
            a, b = rng.sample(range(n), 2); g = rng.choice(["cx", "cry", "crz", "cp", "swap", "cz"])
            if g == "swap" and rng.random() < 0.5:
                from qiskit.circuit.library import UnitaryGate, SwapGate
                qc.append(UnitaryGate(SwapGate().to_matrix()), [a, b])
            elif g in ("cx", "swap", "cz"): getattr(qc, g)(a, b)
            else: getattr(qc, g)(rng.uniform(0, 6.28), a, b)
    return qc
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 7)
tot = [0, 0, 0, 0]; worse = [0, 0, 0]; bad = 0; N = 0
for k in range(60):
    n = rng.choice([3, 4, 5]); qc = rand_dense(n, rng.randint(6, 20), rng)
    row = []
    for m in mods:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = m.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx", layout_search=True, seed_transpiler=0)
        ok, f = phys_equiv(qc, out)
        if not ok: bad += 1; print("NOT EQUIV", m.VERSION, k, f)
        row.append(tq(out))
    l3 = tq(transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0)); row.append(l3)
    N += 1
    for i in range(4): tot[i] += row[i]
    worse[0] += row[2] > row[0]; (print("WORSE", k, row, dict(qc.count_ops())) if row[2] > row[0] or row[1] > row[0] else None); worse[1] += row[2] > row[1]; worse[2] += row[2] > l3
print("N", N, "sum2q rel/c1/c2/L3", tot, "c2>rel", worse[0], "c2>c1", worse[1], "c2>L3", worse[2], "non-equiv", bad)
