"""Item 99 test circuits (W1 of note 7): in-scope native-gate circuits. build(name) -> QuantumCircuit (measured)."""
import math
import random

from qiskit import QuantumCircuit

TESTS = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16",
         "ra_n8_r2", "ra_n8_r4", "ra_n12_r2", "ra_n12_r4", "ra_n16_r2", "ra_n16_r4",
         "su2_n8_r2", "su2_n8_r4", "su2_n12_r2", "su2_n12_r4", "su2_n16_r2", "su2_n16_r4",
         "clf_ring12_L4", "clf_ring12_L8")


def _hh_graph(n, seed):
    """C57-VAL's heavy-hex patch edges (n qubits of FakeFez grown by BFS from qubit 0, labels shuffled by seed)."""
    from qiskit_ibm_runtime import fake_provider
    cm = fake_provider.FakeFez().target.build_coupling_map()
    adj = {}
    for a, b in cm.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    order, seen, fr = [], {0}, [0]
    while fr and len(order) < n:
        u = fr.pop(0)
        order.append(u)
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                fr.append(v)
    S = order[:n]
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    idx = {q: perm[i] for i, q in enumerate(S)}
    return sorted({(min(idx[a], idx[b]), max(idx[a], idx[b])) for a in S for b in adj[a] if b in S})


def _ansatz_cz(name, n, edges, layers=4):
    """W4's CZ ansatz: per layer ry, rz on every qubit, then cz on every edge in the edge list's order."""
    rng = random.Random("W4|" + name)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(0.3 + 0.07 * q, q)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q)
            qc.rz(rng.uniform(-3, 3), q)
        for a, b in edges:
            qc.cz(a, b)
    qc.measure_all()
    return qc


def _library(kind, n, reps):
    from qiskit.circuit.library import efficient_su2, real_amplitudes
    f = real_amplitudes if kind == "ra" else efficient_su2
    c = f(n, entanglement="linear", reps=reps)
    rng = random.Random(f"{kind}|{n}|{reps}")
    c = c.assign_parameters([rng.uniform(-math.pi, math.pi) for _ in c.parameters])
    while any(i.operation.name not in ("ry", "rz", "cx", "rx") for i in c.data):
        c = c.decompose()
    c.measure_all()
    return c


def _classifier(n, L, seed):
    """Note 6's model (DEPTH1 / RING1): per layer RY(pi/2 x_q), RY(a) RZ(b) on every qubit, CZ ring; final RY."""
    rng = random.Random(seed)
    x = [rng.uniform(-1, 1) for _ in range(n)]
    qc = QuantumCircuit(n)
    for _ in range(L):
        for q in range(n):
            qc.ry(math.pi / 2 * x[q], q)
        for q in range(n):
            qc.ry(rng.uniform(-math.pi, math.pi), q)
            qc.rz(rng.uniform(-math.pi, math.pi), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(rng.uniform(-math.pi, math.pi), q)
    qc.measure_all()
    return qc


def build(name):
    if name.startswith("chain_cz"):
        n = int(name.split("_n")[1])
        return _ansatz_cz(name, n, [(q, q + 1) for q in range(n - 1)])
    if name.startswith("ring_cz"):
        n = int(name.split("_n")[1])
        return _ansatz_cz(name, n, [(q, q + 1) for q in range(n - 1)] + [(0, n - 1)])
    if name.startswith("hh_cz"):
        n = int(name.split("_n")[1])
        return _ansatz_cz(name, n, _hh_graph(n, 5400 + n))
    if name.startswith("cx_chain"):
        n = int(name.split("_n")[1])
        rng = random.Random("cx|" + name)
        qc = QuantumCircuit(n)
        for _ in range(4):
            for q in range(n):
                qc.ry(rng.uniform(-3, 3), q)
                qc.rz(rng.uniform(-3, 3), q)
            for q in range(n - 1):
                qc.cx(q, q + 1)
        qc.measure_all()
        return qc
    if name.startswith(("ra_", "su2_")):
        kind, n, r = name.split("_")
        return _library(kind, int(n[1:]), int(r[1:]))
    if name.startswith("clf_ring12"):
        L = int(name.split("_L")[1])
        return _classifier(12, L, 99_000 + L)
    raise KeyError(name)
