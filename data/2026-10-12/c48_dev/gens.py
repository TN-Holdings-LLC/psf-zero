import os
"""Test-circuit generators of c47's breakdown (tfim, heis, heis2d, chem, qaoa3reg) and Benchpress files (bp:)."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import random
from qiskit import QuantumCircuit
root = os.environ.get("BENCHPRESS_QASM", "benchpress/qasm/")
def gen(name):
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter, SuzukiTrotter
    kind, n, reps = name.split("_"); n = int(n[1:]); reps = int(reps[1:])
    rng = random.Random("C47-DEV|" + name)
    def lab(d):
        l = ["I"] * n
        for q, p in d.items(): l[n - 1 - q] = p
        return "".join(l)
    if kind == "tfim":
        terms = [(lab({q: "Z", (q + 1) % n: "Z"}), rng.uniform(-1, 1)) for q in range(n)] + [(lab({q: "X"}), rng.uniform(-1, 1)) for q in range(n)]
    elif kind == "heis":
        terms = [(lab({q: p, (q + 1) % n: p}), rng.uniform(-1, 1)) for q in range(n) for p in "XYZ"]
    elif kind == "heis2d":
        L = int(round(n ** 0.5)); terms = []
        for x in range(L):
            for y in range(L):
                q = x * L + y
                for nb in (((x + 1) % L) * L + y, x * L + (y + 1) % L):
                    for p in "XYZ": terms.append((lab({q: p, nb: p}), rng.uniform(-1, 1)))
    elif kind == "chem":  # random weight-2..6 strings, like a fermionic Hamiltonian under Jordan-Wigner
        d = {}
        for _ in range(6 * n):
            k = rng.choice((2, 2, 4, 4, 6)); qs = sorted(rng.sample(range(n), k))
            d[lab({q: ("Z" if q not in (qs[0], qs[-1]) else rng.choice("XY")) for q in range(qs[0], qs[-1] + 1)})] = rng.uniform(-1, 1)
        terms = sorted(d.items())
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=0.5, synthesis=SuzukiTrotter(order=2, reps=reps) if kind != "chem" else LieTrotter(reps=reps)), range(n))
    qc.measure_all()
    return qc

def qaoa_reg(name):
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    _, n, reps = name.split("_"); n = int(n[1:]); reps = int(reps[1:])
    rng = random.Random("C47-DEV|" + name)
    while True:
        stubs = [v for v in range(n) for _ in range(3)]; rng.shuffle(stubs)
        E = set(); ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or (min(a, b), max(a, b)) in E: ok = False; break
            E.add((min(a, b), max(a, b)))
        if ok: break
    def lab(a, b):
        l = ["I"] * n; l[n - 1 - a] = "Z"; l[n - 1 - b] = "Z"; return "".join(l)
    qa = QAOAAnsatz(SparsePauliOp.from_list([(lab(a, b), 1.0) for a, b in sorted(E)]), reps=reps)
    qc = qa.assign_parameters([rng.uniform(0, 3) for _ in range(qa.num_parameters)])
    qc.measure_all()
    return qc

def build(name):
    if name.startswith("qaoa3reg"):
        return qaoa_reg(name)
    if name.startswith("bp:"):
        qc = QuantumCircuit.from_qasm_file(root + name[3:])
        if not any(i.operation.name == "measure" for i in qc.data): qc.measure_all()
        return qc
    return gen(name)

