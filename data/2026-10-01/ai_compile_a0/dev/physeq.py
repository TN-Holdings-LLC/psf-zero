"""Equivalence of a routed physical circuit to its logical source, on the touched qubits only.
Virtual qubit v starts at physical init[v] and must end at physical fin[v] (TranspileLayout).
Random input states on the virtual qubits (ancillas |0>); the output, read at the final positions, must equal qc|psi>."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector, partial_trace
def phys_equiv(qc, out, trials=3, seed=1, tol=1e-9):
    n = qc.num_qubits
    lay = out.layout
    init = lay.initial_index_layout(filter_ancillas=True)[:n]
    fin = lay.final_index_layout(filter_ancillas=True)[:n]
    touched = set(init) | set(fin)
    for inst in out.data:
        for q in inst.qubits: touched.add(out.find_bit(q).index)
    others = sorted(touched - set(init))
    sidx = {p: v for v, p in enumerate(init)}
    for j, p in enumerate(others): sidx[p] = n + j
    m = n + len(others)
    small = QuantumCircuit(m); small.global_phase = out.global_phase
    for inst in out.data:
        if inst.operation.name in ("barrier", "measure", "delay"): continue
        small.append(inst.operation, [sidx[out.find_bit(q).index] for q in inst.qubits])
    # move virtual v's final position to small index v with explicit swaps
    where = [sidx[fin[v]] for v in range(n)]   # current small index holding virtual v
    occupant = {i: None for i in range(m)}
    for v, i in enumerate(where): occupant[i] = v
    for v in range(n):
        i = where[v]
        if i != v:
            small.swap(i, v)
            u = occupant[v]
            occupant[v], occupant[i] = v, u
            where[v] = v
            if u is not None: where[u] = i
    rng = np.random.default_rng(seed); worst = 1.0
    for _ in range(trials):
        psi = random_statevector(2 ** n, seed=int(rng.integers(1 << 30)))
        phi = psi.evolve(qc)
        full = Statevector.from_label("0" * (m - n)).tensor(psi) if m > n else psi
        rho = partial_trace(full.evolve(small), list(range(n, m))) if m > n else full.evolve(small).to_operator()
        rd = rho.data if m > n else np.outer(full.evolve(small).data, full.evolve(small).data.conj())
        worst = min(worst, float(np.real(phi.data.conj() @ rd @ phi.data)))
    return worst > 1 - tol, worst
