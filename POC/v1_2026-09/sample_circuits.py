"""sample_circuits.py -- PSF-Zero POC kit, sample circuit builders.

Three named circuit families, matching what the papers and the main
project actually measured -- not new constructions invented for this
kit. Each builder is deterministic (fixed seed) so repeated runs are
directly comparable.

Import these into your own scripts:

    from sample_circuits import dense_pair_blocks, saturated_grid

Or run this file directly to see each family's own basic shape:

    python sample_circuits.py
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary


def dense_pair_blocks(n_qubits: int, gates_per_pair: int = 20,
                      seed: int = 0) -> QuantumCircuit:
    """The circuit family used throughout both papers' own gate-synthesis
    and layout timing measurements: disjoint adjacent-qubit pairs, each
    carrying `gates_per_pair` random SU(4) blocks in sequence. This is
    the family PSF-Zero's own gate synthesis is designed for -- deep,
    same-qubit-pair 2-qubit chains (Trotterized Hamiltonian simulation,
    QAOA-style layered entanglers fit this shape naturally).
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    for (a, b) in pairs:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def saturated_grid(rows: int, cols: int, seed: int = 0) -> tuple[QuantumCircuit, "CouplingMap"]:
    """A fully-saturated (spare=0) square-grid instance: every physical
    qubit used by exactly one interacting pair. This is the specific
    regime Paper 1 characterizes -- the point where Qiskit's own
    VF2Layout has a sharp, reproducible failure region. Returns
    (circuit, coupling_map).
    """
    from qiskit.transpiler import CouplingMap

    n = rows * cols
    cm = CouplingMap.from_grid(rows, cols)
    pairs = [(i, i + 1) for i in range(0, n - 1, 2)]

    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)
    for (a, b) in pairs:
        u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
        qc.append(UnitaryGate(u), [a, b])
    return qc, cm


def chain_shaped(n_qubits: int, n_bare_edges: int, dominant_size: int,
                 seed: int = 0) -> QuantumCircuit:
    """The instance family that exposed the layout-search guard defect
    (Paper 2, Section 3): `n_bare_edges` disjoint pairs plus one long
    path of `dominant_size` qubits, sized to exactly saturate the
    remaining qubits. Requires `n_qubits - 2*n_bare_edges - dominant_size`
    to be 0 or >= 2.
    """
    remaining = n_qubits - 2 * n_bare_edges
    filler = remaining - dominant_size
    if filler < 0 or (0 < filler < 2):
        raise ValueError(
            f"invalid sizing: n_qubits={n_qubits}, n_bare_edges={n_bare_edges}, "
            f"dominant_size={dominant_size} leaves filler={filler}"
        )
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    q = 0
    for _ in range(n_bare_edges):
        u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
        qc.append(UnitaryGate(u), [q, q + 1])
        q += 2
    for i in range(dominant_size - 1):
        u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
        qc.append(UnitaryGate(u), [q + i, q + i + 1])
    q += dominant_size
    if filler > 0:
        for i in range(filler - 1):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [q + i, q + i + 1])
    return qc


if __name__ == "__main__":
    print("Sample circuit families available in this kit:")
    print()

    qc1 = dense_pair_blocks(15, gates_per_pair=20)
    print(f"dense_pair_blocks(15, 20)  -> {qc1.num_qubits} qubits, "
          f"{sum(1 for i in qc1.data if len(i.qubits) == 2)} 2-qubit blocks")

    qc2, cm2 = saturated_grid(6, 6)
    print(f"saturated_grid(6, 6)       -> {qc2.num_qubits} qubits, "
          f"{sum(1 for i in qc2.data if len(i.qubits) == 2)} 2-qubit blocks, "
          f"spare=0 on a 6x6 coupling map")

    qc3 = chain_shaped(64, n_bare_edges=17, dominant_size=10)
    print(f"chain_shaped(64, 17, 10)   -> {qc3.num_qubits} qubits, "
          f"{sum(1 for i in qc3.data if len(i.qubits) == 2)} 2-qubit blocks "
          f"(17 disjoint pairs + one 10-qubit chain + filler)")

    print()
    print("See quickstart.py and compare.py for how these are used.")
