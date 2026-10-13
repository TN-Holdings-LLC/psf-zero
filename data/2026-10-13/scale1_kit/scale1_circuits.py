"""SCALE1 circuit families (fixed with the design, 2026-10-13). Both families have a known ideal outcome at any width.

L (placement-dominated): a GHZ chain on n qubits grown from the middle outwards (cx along the chain), then a random Pauli on every qubit, then the chain's
  inverse; ideal outcome: the bitstring of the X/Y components of that Pauli layer, propagated back through the chain.
R (routing-dominated): d layers of random single-qubit Cliffords and CZs on random (non-native) pairs (each qubit in a
  pair with probability 1/2 per layer), a random Pauli layer, then the inverse of the d layers.
A barrier on each side of the Pauli layer keeps compilers from cancelling the two halves.
The ideal outcome is computed from the logical circuit with a stabilizer simulation (qiskit.quantum_info.Clifford)."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, random_clifford

SEED_BASE = {"L": 4_670_000, "R": 4_680_000}


def _pauli_layer(qc, rng, n):
    for q, p in enumerate(rng.integers(0, 4, n)):
        if p == 1:
            qc.x(q)
        elif p == 2:
            qc.y(q)
        elif p == 3:
            qc.z(q)


def family_L(n, inst):
    rng = np.random.default_rng(SEED_BASE["L"] + 1000 * n + inst)
    half = QuantumCircuit(n)  # middle-out chain: depth about n/2
    m = n // 2
    half.h(m)
    for k in range(1, n):
        lo, hi = m - k, m + k
        if lo >= 0:
            half.cx(lo + 1, lo)
        if hi < n:
            half.cx(hi - 1, hi)
        if lo < 0 and hi >= n:
            break
    return _mirror(half, rng, n)


def family_R(n, d, inst):
    rng = np.random.default_rng(SEED_BASE["R"] + 10_000 * n + 100 * d + inst)
    half = QuantumCircuit(n)
    for _ in range(d):
        for q in range(n):
            half.append(random_clifford(1, seed=int(rng.integers(1 << 30))).to_instruction(), [q])
        perm = rng.permutation(n)
        for i in range(0, n - 1, 2):
            if rng.random() < 0.5:
                half.cz(int(perm[i]), int(perm[i + 1]))
    return _mirror(half, rng, n)


def _mirror(half, rng, n):
    qc = QuantumCircuit(n)
    qc.compose(half, inplace=True)
    qc.barrier()
    _pauli_layer(qc, rng, n)
    qc.barrier()
    qc.compose(half.inverse(), inplace=True)
    return qc


def ideal_bitstring(qc):
    """The deterministic outcome of the (unmeasured, Clifford) logical circuit from |0...0>, as a bitstring with qubit 0
    rightmost; raises if the outcome is not deterministic."""
    cl = Clifford(qc.remove_final_measurements(inplace=False) if qc.num_clbits else qc)
    from qiskit.quantum_info import StabilizerState
    st = StabilizerState(cl)
    probs = st.probabilities_dict()
    if len(probs) != 1:
        raise ValueError("outcome not deterministic")
    return next(iter(probs))
