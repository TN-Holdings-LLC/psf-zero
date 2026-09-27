"""Tests for psf_compile.py VERSION 2026-09-27.6 (Addenda 217, 220): circuits taken
from Qiskit's CX decomposer are checked by phase-aligned operator distance and
rebuilt exactly when they are not exact (changelog item 23)."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_unitary

import psf_compile as pc


def frob(u, circ):
    return pc._aligned_errors(u, circ)[2]


def canonical(a, b, c):
    qc = QuantumCircuit(2)
    qc.rxx(-2 * a, 0, 1)
    qc.ryy(-2 * b, 0, 1)
    qc.rzz(-2 * c, 0, 1)
    return Operator(qc).data


def dressed(core, seed):
    k = [random_unitary(2, seed=seed + i).data for i in range(4)]
    return np.kron(k[0], k[1]) @ core @ np.kron(k[2], k[3])


@pytest.mark.parametrize("seed", range(20))
def test_exact_rebuild_random(seed):
    u = random_unitary(4, seed=seed).data
    qc = pc._exact_rebuild(u)
    assert frob(u, qc) <= 1e-13
    assert set(qc.count_ops()) <= {"cx", "rz", "sx", "ry"}
    assert qc.count_ops().get("cx", 0) == 3


@pytest.mark.parametrize("abc", [(np.pi / 4, 0, 0), (np.pi / 4, np.pi / 4, 0), (0.3, 0.3, 0.3),
                                 (0.4, 1e-5, 0), (0.4, 0.2, 1e-6), (0.4, 0.2, 2e-7), (np.pi / 4, 0.1, 1e-9)])
def test_exact_rebuild_near_special(abc):
    for seed in (0, 10, 20):
        u = dressed(canonical(*abc), seed)
        assert frob(u, pc._exact_rebuild(u)) <= 1e-13


@pytest.mark.parametrize("abc", [(0.4, 1e-5, 0), (0.4, 0.2, 1e-6), (0.4, 0.2, 2e-7), (np.pi / 4, 0.1, 1e-9),
                                 (0.5, 0.3, 0.1)])
def test_guarded_synthesis_is_exact(abc):
    for seed in (0, 10, 20):
        u = dressed(canonical(*abc), seed)
        circ, ok = pc._guarded_cx_synthesis(u)
        assert ok
        assert frob(u, circ) <= 1e-13


def test_flag_off_restores_previous_check():
    pc.USE_EXACT_FALLBACK = False
    try:
        u = random_unitary(4, seed=5).data
        circ, ok = pc._guarded_cx_synthesis(u)
        assert ok
    finally:
        pc.USE_EXACT_FALLBACK = True


def test_brick_circuit_floor8_exact():
    """Lap 3600 of Addendum 214 part E, which lost 1.3e-7 in 2026-09-27.3."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import loop_endurance as le
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import CouplingMap
    rng_e = np.random.default_rng(7)
    target_theta = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
    target_theta[8::15] = 0.0
    target = Statevector(le.e_circuit(target_theta))
    rng_t = np.random.default_rng(202)
    for _ in range(3599):
        rng_t.normal(0.0, 0.5, le.E_NPARAMS)
    qc = le.e_circuit(target_theta + rng_t.normal(0.0, 0.5, le.E_NPARAMS))
    pc._CX_CORE_CACHE.clear()
    out = pc.compile_for_hardware(qc, coupling_map=CouplingMap.from_line(12), basis_gates=["cx", "rz", "sx", "x"],
                                  block_gate_floor=8, entangling_basis="cx", initial_layout=list(range(12)),
                                  on_unsupported="keep", seed_transpiler=0)

    def loss(c):
        return 1 - abs(target.inner(Statevector(c))) ** 2

    assert abs(loss(out) - loss(qc)) <= 1e-13


def test_brick_lap_2138_floor8_exact():
    """Lap 2138: a PSF-Zero block off by 5.9e-7 after the polish (Addendum 218);
    rerouted by changelog item 24."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import loop_endurance as le
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import CouplingMap
    rng_e = np.random.default_rng(7)
    target_theta = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
    target_theta[8::15] = 0.0
    target = Statevector(le.e_circuit(target_theta))
    rng_t = np.random.default_rng(202)
    for _ in range(2137):
        rng_t.normal(0.0, 0.5, le.E_NPARAMS)
    qc = le.e_circuit(target_theta + rng_t.normal(0.0, 0.5, le.E_NPARAMS))
    pc._CX_CORE_CACHE.clear()
    out = pc.compile_for_hardware(qc, coupling_map=CouplingMap.from_line(12), basis_gates=["cx", "rz", "sx", "x"],
                                  block_gate_floor=8, entangling_basis="cx", initial_layout=list(range(12)),
                                  on_unsupported="keep", seed_transpiler=0)

    def loss(c):
        return 1 - abs(target.inner(Statevector(c))) ** 2

    assert abs(loss(out) - loss(qc)) <= 1e-13
