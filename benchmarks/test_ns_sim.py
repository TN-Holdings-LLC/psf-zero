"""NS0: implementation checks of ns_sim.py (no device data)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ns_sim as N  # noqa: E402


def _cal(nq, t1=1e3, t2=1e3, e1=0.0, e2=0.0, init=0.0, ro=0.0, pairs=()):
    c = N.Calibration()
    for q in range(nq):
        c.t1[q], c.t2[q], c.e1[q], c.len1[q], c.init[q], c.p10[q], c.p01[q] = t1, t2, e1, 32e-9, init, ro, ro
    for a, b in pairs:
        c.e2[(min(a, b), max(a, b))] = e2
        c.len2[(min(a, b), max(a, b))] = 68e-9
    return c


def _ghz(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n)
    qc.rz(math.pi / 2, 0)
    qc.sx(0)
    qc.rz(math.pi / 2, 0)  # H up to phase
    for q in range(n - 1):
        qc.rz(math.pi / 2, q + 1); qc.sx(q + 1); qc.rz(math.pi / 2, q + 1)
        qc.cz(q, q + 1)
        qc.rz(math.pi / 2, q + 1); qc.sx(q + 1); qc.rz(math.pi / 2, q + 1)
    qc.barrier()
    qc.measure(range(n), range(n))
    return qc


def _ideal(qc):
    from qiskit.quantum_info import Statevector
    body = qc.remove_final_measurements(inplace=False)
    p = Statevector(body).probabilities_dict()
    return {k: v for k, v in p.items() if v > 1e-12}


def test_noiseless_gives_one():
    qc = _ghz(4)
    f, se, _, _, _ = N.simulate_xeb(qc, _ideal(qc), 4, _cal(4, pairs=[(0, 1), (1, 2), (2, 3)]), traj=4)
    assert abs(f - 1.0) < 1e-9 and se < 1e-9


def test_readout_only_matches_closed_form():
    """GHZ on 3 qubits with symmetric readout error r and no other noise: XEB in closed form."""
    qc = _ghz(3)
    r = 0.05
    f, _, _, _, _ = N.simulate_xeb(qc, _ideal(qc), 3, _cal(3, ro=r, pairs=[(0, 1), (1, 2)]), traj=2)
    p_ok = (1 - r) ** 3 + r ** 3  # 000 read as 000 or 111, likewise 111
    # ideal: 1/2 on 000 and 111; sum p_ideal p_meas = 1/2 * P(read in {000,111}) ; 2^3 sum p^2 - 1 = 3
    expect = (8 * 0.5 * p_ok - 1) / 3
    assert abs(f - expect) < 1e-9


def test_depolarizing_lowers_and_is_monotone():
    qc = _ghz(4)
    pr = [(0, 1), (1, 2), (2, 3)]
    f1 = N.simulate_xeb(qc, _ideal(qc), 4, _cal(4, e2=0.01, pairs=pr), traj=96)[0]
    f2 = N.simulate_xeb(qc, _ideal(qc), 4, _cal(4, e2=0.05, pairs=pr), traj=96)[0]
    assert 1.0 > f1 > f2 > 0.0


def test_waiting_costs_fidelity():
    """A qubit put in superposition and left waiting (ALAP: the wait sits between its two gates) loses fidelity with
    a short T2; the same circuit with long T2 does not."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2)
    qc.sx(0)  # q0 into superposition early
    for _ in range(200):
        qc.sx(1)  # q1 keeps the clock running
    qc.sx(0)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    ideal = _ideal(qc)
    long = N.simulate_xeb(qc, ideal, 2, _cal(2, t1=1e3, t2=1e3), traj=8)[0]
    short = N.simulate_xeb(qc, ideal, 2, _cal(2, t1=20e-6, t2=2e-6), traj=96)[0]
    assert abs(long - 1.0) < 1e-6 and short < 0.9


def test_alap_has_no_idle_before_first_gate():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2)
    for _ in range(300):
        qc.sx(0)
    qc.sx(1)  # q1's only gate: ALAP puts it at the end, in |0> before it
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    cal = _cal(2, t1=1e3, t2=1e3)
    cal.t1[1], cal.t2[1] = 1e-6, 1e-6  # q1 would decay a lot if it waited in superposition
    f = N.simulate_xeb(qc, _ideal(qc), 2, cal, traj=16)[0]
    assert f > 0.95


def test_measured_probs_order():
    """Clbit c measures local qubit meas_local[c]; the index is sum_c bit_c 2^c."""
    sv = np.zeros(8, dtype=complex)
    sv[0b011] = 1.0  # local qubits 0 and 1 are |1>
    p = N._measured_probs(sv, 3, {0: 2, 1: 0}, 2)  # clbit 0 <- local 2 (|0>), clbit 1 <- local 0 (|1>)
    assert p[0b10] == pytest.approx(1.0)
