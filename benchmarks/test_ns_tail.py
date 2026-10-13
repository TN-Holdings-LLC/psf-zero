"""Checks of ns_tail.py: without a window before the measurement it equals ns_sim2.py exactly; with one, the window is
charged with the right length and channel; ns_sim2 is left as it was after a call."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ns_sim2 as N2  # noqa: E402
import ns_tail as NT  # noqa: E402
from test_ns_sim import _cal, _ghz, _ideal  # noqa: E402


def _early_finish():
    """q0 is put into |1> and then waits while q1 runs 150 sx gates; both measured at the end."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    for _ in range(150):
        qc.sx(1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    return qc


def _symmetric():
    """Both qubits end with a gate at the same time (no window before the measurement in either schedule); the ideal
    output is not uniform."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2)
    qc.sx(0)
    qc.x(1)
    qc.cz(0, 1)
    qc.sx(0)
    qc.x(1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    return qc


def test_no_window_equals_ns_sim2():
    qc = _symmetric()
    cal = _cal(2, t1=80e-6, t2=60e-6, e1=3e-4, e2=0.01, init=0.01, ro=0.02, pairs=[(0, 1)])
    for m in ("alap", "asap"):
        win, _ = NT.tail_windows(qc, cal, m)
        a = N2.simulate_xeb(qc, _ideal(qc), 2, cal, traj=32, method=m)
        b = NT.simulate_xeb(qc, _ideal(qc), 2, cal, traj=32, method=m)
        assert win == {} and a[:2] == b[:2]


def test_alap_still_has_a_window():
    # the review's point: in a GHZ chain q0's last gate (its CZ) cannot move to the end even as late as possible
    qc = _ghz(4)
    cal = _cal(4, pairs=[(0, 1), (1, 2), (2, 3)])
    assert 0 in NT.tail_windows(qc, cal, "alap")[0]


def test_window_length_and_restore():
    qc = _early_finish()
    cal = _cal(2, t1=50e-6, t2=40e-6)
    win, total = NT.tail_windows(qc, cal, "asap")
    d1 = cal.len1.get(0, 32e-9)
    assert set(win) == {0} and abs(total - 150 * cal.len1.get(1, 32e-9)) < 1e-15 and abs(win[0] - (total - d1)) < 1e-15
    NT.simulate_xeb(qc, _ideal(qc), 2, cal, traj=4, method="asap")
    assert N2.noisy_circuit is NT._NOISY


def test_window_charged_as_t1_decay():
    # q0 in |1> waits for w: P(1) on q0 drops by about w / T1 compared with ns_sim2 (which charges nothing there)
    from qiskit.quantum_info import DensityMatrix
    qc = _early_finish()
    cal = _cal(2, t1=50e-6, t2=40e-6)
    w = NT.tail_windows(qc, cal, "asap")[0][0]
    p1 = []
    for f in (N2.noisy_circuit, NT.noisy_circuit_tail):
        c, active, meas, _ = f(qc, cal, "asap")
        rho = DensityMatrix.from_label("00").evolve(c)
        p1.append(float(np.real(rho.probabilities([active.index(0)])[1])))
    assert abs((p1[0] - p1[1]) / p1[0] - (1 - np.exp(-w / 50e-6))) < 1e-9


def test_tail_lowers_f():
    qc = _early_finish()
    cal = _cal(2, t1=50e-6, t2=40e-6)
    f0 = N2.simulate_xeb(qc, _ideal(qc), 2, cal, traj=64, method="asap")[0]
    f1 = NT.simulate_xeb(qc, _ideal(qc), 2, cal, traj=64, method="asap")[0]
    assert f1 < f0
