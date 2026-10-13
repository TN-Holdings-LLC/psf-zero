"""Checks of ns_sim2.py: its defaults give ns_sim.py's numbers; the options act as described."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ns_sim as N1  # noqa: E402
import ns_sim2 as N2  # noqa: E402
from test_ns_sim import _cal, _ghz, _ideal  # noqa: E402


def _wait_circuit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2)
    qc.sx(0)  # q0 goes into superposition, then must wait for q1 before the CZ (in both schedules)
    for _ in range(150):
        qc.sx(1)
    qc.cz(0, 1)
    qc.sx(0)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    return qc


def test_defaults_equal_ns_sim():
    qc = _ghz(4)
    pr = [(0, 1), (1, 2), (2, 3)]
    c1 = _cal(4, t1=80e-6, t2=60e-6, e1=3e-4, e2=0.01, init=0.01, ro=0.02, pairs=pr)
    c2 = _cal(4, t1=80e-6, t2=60e-6, e1=3e-4, e2=0.01, init=0.01, ro=0.02, pairs=pr)
    for m in ("alap", "asap"):
        a = N1.simulate_xeb(qc, _ideal(qc), 4, c1, traj=32, method=m)
        b = N2.simulate_xeb(qc, _ideal(qc), 4, c2, traj=32, method=m)
        assert abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12


def test_shorter_idle_t2_costs_more():
    qc = _wait_circuit()
    cal = _cal(2, t1=200e-6, t2=20e-6, pairs=[(0, 1)])
    f1 = N2.simulate_xeb(qc, _ideal(qc), 2, cal, traj=512, idle_t2_factor=1.0, method="asap")[0]
    f2 = N2.simulate_xeb(qc, _ideal(qc), 2, cal, traj=512, idle_t2_factor=0.25, method="asap")[0]
    assert f2 < f1


def test_init_on_measured_off_raises_f():
    qc = _ghz(3)
    cal = _cal(3, init=0.05, pairs=[(0, 1), (1, 2)])
    f_on = N2.simulate_xeb(qc, _ideal(qc), 3, cal, traj=512)[0]
    f_off = N2.simulate_xeb(qc, _ideal(qc), 3, cal, traj=8, init_on_measured=False)[0]
    assert abs(f_off - 1.0) < 1e-9 and f_on < 0.95
