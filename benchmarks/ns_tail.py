"""ns_tail.py -- ns_sim2.py (kept unchanged, Addendum 466) with the waiting after each measured qubit's last gate charged.

The workplace's review of the HW3 kit (2026-10-13, section 4) found that ns_sim.py and ns_sim2.py apply relaxation only
to gaps between two gates on the same qubit. Nothing is applied from a qubit's last gate to the common measurement at
the end of the circuit. As soon as possible (HW2-PLUS's P1 and P2), early-finishing qubits wait there, and on HW1's
circuits that window is 7-52% of all waiting time, different between the arms of a test.

This module adds that window and nothing else:

- for every measured qubit whose last gate ends before the circuit's total duration, thermal relaxation for
  (total - end of its last gate), with T2 scaled by `idle_t2_factor` as for the other waiting windows (capped at 2 T1);
- a measured qubit that has no gate stays in |0> and is not charged (relaxation leaves |0> unchanged);
- unmeasured qubits are not charged: a channel on a qubit after its last gate, when the qubit is not measured, does
  not change the measured qubits' state.

Every other part of the model, the schedule, the trajectories and the stopping rule are ns_sim2.py's own, called
unchanged: `simulate_xeb` here runs ns_sim2.simulate_xeb with ns_sim2.noisy_circuit replaced, for that call only, by
`noisy_circuit_tail`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ns_sim2 as N2  # noqa: E402

_NOISY = N2.noisy_circuit  # ns_sim2's own, kept


def tail_windows(circ, cal, method="alap"):
    """{physical qubit: seconds from the end of its last gate to the measurement} for measured qubits that have gates
    and finish before the end; and the circuit's total duration."""
    body, meas, _ = N2.active_body(circ)
    start, total = N2.schedule(cal, body, method)
    end = {}
    for (op, qs), s in zip(body, start):
        d = N2._duration(cal, op, qs)
        for q in qs:
            end[q] = max(end.get(q, 0.0), s + d)
    out = {}
    for q in sorted(set(meas.values())):
        if q in end and total - end[q] > 1e-12:
            out[q] = total - end[q]
    return out, total


def noisy_circuit_tail(circ, cal, method="alap", idle_t2_factor=1.0, init_on_measured=True):
    """ns_sim2.noisy_circuit, then relaxation on each measured qubit for its window before the measurement."""
    qc, active, meas_local, total = _NOISY(circ, cal, method, idle_t2_factor, init_on_measured)
    pos = {q: i for i, q in enumerate(active)}
    win, total2 = tail_windows(circ, cal, method)
    assert abs(total2 - total) < 1e-15
    for q, t in win.items():
        qc.append(N2._kraus(N2._relax(cal, q, t, idle_t2_factor)), [pos[q]])
    return qc, active, meas_local, total


def simulate_xeb(circ, ideal_dist, n_logical, cal, **kw):
    """ns_sim2.simulate_xeb with the window before the measurement charged (same arguments, same return value)."""
    N2.noisy_circuit = noisy_circuit_tail
    try:
        return N2.simulate_xeb(circ, ideal_dist, n_logical, cal, **kw)
    finally:
        N2.noisy_circuit = _NOISY


def idle_share(circ, cal, method="asap"):
    """(uncharged window before the measurement, all waiting from first gate to the end), summed over measured qubits
    with gates, in seconds: the share the review reported."""
    body, meas, _ = N2.active_body(circ)
    start, total = N2.schedule(cal, body, method)
    first, busy = {}, {}
    for (op, qs), s in zip(body, start):
        d = N2._duration(cal, op, qs)
        for q in qs:
            first.setdefault(q, s)
            busy[q] = busy.get(q, 0.0) + d
    win, _ = tail_windows(circ, cal, method)
    mq = [q for q in set(meas.values()) if q in first]
    allw = sum(total - first[q] - busy[q] for q in mq)
    return sum(win.values()), allw
