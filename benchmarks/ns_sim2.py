"""ns_sim2.py -- ns_sim.py (NS1/NS2, unchanged and kept) with three options, for HW2's additional predictors (Addendum 466):

- `method`: "asap" or "alap" (as in ns_sim.py);
- `idle_t2_factor`: the T2 used for relaxation while a qubit waits between gates is this factor times the reported T2
  (capped at 2 T1); gates keep the reported T2. The reported T2 is a Hahn-echo time; a waiting qubit without
  decoupling is not echoed (workplace literature note, 2026-10-13);
- `init_on_measured`: False drops the preparation error on measured qubits, whose reported P(1|0) already includes
  it (workplace review of ns_sim.py, 2026-10-13, point 3).

What follows is ns_sim.py's description.

ns_sim.py -- a calibrated noise simulation of compiled circuits, as a predictor of a real device's XEB fidelity.

The question (Addendum 461's NS tests): HW1 (Addendum 455) found that ESP, the product of the reported gate and readout
fidelities, does not rank PSF-Zero against Qiskit level 3 on a real device. Does a standard Markovian noise model
built from the same day's calibration -- gate errors, T1/T2 relaxation during gates and while waiting, state
preparation and readout errors -- predict the device's fidelities and their ranking? If it does, it can replace ESP
in PSF-Zero's choices; if it does not, what decides fidelity on the device lies outside time and the reported error
rates (crosstalk between simultaneous gates, coherent errors left untwirled, drift).

The model, fixed before it is run on any device data (its form is not fitted):

- **Schedule.** The compiled circuit's gates on its active qubits, as late as possible (ALAP) before a common
  measurement at the end (the circuits end with a barrier and measure every logical qubit together). Durations: each
  qubit's single-qubit gate length, each coupler's CZ length, rz 0.
- **Single-qubit gates (sx, x):** the gate, then thermal relaxation for its duration (T1, T2 of its qubit; T2 capped
  at 2 T1), then depolarizing noise making up the rest of the reported error: lambda = 2 max(0, e - e_relax), with
  e_relax the relaxation channel's average infidelity.
- **CZ:** the gate, then relaxation of both qubits for its duration, then two-qubit depolarizing noise with
  lambda = (4/3) max(0, e - e_relax).
- **Waiting:** between two gates on a qubit, relaxation for the gap. Before a qubit's first gate it is in |0> and
  nothing is applied; with ALAP no qubit waits after its last gate.
- **Preparation:** an X with the qubit's reported initialisation error, before everything.
- **Readout:** each measured qubit's reported P(1|0) and P(0|1), applied to the output distribution.
- **Estimate:** quantum trajectories (Aer statevector; every channel an explicit Kraus instruction) with fixed
  seeds: at least 64, then more until the standard error is at most max(0.01, 0.05 F), 3,000 trajectories or 40
  minutes per circuit. For each trajectory the
  exact output distribution is computed, readout errors are applied, and its normalised linear XEB against the ideal
  distribution is taken: F_t = (2^n sum_x p_ideal(x) p_t(x) - 1) / (2^n sum_x p_ideal(x)^2 - 1). The estimate is the
  mean over trajectories, with its standard error. (This estimates the XEB a device with this noise would give in the
  limit of many shots; it avoids sampling noise.)

Calibrations are read either from an IBM dashboard CSV export (HW1) or from a JSON snapshot of a backend's properties
(`ns_snapshot.py`, HW2)."""
from __future__ import annotations

import csv
import functools
import json
import math

import numpy as np

TRAJ_MIN, TRAJ_MAX, WALL_CAP_S = 64, 3000, 2400.0  # trajectories: the stopping rule of simulate_xeb
SEED = 461_000
MISSING_T_S = 1e-6  # a qubit without a reported T1 or T2: 1 us (as HW1's estimate and HW2's)
BATCH_AMPS = 2 ** 25  # amplitudes kept in memory at once (per-shot statevectors)


# ------------------------------------------------------------------ calibrations

class Calibration:
    """Per-qubit and per-coupler numbers in SI units (seconds)."""

    def __init__(self):
        self.t1, self.t2, self.e1, self.len1, self.init = {}, {}, {}, {}, {}
        self.p10, self.p01 = {}, {}  # P(read 1 | prepared 0), P(read 0 | prepared 1)
        self.e2, self.len2 = {}, {}  # keyed by the sorted pair

    def t1_of(self, q):
        return self.t1.get(q) or MISSING_T_S

    def t2_of(self, q):
        return min(self.t2.get(q) or MISSING_T_S, 2 * self.t1_of(q))


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def calibration_from_csv(path):
    """IBM dashboard export (one row per qubit; couplers as 'other:value;...')."""
    c = Calibration()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        q = int(r["Qubit"])
        t1, t2 = _f(r["T1 (us)"]), _f(r["T2 (us)"])
        c.t1[q] = t1 * 1e-6 if t1 else None
        c.t2[q] = t2 * 1e-6 if t2 else None
        c.e1[q] = _f(r["√x (sx) error"], 1.0)
        c.len1[q] = _f(r["Single-qubit gate length (ns)"], 32.0) * 1e-9
        c.init[q] = _f(r.get("Init error"), 0.0) or 0.0
        c.p10[q] = _f(r.get("Prob meas1 prep0"), _f(r["MEASURE error"], 0.5))
        c.p01[q] = _f(r.get("Prob meas0 prep1"), _f(r["MEASURE error"], 0.5))
        for part in (r.get("CZ error") or "").split(";"):
            if part:
                o, e = part.split(":")
                c.e2[tuple(sorted((q, int(o))))] = float(e)
        for part in (r.get("Gate length (ns)") or "").split(";"):
            if part:
                o, d = part.split(":")
                c.len2[tuple(sorted((q, int(o))))] = float(d) * 1e-9
    return c


def calibration_from_snapshot(path):
    """JSON written by ns_snapshot.py: {"qubits": {q: {...}}, "couplers": [[a, b, error, length_s], ...]}."""
    d = json.load(open(path, encoding="utf-8"))
    c = Calibration()
    for qs, v in d["qubits"].items():
        q = int(qs)
        c.t1[q], c.t2[q] = v.get("t1_s"), v.get("t2_s")
        c.e1[q] = v.get("sx_error") if v.get("sx_error") is not None else 1.0
        c.len1[q] = v.get("sx_length_s") or 32e-9
        c.init[q] = v.get("init_error") or 0.0
        ro = v.get("readout_error")
        c.p10[q] = v.get("prob_meas1_prep0") if v.get("prob_meas1_prep0") is not None else (ro if ro is not None else 0.5)
        c.p01[q] = v.get("prob_meas0_prep1") if v.get("prob_meas0_prep1") is not None else (ro if ro is not None else 0.5)
    for a, b, e, ln in d["couplers"]:
        k = tuple(sorted((int(a), int(b))))
        if e is not None:
            c.e2[k] = max(c.e2.get(k, 0.0), float(e))
        if ln is not None:
            c.len2[k] = float(ln)
    return c


# ------------------------------------------------------------------ channels

@functools.lru_cache(maxsize=None)
def _relax_infidelity(t1, t2, t):
    from qiskit.quantum_info import average_gate_fidelity
    from qiskit_aer.noise import thermal_relaxation_error
    if t <= 0:
        return 0.0
    return 1.0 - average_gate_fidelity(thermal_relaxation_error(t1, t2, t).to_quantumchannel())


@functools.lru_cache(maxsize=None)
def _relax2_infidelity(t1a, t2a, t1b, t2b, t):
    from qiskit.quantum_info import average_gate_fidelity
    from qiskit_aer.noise import thermal_relaxation_error
    if t <= 0:
        return 0.0
    ch = thermal_relaxation_error(t1b, t2b, t).tensor(thermal_relaxation_error(t1a, t2a, t))
    return 1.0 - average_gate_fidelity(ch.to_quantumchannel())


def _relax(cal, q, t, t2_factor=1.0):
    from qiskit_aer.noise import thermal_relaxation_error
    t1 = cal.t1_of(q)
    return thermal_relaxation_error(t1, min(cal.t2_of(q) * t2_factor, 2 * t1), t)


def _kraus(err):
    """The channel as an explicit Kraus instruction. (Aer 0.17.2 failed with 'Kraus is empty' on some circuits when the
    same channels were appended as QuantumError instructions; the Kraus form is the same channel.)"""
    from qiskit.quantum_info import Kraus
    return Kraus(err.to_quantumchannel())


# ------------------------------------------------------------------ the noisy circuit

def active_body(circ):
    """(gates [(name, params, physical qubits)], measured {clbit: physical qubit}, active physical qubits sorted)."""
    meas, body = {}, []
    for ins in circ.data:
        nm = ins.operation.name
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = circ.find_bit(ins.qubits[0]).index
        elif nm in ("barrier", "delay"):
            continue
        else:
            body.append((ins.operation, tuple(circ.find_bit(q).index for q in ins.qubits)))
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    return body, meas, active


def _duration(cal, op, qs):
    nm = op.name
    if nm in ("rz", "id"):
        return 0.0
    if len(qs) == 1:
        return cal.len1.get(qs[0], 32e-9)
    return cal.len2.get(tuple(sorted(qs)), 68e-9)


def schedule(cal, body, method="alap"):
    """Start times of the gates (seconds), and the total duration."""
    n = len(body)
    start = [0.0] * n
    if method == "asap":
        clock = {}
        for i, (op, qs) in enumerate(body):
            s = max(clock.get(q, 0.0) for q in qs)
            start[i] = s
            for q in qs:
                clock[q] = s + _duration(cal, op, qs)
        return start, max(clock.values()) if clock else 0.0
    # ALAP: the ASAP length is the total; then every gate as late as its successors allow
    _, total = schedule(cal, body, "asap")
    back = {}
    for i in range(n - 1, -1, -1):
        op, qs = body[i]
        d = _duration(cal, op, qs)
        f = min(back.get(q, total) for q in qs)
        start[i] = f - d
        for q in qs:
            back[q] = start[i]
    return start, total


def noisy_circuit(circ, cal, method="alap", idle_t2_factor=1.0, init_on_measured=True):
    """The circuit on its active qubits (local indices) with every noise channel as an instruction; returns
    (QuantumCircuit, active, measured {clbit: local index}, total duration)."""
    from qiskit import QuantumCircuit
    from qiskit_aer.noise import depolarizing_error, pauli_error
    body, meas, active = active_body(circ)
    pos = {q: i for i, q in enumerate(active)}
    start, total = schedule(cal, body, method)
    qc = QuantumCircuit(len(active))
    measured = set(meas.values())
    for q in active:
        if not init_on_measured and q in measured:
            continue
        p = cal.init.get(q, 0.0)
        if p > 0:
            qc.append(_kraus(pauli_error([("X", p), ("I", 1 - p)])), [pos[q]])
    busy_until = {}
    for (op, qs), s in zip(body, start):
        d = _duration(cal, op, qs)
        for q in qs:  # waiting since the qubit's previous gate (none before its first)
            if q in busy_until and s - busy_until[q] > 1e-12:
                qc.append(_kraus(_relax(cal, q, s - busy_until[q], idle_t2_factor)), [pos[q]])
        qc.append(op, [pos[q] for q in qs])
        if d > 0:
            if len(qs) == 1:
                q = qs[0]
                e = cal.e1.get(q, 1.0)
                er = _relax_infidelity(cal.t1_of(q), cal.t2_of(q), d)
                qc.append(_kraus(_relax(cal, q, d)), [pos[q]])
                lam = min(2.0 * max(0.0, e - er), 4.0 / 3.0)
                if lam > 0:
                    qc.append(_kraus(depolarizing_error(lam, 1)), [pos[q]])
            else:
                a, b = qs
                e = cal.e2.get(tuple(sorted(qs)), 1.0)
                er = _relax2_infidelity(cal.t1_of(a), cal.t2_of(a), cal.t1_of(b), cal.t2_of(b), d)
                qc.append(_kraus(_relax(cal, a, d)), [pos[a]])
                qc.append(_kraus(_relax(cal, b, d)), [pos[b]])
                lam = min((4.0 / 3.0) * max(0.0, e - er), 16.0 / 15.0)
                if lam > 0:
                    qc.append(_kraus(depolarizing_error(lam, 2)), [pos[a], pos[b]])
        for q in qs:
            busy_until[q] = s + d
    return qc, active, {c: pos[q] for c, q in meas.items()}, total


# ------------------------------------------------------------------ the estimate

def _ideal_array(dist, nclb):
    arr = np.zeros(2 ** nclb)
    for k, v in dist.items():
        arr[int(k.replace(" ", ""), 2)] = v
    return arr


def _measured_probs(sv, nact, meas_local, nclb):
    """Probabilities over the classical bits (index sum_c bit_c 2^c) from a statevector on nact local qubits."""
    p = np.abs(np.asarray(sv)) ** 2
    p = p.reshape([2] * nact)  # axis k is local qubit nact - 1 - k
    keep = [meas_local[c] for c in range(nclb)]
    drop = tuple(nact - 1 - q for q in range(nact) if q not in keep)
    if drop:
        p = p.sum(axis=drop)
    rem = [q for q in range(nact) if q in keep]  # remaining axes, in decreasing local index order
    rem_axes = sorted(rem, reverse=True)
    order = [rem_axes.index(meas_local[c]) for c in range(nclb - 1, -1, -1)]  # clbit nclb-1 first
    return np.transpose(p, order).reshape(-1)


def _apply_readout(p, nclb, conf):
    """conf[c] = (p10, p01) for clbit c; p indexed sum_c bit_c 2^c."""
    t = p.reshape([2] * nclb)  # axis k is clbit nclb - 1 - k
    for c in range(nclb):
        p10, p01 = conf[c]
        m = np.array([[1 - p10, p01], [p10, 1 - p01]])
        t = np.moveaxis(np.tensordot(m, t, axes=([1], [nclb - 1 - c])), 0, nclb - 1 - c)
    return t.reshape(-1)


def simulate_xeb(circ, ideal_dist, n_logical, cal, traj=None, seed=SEED, method="alap", traj_min=TRAJ_MIN,
                 traj_max=TRAJ_MAX, wall_cap_s=WALL_CAP_S, idle_t2_factor=1.0, init_on_measured=True):
    """(F estimate, standard error, total duration in s, active qubits, trajectories used). With traj given, exactly
    that many trajectories; otherwise at least traj_min, then more in batches until the standard error is at most
    max(0.01, 0.05 |F|), traj_max trajectories are reached, or wall_cap_s seconds have passed (fixed stopping rule;
    it looks only at the simulation itself)."""
    import time
    from qiskit_aer import AerSimulator
    qc, active, meas_local, total = noisy_circuit(circ, cal, method, idle_t2_factor, init_on_measured)
    nclb, nact = circ.num_clbits, len(active)
    ideal = _ideal_array(ideal_dist, nclb)
    s2 = float(np.dot(ideal, ideal))
    den = (2 ** n_logical) * s2 - 1.0
    conf = {c: (cal.p10.get(active[meas_local[c]], 0.0), cal.p01.get(active[meas_local[c]], 0.0)) for c in range(nclb)}
    qc.save_statevector(pershot=True)
    sim = AerSimulator(method="statevector")
    batch = max(1, min(64, BATCH_AMPS // (2 ** nact)))
    vals = []
    k = 0
    t0 = time.time()
    while True:
        if traj is not None:
            if len(vals) >= traj:
                break
            b = min(batch, traj - len(vals))
        else:
            if len(vals) >= traj_min:
                m = float(np.mean(vals))
                se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                if se <= max(0.01, 0.05 * abs(m)) or len(vals) >= traj_max or time.time() - t0 > wall_cap_s:
                    break
            b = batch
        res = sim.run(qc, shots=b, seed_simulator=seed + k).result()
        k += 1
        for sv in res.data(0)["statevector"]:
            p = _apply_readout(_measured_probs(sv, nact, meas_local, nclb), nclb, conf)
            vals.append(((2 ** n_logical) * float(np.dot(ideal, p)) - 1.0) / den)
    v = np.array(vals)
    se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return float(v.mean()), se, total, nact, len(v)
