"""Prototype: state-aware first-order infidelity estimate of a routed circuit from a Target (T1/T2, gate error, duration)."""
import math
import numpy as np

_P = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
      "Y": np.array([[0, -1j], [1j, 0]], dtype=complex), "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def _thermal_pauli(qp, q, dur):
    p = qp[q] if qp and q < len(qp) else None
    t1 = getattr(p, "t1", None) if p is not None else None
    if not t1 or not dur:
        return 0.0, 0.0, 0.0
    t2 = getattr(p, "t2", None)
    t2 = min(t2, 2 * t1) if t2 else 2 * t1
    l1, l2 = math.exp(-dur / t1), math.exp(-dur / t2)
    px = (1 - l1) / 4.0
    pz = max(0.0, (1 - l2) / 2.0 - px)
    return px, px, pz


def _fpro(qp, q, dur):
    px, py, pz = _thermal_pauli(qp, q, dur)
    return 1 - (px + py + pz)  # process fidelity of the twirled single-qubit thermal channel


def state_aware_cost(circ, target):
    """First-order loss of global fidelity, summing over every gate's Pauli error components the probability times
    (1 - <P>^2) evaluated on the ideal state right after the gate (Pauli-twirled thermal relaxation per qubit plus
    the depolarizing remainder up to the reported gate error)."""
    qp = getattr(target, "qubit_properties", None)
    used = sorted({circ.find_bit(q).index for inst in circ.data for q in inst.qubits})
    idx = {q: i for i, q in enumerate(used)}
    m = len(used)
    psi = np.zeros(2 ** m, dtype=complex)
    psi[0] = 1.0
    psi = psi.reshape([2] * m)  # axis k <-> used[k]
    total = 0.0
    for inst in circ.data:
        name = inst.operation.name
        qs = [circ.find_bit(q).index for q in inst.qubits]
        if name in ("barrier", "measure", "delay"):
            continue
        U = inst.operation.to_matrix()
        ax = [idx[q] for q in qs]
        k = len(ax)
        # Qiskit little-endian: matrix index bit j <-> qs[j]; tensor axes ordered (qs[k-1], ..., qs[0])
        Ut = U.reshape([2] * (2 * k))
        in_axes = [ax[j] for j in reversed(range(k))]
        psi = np.tensordot(Ut, psi, axes=(list(range(k, 2 * k)), in_axes))
        psi = np.moveaxis(psi, list(range(k)), in_axes)
        if name in ("rz", "id"):
            continue
        try:
            props = target[name].get(tuple(qs))
        except KeyError:
            props = None
        if props is None:
            continue
        dur = getattr(props, "duration", None) or 0.0
        rep = getattr(props, "error", None) or 0.0
        d = 2 ** k
        fp_th = 1.0
        for q in qs:
            fp_th *= _fpro(qp, q, dur)
        e_th = 1 - (d * fp_th + 1) / (d + 1)
        e_dep = max(0.0, rep - e_th)
        p_dep_each = e_dep * (d + 1) / d / (d * d - 1)
        # reduced density matrix of the gate's qubits (ordered as qs)
        axes_keep = [idx[q] for q in qs]
        rest = [a for a in range(m) if a not in axes_keep]
        v = np.moveaxis(psi, axes_keep, list(range(k))).reshape(2 ** k, -1)
        rho = v @ v.conj().T  # index bits: first axis = qs[0] is the MOST significant here
        loss = 0.0
        if k == 1:
            for pn in ("X", "Y", "Z"):
                ev = float(np.real(np.trace(rho @ _P[pn])))
                loss += p_dep_each * (1 - ev * ev)
            px, py, pz = _thermal_pauli(qp, qs[0], dur)
            for pn, p in (("X", px), ("Y", py), ("Z", pz)):
                ev = float(np.real(np.trace(rho @ _P[pn])))
                loss += p * (1 - ev * ev)
        else:
            for a in "IXYZ":
                for b in "IXYZ":
                    if a == "I" and b == "I":
                        continue
                    P = np.kron(_P[a], _P[b])  # a on qs[0], b on qs[1]
                    ev = float(np.real(np.trace(rho @ P)))
                    loss += p_dep_each * (1 - ev * ev)
            for j, q in enumerate(qs):
                px, py, pz = _thermal_pauli(qp, q, dur)
                for pn, p in (("X", px), ("Y", py), ("Z", pz)):
                    P = np.kron(_P[pn], _P["I"]) if j == 0 else np.kron(_P["I"], _P[pn])
                    ev = float(np.real(np.trace(rho @ P)))
                    loss += p * (1 - ev * ev)
        total += loss
    return total
