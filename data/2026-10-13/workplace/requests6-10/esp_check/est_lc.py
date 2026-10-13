"""a14's estimate (gate + readout + idle) with the reduced states from a provider: 'aer' (exact, qiskit-aer density-matrix
saves on the compact circuit) or 'rule' (rule R6/16 of docs/findings/a14-lightcone-rule-2026-10-13.md). The formulas
are a14's own (_gate_params, _score_weights, idle_loss_rate_terms, idle_cost, _duration_of)."""
import numpy as np

K_2Q, L_CONE = 6, 16
_P = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
      "Y": np.array([[0, -1j], [1j, 0]], dtype=complex), "Z": np.diag([1, -1]).astype(complex)}
SKIP = ("barrier", "measure", "delay", "reset")


def _ops(circ):
    pos = {q: i for i, q in enumerate(circ.qubits)}
    return [(ins.operation, tuple(pos[q] for q in ins.qubits)) for ins in circ.data]


def _events(ops):
    """indices of instructions whose reduced states the estimate needs (gates on 1-2 qubits, not rz/id)."""
    return [i for i, (op, qs) in enumerate(ops) if op.name not in SKIP and op.name not in ("rz", "id") and len(qs) <= 2]


def states_aer(circ):
    """{instruction index: reduced density matrix of its qubits (qs[0] most significant)} exactly."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    ops = _ops(circ)
    used = sorted({q for op, qs in ops if op.name not in ("barrier",) for q in qs})
    idx = {q: i for i, q in enumerate(used)}
    c = QuantumCircuit(len(used))
    ev = set(_events(ops))
    for i, (op, qs) in enumerate(ops):
        if op.name in SKIP:
            continue
        c.append(op, [idx[q] for q in qs])
        if i in ev:
            c.save_density_matrix(qubits=[idx[q] for q in qs], label=f"e{i}")
    sim = AerSimulator(method="statevector", enable_truncation=False)
    data = sim.run(c, shots=1).result().data()
    out = {}
    for i in ev:
        r = np.asarray(data[f"e{i}"])
        if len(ops[i][1]) == 2:  # Aer: first listed qubit is the least significant -> swap to qs[0] most significant
            r = r.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
        out[i] = r
    return out


def _simulate(sub_ops, qubits):
    idx = {q: i for i, q in enumerate(qubits)}
    m = len(qubits)
    psi = np.zeros(2 ** m, dtype=complex); psi[0] = 1
    psi = psi.reshape([2] * m)
    for op, qs in sub_ops:
        k = len(qs)
        U = op.to_matrix().reshape([2] * (2 * k))
        ax = [idx[qs[j]] for j in reversed(range(k))]
        psi = np.tensordot(U, psi, axes=(list(range(k, 2 * k)), ax))
        psi = np.moveaxis(psi, list(range(k)), ax)
    return psi, idx


def _rho(psi, idx, qs):
    k = len(qs)
    v = np.moveaxis(psi, [idx[q] for q in qs], list(range(k))).reshape(2 ** k, -1)
    return v @ v.conj().T


def states_rule(circ, K=K_2Q, L=L_CONE):
    """{instruction index: reduced density matrix or None (= maximally mixed)} by rule R(K, L)."""
    ops = _ops(circ)
    count = {}
    cnt_at = {}
    for i, (op, qs) in enumerate(ops):
        if op.name not in SKIP and len(qs) == 2:
            for q in qs:
                count[q] = count.get(q, 0) + 1
        cnt_at[i] = {q: count.get(q, 0) for q in qs}
    out, stats = {}, {"exact": 0, "mixed_count": 0, "mixed_cone": 0}
    for g in _events(ops):
        qs = ops[g][1]
        if any(cnt_at[g][q] > K for q in qs):
            out[g] = None; stats["mixed_count"] += 1; continue
        Q, inc = set(qs), [g]
        for i in range(g - 1, -1, -1):
            op, q2 = ops[i]
            if op.name in SKIP:
                continue
            if Q.intersection(q2):
                Q |= set(q2); inc.append(i)
                if len(Q) > L:
                    break
        if len(Q) > L:
            out[g] = None; stats["mixed_cone"] += 1; continue
        psi, idx = _simulate([ops[i] for i in sorted(inc)], sorted(Q))
        out[g] = _rho(psi, idx, qs); stats["exact"] += 1
    return out, stats


def estimate(circ, target, states, a14):
    """(gate+readout, idle, makespan) from the reduced states (None = maximally mixed)."""
    ops = _ops(circ)
    acc = {}
    st1 = {}
    avail, state1, measured, A, B = {}, {}, set(), {}, {}
    used = sorted({q for op, qs in ops for q in qs})
    for q in used:
        avail[q] = 0.0; state1[q] = (0.0, 1.0); A[q] = 0.0; B[q] = 0.0

    def charge(q, t):
        if t > 0.0 and q not in measured:
            a, b = a14.idle_loss_rate_terms(*state1[q]); A[q] += t * a; B[q] += t * b

    for i, (op, qs) in enumerate(ops):
        name = op.name
        if name == "measure":
            acc.setdefault(("measure", (qs[0],)), [1.0, [[0.0, 0.0]]])
        if not qs:
            continue
        s = max(avail[q] for q in qs)
        for q in qs:
            charge(q, s - avail[q]); avail[q] = s
        if name == "barrier":
            continue
        d = a14._duration_of(target, name, qs)
        if name == "measure":
            for q in qs:
                measured.add(q); avail[q] = s + d
            continue
        if name in ("delay", "reset"):
            raise ValueError("not used in HW1")
        if i in states:
            r = states[i]
            k = len(qs)
            if k == 1:
                ev = {p: (0.0 if r is None else float(np.real(np.trace(r @ _P[p])))) for p in "XYZ"}
                ent = acc.setdefault((name, qs), [0.0, [[0.0, 0.0]]])
                ent[0] += sum(1 - ev[p] ** 2 for p in "XYZ")
                ent[1][0][0] += (1 - ev["X"] ** 2) + (1 - ev["Y"] ** 2)
                ent[1][0][1] += 1 - ev["Z"] ** 2
                st1[qs[0]] = None if r is None else r
            else:
                ent = acc.setdefault((name, qs), [0.0, [[0.0, 0.0], [0.0, 0.0]]])
                for a in "IXYZ":
                    for b in "IXYZ":
                        if a == b == "I":
                            continue
                        e = 0.0 if r is None else float(np.real(np.trace(r @ np.kron(_P[a], _P[b]))))
                        ent[0] += 1 - e * e
                        if b == "I":
                            ent[1][0][0 if a in "XY" else 1] += 1 - e * e
                        if a == "I":
                            ent[1][1][0 if b in "XY" else 1] += 1 - e * e
            for j, q in enumerate(qs):
                if r is None:
                    state1[q] = (0.0, 0.0)
                else:
                    rq = r if k == 1 else (r.reshape(2, 2, 2, 2).trace(axis1=1, axis2=3) if j == 0
                                           else r.reshape(2, 2, 2, 2).trace(axis1=0, axis2=2))
                    state1[q] = (float(abs(rq[0, 1]) ** 2), float(np.real(rq[0, 0] - rq[1, 1])))
        for q in qs:
            avail[q] = s + d
    T = max(avail.values())
    for q in used:
        charge(q, T - avail[q])
    ident = {q: q for q in used}
    gate = a14._score_weights(acc, target, ident)
    idle = a14.idle_cost({q: (A[q], B[q]) for q in used}, target, ident)
    return gate, idle, T
