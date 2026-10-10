"""Cross-check (prototype): random problems; the NumPy reference copies psf_compile's excitation_cost / hybrid_cost /
_apply_ops bodies (c29), with the circuit and Target replaced by plain lists; compared with the Rust kernel."""
import math
import struct
import subprocess
import sys

import numpy as np

_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
_EYE2 = np.eye(2, dtype=complex)


def _embed_1q(mats, qubits):
    if len(qubits) == 1:
        return mats.get(qubits[0], _EYE2)
    if len(qubits) == 2:
        a, b = mats.get(qubits[1], _EYE2), mats.get(qubits[0], _EYE2)
        return (a[:, None, :, None] * b[None, :, None, :]).reshape(4, 4)
    m = np.ones((1, 1), dtype=complex)
    for q in reversed(qubits):
        m = np.kron(m, mats.get(q, _EYE2))
    return m


def _rho1(psi, ax):
    m = np.moveaxis(psi, ax, 0).reshape(2, -1)
    a, b = m[0], m[1]
    c = np.vdot(b, a)
    return np.array([[np.vdot(a, a).real, c], [np.conj(c), np.vdot(b, b).real]], dtype=complex)


def _p1(rho, i):
    r = rho.get(i)
    return 0.0 if r is None else float(r[1, 1].real)


class P:  # InstructionProperties stand-in
    def __init__(self, error, duration):
        self.error, self.duration = error, duration


class Q:  # QubitProperties stand-in
    def __init__(self, t1, t2):
        self.t1, self.t2 = t1, t2


def excitation_ref(ops, qp, k):
    """ops: [(mat, q, props or None)], q = physical = positions here (active = range(k))."""
    pos = {p: p for p in range(k)}
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0
    cost, rho, pend = 0.0, {}, {}
    for mat, q, props in ops:
        if props is not None and props.error is not None:
            cost += -math.log(max(1.0 - props.error, 1e-300))
        dur = props.duration if props is not None and props.duration else 0.0
        if dur:
            for i in q:
                t1 = getattr(qp[i], "t1", None) if i < len(qp) and qp[i] is not None else None
                if t1:
                    cost += dur / t1 * _p1(rho, i)
        if len(q) == 1:
            i = q[0]
            p = pend.get(i)
            pend[i] = mat if p is None else mat @ p
            r = rho.get(i, _RHO0)
            rho[i] = mat @ r @ mat.conj().T
            continue
        pre = {x: pend.pop(x) for x in q if x in pend}
        if pre:
            mat = mat @ _embed_1q(pre, q)
        axes = [pos[i] for i in q]
        m = len(axes)
        rev = axes[::-1]
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
        for i, ax in zip(q, axes):
            rho[i] = _rho1(psi, ax)
    return cost


def hybrid_ref(ops, qp, k):
    pos = {p: p for p in range(k)}
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0

    def thermal(i, t):
        p = qp[i] if i < len(qp) else None
        t1 = getattr(p, "t1", None) if p is not None else None
        if not t or not t1:
            return None
        t2 = getattr(p, "t2", None)
        return t1, (min(t2, 2 * t1) if t2 else 2 * t1)

    cost, rho, pend = 0.0, {}, {}
    for mat, q, props in ops:
        m = len(q)
        t = (props.duration or 0.0) if props is not None else 0.0
        if props is not None:
            for i in q:
                th = thermal(i, t)
                if th:
                    cost += t / th[0] * _p1(rho, i)
        if m == 1:
            i = q[0]
            p = pend.get(i)
            pend[i] = mat if p is None else mat @ p
            rho[i] = mat @ rho.get(i, _RHO0) @ mat.conj().T
        else:
            pre = {x: pend.pop(x) for x in q if x in pend}
            if pre:
                mat = mat @ _embed_1q(pre, q)
            axes = [pos[i] for i in q]
            rev = axes[::-1]
            psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
            psi = np.moveaxis(psi, list(range(m)), rev)
            for i, ax in zip(q, axes):
                rho[i] = _rho1(psi, ax)
        if props is None:
            continue
        thermal_f = 1.0
        for i in q:
            th = thermal(i, t)
            if not th:
                continue
            t1, t2 = th
            rate = max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0)
            ez = float(rho[i][0, 0].real - rho[i][1, 1].real) if i in rho else 1.0
            cost += (1.0 - math.exp(-t * rate)) / 2.0 * (1.0 - ez * ez)
            thermal_f *= (1.0 + 2.0 * math.exp(-t / t2) + math.exp(-t / t1)) / 4.0
        d = 2 ** m
        cost += max((props.error or 0.0) - (1.0 - (d * thermal_f + 1.0) / (d + 1.0)), 0.0) * (d + 1) / d
    return cost


def apply_ref(psi, ops, k):
    pos = {p: p for p in range(k)}
    for mat, q, _ in ops:
        axes = [pos[i] for i in q]
        m = len(axes)
        rev = axes[::-1]
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
    return psi


def encode(kind, ops, qp, k, state=None):
    nan = float("nan")
    t1 = [nan if qp[i] is None or qp[i].t1 is None else qp[i].t1 for i in range(k)]
    t2 = [nan if qp[i] is None or qp[i].t2 is None else qp[i].t2 for i in range(k)]
    ms = bytes(len(q) for _, q, _ in ops)
    pos = bytes(i for _, q, _ in ops for i in q)
    hp = bytes(0 if p is None else 1 for _, _, p in ops)
    err = np.array([nan if p is None or p.error is None else p.error for _, _, p in ops], dtype="<f8")
    dur = np.array([nan if p is None or p.duration is None else p.duration for _, _, p in ops], dtype="<f8")
    mat = (np.concatenate([np.ascontiguousarray(m, dtype=complex).ravel() for m, _, _ in ops]).view("<f8")
           if ops else np.zeros(0, dtype="<f8"))
    head = struct.pack("<7I", 0x37354650, 1, kind, k, len(ops), len(pos), mat.size)
    out = head + np.array(t1, "<f8").tobytes() + np.array(t2, "<f8").tobytes() + ms + pos + hp
    out += err.tobytes() + dur.tobytes() + mat.tobytes()
    if state is not None:
        out += np.ascontiguousarray(state, dtype=complex).ravel().view("<f8").tobytes()
    return out


def rand_unitary(rng, d):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def problem(rng, k, n):
    qp = []
    for i in range(k):
        u = rng.random()
        qp.append(None if u < 0.1 else Q(None if u < 0.2 else rng.uniform(50e-6, 300e-6),
                                         None if u < 0.3 else (0.0 if u < 0.35 else rng.uniform(20e-6, 400e-6))))
    ops = []
    for _ in range(n):
        m = rng.choice([1, 1, 1, 2, 2, 3]) if k >= 3 else rng.choice([1, 2]) if k == 2 else 1
        q = list(rng.choice(k, size=m, replace=False))
        q = [int(x) for x in q]
        u = rng.random()
        props = None if u < 0.15 else P(None if u < 0.25 else rng.uniform(0, 0.02),
                                        None if u < 0.3 else (0.0 if u < 0.35 else rng.uniform(20e-9, 600e-9)))
        mat = rand_unitary(rng, 2 ** m)
        if rng.random() < 0.2 and m == 1:
            mat = np.diag(np.exp(1j * rng.uniform(0, 6.3, 2)))
        ops.append((mat, q, props))
    return ops, qp


def main(binary, n_cases):
    rng = np.random.default_rng(57_000)
    files, refs = [], []
    import os
    import tempfile
    d = tempfile.mkdtemp()
    for c in range(n_cases):
        k = int(rng.integers(1, 11))
        ops, qp = problem(rng, k, int(rng.integers(0, 60)))
        for kind, f in ((0, excitation_ref), (1, hybrid_ref)):
            path = os.path.join(d, f"c{c}_{kind}")
            open(path, "wb").write(encode(kind, ops, qp, k))
            files.append(path)
            refs.append(f(ops, qp, k))
        psi = (rng.normal(size=(2,) * k) + 1j * rng.normal(size=(2,) * k))
        psi /= np.linalg.norm(psi)
        path = os.path.join(d, f"c{c}_2")
        open(path, "wb").write(encode(2, ops, qp, k, psi))
        files.append(path)
        refs.append(apply_ref(psi, ops, k))
    out = subprocess.run([binary] + files, capture_output=True, text=True, check=True).stdout.split("\n")
    worst = {0: 0.0, 1: 0.0, 2: 0.0}
    for path, line, ref in zip(files, out, refs):
        kind = int(path[-1])
        if kind == 2:
            got = np.frombuffer(open(path + ".out", "rb").read(), dtype="<f8").view(complex).reshape(ref.shape)
            dev = float(np.max(np.abs(got - ref))) if ref.size else 0.0
        else:
            val = float(line.split()[1])
            dev = 0.0 if val == ref else abs(val - ref) / max(abs(ref), 1e-300)
        worst[kind] = max(worst[kind], dev)
    print(f"{n_cases} cases: largest relative difference excitation {worst[0]:.2e}, hybrid {worst[1]:.2e}; "
          f"largest amplitude difference after apply_ops {worst[2]:.2e}")
    return worst


if __name__ == "__main__":
    w = main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    sys.exit(0 if max(w[0], w[1]) < 1e-12 and w[2] < 1e-12 else 1)
