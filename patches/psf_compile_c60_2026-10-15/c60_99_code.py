

# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c59 (2026-10-15): item 99, native two-qubit gate circuits on an embeddable graph built
# by formula as written -- placement by item 91's first VF2 embedding, each gate in a fixed native form, commuting
# diagonal gates reordered by item 91's edge colouring, single-qubit runs merged in closed form (ZYZ) by PSF-Zero
# itself. One fixed pass: nothing is searched, compared or simulated.
# ---------------------------------------------------------------------------------------------------------------------

import cmath  # noqa: E402  (item 99)
from qiskit.circuit import Gate, QuantumRegister  # noqa: E402  (item 99)

NATIVE_BRICKWORK = True  # item 99
NATIVE99_STATS = {"tried": 0, "not_read": 0, "degree": 0, "no_colouring": 0, "no_embedding": 0, "not_cz": 0,
                  "block_not_exact": 0, "failed_check": 0, "built": 0, "gates": {}}
_N99_TOL = 1e-12  # item 99: angles and matrix entries within this of a special value are taken as that value
_N99_BLOCK_TOL = 1e-9  # item 99: every two-qubit gate's native form is multiplied out and checked to this (up to phase)
# item 99: gates diagonal for every angle (the diagonal class is decided by the name for gates with a parameter, so
# that it never depends on the angle's value); a gate without parameters is diagonal when its matrix is
_N99_DIAG = frozenset(("id", "rz", "p", "u1", "z", "s", "sdg", "t", "tdg", "cz", "cp", "cu1", "crz", "rzz",
                       "ccz"))
_N99_PI = math.pi


def _n99_t(op):
    """Item 99: a single-qubit operation as a tuple matrix (a, b, c, d) = [[a, b], [c, d]] (closed forms for rz, ry,
    rx, p; else the operation's matrix)."""
    nm = op.name
    if nm in ("rz", "ry", "rx", "p", "u1"):
        t = float(op.params[0])
        if nm == "rz":
            return (cmath.exp(-0.5j * t), 0j, 0j, cmath.exp(0.5j * t))
        if nm in ("p", "u1"):
            return (1 + 0j, 0j, 0j, cmath.exp(1j * t))
        c, s = math.cos(t / 2), math.sin(t / 2)
        if nm == "ry":
            return (complex(c), complex(-s), complex(s), complex(c))
        return (complex(c), -1j * s, -1j * s, complex(c))
    m = op.to_matrix()
    return (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))


def _n99_mul(A, B):
    """Item 99: the tuple-matrix product A @ B."""
    a, b, c, d = A
    e, f, g, h = B
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def _n99_rz(t):
    return (cmath.exp(-0.5j * t), 0j, 0j, cmath.exp(0.5j * t))


_N99_H = (complex(1 / math.sqrt(2)), complex(1 / math.sqrt(2)), complex(1 / math.sqrt(2)), complex(-1 / math.sqrt(2)))
_N99_X = (0j, 1 + 0j, 1 + 0j, 0j)


def _n99_zyz(U):
    """Item 99: (theta, phi, lam) with U = e^{i alpha} Rz(phi) Ry(theta) Rz(lam) (as a20's `_t_zyz`)."""
    a, b, c, d = U
    sq = cmath.sqrt(a * d - b * c)
    a, c, d = a / sq, c / sq, d / sq
    if abs(c) <= 1e-15:  # diagonal: only phi + lam is defined
        return 0.0, 2 * cmath.phase(d), 0.0
    if abs(a) <= 1e-15:  # anti-diagonal: only phi - lam is defined
        return _N99_PI, 2 * cmath.phase(c), 0.0
    p11, p10 = cmath.phase(d), cmath.phase(c)
    return 2 * math.atan2(abs(c), abs(a)), p11 + p10, p11 - p10


def _n99_wrap(t):
    """Item 99: t reduced to (-pi, pi]."""
    t = math.fmod(t, 2 * _N99_PI)
    if t <= -_N99_PI:
        t += 2 * _N99_PI
    elif t > _N99_PI:
        t -= 2 * _N99_PI
    return t


def _n99_run_class(U, use_rx):
    """Item 99: how a merged single-qubit run is emitted -- (class, angles): "none" (identity up to phase), "z"
    (one rz), "half" (rz sx rz, theta = pi/2, without rx), "full" (rz sx rz sx rz, or rz rx rz with rx)."""
    theta, phi, lam = _n99_zyz(U)
    if abs(theta) < _N99_TOL:
        z = _n99_wrap(phi + lam)
        return ("none", ()) if abs(z) < _N99_TOL else ("z", (z,))
    if use_rx:
        return "full", (lam - _N99_PI / 2, theta, phi + _N99_PI / 2)
    if abs(theta - _N99_PI / 2) < _N99_TOL:
        return "half", (lam - _N99_PI / 2, phi + _N99_PI / 2)
    return "full", (lam, theta + _N99_PI, phi + _N99_PI)


def _n99_emit_run(out, p, cls, ang, use_rx):
    """Item 99: append a run's gates (`_n99_run_class`) on physical qubit p."""
    if cls == "z":
        out.rz(ang[0], p)
    elif cls == "half":
        out.rz(ang[0], p)
        out.sx(p)
        out.rz(ang[1], p)
    elif cls == "full":
        out.rz(ang[0], p)
        if use_rx:
            out.rx(ang[1], p)
        else:
            out.sx(p)
            out.rz(ang[1], p)
            out.sx(p)
        out.rz(ang[2], p)


def _n99_diag(op, nq):
    """Item 99: True if `op` is diagonal (by name for the gates of _N99_DIAG; by its matrix for a gate without
    parameters; never for any other gate with a parameter)."""
    if op.name in _N99_DIAG:
        return True
    if op.params:
        return False
    try:
        m = np.asarray(op.to_matrix())
    except Exception:  # noqa: BLE001
        return False
    return bool(np.all(np.abs(m - np.diag(np.diag(m))) < _N99_TOL))


def _n99_read(qc):
    """Item 99: ([(name, qubits, op, diagonal)], [(qubit, clbit)]) when `qc` is gates on one or two qubits (each with
    a matrix and bound parameters), barriers and final measurements; else None."""
    ops, meas, measured = [], [], set()
    for ins in qc.data:
        op = ins.operation
        nm = op.name
        if nm == "barrier":
            continue
        if getattr(op, "condition", None) is not None or getattr(op, "blocks", None) is not None:
            return None
        qs = tuple(qc.find_bit(q).index for q in ins.qubits)
        if nm == "measure":
            if qs[0] in measured:
                return None
            measured.add(qs[0])
            meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))
            continue
        if ins.clbits or measured & set(qs) or len(qs) > 2 or not isinstance(op, Gate):
            return None
        if getattr(op, "is_parameterized", lambda: False)():
            return None
        if not qs:  # a global phase: no effect on outcomes
            continue
        ops.append((nm, qs, op, _n99_diag(op, len(qs))))
    return ops, meas


def _n99_zz(q0, q1, th, frac, native_rzz):
    """Item 99: steps for RZZ(th) = exp(-i th/2 Z Z) on (q0, q1), up to a global phase. RZZ(pi) is local (Rz(pi) on
    both), so th is first reduced to t in (-pi/2, pi/2]. Then: t = 0, no two-qubit gate; |t| = pi/2, one cz (unless
    `native_rzz`); otherwise, with rzz in the Target (`frac`), one rzz(|t|) in IBM's range, an X on q0 on both sides
    when t < 0; without it, cz, rz(t), cz with H on q1 (CX = H cz H)."""
    k = round(th / _N99_PI)
    t = th - k * _N99_PI
    if t <= -_N99_PI / 2 + _N99_TOL:
        t += _N99_PI
        k -= 1
    steps = []
    if k % 2:
        steps += [("1", q0, _n99_rz(_N99_PI)), ("1", q1, _n99_rz(_N99_PI))]
    if abs(t) < _N99_TOL:
        return steps
    if frac and (native_rzz or abs(abs(t) - _N99_PI / 2) >= _N99_TOL):
        if t > 0:
            return steps + [("rzz", q0, q1, t)]
        return steps + [("1", q0, _N99_X), ("rzz", q0, q1, -t), ("1", q0, _N99_X)]
    if abs(abs(t) - _N99_PI / 2) < _N99_TOL:
        s = _N99_PI / 2 if t > 0 else -_N99_PI / 2
        return steps + [("1", q0, _n99_rz(s)), ("1", q1, _n99_rz(s)), ("cz", q0, q1)]
    return steps + [("1", q1, _N99_H), ("cz", q0, q1), ("1", q1, _N99_H), ("1", q1, _n99_rz(t)), ("1", q1, _N99_H),
                    ("cz", q0, q1), ("1", q1, _N99_H)]


def _n99_kron_steps(steps, q0, q1):
    """Item 99: the 4 x 4 matrix (qubit q0 least significant) of a list of steps on (q0, q1), for the check."""
    U = np.eye(4, dtype=complex)
    for st in steps:
        if st[0] == "1":
            a, b, c, d = st[2]
            m = np.array([[a, b], [c, d]])
            U = (np.kron(np.eye(2), m) if st[1] == q0 else np.kron(m, np.eye(2))) @ U
        elif st[0] == "cz":
            U = np.diag([1, 1, 1, -1]).astype(complex) @ U
        else:
            t = st[3]
            U = np.diag(np.exp(-0.5j * t * np.array([1, -1, -1, 1]))) @ U
    return U


def _n99_same(U, W):
    k = int(np.argmax(np.abs(U)))
    ph = U.flat[k] / W.flat[k]
    return abs(abs(ph) - 1) < _N99_BLOCK_TOL and float(np.max(np.abs(U - ph * W))) < _N99_BLOCK_TOL


def _n99_steps(nm, qs, op, diag, frac):
    """Item 99: the native form of one two-qubit gate on logical qubits qs = (q0, q1): [("1", q, 2 x 2 tuple) |
    ("cz", q0, q1) | ("rzz", q0, q1, t)] in time order, or None if no form passes the check. cz stays cz. A diagonal
    gate is Rz (x) Rz times RZZ(theta), its angles read off its four phases (closed form), RZZ by `_n99_zz`. Any other
    gate by its Weyl decomposition (Qiskit's, no specialisation): a single interaction (b = c = 0: cx, rxx, ryy, crx,
    ...) is local gates around RZZ(-2a) (XX = (H x H) ZZ (H x H)), by `_n99_zz`; otherwise Qiskit's closed-form CZ
    decomposition (KAK, two or three cz)."""
    q0, q1 = qs
    if nm == "cz":
        return [("cz", q0, q1)]
    U = np.asarray(op.to_matrix(), dtype=complex)
    if diag:
        ph = np.angle(np.diag(U))  # index x0 + 2 x1
        z0, z1 = np.array([1, -1, 1, -1]), np.array([1, 1, -1, -1])
        b0, b1, th = -0.5 * float(ph @ z0), -0.5 * float(ph @ z1), -0.5 * float(ph @ (z0 * z1))
        steps = [("1", q0, _n99_rz(b0)), ("1", q1, _n99_rz(b1))] + _n99_zz(q0, q1, th, frac, nm == "rzz")
        return steps if _n99_same(U, _n99_kron_steps(steps, q0, q1)) else None
    from qiskit.synthesis import TwoQubitWeylDecomposition
    try:
        d = TwoQubitWeylDecomposition(U, fidelity=None)
    except Exception:  # noqa: BLE001
        d = None
    if d is not None and abs(d.b) < _N99_TOL and abs(d.c) < _N99_TOL:
        t = lambda m: (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))  # noqa: E731
        steps = ([("1", q0, t(d.K2r)), ("1", q1, t(d.K2l)), ("1", q0, _N99_H), ("1", q1, _N99_H)]
                 + _n99_zz(q0, q1, -2 * d.a, frac, False)
                 + [("1", q0, _N99_H), ("1", q1, _N99_H), ("1", q0, t(d.K1r)), ("1", q1, t(d.K1l))])
        if _n99_same(U, _n99_kron_steps(steps, q0, q1)):
            return steps
    from qiskit.circuit.library import CZGate
    from qiskit.synthesis import TwoQubitBasisDecomposer
    for approximate in (True, False):
        dec = TwoQubitBasisDecomposer(CZGate(), euler_basis="ZSX")(U, approximate=approximate)
        steps = []
        for ins in dec.data:
            w = [dec.find_bit(q).index for q in ins.qubits]
            if len(w) == 2:
                steps.append(("cz", q0, q1))
            else:
                steps.append(("1", q0 if w[0] == 0 else q1, _n99_t(ins.operation)))
        if _n99_same(U, _n99_kron_steps(steps, q0, q1)):
            return steps
    return None


def _n99_layout(qc, out, phys):
    """Item 99: the output's TranspileLayout (input qubit q on physical qubit phys[q]; no routing)."""
    from qiskit.transpiler import Layout, TranspileLayout
    N = out.num_qubits
    used = set(phys)
    anc = QuantumRegister(N - qc.num_qubits, "ancilla") if N > qc.num_qubits else None
    mapping = {qc.qubits[q]: phys[q] for q in range(qc.num_qubits)}
    if anc is not None:
        rest = [p for p in range(N) if p not in used]
        mapping.update({anc[i]: p for i, p in enumerate(rest)})
    lay = Layout(mapping)
    for r in qc.qregs:
        lay.add_register(r)
    if anc is not None:
        lay.add_register(anc)
    iq = {qc.qubits[q]: q for q in range(qc.num_qubits)}
    if anc is not None:
        iq.update({anc[i]: qc.num_qubits + i for i in range(len(anc))})
    return TranspileLayout(initial_layout=lay, input_qubit_mapping=iq, final_layout=None,
                           _input_qubit_count=qc.num_qubits, _output_qubit_list=list(out.qubits))


def _native_brickwork(qc, target, max_error, seed):
    """Item 99 (workplace proposal on c59): a circuit of gates on one or two qubits whose interaction graph embeds in
    the cut device (max degree <= 3, bipartite), built by formula as written, in one fixed pass:
      1. placement: item 91's -- the first VF2 embedding (`_graph_place`), on the couplers of the device's most common
         two-qubit gate length first, then on all of item 87's cut device; qubits without a two-qubit gate take the
         cut device's free qubits in order;
      2. order: every gate at level 1 + the largest level of an earlier gate on a shared qubit that does not commute
         with it (diagonal gates commute with each other; nothing else is assumed to commute) -- exact; within a level
         the two-qubit gates by the colour of item 91's edge colouring (`_graph_colours`), then the single-qubit ones;
      3. each two-qubit gate in its native form (`_n99_steps`: cz; rzz native and folded into IBM's range when the
         Target has rzz, else cz rz cz; a diagonal or single-interaction gate through RZZ; any other by the closed-form
         CZ decomposition), every form multiplied out and checked to 1e-9;
      4. every single-qubit run merged and emitted in closed form (ZYZ): rz sx rz sx rz (rz sx rz at theta = pi/2,
         one rz when diagonal), or rz rx rz with rx in the Target;
      5. measurements as written (item 89 not applied).
    No Qiskit pass manager runs and nothing is simulated. None when not applicable (counted in NATIVE99_STATS)."""
    NATIVE99_STATS["tried"] += 1
    if "cz" not in target.operation_names:
        NATIVE99_STATS["not_cz"] += 1
        return None
    got = _n99_read(qc)
    if got is None:
        NATIVE99_STATS["not_read"] += 1
        return None
    ops, meas = got
    n, N = qc.num_qubits, target.num_qubits
    edges = sorted({(min(qs), max(qs)) for _, qs, _, _ in ops if len(qs) == 2})
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    if deg and max(deg.values()) > 3:
        NATIVE99_STATS["degree"] += 1
        return None
    col = _graph_colours(edges) if edges else {}
    if col is None:
        NATIVE99_STATS["no_colouring"] += 1
        return None
    lat = _chain_lattice(target, max_error)
    if lat is None:
        NATIVE99_STATS["no_embedding"] += 1
        return None
    K, kadj, fadj, _ = lat
    place = {}
    if edges:
        place = _graph_place(n, edges, K, fadj) or _graph_place(n, edges, K, kadj)
        if place is None:
            NATIVE99_STATS["no_embedding"] += 1
            return None
    free = [q for q in sorted(K) if q not in set(place.values())]
    if n - len(place) > len(free):
        NATIVE99_STATS["no_embedding"] += 1
        return None
    phys = [place[q] if q in place else free.pop(0) for q in range(n)]
    # 2. levels and order
    maxall, maxnd, level = {}, {}, []
    for _, qs, _, dg in ops:
        if dg:
            lv = max([maxnd.get(q, -1) for q in qs]) + 1
        else:
            lv = max([maxall.get(q, -1) for q in qs]) + 1
        level.append(lv)
        for q in qs:
            maxall[q] = max(maxall.get(q, -1), lv)
            if not dg:
                maxnd[q] = max(maxnd.get(q, -1), lv)

    def key(k):
        qs = ops[k][1]
        if len(qs) == 2:
            return (level[k], 0, col[(min(qs), max(qs))], k)
        return (level[k], 1, 0, k)
    order = sorted(range(len(ops)), key=key)
    # 3-4. emission
    frac = "rzz" in target.operation_names
    use_rx = "rx" in target.operation_names
    out = QuantumCircuit(N, name=qc.name)
    for cr in qc.cregs:
        out.add_register(cr)
    one = (1 + 0j, 0j, 0j, 1 + 0j)
    pending = {}

    def flush(q):
        U = pending.pop(q, None)
        if U is not None:
            cls, ang = _n99_run_class(U, use_rx)
            _n99_emit_run(out, phys[q], cls, ang, use_rx)

    def two(name, a, b):
        pa, pb = phys[a], phys[b]
        if (pa, pb) not in target[name]:
            pa, pb = pb, pa
        return pa, pb
    for k in order:
        nm, qs, op, dg = ops[k]
        if len(qs) == 1:
            pending[qs[0]] = _n99_mul(_n99_t(op), pending.get(qs[0], one))
            continue
        steps = _n99_steps(nm, qs, op, dg, frac)
        if steps is None:
            NATIVE99_STATS["block_not_exact"] += 1
            return None
        NATIVE99_STATS["gates"][nm] = NATIVE99_STATS["gates"].get(nm, 0) + 1
        for st in steps:
            if st[0] == "1":
                pending[st[1]] = _n99_mul(st[2], pending.get(st[1], one))
                continue
            flush(st[1])
            flush(st[2])
            if st[0] == "cz":
                out.cz(*two("cz", st[1], st[2]))
            else:
                out.rzz(st[3], *two("rzz", st[1], st[2]))
    for q in sorted(pending):
        flush(q)
    for q, cb in meas:
        out.measure(phys[q], cb)
    out._layout = _n99_layout(qc, out, phys)
    edges_f, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges_f, bad):
        NATIVE99_STATS["failed_check"] += 1
        return None
    NATIVE99_STATS["built"] += 1
    return out
