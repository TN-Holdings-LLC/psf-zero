

# ---------------------------------------------------------------------------------------------------------------------
# Prototype a20 (home, 2026-10-15): item 35, the ring template -- the ring formula built once by PSF-Zero itself (no
# Qiskit pass manager), then bound to new angles by closed-form single-qubit synthesis; item 36, the gate closed --
# no input is handed to Qiskit's level 3.
# ---------------------------------------------------------------------------------------------------------------------

GATE_CLOSED = True  # item 36
TEMPLATE_STATS = {"built": 0, "bound": 0, "structure_mismatch": 0}


class OutOfScopeError(pc.TranspilerError):
    """Item 36: the input is not built by PSF-Zero's formulas; compile it with Qiskit (or another compiler) yourself."""


def _m_rz(t):
    import numpy as np
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]])


def _m_ry(t):
    import numpy as np
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _m_rx(t):
    import numpy as np
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


_FAST_1Q = {"rz": _m_rz, "ry": _m_ry, "rx": _m_rx}


def _m_op(op):
    """Item 35: the 2 x 2 matrix of a single-qubit operation (closed forms for rz, ry, rx; else Qiskit's to_matrix)."""
    f = _FAST_1Q.get(op.name)
    if f is not None:
        return f(float(op.params[0]))
    import numpy as np
    return np.asarray(op.to_matrix(), dtype=complex)


def _zyz(U):
    """Item 35: (theta, phi, lam) with U = e^{i alpha} Rz(phi) Ry(theta) Rz(lam)."""
    import numpy as np
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    V = U / np.sqrt(det)
    theta = 2 * np.arctan2(abs(V[1, 0]), abs(V[0, 0]))
    ph11 = np.angle(V[1, 1])
    ph10 = np.angle(V[1, 0])
    phi = ph11 + ph10
    lam = ph11 - ph10
    return float(theta), float(phi), float(lam)


def _mul(A, B):
    """Item 35: the 2 x 2 product A @ B on tuples (a, b, c, d) = [[a, b], [c, d]] (plain Python: faster than numpy at
    this size)."""
    a, b, c, d = A
    e, f, g, h = B
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def _t_op(op):
    """Item 35: a single-qubit operation as a tuple matrix (closed forms for rz, ry, rx; else Qiskit's to_matrix)."""
    import cmath
    import math
    nm = op.name
    if nm == "rz":
        t = float(op.params[0])
        return (cmath.exp(-0.5j * t), 0j, 0j, cmath.exp(0.5j * t))
    if nm == "ry":
        t = float(op.params[0])
        c, s = math.cos(t / 2), math.sin(t / 2)
        return (complex(c), complex(-s), complex(s), complex(c))
    if nm == "rx":
        t = float(op.params[0])
        c, s = math.cos(t / 2), math.sin(t / 2)
        return (complex(c), -1j * s, -1j * s, complex(c))
    m = op.to_matrix()
    return (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))


def _t_zyz(U):
    """Item 35: (theta, phi, lam) with U = e^{i alpha} Rz(phi) Ry(theta) Rz(lam), U a tuple matrix."""
    import cmath
    import math
    a, b, c, d = U
    sq = cmath.sqrt(a * d - b * c)
    a, c, d = a / sq, c / sq, d / sq
    theta = 2 * math.atan2(abs(c), abs(a))
    p11, p10 = cmath.phase(d), cmath.phase(c) if abs(c) > 1e-15 else 0.0
    if abs(c) <= 1e-15:  # diagonal: only phi + lam is defined
        return 0.0, 2 * p11, 0.0
    if abs(a) <= 1e-15:  # anti-diagonal: only phi - lam is defined
        return math.pi, 2 * p10, 0.0
    return theta, p11 + p10, p11 - p10


def _rot(kind, t):
    """Item 35: rz / ry / rx(t) as a tuple matrix."""
    import cmath
    import math
    if kind == "rz":
        return (cmath.exp(-0.5j * t), 0j, 0j, cmath.exp(0.5j * t))
    c, s = math.cos(t / 2), math.sin(t / 2)
    if kind == "ry":
        return (complex(c), complex(-s), complex(s), complex(c))
    return (complex(c), -1j * s, -1j * s, complex(c))


class RingTemplate:
    """Item 35: a ring classifier compiled once by the ring formula into a parametrised circuit.

    `bind_values(values)` compiles the same classifier with new angles: `values` are the angles of the input's rz /
    ry / rx gates in the input's order (`values_of(qc)` reads them from a circuit; any other single-qubit gate of the
    input is fixed as it was at build time). The angles of every single-qubit run that holds such a gate are computed
    in closed form (ZYZ) and assigned with Qiskit's assign_parameters; runs of fixed gates only were synthesised once,
    with the fewest pulses. `bind(qc)` does the same from a circuit, after checking its structure."""

    def __init__(self, sig, circuit, var_runs, params, use_rx, layout, var_kinds):
        self.sig, self.circuit, self.var_runs, self.params = sig, circuit, var_runs, params
        self.use_rx, self.layout, self.var_kinds = use_rx, layout, var_kinds

    def values_of(self, qc):
        ops = qc.data
        return [float(ops[k].operation.params[0]) for k, _ in self.var_kinds]

    def angles(self, values):
        import math
        out = []
        half = math.pi / 2
        for factors in self.var_runs:
            U = (1 + 0j, 0j, 0j, 1 + 0j)
            for f in factors:
                U = _mul(_rot(f[2], values[f[1]]) if len(f) == 3 else f, U)
            theta, phi, lam = _t_zyz(U)
            if self.use_rx:
                out += (lam - half, theta, phi + half)
            else:
                out += (lam, theta + math.pi, phi + math.pi)
        return out

    def bind_values(self, values):
        if len(values) != len(self.var_kinds):
            raise ValueError(f"item 35: {len(self.var_kinds)} angles expected, {len(values)} given")
        out = self.circuit.assign_parameters(self.angles(values))
        TEMPLATE_STATS["bound"] += 1
        return out

    def bind(self, qc):
        if _structure_sig(qc) != self.sig:
            TEMPLATE_STATS["structure_mismatch"] += 1
            raise ValueError("item 35: the circuit's structure differs from the template's")
        return self.bind_values(self.values_of(qc))


def _structure_sig(qc):
    """Item 35: the circuit with its angles left out (gate names, qubits, clbits) -- what a template must match."""
    return tuple((i.operation.name, tuple(qc.find_bit(q).index for q in i.qubits),
                  tuple(qc.find_bit(c).index for c in i.clbits)) for i in qc.data)


_BLOCK_CACHE = {}


def _block_ops(kinds):
    """Item 35: a fixed two-qubit block (a sequence of "cz" and "swap" on wires (0, 1)) as time-ordered
    [("1q", wire, 2 x 2 matrix) | ("cz",)] with the fewest CZ, decomposed once and checked to 1e-12 up to phase."""
    if kinds in _BLOCK_CACHE:
        return _BLOCK_CACHE[kinds]
    import numpy as np
    from qiskit.circuit.library import CZGate
    from qiskit.quantum_info import Operator
    from qiskit.synthesis import TwoQubitBasisDecomposer
    blk = QuantumCircuit(2)
    for k in kinds:
        blk.cz(0, 1) if k == "cz" else blk.swap(0, 1)
    U = Operator(blk).data
    dec = TwoQubitBasisDecomposer(CZGate(), euler_basis="ZSX")(U)
    seq = []
    for ins in dec.data:
        qs = [dec.find_bit(q).index for q in ins.qubits]
        if len(qs) == 2:
            seq.append(("cz",))
        else:
            seq.append(("1q", qs[0], np.asarray(ins.operation.to_matrix(), dtype=complex)))
    W = Operator(dec).data
    k = np.argmax(abs(U))
    ph = U.flat[k] / W.flat[k]
    if np.max(abs(U - ph * W)) > 1e-12:
        raise RuntimeError(f"item 35: block {kinds} not exact")
    _BLOCK_CACHE[kinds] = seq
    return seq


def _ring_template_one(qc, target, basis_gates=None, seed=0, rounds_fn=None):
    """Item 35: a RingTemplate for the ring classifier `qc` on the device `target` (the ring formula of item 28, or
    item 31's schedule with `rounds_fn=ring_rounds_q4`, placed by item 87's rule as in item 28), or None when `qc` is
    not read as a ring classifier or no line is found. No Qiskit pass manager runs: the fixed CZ / SWAP blocks are
    decomposed once (`_block_ops`), and every single-qubit run between CZs is re-synthesised at `bind` time."""
    read = _ring_read(qc)
    if read is None:
        return None
    n, N = qc.num_qubits, target.num_qubits
    lat = pc._chain_lattice(target, a14.FAILED_MAX_ERROR)
    if lat is None:
        return None
    K, kadj, fadj, snake = lat
    line, _ = pc._chain_place(n, K, fadj, snake)
    if line is None:
        line, _ = pc._chain_place(n, K, kadj, snake)
    if line is None:
        return None
    built = _ring_logical(qc, read, rounds_fn)
    if built is None:
        return None
    logical, p_of_wire = built
    phys = {w: line[p_of_wire[w]] for w in range(n)}
    # the logical circuit's single-qubit operations are qc's, in qc's order: map each to its index in qc.data
    src = [k for k, ins in enumerate(qc.data) if ins.operation.name not in ("barrier", "measure", "cz")]
    pending = {p: [] for p in phys.values()}
    items = []
    lg = [(ins.operation.name, [logical.find_bit(q).index for q in ins.qubits],
           [logical.find_bit(c).index for c in ins.clbits]) for ins in logical.data if ins.operation.name != "barrier"]
    used = [False] * len(lg)
    si = 0
    var_kinds = []  # (index in qc.data, gate name) of every rz / ry / rx of the input, in order: bind_values' order

    def flush(p):
        if pending[p]:
            items.append(("run", p, list(pending[p])))
            pending[p] = []
    for i, (nm, ws, cs) in enumerate(lg):
        if used[i]:
            continue
        used[i] = True
        if len(ws) == 1 and nm != "measure":
            op = qc.data[src[si]].operation
            if op.name in ("rz", "ry", "rx") and not getattr(op, "is_parameterized", lambda: False)():
                pending[phys[ws[0]]].append(("v", len(var_kinds), op.name))
                var_kinds.append((src[si], op.name))
            else:
                pending[phys[ws[0]]].append(_t_op(op))
            si += 1
            continue
        if nm == "measure":
            p = phys[ws[0]]
            flush(p)
            items.append(("measure", p, cs[0]))
            continue
        # a block: this cz / swap and the next ones on the same pair, up to any other operation on either wire
        a, b = sorted(ws)
        kinds = [nm]
        for j in range(i + 1, len(lg)):
            if used[j]:
                continue
            nj, wj, _ = lg[j]
            if not ({a, b} & set(wj)):
                continue
            if nj in ("cz", "swap") and sorted(wj) == [a, b]:
                kinds.append(nj)
                used[j] = True
                continue
            break
        pa, pb = phys[a], phys[b]
        for op in _block_ops(tuple(kinds)):
            if op[0] == "cz":
                flush(pa)
                flush(pb)
                items.append(("cz", pa, pb))
            else:
                m = op[2]
                pending[pa if op[1] == 0 else pb].append((complex(m[0, 0]), complex(m[0, 1]), complex(m[1, 0]),
                                                          complex(m[1, 1])))
    for p in list(pending):
        flush(p)
    if si != len(src):
        return None
    use_rx = "rx" in target.operation_names
    from qiskit.circuit import ParameterVector
    isvar = lambda f: len(f) == 3  # noqa: E731 - ("v", position, kind); a fixed matrix is a 4-tuple
    nvar = sum(1 for it in items if it[0] == "run" and any(isvar(f) for f in it[2]))
    pv = ParameterVector("psf_t", 3 * nvar)
    out = QuantumCircuit(N, name=qc.name)
    for cr in qc.cregs:
        out.add_register(cr)
    var_runs, k = [], 0
    half = 3.141592653589793 / 2
    for it in items:
        if it[0] == "cz":
            out.cz(it[1], it[2])
        elif it[0] == "measure":
            out.measure(it[1], it[2])
        else:
            _, p, factors = it
            merged, acc = [], None  # consecutive fixed matrices pre-multiplied
            for f in factors:
                if isvar(f):
                    if acc is not None:
                        merged.append(acc)
                        acc = None
                    merged.append(f)
                else:
                    acc = f if acc is None else _mul(f, acc)
            if acc is not None:
                merged.append(acc)
            if any(isvar(f) for f in merged):
                var_runs.append(merged)
                a, t, c = pv[k], pv[k + 1], pv[k + 2]
                k += 3
                if use_rx:
                    out.rz(a, p)
                    out.rx(t, p)
                    out.rz(c, p)
                else:
                    out.rz(a, p)
                    out.sx(p)
                    out.rz(t, p)
                    out.sx(p)
                    out.rz(c, p)
            else:  # fixed only: the fewest pulses, decided now
                theta, phi, lam = _t_zyz(merged[0])
                if abs(theta) < 1e-12:
                    if abs(((phi + lam + 3.141592653589793) % (2 * 3.141592653589793)) - 3.141592653589793) > 1e-12:
                        out.rz(phi + lam, p)
                elif use_rx:
                    out.rz(lam - half, p)
                    out.rx(theta, p)
                    out.rz(phi + half, p)
                elif abs(theta - half) < 1e-12:
                    out.rz(lam - half, p)
                    out.sx(p)
                    out.rz(phi + half, p)
                else:
                    out.rz(lam, p)
                    out.sx(p)
                    out.rz(theta + 3.141592653589793, p)
                    out.sx(p)
                    out.rz(phi + 3.141592653589793, p)
    TEMPLATE_STATS["built"] += 1
    return RingTemplate(_structure_sig(qc), out, var_runs, list(pv), use_rx, [phys[w] for w in range(n)], var_kinds)


def ring_template(qc, target, basis_gates=None, seed=0, rounds_fn=None):
    """Item 35: the RingTemplate of the shorter ring schedule for `qc` on `target` -- N2's (item 28) and item 31's,
    as item 25 chooses between the two ring candidates (duration of the example bound, then two-qubit count) -- or
    the one schedule `rounds_fn` names; None when neither builds."""
    if rounds_fn is not None:
        return _ring_template_one(qc, target, basis_gates, seed, rounds_fn)
    best, bc = None, None
    for fn in (None, ring_rounds_q4):
        try:
            t = _ring_template_one(qc, target, basis_gates, seed, fn)
        except RuntimeError:
            t = None
        if t is None:
            continue
        c = t.bind(qc)
        if best is None or pc._better(c, bc, target):
            best, bc = t, c
    return best
