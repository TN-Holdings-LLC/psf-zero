

# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c60 (2026-10-15): item 101, the CZ path of items 87 and 91 emitted by PSF-Zero
# itself -- the last Qiskit level-3 stage inside the scope removed. The rule is by block kind: a single interaction
# (one two-qubit term: a local Clifford around RZZ) in closed form; any other block (several terms on one pair, with
# the single-qubit operations between them) by its Weyl decomposition and closed-form circuits. Nothing is compared.
# ---------------------------------------------------------------------------------------------------------------------

CZ_OWN_EMIT = True  # item 101
CZ101_STATS = {"emitted": 0, "single_blocks": 0, "weyl_blocks": 0, "removed_diagonal": 0}
_T100_ONE = (1 + 0j, 0j, 0j, 1 + 0j)


def _t100_emit(items, phys, target, use_rx, use_x=False):
    """Item 100: the output's shape [(gate, physical qubits...)] and its angle slots, from emitter items
    ("1", q, 2 x 2 tuple) | ("cz", a, b) | ("rzz", a, b, t) | ("end",) | ("meas", q, clbit) -- single-qubit runs merged
    and emitted exactly as item 99's `_native_brickwork` does (`_n99_run_class`)."""
    shape, slots, pending = [], [], {}
    cz_q, rzz_q = target["cz"], (target["rzz"] if "rzz" in target.operation_names else None)

    def flush(q):
        U = pending.pop(q, None)
        if U is None:
            return
        cls, ang = _n99_run_class(U, use_rx)
        p = phys[q]
        if use_x and cls == "full" and abs(ang[1] - 2 * math.pi) < _N99_TOL:  # theta = pi: one x pulse
            shape.extend((("rz", p), ("x", p), ("rz", p)))
            slots.extend((ang[0] - math.pi / 2, ang[2] - math.pi / 2))
            return
        if cls == "z":
            shape.append(("rz", p))
            slots.append(ang[0])
        elif cls == "half":
            shape.extend((("rz", p), ("sx", p), ("rz", p)))
            slots.extend(ang)
        elif cls == "full":
            if use_rx:
                shape.extend((("rz", p), ("rx", p), ("rz", p)))
            else:
                shape.extend((("rz", p), ("sx", p), ("rz", p), ("sx", p), ("rz", p)))
            slots.extend(ang)
    for it in items:
        k = it[0]
        if k == "1":
            q = it[1]
            pending[q] = _n99_mul(it[2], pending.get(q, _T100_ONE))
        elif k == "cz" or k == "rzz":
            a, b = it[1], it[2]
            flush(a)
            flush(b)
            pa, pb = phys[a], phys[b]
            if (pa, pb) not in (cz_q if k == "cz" else rzz_q):
                pa, pb = pb, pa
            shape.append((k, pa, pb))
            if k == "rzz":
                slots.append(it[3])
        elif k == "end":
            for q in sorted(pending):
                flush(q)
        else:  # ("meas", q, clbit)
            flush(it[1])
            shape.append(("measure", phys[it[1]], it[2]))
    return tuple(shape), slots


_T100_PAULI = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
               "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]).astype(complex)}
_T100_RX90 = None


def _t100_rot1(ch, a):
    """Item 100: exp(-i a P) for a single-qubit Pauli P as a tuple matrix (rx / ry / rz(2a))."""
    if ch == "Z":
        return _n99_rz(2 * a)
    c, s = math.cos(a), math.sin(a)
    if ch == "X":
        return (complex(c), -1j * s, -1j * s, complex(c))
    return (complex(c), complex(-s), complex(s), complex(c))


def _t100_k1(a, b, c, d, low):
    """Item 100: a 2 x 2 matrix [[a, b], [c, d]] on one qubit of a pair as 4 x 4 (low: the less significant qubit)."""
    if low:
        return np.array([[a, b, 0, 0], [c, d, 0, 0], [0, 0, a, b], [0, 0, c, d]], dtype=complex)
    return np.array([[a, 0, b, 0], [0, a, 0, b], [c, 0, d, 0], [0, c, 0, d]], dtype=complex)


_N99_CZ4 = np.diag([1, 1, 1, -1]).astype(complex)


def _n99_kron_steps(steps, q0, q1):  # noqa: F811 - item 100: item 99's check, the same matrices without np.kron
    """Item 99 (faster form, c61): the 4 x 4 matrix (qubit q0 least significant) of a list of steps on (q0, q1)."""
    U = np.eye(4, dtype=complex)
    for st in steps:
        if st[0] == "1":
            U = _t100_k1(*st[2], st[1] == q0) @ U
        elif st[0] == "cz":
            U = _N99_CZ4 @ U
        else:
            t = st[3]
            e, f = cmath.exp(-0.5j * t), cmath.exp(0.5j * t)
            U = np.array([e, f, f, e])[:, None] * U
    return U


_T100_MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / math.sqrt(2)
_T100_XYZ = np.array([np.real(np.diag(_T100_MAGIC.conj().T @ np.kron(_T100_PAULI[p], _T100_PAULI[p]) @ _T100_MAGIC))
                      for p in "XYZ"])  # eigenvalues of XX, YY, ZZ on the magic basis (all three diagonal there)


def _t100_canonical(q0, q1, a, b, c):
    """Item 100: steps for exp(i(a XX + b YY + c ZZ)) on (q0, q1), any real a, b, c, up to a global phase: Rz(pi/2)
    on q1; CX(q1 -> q0); Rz(-pi/2 - 2c) on q0, Ry(-pi/2 - 2a) on q1; CX(q0 -> q1); Ry(pi/2 + 2b) on q1;
    CX(q1 -> q0); Rz(-pi/2) on q0 -- CX = H cz H (three cz). An identity, checked in test_c61."""
    hp, H = math.pi / 2, _N99_H
    return [("1", q1, _n99_rz(hp)),
            ("1", q0, H), ("cz", q0, q1), ("1", q0, H),
            ("1", q0, _n99_rz(-hp - 2 * c)), ("1", q1, _T100_RY(-hp - 2 * a)),
            ("1", q1, H), ("cz", q0, q1), ("1", q1, H),
            ("1", q1, _T100_RY(hp + 2 * b)),
            ("1", q0, H), ("cz", q0, q1), ("1", q0, H),
            ("1", q0, _n99_rz(-hp))]


def _t100_weyl_ok(U, d):
    """Item 100: True if Qiskit's Weyl decomposition `d` reproduces U to 1e-9 (the closed-form circuits built from it
    are identities, so this checks the step list without multiplying it out)."""
    a, b = d.K1l, d.K1r
    k1 = (a[:, None, :, None] * b[None, :, None, :]).reshape(4, 4)
    a, b = d.K2l, d.K2r
    k2 = (a[:, None, :, None] * b[None, :, None, :]).reshape(4, 4)
    Nd = np.exp(1j * (d.a * _T100_XYZ[0] + d.b * _T100_XYZ[1] + d.c * _T100_XYZ[2]))
    W = k1 @ ((_T100_MAGIC * Nd) @ _T100_MAGIC.conj().T) @ k2
    return _n99_same(U, W)


_T100_RY = lambda t: (complex(math.cos(t / 2)), complex(-math.sin(t / 2)), complex(math.sin(t / 2)),  # noqa: E731
                      complex(math.cos(t / 2)))


def _t100_block_steps(U, q0, q1):
    """Item 100: steps for a two-qubit block U (q0 least significant), from its Weyl decomposition (Qiskit's, without
    specialisation): U = e^{i phi} (K1l x K1r) exp(i(a XX + b YY + c ZZ)) (K2l x K2r). A single interaction
    (b = c = 0) through RZZ (`_n99_zz`); c = 0 by Qiskit's closed-form CZ decomposition (two cz); otherwise the
    closed-form three-CNOT circuit of exp(i(a XX + b YY + c ZZ)) -- Rz(pi/2) on q1; CX(q1 -> q0); Rz(-pi/2 - 2c) on q0,
    Ry(-pi/2 - 2a) on q1; CX(q0 -> q1); Ry(pi/2 + 2b) on q1; CX(q1 -> q0); Rz(-pi/2) on q0 -- with CX = H cz H. Every
    form is multiplied out and checked to 1e-9."""
    from qiskit.synthesis import TwoQubitWeylDecomposition
    d = TwoQubitWeylDecomposition(U, fidelity=None)
    t = lambda m: (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))  # noqa: E731
    pre = [("1", q0, t(d.K2r)), ("1", q1, t(d.K2l))]
    post = [("1", q0, t(d.K1r)), ("1", q1, t(d.K1l))]
    H = _N99_H
    exact_form = True  # closed forms of the Weyl decomposition: the decomposition itself is checked
    if abs(d.b) < _N99_TOL and abs(d.c) < _N99_TOL:
        steps = pre + [("1", q0, H), ("1", q1, H)] + _n99_zz(q0, q1, -2 * d.a, False, False) + \
            [("1", q0, H), ("1", q1, H)] + post
    elif abs(d.c) < _N99_TOL:
        exact_form = False
        steps = _n99_steps("unitary", (q0, q1), _T100Mat(U), False, False)
        if steps is not None:
            return steps
        steps = []
    else:
        steps = pre + _t100_canonical(q0, q1, d.a, d.b, d.c) + post
    if steps and (_t100_weyl_ok(U, d) if exact_form else _n99_same(U, _n99_kron_steps(steps, q0, q1))):
        return steps
    TEMPLATE100_STATS["block_not_exact"] += 1
    raise TranspilerError("item 100: a block's native form failed its check")


class _T100Op:
    """An operation of the input with new parameters (name and params as the original's; matrix from a new instance
    of its class), for `_n99_t`."""
    __slots__ = ("name", "params", "_cls")

    def __init__(self, op, params):
        self.name, self.params, self._cls = op.name, params, op.__class__

    def to_matrix(self):
        return self._cls(*self.params).to_matrix()


class _T100Mat:
    """Item 100: a fixed 4 x 4 block for `_n99_steps`."""
    __slots__ = ("U",)
    name = "unitary"
    params = ()

    def __init__(self, U):
        self.U = U

    def to_matrix(self):
        return self.U




class _CZ101Program:
    """Item 101: the program items 87/91 build (prefix gates, terms in their order, suffix gates, measurements, on
    their fixed layout) as PSF-Zero emits it on the CZ path: every two-qubit term a local Clifford around RZZ
    (`_n99_zz`); consecutive two-qubit terms on one pair one block (single-qubit operations between two of its terms
    absorbed); a block of one term in closed form, any other by its Weyl decomposition (`_t100_block_steps`);
    diagonal operations directly before the measurements left out (repeatedly; exact for the outcome distribution).
    Values: the prefix gates' parameters, the terms' angles (reader order), the suffix gates' parameters."""

    def __init__(self, info):
        self.n, self.phys = info["n"], list(info["layout"])
        self.prefix, self.terms, self.order = info["prefix"], info["terms"], info["order"]
        self.suffix, self.meas, self.perm = info["suffix"], info["meas"], info["perm"]
        self.pre_sl, pos = [], 0
        for op, _ in self.prefix:
            m = len(op.params)
            self.pre_sl.append((pos, pos + m) if m else None)
            pos += m
        self.term0 = pos
        pos += len(self.terms)
        self.suf_sl = []
        for op, _ in self.suffix:
            m = len(op.params)
            self.suf_sl.append((pos, pos + m) if m else None)
            pos += m
        self.nvalues = pos
        self.values0 = ([float(p) for op, _ in self.prefix for p in op.params] + [float(a) for _, a in self.terms]
                        + [float(p) for op, _ in self.suffix for p in op.params])
        self._build_blocks()

    def _gate1(self, op, sl, values):
        return _n99_t(op if sl is None else _T100Op(op, values[sl[0]:sl[1]]))

    def _build_blocks(self):
        """The logical program: prefix gates, terms in items 87/91's order, suffix gates, measurements; diagonal
        operations directly before the measurements left out (repeatedly); consecutive two-qubit terms on one pair
        grouped into a block."""
        prog = []  # (kind, qubits, ref, diagonal)
        for j, (op, q) in enumerate(self.prefix):
            prog.append(("pre", (q,), j, _n99_diag(op, 1)))
        for k in self.order:
            lb, _ = self.terms[k]
            sup = _chain_support(lb)
            prog.append(("term", tuple(q for q, _ in sup), k, all(ch == "Z" for _, ch in sup)))
        for j, (op, q) in enumerate(self.suffix):
            prog.append(("suf", (q,), j, _n99_diag(op, 1)))
        measured = {self.perm[q] for q, _ in self.meas}
        if measured:
            while True:
                last = {}
                for i, (_, qs, _, _) in enumerate(prog):
                    for q in qs:
                        last[q] = i
                drop = [i for i in sorted(set(last.values()), reverse=True)
                        if prog[i][3] and all(q in measured and last[q] == i for q in prog[i][1])]
                if not drop:
                    break
                for i in drop:
                    del prog[i]
                CZ101_STATS["removed_diagonal"] += len(drop)
        # blocks: consecutive two-qubit terms on one pair; single-qubit operations on the pair's qubits between two
        # of its terms are absorbed into the block, those after its last term stay outside (decided here, once)
        seq, open_ = [], {}

        def close(blk):
            for x in blk[1]:
                if open_.get(x) is blk:
                    del open_[x]
            seq.extend(blk[3])
            blk[3] = []
        for kind, qs, ref, _ in prog:
            if kind == "term" and len(qs) == 2:
                pair = tuple(sorted(qs))
                blk = open_.get(qs[0])
                if blk is not None and blk is open_.get(qs[1]) and blk[1] == pair:
                    blk[2].extend(("g",) + h for h in blk[3])
                    blk[3] = []
                    blk[2].append(("t", ref))
                    continue
                for q in qs:
                    if open_.get(q) is not None:
                        close(open_[q])
                blk = ["block", pair, [("t", ref)], []]
                open_[qs[0]] = open_[qs[1]] = blk
                seq.append(blk)
                continue
            q = qs[0] if len(qs) == 1 else None
            if q is not None and open_.get(q) is not None:
                open_[q][3].append((kind, qs, ref))
                continue
            for x in qs:
                if open_.get(x) is not None:
                    close(open_[x])
            seq.append((kind, qs, ref))
        for blk in list({id(v): v for v in open_.values()}.values()):
            close(blk)
        self.prog = seq
        self.paulis = {}
        for blk in seq:
            if blk[0] == "block":
                qa, qb = blk[1]
                for m in blk[2]:
                    if m[0] == "t":
                        ch = dict(_chain_support(self.terms[m[1]][0]))
                        self.paulis[m[1]] = np.kron(_T100_PAULI[ch.get(qb, "I")], _T100_PAULI[ch.get(qa, "I")])

    def _one(self, kind, ref, values):
        """A single-qubit operation of the program: (qubit, 2 x 2 tuple)."""
        if kind == "pre" or kind == "suf":
            op, q = (self.prefix if kind == "pre" else self.suffix)[ref]
            return q, self._gate1(op, (self.pre_sl if kind == "pre" else self.suf_sl)[ref], values)
        (q, ch), = _chain_support(self.terms[ref][0])
        return q, _t100_rot1(ch, values[self.term0 + ref])

    def _block_items(self, values):
        t0 = self.term0
        for b in self.prog:
            if b[0] != "block":
                q, M = self._one(b[0], b[2], values)
                yield ("1", q, M)
            elif len(b[2]) == 1:  # one two-qubit term: a local Clifford around RZZ(2a), exact in closed form
                CZ101_STATS["single_blocks"] += 1
                k = b[2][0][1]
                (q1, c1), (q2, c2) = _chain_support(self.terms[k][0])
                for q, ch in ((q1, c1), (q2, c2)):
                    if ch == "X":
                        yield ("1", q, _N99_H)
                    elif ch == "Y":
                        yield ("1", q, _t100_rot1("X", math.pi / 4))
                yield from _n99_zz(q1, q2, 2 * values[t0 + k], False, False)
                for q, ch in ((q1, c1), (q2, c2)):
                    if ch == "X":
                        yield ("1", q, _N99_H)
                    elif ch == "Y":
                        yield ("1", q, _t100_rot1("X", -math.pi / 4))
            else:  # several terms (and the single-qubit operations between them) on one pair: the Weyl form (item 101)
                CZ101_STATS["weyl_blocks"] += 1
                qa, qb = b[1]
                U = np.eye(4, dtype=complex)
                for m in b[2]:
                    if m[0] == "t":
                        a = values[t0 + m[1]]
                        U = (math.cos(a) * np.eye(4) - 1j * math.sin(a) * self.paulis[m[1]]) @ U
                    else:
                        q, (x, y, z, w) = self._one(m[1], m[3], values)
                        U = _t100_k1(x, y, z, w, q == qa) @ U
                yield from _t100_block_steps(U, qa, qb)
        yield ("end",)
        for q, cb in self.meas:
            yield ("meas", self.perm[q], cb)

    def items(self, values):
        return self._block_items(values)


def _cz101_circuit(qc, target, shape, slots, phys):
    """Item 101: the concrete circuit of an emitted shape and its angles (the same gates, in the same order, as the
    compile template's parametrised circuit bound to them)."""
    out = QuantumCircuit(target.num_qubits, name=qc.name)
    for cr in qc.cregs:
        out.add_register(cr)
    k = 0
    for g in shape:
        nm = g[0]
        if nm == "rz":
            out.rz(slots[k], g[1])
            k += 1
        elif nm == "sx":
            out.sx(g[1])
        elif nm == "x":
            out.x(g[1])
        elif nm == "rx":
            out.rx(slots[k], g[1])
            k += 1
        elif nm == "cz":
            out.cz(g[1], g[2])
        elif nm == "rzz":
            out.rzz(slots[k], g[1], g[2])
            k += 1
        else:
            out.measure(g[1], g[2])
    out._layout = _n99_layout(qc, out, phys)
    return out


def _cz101_emit(qc, target, info):
    """Item 101: items 87/91's CZ path emitted by PSF-Zero (`_CZ101Program`, runs merged as item 99's, a run with
    theta = pi as one x pulse when the Target has x). No pass manager, no comparison, no simulation."""
    p = _CZ101Program(info)
    use_x = "x" in target.operation_names
    shape, slots = _t100_emit(p.items(p.values0), p.phys, target, "rx" in target.operation_names, use_x)
    CZ101_STATS["emitted"] += 1
    return _cz101_circuit(qc, target, shape, slots, p.phys)

