

# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c60 (2026-10-15): item 100, the compile template -- a circuit inside the scope is
# built once (the formula path c60 takes: item 87's chain, item 91's graph or item 99's native gates); new angles are
# bound in closed form only (no pass manager, no simulation); a shape signature guards every value-dependent branch.
# ---------------------------------------------------------------------------------------------------------------------

TEMPLATE100_STATS = {"built": 0, "bound": 0, "rebuilt": 0, "switched": 0, "structure_mismatch": 0,
                     "block_not_exact": 0}
TEMPLATE_CACHE = 256  # item 100: shapes kept per template (a shape seen before is not built again)
_T100_HOOK = [None]  # item 100: set while compile_template reads items 87/91's structure (they stop before Qiskit)
_T100_ONE = (1 + 0j, 0j, 0j, 1 + 0j)


class _T100Captured(Exception):
    """Item 100: raised by the hook inside items 87/91 with the structure they built (nothing is compiled)."""

    def __init__(self, info):
        super().__init__("item 100: structure captured")
        self.info = info


def _t100_hook(info):
    raise _T100Captured(info)


class _T100Op:
    """Item 100: an operation of the input with new parameters (name and params as the original's; matrix from a new
    instance of its class), for `_n99_t` and `_n99_steps`."""
    __slots__ = ("name", "params", "_cls")

    def __init__(self, op, params):
        self.name, self.params, self._cls = op.name, params, op.__class__

    def to_matrix(self):
        return self._cls(*self.params).to_matrix()


def _t100_params(op):
    return [float(p) for p in op.params] if op.params else []


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


class _T100Native:
    """Item 100: item 99's construction (`_native_brickwork`) split into its structure (placement, order, fixed gates'
    native forms -- decided once) and its values (the input's gate parameters, in the input's order)."""
    kind = "native"

    def __init__(self, qc, target, max_error):
        got = _n99_read(qc)
        if got is None:
            raise _T100NotHere("native: not read")
        ops, meas = got
        n, N = qc.num_qubits, target.num_qubits
        edges = sorted({(min(qs), max(qs)) for _, qs, _, _ in ops if len(qs) == 2})
        deg = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        if deg and max(deg.values()) > 3:
            raise _T100NotHere("native: degree")
        col = _graph_colours(edges) if edges else {}
        lat = _chain_lattice(target, max_error)
        if col is None or lat is None:
            raise _T100NotHere("native: no colouring or no lattice")
        K, kadj, fadj, _ = lat
        place = {}
        if edges:
            place = _graph_place(n, edges, K, fadj) or _graph_place(n, edges, K, kadj)
            if place is None:
                raise _T100NotHere("native: no embedding")
        free = [q for q in sorted(K) if q not in set(place.values())]
        if n - len(place) > len(free):
            raise _T100NotHere("native: no embedding")
        self.phys = [place[q] if q in place else free.pop(0) for q in range(n)]
        maxall, maxnd, level = {}, {}, []
        for _, qs, _, dg in ops:
            lv = max([(maxnd if dg else maxall).get(q, -1) for q in qs]) + 1
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
        self.order = sorted(range(len(ops)), key=key)
        self.ops, self.meas, self.qc = ops, meas, qc
        self.frac = "rzz" in target.operation_names
        self.slices, pos = [], 0
        for _, _, op, _ in ops:
            m = len(op.params)
            self.slices.append((pos, pos + m) if m else None)
            pos += m
        self.nvalues = pos
        self.fixed = {}
        self.fixed1 = {k: _n99_t(op) for k, (_, qs, op, _) in enumerate(ops) if len(qs) == 1 and not op.params}
        for k, (nm, qs, op, dg) in enumerate(ops):
            if len(qs) == 2 and not op.params:
                st = _n99_steps(nm, qs, op, dg, self.frac)
                if st is None:
                    raise _T100NotHere("native: block not exact")
                self.fixed[k] = st
        self.input_sig = ("native",) + tuple((nm, qs, dg) for nm, qs, _, dg in ops) + (tuple(meas),)

    def values_of(self, qc):
        got = _n99_read(qc)
        if got is None or ("native",) + tuple((nm, qs, dg) for nm, qs, _, dg in got[0]) + (tuple(got[1]),) != \
                self.input_sig:
            TEMPLATE100_STATS["structure_mismatch"] += 1
            raise ValueError("item 100: the circuit's structure differs from the template's")
        return [v for _, _, op, _ in got[0] for v in _t100_params(op)]

    def items(self, values):
        ops, sl, frac = self.ops, self.slices, self.frac
        for k in self.order:
            nm, qs, op, dg = ops[k]
            s = sl[k]
            if len(qs) == 1:
                yield ("1", qs[0], self.fixed1[k] if s is None else _n99_t(_T100Op(op, values[s[0]:s[1]])))
                continue
            st = self.fixed.get(k)
            if st is None:
                st = _n99_steps(nm, qs, _T100Op(op, values[s[0]:s[1]]), dg, frac)
                if st is None:
                    TEMPLATE100_STATS["block_not_exact"] += 1
                    raise TranspilerError("item 100: a gate's native form failed its check")
            yield from st
        yield ("end",)
        for q, cb in self.meas:
            yield ("meas", q, cb)


class _T100NotHere(Exception):
    """Item 100: this formula path does not build the input (the next one is tried)."""


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


class _T100Formula:
    """Item 100: items 87/91's structure (as they read and place the input; captured before their Qiskit stage) and a
    PSF-Zero emission of it that needs no pass manager.

    With rzz and rx in the Target (item 88's frame path) the logical circuit, Qiskit's RemoveDiagonalGatesBeforeMeasure
    (applied once at build; the same positions are left out at every bind) and the Pauli X frame are exactly c60's;
    the single-qubit runs are merged by PSF-Zero (`_n99_run_class`) instead of Qiskit's Optimize1qGatesDecomposition.
    Without them (c60: Qiskit's level 3 on the fixed layout), every two-qubit term is a local Clifford around RZZ (by
    `_n99_zz`), consecutive two-qubit terms on one pair are one block (closed form when it is a single term; Qiskit's
    Weyl decomposition or closed-form CZ decomposition otherwise, as in item 99), and diagonal terms directly before
    the measurements are left out (repeatedly; exact for the outcome distribution)."""

    def __init__(self, kind, qc, target, info):
        self.kind, self.qc = kind, qc
        self.n = info["n"]
        self.frame = info["frame"]
        self.phys = list(info["layout"])
        self.prefix, self.terms, self.order = info["prefix"], info["terms"], info["order"]
        self.suffix, self.meas, self.perm = info["suffix"], info["meas"], info["perm"]
        self.use_rx = "rx" in target.operation_names
        # values: prefix gates' parameters, the terms' angles (reader order), suffix gates' parameters
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
        self.input_sig = self._read_sig(self.prefix, self.terms, self.suffix, self.meas, self.perm, info.get("extra"))
        self.values0 = self._flat(self.prefix, self.terms, self.suffix)
        if self.frame:
            self._build_frame()
        else:
            self._build_blocks()

    @staticmethod
    def _read_sig(prefix, terms, suffix, meas, perm, extra):
        return (tuple((op.name, q, len(op.params)) for op, q in prefix), tuple(lb for lb, _ in terms),
                tuple((op.name, q, len(op.params)) for op, q in suffix), tuple(meas), tuple(perm),
                tuple(extra) if extra is not None else None)

    @staticmethod
    def _flat(prefix, terms, suffix):
        return ([v for op, _ in prefix for v in _t100_params(op)] + [float(a) for _, a in terms]
                + [v for op, _ in suffix for v in _t100_params(op)])

    def values_of(self, qc):
        got = (_chain_read if self.kind == "chain" else _graph_read)(qc)
        if got is not None:
            prefix, meas, terms, _, suffix, perm, extra = got
            if self._read_sig(prefix, terms, suffix, meas, perm, extra) == self.input_sig:
                return self._flat(prefix, terms, suffix)
        TEMPLATE100_STATS["structure_mismatch"] += 1
        raise ValueError("item 100: the circuit's structure differs from the template's")

    def _gate1(self, op, sl, values):
        return _n99_t(op if sl is None else _T100Op(op, values[sl[0]:sl[1]]))

    # -- item 88's frame path ----------------------------------------------------------------------------------------
    def _build_frame(self):
        """The logical circuit exactly as items 87/91 write it (`_chain_rotation`, frame=True), each instruction's
        parameter traced to its source, and the positions Qiskit's RemoveDiagonalGatesBeforeMeasure removes."""
        from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
        log = QuantumCircuit(*self.qc.qregs, *self.qc.cregs)
        src = []
        for j, (op, q) in enumerate(self.prefix):
            log.append(op, [q])
            src.append(("pre", j))
        for k in self.order:
            lb, a = self.terms[k]
            before = len(log.data)
            _chain_rotation(log, lb, a, frame=True)
            new = log.data[before:]
            rot = [i for i, ins in enumerate(new) if ins.operation.name in ("rx", "ry", "rz", "rzz")
                   and len(new) == 1 or ins.operation.name == "rzz"]
            for i, ins in enumerate(new):
                src.append(("term", k) if i in rot else ("fixed",))
        for j, (op, q) in enumerate(self.suffix):
            log.append(op, [q])
            src.append(("suf", j))
        for q, cb in self.meas:
            log.measure(self.perm[q], cb)
            src.append(("meas",))
        # Qiskit's rule, applied here by position: a measurement's predecessor on its qubit is removed when it is a
        # diagonal single-qubit gate, or a diagonal two-qubit gate whose other qubit's successor is a measurement too
        d1 = {"rz", "z", "t", "s", "tdg", "sdg", "u1", "p"}
        d2 = {"cz", "crz", "cu1", "rzz", "cp"}
        qs_of = [tuple(log.find_bit(x).index for x in ins.qubits) for ins in log.data]
        nxt, prev, last = {}, {}, {}
        for i, qs in enumerate(qs_of):
            for q in qs:
                if q in last:
                    nxt[(last[q], q)] = i
                    prev[(i, q)] = last[q]
                last[q] = i
        names = [ins.operation.name for ins in log.data]
        drop = set()
        for i, nm in enumerate(names):
            if nm != "measure":
                continue
            p = prev.get((i, qs_of[i][0]))
            if p is None:
                continue
            if names[p] in d1:
                drop.add(p)
            elif names[p] in d2 and all(names[nxt.get((p, q), -1)] == "measure" if (p, q) in nxt else False
                                        for q in qs_of[p]):
                drop.add(p)
        keep = [i for i in range(len(names)) if i not in drop]
        kept = PassManager([RemoveDiagonalGatesBeforeMeasure()]).run(log)  # checked against Qiskit's own result

        def perq(c, idx=None):
            out = {}
            for k, ins in enumerate(c.data):
                if idx is not None and k not in idx:
                    continue
                for x in ins.qubits:
                    out.setdefault(c.find_bit(x).index, []).append(ins.operation.name)
            return out
        if perq(log, set(keep)) != perq(kept):
            raise _T100NotHere("frame: removal differs from Qiskit's")
        a = qs_of
        self.fixed_m = {}
        for i in keep:
            if src[i][0] == "fixed" and len(a[i]) == 1:
                self.fixed_m[i] = _n99_t(log.data[i].operation)
        self.log = [(log.data[i].operation, a[i], log.find_bit(log.data[i].clbits[0]).index
                     if log.data[i].clbits else None, src[i], self.fixed_m.get(i)) for i in keep]

    def _frame_items(self, values):
        f = [0] * self.n
        X = _N99_X
        t0 = self.term0
        for op, qs, cb, s, fm in self.log:
            nm = op.name
            if nm == "measure":
                q = qs[0]
                if f[q]:
                    yield ("1", q, X)
                    f[q] = 0
                yield ("meas", q, cb)
                continue
            if s[0] == "term":
                prm = 2 * values[t0 + s[1]]
            elif s[0] == "pre" or s[0] == "suf":
                sl = (self.pre_sl if s[0] == "pre" else self.suf_sl)[s[1]]
                prm = None if sl is None else values[sl[0]:sl[1]]
            else:
                prm = None
            if nm == "rzz":
                a, b = qs
                th = prm
                th = math.pi - (math.pi - th) % (2 * math.pi)
                if abs(th) > math.pi / 2:
                    yield ("1", a, _n99_rz(math.pi))
                    yield ("1", b, _n99_rz(math.pi))
                    th -= math.copysign(math.pi, th)
                if abs(th) < 1e-12:
                    continue
                eff = -th if f[a] != f[b] else th
                if eff < 0:
                    yield ("1", a, X)
                    f[a] ^= 1
                    eff = -eff
                yield ("rzz", a, b, eff)
                continue
            q = qs[0]
            if s[0] == "term":
                M = _t100_rot1({"rx": "X", "ry": "Y", "rz": "Z"}[nm], prm / 2)
            elif fm is not None:
                M = fm
            elif prm is None:
                M = _n99_t(op)
            else:
                M = _n99_t(_T100Op(op, prm))
            if f[q]:  # X M X, exactly: the matrix's entries permuted
                yield ("1", q, (M[3], M[2], M[1], M[0]))
            else:
                yield ("1", q, M)
        for q in range(self.n):
            if f[q]:
                yield ("1", q, X)
        yield ("end",)

    # -- the CZ path: PSF-Zero's own blocks ---------------------------------------------------------------------------
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
        for blk in seq:  # item 100: a block of exactly XX, YY and ZZ (no single-qubit operation) is the canonical gate
            if blk[0] == "block":
                byp = {}
                if all(m[0] == "t" for m in blk[2]) and len(blk[2]) == 3:
                    for m in blk[2]:
                        sup = _chain_support(self.terms[m[1]][0])
                        if len(sup) == 2 and sup[0][1] == sup[1][1]:
                            byp[sup[0][1]] = m[1]
                blk.append((byp["X"], byp["Y"], byp["Z"]) if sorted(byp) == ["X", "Y", "Z"] else None)
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
            elif b[4] is not None:  # exactly XX, YY and ZZ on one pair (they commute): the canonical gate itself
                al, be, ga = (values[t0 + k] for k in b[4])
                yield from _t100_canonical(b[1][0], b[1][1], -al, -be, -ga)
            else:  # several terms (and the single-qubit operations between them) on one pair: item 99's general rule
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
        return self._frame_items(values) if self.frame else self._block_items(values)


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


class _T100Mat:
    """Item 100: a fixed 4 x 4 block for `_n99_steps`."""
    __slots__ = ("U",)
    name = "unitary"
    params = ()

    def __init__(self, U):
        self.U = U

    def to_matrix(self):
        return self.U


class CompileTemplate:
    """Item 100: a circuit inside PSF-Zero's scope compiled once into a parametrised circuit on the device.

    `bind_values(values)` returns the compiled circuit for new values, computed in closed form (single-qubit runs by
    ZYZ, two-qubit forms as at build) and assigned with Qiskit's assign_parameters: no pass manager, no simulation.
    `values_of(qc)` reads the values from a circuit of the same structure (item 99: every gate parameter, in the
    input's order; items 87/91: the reader's prefix-gate parameters, term angles and suffix-gate parameters).
    `bind(qc)` = `bind_values(values_of(qc))`. `signature` is the input's structure and the output's shape; if new
    values change the shape (a value-dependent branch: a single-qubit run's class, an rzz folded or dropped, item 88's
    frame, a block's CZ count), the circuit of the new shape is built and counted (TEMPLATE100_STATS["rebuilt"]), or
    taken from the template's cache of shapes already built (["switched"]); a wrong output is never returned for the
    old shape."""

    def __init__(self, gen, target, seed=0):
        self._gen, self.target, self.kind, self.seed = gen, target, gen.kind, seed
        self.use_rx = "rx" in target.operation_names
        # item 100: on the CZ path of items 87/91 a run with theta = pi is one x pulse (as Qiskit's single-qubit
        # synthesis there gives); item 99's own runs stay exactly as c60 emits them
        self.use_x = gen.kind != "native" and not gen.frame and "x" in target.operation_names
        self._cache = {}  # shape -> parametrised circuit (TEMPLATE_CACHE shapes at most, oldest dropped first)
        self._build(gen.values0 if hasattr(gen, "values0") else None)
        TEMPLATE100_STATS["built"] += 1

    def _circuit(self, shape, nslots):
        """The parametrised circuit of an output shape."""
        from qiskit.circuit import ParameterVector
        pv = ParameterVector("psf_t100", nslots)
        qc = self._gen.qc
        out = QuantumCircuit(self.target.num_qubits, name=qc.name)
        for cr in qc.cregs:
            out.add_register(cr)
        k = 0
        for g in shape:
            nm = g[0]
            if nm == "rz":
                out.rz(pv[k], g[1])
                k += 1
            elif nm == "sx":
                out.sx(g[1])
            elif nm == "x":
                out.x(g[1])
            elif nm == "rx":
                out.rx(pv[k], g[1])
                k += 1
            elif nm == "cz":
                out.cz(g[1], g[2])
            elif nm == "rzz":
                out.rzz(pv[k], g[1], g[2])
                k += 1
            else:
                out.measure(g[1], g[2])
        out._layout = _n99_layout(qc, out, self._gen.phys)
        return out

    def _build(self, values, shape=None, slots=None):
        if values is None:
            values = self._gen.values_of(self._gen.qc)
        if shape is None:
            shape, slots = _t100_emit(self._gen.items(values), self._gen.phys, self.target, self.use_rx, self.use_x)
        circ = self._cache.get(shape)
        if circ is None:
            circ = self._circuit(shape, len(slots))
            if len(self._cache) >= TEMPLATE_CACHE:
                self._cache.pop(next(iter(self._cache)))
            self._cache[shape] = circ
            new = True
        else:
            new = False
        self.circuit, self._shape, self._slots0, self._values0 = circ, shape, slots, list(values)
        self.signature = (self.kind, self._gen.input_sig, shape)
        return new

    def values_of(self, qc):
        return self._gen.values_of(qc)

    def bind_values(self, values):
        values = [float(v) for v in values]
        if len(values) != self._gen.nvalues:
            raise ValueError(f"item 100: {self._gen.nvalues} values expected, {len(values)} given")
        shape, slots = _t100_emit(self._gen.items(values), self._gen.phys, self.target, self.use_rx, self.use_x)
        if shape != self._shape:  # a value-dependent branch changed the shape: built anew, or taken from the cache
            if self._build(values, shape, slots):
                TEMPLATE100_STATS["rebuilt"] += 1
            else:
                TEMPLATE100_STATS["switched"] += 1
        out = self.circuit.assign_parameters(slots)
        TEMPLATE100_STATS["bound"] += 1
        return out

    def bind(self, qc):
        return self.bind_values(self.values_of(qc))


def compile_template(qc, target, seed_transpiler=0, prune_max_error=0.5):
    """Item 100: the CompileTemplate of `qc` on the device `target`, by the formula path c60's default call takes
    (item 96's scope, then item 87's chain, item 91's graph, item 99's native gates). OutOfScopeError where c60
    refuses."""
    if target is None:
        raise OutOfScopeError("item 100: a template needs the device's Target (changelog item 98)")
    max_error = prune_max_error
    edges, reason = _scope96_edges(qc)
    if not (reason is None and (not edges or _scope96_embeds(edges, qc.num_qubits, target, max_error))):
        raise OutOfScopeError("item 100: outside PSF-Zero's scope (changelog item 98)")
    if "cz" not in target.operation_names:
        raise OutOfScopeError("item 100: the device's two-qubit gate is not cz")
    fe, fq = _failed_elements(target, max_error)
    for kind, fn in (("chain", _chain_brickwork), ("graph", _graph_brickwork)):
        info = None
        _T100_HOOK[0] = _t100_hook
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn(qc, target, max_error, seed_transpiler)
        except _T100Captured as e:
            info = e.info
        finally:
            _T100_HOOK[0] = None
        if info is None:
            continue
        try:
            tpl = CompileTemplate(_T100Formula(kind, qc, target, info), target, seed_transpiler)
        except _T100NotHere:
            continue
        if not _uses_failed(tpl.circuit, fe, fq):
            return tpl
    if NATIVE_BRICKWORK:
        try:
            tpl = CompileTemplate(_T100Native(qc, target, max_error), target, seed_transpiler)
        except _T100NotHere:
            tpl = None
        if tpl is not None and not _uses_failed(tpl.circuit, fe, fq):
            return tpl
    raise OutOfScopeError("item 100: inside the scope, but no formula reads this circuit (changelog items 98-99)")
