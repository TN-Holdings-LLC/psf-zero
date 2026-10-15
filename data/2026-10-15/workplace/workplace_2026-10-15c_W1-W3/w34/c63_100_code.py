

# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c62 (2026-10-15): item 100 as in c61 with a planned binding (c63) -- a circuit inside
# the scope is built once (the formula path: item 87's chain, item 91's graph or item 99's native gates); new angles are
# bound in closed form only (no pass manager, no simulation); a shape signature guards every value-dependent branch;
# each shape keeps its emission plan, so a bind multiplies each run's factors and merges it without re-emitting.
# ---------------------------------------------------------------------------------------------------------------------

TEMPLATE100_STATS = {"built": 0, "bound": 0, "rebuilt": 0, "switched": 0, "structure_mismatch": 0,
                     "block_not_exact": 0}
TEMPLATE_BATCH = True  # item 100 (c63): items 87/91's runs in one numpy batch per bind (False: CPython, per run)
TEMPLATE_CACHE = 256  # item 100: shapes kept per template (a shape seen before is not built again)
_T100_HOOK = [None]  # item 100: set while compile_template reads items 87/91's structure (they stop before Qiskit)


_T100_RZPI = _n99_rz(math.pi)  # c63: a kept object, so that runs of constant factors are recognised


class _T100Captured(Exception):
    """Item 100: raised by the hook inside items 87/91 with the structure they built (nothing is compiled)."""

    def __init__(self, info):
        super().__init__("item 100: structure captured")
        self.info = info


def _t100_hook(info):
    raise _T100Captured(info)


def _t100_params(op):
    return [float(p) for p in op.params] if op.params else []


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
        self.fixed_m, self.fixed_mc = {}, {}
        for i in keep:
            fixed = src[i][0] == "fixed" or (src[i][0] in ("pre", "suf") and
                                             (self.pre_sl if src[i][0] == "pre" else self.suf_sl)[src[i][1]] is None)
            if fixed and len(a[i]) == 1 and log.data[i].operation.name != "measure":
                m = _n99_t(log.data[i].operation)
                self.fixed_m[i], self.fixed_mc[i] = m, (m[3], m[2], m[1], m[0])  # c63: X m X kept too
        self.log = [(log.data[i].operation, a[i], log.find_bit(log.data[i].clbits[0]).index
                     if log.data[i].clbits else None, src[i], self.fixed_m.get(i), self.fixed_mc.get(i))
                    for i in keep]

    def _frame_items(self, values):
        f = [0] * self.n
        X = _N99_X
        t0 = self.term0
        for op, qs, cb, s, fm, fmc in self.log:
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
                    yield ("1", a, _T100_RZPI)
                    yield ("1", b, _T100_RZPI)
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
            if fm is not None:  # c63: a fixed gate's matrix, and its X-conjugate, are kept objects (constant)
                yield ("1", q, fmc if f[q] else fm)
                continue
            if s[0] == "term":
                M = _t100_rot1({"rx": "X", "ry": "Y", "rz": "Z"}[nm], prm / 2)
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
        self.sup = [tuple(_chain_support(lb)) for lb, _ in self.terms]  # c63: each term's support, read once
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
        (q, ch), = self.sup[ref]
        return q, _t100_rot1(ch, values[self.term0 + ref])

    def _block_items(self, values):
        t0 = self.term0
        # c63: every general block's 4 x 4 matrix first, then their Weyl decompositions with one batched check
        gen = []
        for b in self.prog:
            if b[0] == "block" and len(b[2]) > 1 and b[4] is None:
                qa, qb = b[1]
                U = _T63_I4
                for m in b[2]:
                    if m[0] == "t":
                        a = values[t0 + m[1]]
                        U = (math.cos(a) * _T63_I4 - 1j * math.sin(a) * self.paulis[m[1]]) @ U
                    else:
                        q, (x, y, z, w) = self._one(m[1], m[3], values)
                        U = _t100_k1(x, y, z, w, q == qa) @ U
                gen.append((U, qa, qb))
        gen = iter(_t63_blocks(gen))
        for b in self.prog:
            if b[0] != "block":
                q, M = self._one(b[0], b[2], values)
                yield ("1", q, M)
            elif len(b[2]) == 1:  # one two-qubit term: a local Clifford around RZZ(2a), exact in closed form
                k = b[2][0][1]
                (q1, c1), (q2, c2) = self.sup[k]
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
                for st in next(gen):  # c63: a Weyl block's single-qubit factors pin their run
                    yield st + (1,) if st[0] == "1" else st
        yield ("end",)
        for q, cb in self.meas:
            yield ("meas", self.perm[q], cb)

    def items(self, values):
        return self._frame_items(values) if self.frame else self._block_items(values)


import itertools as _itertools

_T63_I4 = np.eye(4, dtype=complex)


def _t63_blocks(blocks):
    """Item 100 (c63): `_t100_block_steps` for several blocks [(U, q0, q1)], with the check of every closed-form
    three-CNOT block (the Weyl decomposition reproduces U to item 99's block tolerance) done in one numpy batch.
    A block with c = 0 (one interaction, or two cz) takes `_t100_block_steps` itself, with its own check."""
    from qiskit.synthesis import TwoQubitWeylDecomposition
    out, chk = [], []
    t = lambda m: (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))  # noqa: E731
    for U, q0, q1 in blocks:
        d = TwoQubitWeylDecomposition(U, fidelity=None)
        if abs(d.c) < _N99_TOL:
            out.append(_t100_block_steps(U, q0, q1))
            continue
        out.append([("1", q0, t(d.K2r)), ("1", q1, t(d.K2l))] + _t100_canonical(q0, q1, d.a, d.b, d.c) +
                   [("1", q0, t(d.K1r)), ("1", q1, t(d.K1l))])
        chk.append((U, d))
    if chk:
        n = len(chk)
        st = lambda f: np.array([f(d) for _, d in chk])  # noqa: E731
        k1l, k1r, k2l, k2r = st(lambda d: d.K1l), st(lambda d: d.K1r), st(lambda d: d.K2l), st(lambda d: d.K2r)
        K1 = (k1l[:, :, None, :, None] * k1r[:, None, :, None, :]).reshape(n, 4, 4)
        K2 = (k2l[:, :, None, :, None] * k2r[:, None, :, None, :]).reshape(n, 4, 4)
        abc = np.array([(d.a, d.b, d.c) for _, d in chk])
        Nd = np.exp(1j * (abc @ _T100_XYZ))
        W = K1 @ ((_T100_MAGIC[None] * Nd[:, None, :]) @ _T100_MAGIC.conj().T) @ K2
        Us = np.array([U for U, _ in chk])
        fu, fw = Us.reshape(n, 16), W.reshape(n, 16)
        k = np.argmax(np.abs(fu), axis=1)
        r = np.arange(n)
        ph = fu[r, k] / fw[r, k]
        ok = (np.abs(np.abs(ph) - 1) < _N99_BLOCK_TOL) & \
            (np.max(np.abs(fu - ph[:, None] * fw), axis=1) < _N99_BLOCK_TOL)
        if not ok.all():
            TEMPLATE100_STATS["block_not_exact"] += 1
            raise TranspilerError("item 100: a block's native form failed its check")
    return out


def _t63_full(U, use_rx):
    """Item 100 (c63): a pinned run -- one with a factor of a Weyl block -- in the full form whatever its values
    (rz sx rz sx rz, or rz rx rz with rx; exact for every U, theta = 0 included), so that a Weyl decomposition's
    discrete branches do not change the template's shape."""
    theta, phi, lam = _n99_zyz(U)
    if use_rx:
        return "full", (lam - _N99_PI / 2, theta, phi + _N99_PI / 2)
    return "full", (lam, theta + _N99_PI, phi + _N99_PI)


def _t63_emit(items, phys, target, use_rx, use_x=False):
    """Item 100 (c63): `_t100_emit` for templates. An item ("1", q, M, 1) pins its run to the full form
    (`_t63_full`); every other run is emitted exactly as `_t100_emit` emits it."""
    shape, slots, pending, pinned = [], [], {}, set()
    cz_q, rzz_q = target["cz"], (target["rzz"] if "rzz" in target.operation_names else None)

    def flush(q):
        U = pending.pop(q, None)
        if U is None:
            return
        p = phys[q]
        if q in pinned:
            pinned.discard(q)
            _, ang = _t63_full(U, use_rx)
            if use_rx:
                shape.extend((("rz", p), ("rx", p), ("rz", p)))
            else:
                shape.extend((("rz", p), ("sx", p), ("rz", p), ("sx", p), ("rz", p)))
            slots.extend(ang)
            return
        cls, ang = _n99_run_class(U, use_rx)
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
            if len(it) > 3:
                pinned.add(q)
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
        # c63: items 87/91's runs are merged in one numpy batch per bind (their outputs are checked for exactness, not
        # compared value for value); item 99's runs keep CPython's arithmetic, bit for bit c60's
        self.batch = gen.kind != "native" and TEMPLATE_BATCH
        self._build(gen.values0 if hasattr(gen, "values0") else None)
        TEMPLATE100_STATS["built"] += 1

    def _circuit(self, shape, plan):
        """The parametrised circuit of an output shape: a run of constant factors (`plan`) with its angles as numbers,
        every other angle a parameter (in slot order)."""
        from qiskit.circuit import ParameterVector
        pv = ParameterVector("psf_t100", plan["nslots"])
        fixed = plan["fixed"]  # shape slot position -> constant angle
        remap = plan["remap"]  # shape slot position -> parameter index
        qc = self._gen.qc
        out = QuantumCircuit(self.target.num_qubits, name=qc.name)
        for cr in qc.cregs:
            out.add_register(cr)
        k = 0

        def ang():
            nonlocal k
            v = fixed[k] if k in fixed else pv[remap[k]]
            k += 1
            return v
        for g in shape:
            nm = g[0]
            if nm == "rz":
                out.rz(ang(), g[1])
            elif nm == "sx":
                out.sx(g[1])
            elif nm == "x":
                out.x(g[1])
            elif nm == "rx":
                out.rx(ang(), g[1])
            elif nm == "cz":
                out.cz(g[1], g[2])
            elif nm == "rzz":
                out.rzz(ang(), g[1], g[2])
            else:
                out.measure(g[1], g[2])
        out._layout = _n99_layout(qc, out, self._gen.phys)
        return out

    def _build(self, values, items=None, shape=None, slots=None):
        if values is None:
            values = self._gen.values_of(self._gen.qc)
        if items is None:
            items = list(self._gen.items(values))
        if shape is None:
            shape, slots = _t63_emit(iter(items), self._gen.phys, self.target, self.use_rx, self.use_x)
        kinds = tuple(it[0] for it in items)
        key = (kinds, shape)  # c63: constant runs depend on the items' kinds as well as on the shape
        hit = self._cache.get(key)
        if hit is None:
            again = list(self._gen.items(values))  # a kept object is the same object in both: a constant factor
            const = [it[0] == "1" and it[2] is ag[2] for it, ag in zip(items, again)]
            plan = self._plan(items, const, slots)
            hit = (self._circuit(shape, plan), plan)
            if len(self._cache) >= TEMPLATE_CACHE:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = hit
            new = True
        else:
            new = False
        self.circuit, self._planned = hit
        self._shape, self._values0 = shape, list(values)
        self._slots0 = [slots[i] for i in sorted(self._planned["remap"], key=self._planned["remap"].get)]
        self.signature = (self.kind, self._gen.input_sig, shape)
        return new

    def _run(self, U, pin=False):
        """Item 100 (c63): a merged run's class and angles exactly as `_t63_emit` emits them."""
        if pin:
            return _t63_full(U, self.use_rx)
        cls, ang = _n99_run_class(U, self.use_rx)
        if self.use_x and cls == "full" and abs(ang[1] - 2 * math.pi) < _N99_TOL:
            return "x", (ang[0] - math.pi / 2, ang[2] - math.pi / 2)
        return cls, ang

    def _plan(self, items, const, slots):
        """Item 100 (c63): the emission plan of a shape -- every run's factors (item positions, in the order
        `_t100_emit` multiplies them), class and slots; a run of constant factors only is fixed now (its angles become
        numbers in the circuit); every rzz's item and slot; the items' kinds."""
        pending, runs, rzz, pos = {}, [], [], 0
        fixed, remap, nvar = {}, {}, 0

        def flush(q):
            nonlocal pos, nvar
            idx = pending.pop(q, None)
            if idx is None:
                return
            U = _T100_ONE
            for i in idx:
                U = _n99_mul(items[i][2], U)
            pin = any(len(items[i]) > 3 for i in idx)
            cls, ang = self._run(U, pin)
            if all(const[i] for i in idx):
                for j, x in enumerate(ang):
                    fixed[pos + j] = x
            else:
                runs.append((tuple(idx), cls, nvar, len(ang), pin))
                for j in range(len(ang)):
                    remap[pos + j] = nvar + j
                nvar += len(ang)
            pos += len(ang)
        for i, it in enumerate(items):
            k = it[0]
            if k == "1":
                pending.setdefault(it[1], []).append(i)
            elif k == "cz" or k == "rzz":
                flush(it[1])
                flush(it[2])
                if k == "rzz":
                    rzz.append((i, nvar))
                    remap[pos] = nvar
                    nvar += 1
                    pos += 1
            elif k == "end":
                for q in sorted(pending):
                    flush(q)
            else:
                flush(it[1])
        if pos != len(slots):
            raise TranspilerError("item 100: the emission plan does not match the shape")
        return dict(kinds=[it[0] for it in items], runs=runs, rzz=rzz, nslots=nvar, fixed=fixed, remap=remap)

    def _batch_plan(self, p):
        """Item 100 (c63): a plan's runs as index arrays for `_batch` (made once per shape)."""
        runs = p["runs"]
        need, rows = [], []
        for idx, *_ in runs:
            rows.append(list(range(len(need), len(need) + len(idx))))
            need.extend(idx)
        lmax = max((len(r) for r in rows), default=0)
        IDX = np.full((len(runs), lmax), len(need), dtype=np.intp)  # padded with the identity (the last row)
        for k, r in enumerate(rows):
            IDX[k, :len(r)] = r
        code = {"none": 0, "z": 1, "half": 2, "full": 3, "x": 4}
        want = np.array([5 if r[4] else code[r[1]] for r in runs], dtype=np.int8)
        groups = {}
        for k, r in enumerate(runs):
            groups.setdefault(int(want[k]), []).append(k)
        groups = {c: (np.array(ks, dtype=np.intp), np.array([runs[k][2] for k in ks], dtype=np.intp))
                  for c, ks in groups.items()}
        b = dict(need=need, IDX=IDX.T.copy(), want=want, groups=groups,
                 rzz_i=[i for i, _ in p["rzz"]], rzz_pos=np.array([q for _, q in p["rzz"]], dtype=np.intp))
        p["batch"] = b
        return b

    def _batch(self, items, p):
        """Item 100 (c63): every planned run's product, ZYZ angles and class in one numpy batch (as `_n99_zyz` and
        `_n99_run_class`, to floating-point rounding); None if a run's class differs from the plan's (the shape
        changed: the caller re-emits)."""
        b = p.get("batch") or self._batch_plan(p)
        slots = np.zeros(p["nslots"])
        if b["rzz_i"]:
            slots[b["rzz_pos"]] = [items[i][3] for i in b["rzz_i"]]
        if not b["need"]:
            return slots.tolist()
        need = b["need"]
        A = np.fromiter(_itertools.chain.from_iterable([items[i][2] for i in need] + [_T100_ONE]), complex,
                        4 * len(need) + 4).reshape(-1, 4)
        IDX = b["IDX"]
        M = A[IDX[0]]
        a, bb, c, d = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
        for row in IDX[1:]:
            M = A[row]
            m0, m1, m2, m3 = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
            a, bb, c, d = m0 * a + m1 * c, m0 * bb + m1 * d, m2 * a + m3 * c, m2 * bb + m3 * d
        sq = np.sqrt(a * d - bb * c)
        a, c, d = a / sq, c / sq, d / sq
        ac, aa = np.abs(c), np.abs(a)
        pd, pc = np.angle(d), np.angle(c)
        diag, anti = ac <= 1e-15, (aa <= 1e-15) & (ac > 1e-15)
        theta = np.where(diag, 0.0, np.where(anti, _N99_PI, 2 * np.arctan2(ac, aa)))
        phi = np.where(diag, 2 * pd, np.where(anti, 2 * pc, pd + pc))
        lam = np.where(diag | anti, 0.0, pd - pc)
        z = np.fmod(phi + lam, 2 * _N99_PI)
        z = np.where(z <= -_N99_PI, z + 2 * _N99_PI, np.where(z > _N99_PI, z - 2 * _N99_PI, z))
        small = np.abs(theta) < _N99_TOL
        if self.use_rx:
            got = np.where(small, np.where(np.abs(z) < _N99_TOL, 0, 1), 3)
        else:
            got = np.where(small, np.where(np.abs(z) < _N99_TOL, 0, 1),
                           np.where(np.abs(theta - _N99_PI / 2) < _N99_TOL, 2, 3))
            if self.use_x:
                got = np.where((got == 3) & (np.abs(theta + _N99_PI - 2 * _N99_PI) < _N99_TOL), 4, got)
        want = b["want"]
        got = np.where(want == 5, 5, got)
        if not np.array_equal(got, want):
            return None
        for cl, (ks, pos) in b["groups"].items():
            if cl == 1:
                slots[pos] = z[ks]
            elif cl == 2:
                slots[pos] = lam[ks] - _N99_PI / 2
                slots[pos + 1] = phi[ks] + _N99_PI / 2
            elif cl == 4:
                slots[pos] = lam[ks] - _N99_PI / 2
                slots[pos + 1] = phi[ks] + _N99_PI / 2
            elif cl == 3 or cl == 5:
                if self.use_rx:
                    slots[pos] = lam[ks] - _N99_PI / 2
                    slots[pos + 1] = theta[ks]
                    slots[pos + 2] = phi[ks] + _N99_PI / 2
                else:
                    slots[pos] = lam[ks]
                    slots[pos + 1] = theta[ks] + _N99_PI
                    slots[pos + 2] = phi[ks] + _N99_PI
        return slots.tolist()

    def values_of(self, qc):
        return self._gen.values_of(qc)

    def bind_values(self, values):
        values = [float(v) for v in values]
        if len(values) != self._gen.nvalues:
            raise ValueError(f"item 100: {self._gen.nvalues} values expected, {len(values)} given")
        items = list(self._gen.items(values))
        p = self._planned
        slots = None
        if self.batch and [it[0] for it in items] == p["kinds"]:
            slots = self._batch(items, p)
        elif [it[0] for it in items] == p["kinds"]:  # c63: the planned emission, the same arithmetic as _t100_emit
            slots = [0.0] * p["nslots"]  # constant runs are already numbers in the circuit
            mul, one, run = _n99_mul, _T100_ONE, self._run
            for idx, cls, pos, m, pin in p["runs"]:
                U = one
                for i in idx:
                    U = mul(items[i][2], U)
                c, ang = run(U, pin)
                if c != cls:
                    slots = None
                    break
                slots[pos:pos + m] = ang
            if slots is not None:
                for i, pos in p["rzz"]:
                    slots[pos] = items[i][3]
        if slots is None:  # a value-dependent branch changed the shape: built anew, or taken from the cache
            shape, full = _t63_emit(iter(items), self._gen.phys, self.target, self.use_rx, self.use_x)
            if self._build(values, items, shape, full):
                TEMPLATE100_STATS["rebuilt"] += 1
            else:
                TEMPLATE100_STATS["switched"] += 1
            slots = self._slots0
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
