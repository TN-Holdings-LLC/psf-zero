
# ======================================================================================================================
# Item 103 (prototype c66, home, 2026-10-16): item 88's frame in a template by segments, solved over GF(2).
#
# Option A (c63a, kept by the owner) makes every rzz's frame local: a negative angle becomes X RZZ(|theta|) X on its
# first qubit, and the runs on both sides are pinned to the full form, so a sign change never changes the shape. Its
# cost is an extra pulse in every pinned run that would otherwise be z-only or empty: on fractional targets, same-sign
# inputs are up to 1.21-1.35 x c62's duration (C63A-VAL, Addendum 507).
#
# Item 103 keeps the shape fixed without those pulses. On every qubit, the runs that are not diagonal ("anchors": a
# term's rx or ry, a non-diagonal prefix or suffix gate, a non-diagonal fixed gate) cut its timeline into segments. A
# segment between two anchors may be conjugated by X as a whole: the two X's merge into the anchors' runs (their class
# does not change: X U has the same |U_00| as U up to the order of the factors) and every rzz inside it on that qubit
# changes sign. Which segments to conjugate, so that every rzz angle is positive, is a linear system over GF(2): one
# variable per inner segment, one equation per rzz (c_a + c_b = [theta < 0]). Its spanning forest is found once, at
# build; a bind solves it by one pass over the forest's edges.
#
# - An rzz whose qubits are each before their first anchor (a basis state: the rzz is a phase) or, in a measured
#   circuit, after their last anchor (only diagonal operations follow until the measurement) needs no positive angle:
#   it is emitted with |theta|.
# - An rzz that closes a cycle of the forest, or joins two fixed segments, is emitted as in option A (pinned, local X).
#   On a chain there are none.
# - A fold (|theta| > pi/2) is Rz(pi) on both qubits, a Z frame carried to the next anchor (or dropped on a basis state
#   or before a measurement); in the pinned case, as in option A.
#
# The items' kinds and every run's class are the same for every value (up to values of measure zero), so a template
# never changes its shape. Exactness is checked by the prototype's tests against the input, not assumed.
# ----------------------------------------------------------------------------------------------------------------------
_T103_I = (1 + 0j, 0j, 0j, 1 + 0j)
_T103_Z = (1 + 0j, 0j, 0j, -1 + 0j)
_T103_TOL = 1e-12
T103_STATS = {"plans": 0, "tree": 0, "free": 0, "pinned": 0, "anchors": 0}


def _t103_new(M):
    """A new tuple object with M's entries (never a kept object: `CompileTemplate._build` treats an item that is the
    same object in two generations as a constant factor)."""
    return (M[0] + 0j, M[1] + 0j, M[2] + 0j, M[3] + 0j)


def _t103_xc(M):
    """X M X, as a new tuple."""
    return (M[3] + 0j, M[2] + 0j, M[1] + 0j, M[0] + 0j)


def _t103_bound(c, z):
    """X^c Z^z as a new tuple (the order only changes a global phase)."""
    if c and z:
        return (0j, -1 + 0j, 1 + 0j, 0j)
    if c:
        return (0j, 1 + 0j, 1 + 0j, 0j)
    if z:
        return (1 + 0j, 0j, 0j, -1 + 0j)
    return (1 + 0j, 0j, 0j, 1 + 0j)


def _t103_matrix(self, op, s, fm, values):
    """A single-qubit log entry's matrix at `values` (as `_frame_items` emits it, without a frame)."""
    nm = op.name
    if fm is not None:
        return fm
    if s[0] == "term":
        return _t100_rot1({"rx": "X", "ry": "Y", "rz": "Z"}[nm], values[self.term0 + s[1]])
    if s[0] == "pre" or s[0] == "suf":
        sl = (self.pre_sl if s[0] == "pre" else self.suf_sl)[s[1]]
        return _n99_t(op) if sl is None else _n99_t(_T100Op(op, values[sl[0]:sl[1]]))
    return _n99_t(op)


def _t103_plan(self):
    """Item 103: the segments, the anchors and the rzz equations of this template's log, and the spanning forest that
    solves them (made once, from the build values)."""
    values = self.values0
    n = self.n
    seg = [0] * n
    anchor = set()
    edges = []  # (log index, a, seg_a, b, seg_b)
    measured = False
    # the single-qubit runs (consecutive single-qubit entries on a qubit between its two-qubit entries); a run is an
    # anchor when its product at the build values is neither diagonal nor anti-diagonal (X U, U X and X U X then have
    # its class), and its first non-diagonal entry is where the segments meet
    runs, cur = [], {}
    for i, (op, qs, cb, s, fm, fmc) in enumerate(self.log):
        if op.name == "measure" or len(qs) == 2:
            for q in qs:
                if cur.get(q):
                    runs.append(cur.pop(q))
            continue
        cur.setdefault(qs[0], []).append(i)
    runs.extend(r for r in cur.values() if r)
    for r in runs:
        U, first = _T103_I, None
        for i in r:
            op, qs, cb, s, fm, fmc = self.log[i]
            M = _t103_matrix(self, op, s, fm, values)
            U = _n99_mul(M, U)
            if first is None and (abs(M[1]) > _T103_TOL or abs(M[2]) > _T103_TOL):
                first = i
        if first is not None and min(abs(U[0]), abs(U[1])) > 1e-6:
            anchor.add(first)
    for i, (op, qs, cb, s, fm, fmc) in enumerate(self.log):
        nm = op.name
        if nm == "measure":
            measured = True
            continue
        if nm == "rzz":
            a, b = qs
            edges.append((i, a, seg[a], b, seg[b]))
            continue
        if i in anchor:
            seg[qs[0]] += 1
    last = seg  # number of anchors per qubit: segments 0 .. last[q]; 0 is "start", last[q] (> 0) is "end"

    def kind(q, k):
        if k == 0:
            return "S"
        if k == last[q]:
            return "E"
        return "I"
    node = {}
    for q in range(n):
        for k in range(1, last[q]):
            node[(q, k)] = len(node) + 1  # 0 is the ground (fixed segments, c = 0)
    parent = list(range(len(node) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ekind, enodes, tree_adj = {}, {}, {}
    for i, a, ka, b, kb in edges:
        ta, tb = kind(a, ka), kind(b, kb)
        free_a = ta == "S" or (ta == "E" and measured)
        free_b = tb == "S" or (tb == "E" and measured)
        if free_a and free_b:
            ekind[i] = "free"
            continue
        u = node.get((a, ka), 0)
        v = node.get((b, kb), 0)
        enodes[i] = (u, v)
        ru, rv = find(u), find(v)
        if ru == rv:
            ekind[i] = "pin"
            continue
        parent[ru] = rv
        ekind[i] = "tree"
        tree_adj.setdefault(u, []).append((v, i))
        tree_adj.setdefault(v, []).append((u, i))
    order, seen = [], set()
    for root in [0] + list(range(1, len(node) + 1)):
        if root in seen:
            continue
        seen.add(root)
        stack = [root]
        while stack:
            x = stack.pop()
            for y, i in tree_adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    order.append((y, x, i))
                    stack.append(y)
    # Z frame at the end of a qubit whose last segment is not dropped (an unmeasured circuit): emitted pinned
    end_pin = set()
    if not measured:
        for i, a, ka, b, kb in edges:
            if ekind[i] == "tree":
                for q, k in ((a, ka), (b, kb)):
                    if kind(q, k) == "E":
                        end_pin.add(q)
    T103_STATS["plans"] += 1
    T103_STATS["anchors"] += len(anchor)
    for k in ("tree", "free", "pin"):
        T103_STATS["pinned" if k == "pin" else k] += sum(1 for v in ekind.values() if v == k)
    return dict(anchor=anchor, last=last, node=node, ekind=ekind, enodes=enodes, order=order, measured=measured,
                end_pin=end_pin, nnodes=len(node) + 1)


def _t103_frame_items(self, values):
    """Item 103: item 88's frame by segments (see the comment above); the same output distribution (or state) as
    c63a's, with no pinned run but where a cycle or a fixed segment needs one."""
    plan = getattr(self, "_t103", None)
    if plan is None:
        plan = self._t103 = _t103_plan(self)
    X = _N99_X
    t0 = self.term0
    log = self.log
    ekind, last, node, anchor = plan["ekind"], plan["last"], plan["node"], plan["anchor"]
    # each rzz's angle after the fold, then the segments' conjugations
    ang = {}
    neg = {}
    for i, k in ekind.items():
        th = 2 * values[t0 + log[i][3][1]] if log[i][3][0] == "term" else None
        if th is None:  # an rzz not traced to a term: as written
            th = float(log[i][0].params[0])
        th = math.pi - (math.pi - th) % (2 * math.pi)
        fold = abs(th) > math.pi / 2
        if fold:
            th -= math.copysign(math.pi, th)
        ang[i] = (th, fold)
        neg[i] = th < 0
    c = [0] * plan["nnodes"]
    for y, x, i in plan["order"]:
        c[y] = c[x] ^ neg[i]
    n = self.n
    seg = [0] * n
    zf = [0] * n

    def cseg(q, k):
        v = node.get((q, k))
        return c[v] if v is not None else 0

    def is_inner(q, k):
        return 0 < k < last[q]
    for i, (op, qs, cb, s, fm, fmc) in enumerate(log):
        nm = op.name
        if nm == "measure":
            yield ("meas", qs[0], cb)
            continue
        if nm == "rzz":
            a, b = qs
            th, fold = ang[i]
            k = ekind[i]
            if abs(th) < 1e-12 and not fold:
                continue
            if k == "free":
                yield ("rzz", a, b, abs(th))
                continue
            ca, cb_ = cseg(a, seg[a]), cseg(b, seg[b])
            eff = -th if ca != cb_ else th
            if k == "tree":
                if fold:
                    for q in (a, b):
                        if is_inner(q, seg[q]) or q in plan["end_pin"]:
                            zf[q] ^= 1
                yield ("rzz", a, b, abs(eff))
                continue
            # "pin": option A's local form, with this bind's effective sign
            Z = _T100_RZPI if fold else _T103_I
            Pa = _n99_mul((X[0], X[1], X[2], X[3]), Z) if eff < 0 else _t103_new(Z)
            yield ("1", a, Pa, 1)
            yield ("1", b, _t103_new(Z), 1)
            yield ("rzz", a, b, abs(eff))
            Q = X if eff < 0 else _T103_I
            yield ("1", a, _t103_new(Q), 1)
            continue
        q = qs[0]
        k = seg[q]
        if i in anchor:
            if is_inner(q, k):  # closing a conjugated segment: X^c and the segment's Z frame, into the anchor's run
                yield ("1", q, _t103_bound(cseg(q, k), zf[q]))
            zf[q] = 0
            M = _t103_matrix(self, op, s, fm, values)
            yield ("1", q, M if fm is not None and not is_inner(q, k) and not is_inner(q, k + 1) else _t103_new(M))
            seg[q] = k + 1
            if is_inner(q, k + 1):  # opening the next segment
                yield ("1", q, _t103_bound(cseg(q, k + 1), 0))
            continue
        M = _t103_matrix(self, op, s, fm, values)  # any other single-qubit operation inside segment k
        if is_inner(q, k):
            yield ("1", q, _t103_xc(M) if cseg(q, k) else _t103_new(M))
        else:
            yield ("1", q, M)
    for q in sorted(plan["end_pin"]):
        yield ("1", q, _t103_bound(0, zf[q]), 1)
    yield ("end",)


TEMPLATE_FRAME_SEGMENTS = True  # item 103 (prototype switch for the comparison with option A; removed on release)
_t103_option_a = _T100Formula._frame_items


def _t103_items(self, values):
    if TEMPLATE_FRAME_SEGMENTS:
        return _t103_frame_items(self, values)
    return _t103_option_a(self, values)


_T100Formula._frame_items = _t103_items
