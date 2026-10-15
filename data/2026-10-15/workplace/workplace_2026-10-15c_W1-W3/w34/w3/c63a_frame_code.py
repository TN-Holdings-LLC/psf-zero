
# ======================================================================================================================
# W3 option A (measurement variant c63a; not a proposal by itself): item 88's frame fixed at build in the template.
# A negative rzz angle becomes X RZZ(|theta|) X on its first qubit, locally, instead of flipping the Pauli frame; the
# items on both sides are pinned (`_t63_emit`), so a sign change alters angles only, never the shape. A fold (|theta|
# crossing pi/2) is pinned too; only an angle of exactly 0 still changes the shape. c63's default call is unchanged.
TEMPLATE_FRAME_FIXED = True
_t63a_orig_frame_items = _T100Formula._frame_items
_T63A_I = (1 + 0j, 0j, 0j, 1 + 0j)


def _t63a_frame_items(self, values):
    if not TEMPLATE_FRAME_FIXED:
        yield from _t63a_orig_frame_items(self, values)
        return
    X = _N99_X
    t0 = self.term0
    for op, qs, cb, s, fm, fmc in self.log:
        nm = op.name
        if nm == "measure":
            yield ("meas", qs[0], cb)
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
            th = math.pi - (math.pi - prm) % (2 * math.pi)
            fold = abs(th) > math.pi / 2
            if fold:
                th -= math.copysign(math.pi, th)
            if abs(th) < 1e-12 and not fold:
                continue
            # new tuple objects (never constant): the sign and the fold may change from bind to bind. Time order on
            # qubit a: the fold's Rz(pi), then X if theta < 0; on qubit b: the fold's Rz(pi). All pinned.
            Z = _T100_RZPI if fold else _T63A_I
            Pa = _n99_mul((X[0], X[1], X[2], X[3]), Z) if th < 0 else (Z[0], Z[1], Z[2], Z[3])
            yield ("1", a, Pa, 1)
            yield ("1", b, (Z[0], Z[1], Z[2], Z[3]), 1)
            yield ("rzz", a, b, abs(th))
            Q = X if th < 0 else _T63A_I
            yield ("1", a, (Q[0], Q[1], Q[2], Q[3]), 1)  # built anew: a literal tuple would be one kept object
            continue
        q = qs[0]
        if fm is not None:
            yield ("1", q, fm)
            continue
        if s[0] == "term":
            yield ("1", q, _t100_rot1({"rx": "X", "ry": "Y", "rz": "Z"}[nm], prm / 2))
        elif prm is None:
            yield ("1", q, _n99_t(op))
        else:
            yield ("1", q, _n99_t(_T100Op(op, prm)))
    yield ("end",)


_T100Formula._frame_items = _t63a_frame_items
