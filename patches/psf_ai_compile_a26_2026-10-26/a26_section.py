

# ---------------------------------------------------------------------------------------------------------------------
# a26 (2026-10-26): item 50, the ring folded onto one of the device's own cycles, the default (PSF_RING_FOLD=0: a25).
# Ported from TOQB toqb/fold.py (fold:rule, 145f38b; results/flow-e-2026-10-26, flow-f-2026-10-26).
# ---------------------------------------------------------------------------------------------------------------------

from collections import defaultdict  # noqa: E402

A26_STATS = {"fold_pend": 0, "fold_holes": 0, "fold_zipper_fewer": 0, "fold_no_cycle": 0, "fold_not_ring": 0,
             "fold_off": 0, "fold_permuted": 0}


class FoldRefused(RuntimeError):
    pass


def _fold_on():
    """Item 50: the fold is the default; PSF_RING_FOLD=0 gives a25's output."""
    return os.environ.get("PSF_RING_FOLD", "1") != "0"


def _fold_works(target, name, qa, max_error):
    if name not in target.operation_names:
        return False
    p = target[name].get(qa)
    return p is not None and (p.error is None or p.error < max_error)


def _fold_edges(target, max_error):
    """Working CZ couplers as sorted pairs: every direction the Target lists must work (a direction listed and
    failed removes the pair)."""
    pairs = defaultdict(list)
    for qa in target["cz"]:
        if qa is not None:
            pairs[tuple(sorted(qa))].append(_fold_works(target, "cz", qa, max_error))
    return {e for e, oks in pairs.items() if all(oks)}


def _fold_qubit_ok(target, q, max_error):
    return all(_fold_works(target, g, (q,), max_error) for g in ("sx", "measure")) and (
        "x" not in target.operation_names or _fold_works(target, "x", (q,), max_error))


def fold_lattice(edges, nq):
    """Item 50: heavy-hex read off the coupling graph: bridges (two neighbours, indices not consecutive), rows (what
    is left without them), and per row pair its bridges (upper qubit, bridge, lower qubit), left to right."""
    nb = defaultdict(set)
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    bridges = {q for q in range(nq) if len(nb[q]) == 2 and all(abs(x - q) > 1 for x in nb[q])}
    row_of, nrows = {}, 0
    for q in range(nq):
        if q in bridges or q in row_of or not nb[q]:
            continue
        stack = [q]
        row_of[q] = nrows
        while stack:
            u = stack.pop()
            for v in nb[u]:
                if v not in bridges and v not in row_of:
                    row_of[v] = nrows
                    stack.append(v)
        nrows += 1
    pairs = defaultdict(list)
    for b in bridges:
        u, v = sorted(nb[b], key=lambda x: row_of.get(x, -1))
        if u in row_of and v in row_of and row_of[u] != row_of[v]:
            pairs[(row_of[u], row_of[v])].append((u, b, v))
    for k in pairs:
        pairs[k].sort()
    return pairs


def _fold_cell(rp, j, pairs):
    (u1, b1, v1), (u2, b2, v2) = pairs[rp][j], pairs[rp][j + 1]
    e = set()
    for a in range(min(u1, u2), max(u1, u2)):
        e.add((a, a + 1))
    for a in range(min(v1, v2), max(v1, v2)):
        e.add((a, a + 1))
    for x, y in ((u1, b1), (b1, v1), (u2, b2), (b2, v2)):
        e.add((min(x, y), max(x, y)))
    return e


def _fold_as_cycle(e):
    nb = defaultdict(list)
    for a, b in e:
        nb[a].append(b)
        nb[b].append(a)
    if not nb or any(len(v) != 2 for v in nb.values()):
        return None
    start = min(nb)
    cyc, prev, cur = [start], None, start
    while True:
        nxt = nb[cur][0] if nb[cur][0] != prev else nb[cur][1]
        if nxt == start:
            break
        cyc.append(nxt)
        prev, cur = cur, nxt
    return cyc if len(cyc) == len(nb) else None


def fold_shapes(pairs):
    """Item 50: outlines of k cells side by side in a row pair (8k + 4 qubits), and of k cells plus k2 side by side
    in the next row pair down when the outline is one cycle: [(length, name, qubits in cycle order)], in a fixed order
    (row pairs top to bottom, cells left to right)."""
    out, keys = [], sorted(pairs)
    for rp in keys:
        nc = len(pairs[rp]) - 1
        for k in range(1, nc + 1):
            for j in range(nc - k + 1):
                e = set()
                for c in range(j, j + k):
                    e ^= _fold_cell(rp, c, pairs)
                cyc = _fold_as_cycle(e)
                if cyc:
                    out.append((len(cyc), ("row", rp, j, k), cyc))
                for rb in (kk for kk in keys if kk[0] == rp[1]):
                    nb2 = len(pairs[rb]) - 1
                    for k2 in range(1, nb2 + 1):
                        for c in range(nb2 - k2 + 1):
                            e2 = set(e)
                            for cc in range(c, c + k2):
                                e2 ^= _fold_cell(rb, cc, pairs)
                            cyc2 = _fold_as_cycle(e2)
                            if cyc and cyc2:
                                out.append((len(cyc2), ("row+row", rp, j, k, rb, c, k2), cyc2))
    return out


def _fold_cands(target, max_error):
    edges = _fold_edges(target, max_error)
    good = []
    for length, name, cyc in fold_shapes(fold_lattice(edges, target.num_qubits)):
        if all(_fold_qubit_ok(target, q, max_error) for q in cyc) and all(
                tuple(sorted((cyc[i], cyc[(i + 1) % length]))) in edges for i in range(length)):
            good.append((length, name, cyc))
    return edges, good


def fold_place_holes(target, n, max_error=FAILED_MAX_ERROR):
    """Item 50: (m, name, cycle) -- the shortest working outline of m >= n qubits, the first in the fixed order."""
    _, good = _fold_cands(target, max_error)
    for m in sorted({g[0] for g in good}):
        if m >= n:
            return next(g for g in good if g[0] == m)
    return None


def fold_place_pend(target, n, max_error=FAILED_MAX_ERROR):
    """Item 50: (m, name, cycle, {cycle position: pendant qubit}) -- the longest working outline of m <= n qubits
    with n - m cycle nodes that have a working outside neighbour; pendants spread evenly over those nodes in cycle
    order, each node's smallest working outside neighbour not used before."""
    edges, good = _fold_cands(target, max_error)
    nb = defaultdict(set)
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    for m in sorted({g[0] for g in good}, reverse=True):
        if m > n:
            continue
        k = n - m
        if k >= m:
            continue
        for length, name, cyc in good:
            if length != m:
                continue
            inside, used, spots = set(cyc), set(), []
            for p, q in enumerate(cyc):
                outs = sorted(x for x in nb[q] if x not in inside and _fold_qubit_ok(target, x, max_error))
                if outs:
                    spots.append((p, outs))
            if len(spots) < k:
                continue
            pick = [spots[(j * len(spots)) // k] for j in range(k)] if k else []
            pend = {}
            for p, outs in pick:
                free = [x for x in outs if x not in used]
                if not free:
                    break
                pend[p] = free[0]
                used.add(free[0])
            if len(pend) == k:
                return m, name, cyc, pend
    return None


def _fold_read(qc):
    """Item 50: a ring classifier (layers of single-qubit gates, each followed by the n CZs of one cycle through all
    qubits, the same cycle every layer; then single-qubit gates and measurements) as (cycle order, layers, final
    gates, measurements) with qubits relabelled along the cycle; FoldRefused otherwise."""
    n = qc.num_qubits
    if n < 4:
        raise FoldRefused("fewer than 4 qubits")
    layers, cur, seen, meas, ring = [], [[] for _ in range(n)], set(), [], None
    for ins in qc.data:
        nm = ins.operation.name
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if nm == "cz":
            e = frozenset(qs)
            if e in seen or (ring is not None and e not in ring):
                raise FoldRefused("not one CZ cycle per layer")
            seen.add(e)
            if len(seen) == n:
                if ring is None:
                    ring = set(seen)
                layers.append(cur)
                cur, seen = [[] for _ in range(n)], set()
        elif nm == "measure":
            meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))
        elif nm == "barrier":
            continue
        elif len(qs) == 1 and not seen:
            if meas:
                raise FoldRefused("a gate after a measurement")
            cur[qs[0]].append(ins.operation)
        else:
            raise FoldRefused(f"{nm} on {qs} inside a ring layer")
    if seen or ring is None:
        raise FoldRefused("an incomplete ring layer")
    nb = defaultdict(list)
    for e in ring:
        a, b = sorted(e)
        nb[a].append(b)
        nb[b].append(a)
    if any(len(nb[q]) != 2 for q in range(n)):
        raise FoldRefused("the CZs do not form one cycle")
    for q in nb:
        nb[q].sort()  # the cycle is read from qubit 0 towards its smaller neighbour (0, 1, 2, ... for TOQB's ring)
    order, prev, c = [0], None, 0
    while True:
        nx = nb[c][0] if nb[c][0] != prev else nb[c][1]
        if nx == 0:
            break
        order.append(nx)
        prev, c = c, nx
    if len(order) != n:
        raise FoldRefused("the CZs do not form one cycle")
    pos = {q: i for i, q in enumerate(order)}
    layers = [[layer[q] for q in order] for layer in layers]
    final = [cur[q] for q in order]
    meas = [(pos[q], c) for q, c in meas]
    return order, layers, final, meas


def _fold_spread(n, m):
    h = m - n
    at = {round((k + 0.5) * m / h) % m for k in range(h)} if h else set()
    while len(at) < h:
        at.add(max(set(range(m)) - at))
    occ, q = [], 0
    for p in range(m):
        if p in at:
            occ.append(None)
        else:
            occ.append(q)
            q += 1
    return occ


def _fold_holes(n, layers, final, meas, cyc, nq, ncl):
    """Item 50, holes: the m - n spare qubits of the cycle start in |0>; per layer CZ on every ring edge already
    adjacent (rounds of disjoint pairs), then for each run of holes the ring node before it moves across (CX a->h,
    CX h->a), then CZ across: n + 2(m - n) two-qubit gates per layer."""
    m = len(cyc)
    occ = _fold_spread(n, m)
    start = {r: cyc[p] for p, r in enumerate(occ) if r is not None}
    out = QuantumCircuit(nq, ncl)
    for layer in layers:
        pos = {r: p for p, r in enumerate(occ) if r is not None}
        for r in range(n):
            for op in layer[r]:
                out.append(op, [cyc[pos[r]]])
        edges = [p for p in range(m) if occ[p] is not None and occ[(p + 1) % m] is not None
                 and ((occ[p] + 1) % n == occ[(p + 1) % m] or (occ[(p + 1) % m] + 1) % n == occ[p])]
        rounds = []
        for p in edges:
            for rd in rounds:
                if all({p, (p + 1) % m}.isdisjoint({q, (q + 1) % m}) for q in rd):
                    rd.append(p)
                    break
            else:
                rounds.append([p])
        for rd in rounds:
            for p in rd:
                out.cz(cyc[p], cyc[(p + 1) % m])
        for p in [p for p in range(m) if occ[p] is None and occ[(p - 1) % m] is not None]:
            cur, pb = (p - 1) % m, (p + 1) % m
            while occ[pb] is None:
                pb = (pb + 1) % m
            while occ[(cur + 1) % m] is None:
                nx = (cur + 1) % m
                out.cx(cyc[cur], cyc[nx])
                out.cx(cyc[nx], cyc[cur])
                occ[nx], occ[cur] = occ[cur], None
                cur = nx
            out.cz(cyc[cur], cyc[pb])
    pos = {r: p for p, r in enumerate(occ) if r is not None}
    for r in range(n):
        for op in final[r]:
            out.append(op, [cyc[pos[r]]])
    for q, c in meas:
        out.measure(cyc[pos[q]], c)
    return out, [start[r] for r in range(n)], [cyc[pos[r]] for r in range(n)]


def _fold_pend(n, layers, final, meas, cyc, pend, nq, ncl):
    """Item 50, pendants: the ring on the cycle, its n - m extra nodes hung off it; per layer each pendant node
    trades places with its cycle node by one CZ + SWAP block (two two-qubit gates): n + (n - m) per layer."""
    m = len(cyc)
    if len(pend) >= m:
        raise FoldRefused("every cycle node has a pendant")
    oc, op, r = [None] * m, {}, 0
    for p in range(m):
        oc[p] = r
        r += 1
        if p in pend:
            op[p] = r
            r += 1
    if r != n:
        raise FoldRefused("pendants do not match the ring size")

    def where():
        w = {v: cyc[p] for p, v in enumerate(oc)}
        w.update({v: pend[p] for p, v in op.items()})
        return w

    start = where()
    out = QuantumCircuit(nq, ncl)
    for layer in layers:
        w = where()
        for i in range(n):
            for o in layer[i]:
                out.append(o, [w[i]])
        need = {frozenset((i, (i + 1) % n)) for i in range(n)}
        while need:
            progress = False
            rounds = []
            for p in range(m):
                e = frozenset((oc[p], oc[(p + 1) % m]))
                if e in need:
                    for rd in rounds:
                        if all({p, (p + 1) % m}.isdisjoint({q, (q + 1) % m}) for q in rd):
                            rd.append(p)
                            break
                    else:
                        rounds.append([p])
                    need.discard(e)
            for rd in rounds:
                for p in rd:
                    out.cz(cyc[p], cyc[(p + 1) % m])
                progress = True
            for p in sorted(op):
                u, v = oc[p], op[p]
                e = frozenset((u, v))
                other = frozenset((u, (u - 1) % n)) if (u + 1) % n == v else frozenset((u, (u + 1) % n))
                if e in need and other not in need:
                    a, b = cyc[p], pend[p]
                    out.h(b)  # CZ then SWAP on (a, b) = H(b), CX(b, a), CX(a, b), H(a)
                    out.cx(b, a)
                    out.cx(a, b)
                    out.h(a)
                    oc[p], op[p] = v, u
                    need.discard(e)
                    progress = True
            if not progress:
                raise FoldRefused("no schedule step left")
    w = where()
    for i in range(n):
        for o in final[i]:
            out.append(o, [w[i]])
    for q, c in meas:
        out.measure(w[q], c)
    return out, [start[i] for i in range(n)], [w[i] for i in range(n)]


def fold_choice(target, n, max_error=FAILED_MAX_ERROR):
    """Item 50's rule: of the extra two-qubit gates per layer -- pendants n - m_below, holes 2 (m_above - n), the
    zipper (5n - 8)/2 - n or (5n - 7)/2 - n -- the fewest decides (ties: pendants, then holes). ("pend", placement),
    ("holes", placement) or ("zipper", None)."""
    below, above = fold_place_pend(target, n, max_error), fold_place_holes(target, n, max_error)
    ep = (n - below[0]) if below else None
    eh = 2 * (above[0] - n) if above else None
    ez = ((5 * n - 8) // 2 if n % 2 == 0 else (5 * n - 7) // 2) - n
    best = min(e for e in (ep, eh, ez) if e is not None)
    if ep is not None and ep == best:
        return "pend", below
    if eh is not None and eh == best:
        return "holes", above
    return "zipper", None


class FoldTemplate:
    """Item 50: a ring classifier folded onto a device cycle, compiled once into a parametrised circuit (Qiskit's
    level-1 preset on the fixed layout, no routing); `bind(qc)` checks the structure and assigns the input's angles.
    `layout`: the device qubit of each input qubit at the start; `final_layout`: at the end; `schedule`: "pend" or
    "holes"; `cycle`: the cycle's length."""

    def __init__(self, sig, circuit, order, layout, final_layout, schedule, cycle):
        self.sig, self.circuit, self.order = sig, circuit, order
        self.layout, self.final_layout, self.schedule, self.cycle = layout, final_layout, schedule, cycle

    def values_of(self, qc):
        return [float(p) for i in qc.data for p in i.operation.params]

    def bind_values(self, values):
        present = set(self.circuit.parameters)
        out = self.circuit.assign_parameters({p: v for p, v in zip(self.order, values) if p in present})
        TEMPLATE_STATS["bound"] += 1
        return out

    def bind(self, qc):
        if _structure_sig(qc) != self.sig:
            TEMPLATE_STATS["structure_mismatch"] += 1
            raise ValueError("item 50: the circuit's structure differs from the template's")
        return self.bind_values(self.values_of(qc))


def fold_template(qc, target, seed=0, max_error=FAILED_MAX_ERROR):
    """Item 50: the FoldTemplate of `qc` on `target`, or None (not a ring classifier, no CZ, the zipper has fewer
    extra gates, no cycle by the formula, or an unmeasured circuit whose qubits would end permuted)."""
    from qiskit.circuit import ParameterVector
    if target is None or "cz" not in target.operation_names:
        return None
    try:
        order, _, _, _ = _fold_read(qc)
    except FoldRefused:
        A26_STATS["fold_not_ring"] += 1
        return None
    n = qc.num_qubits
    kind, placed = fold_choice(target, n, max_error)
    if kind == "zipper":
        A26_STATS["fold_zipper_fewer"] += 1
        return None
    values = [float(p) for i in qc.data for p in i.operation.params]
    pv = ParameterVector("f", len(values))
    par = qc.copy_empty_like()
    k = 0
    for ins in qc.data:
        op = ins.operation
        if op.params:
            op = op.to_mutable()
            op.params = [pv[k + j] for j in range(len(op.params))]
            k += len(op.params)
        par.append(op, ins.qubits, ins.clbits)
    _, layers, final, meas = _fold_read(par)
    try:
        if kind == "pend":
            m, _, cyc, pend = placed
            hc, init, fin = _fold_pend(n, layers, final, meas, cyc, pend, target.num_qubits, qc.num_clbits)
        else:
            m, _, cyc = placed
            hc, init, fin = _fold_holes(n, layers, final, meas, cyc, target.num_qubits, qc.num_clbits)
    except FoldRefused:
        A26_STATS["fold_no_cycle"] += 1
        return None
    measured = {q for q, _ in meas}
    if init != fin and len(measured) < n:
        A26_STATS["fold_permuted"] += 1
        return None
    pm = generate_preset_pass_manager(1, target=target, initial_layout=list(range(target.num_qubits)),
                                      routing_method="none", seed_transpiler=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tq = pm.run(hc)
    inv = {q: i for i, q in enumerate(order)}
    layout = [init[inv[q]] for q in range(n)]
    final_layout = [fin[inv[q]] for q in range(n)]
    A26_STATS["fold_pend" if kind == "pend" else "fold_holes"] += 1
    return FoldTemplate(_structure_sig(qc), tq, [pv[j] for j in range(len(values))], layout, final_layout, kind, m)


_A25_MODEL_TEMPLATE = model_template
_A25_COMPILE = compile_for_model_circuit


def model_template(qc, target, basis_gates=None, seed=0, max_error=0.5):  # noqa: F811
    """Item 50 (a26): an embeddable ring and every other input as a25; a ring classifier that does not embed is
    folded onto a device cycle by item 50's rule (FoldTemplate) when the rule picks pendants or holes; otherwise, and
    with PSF_RING_FOLD=0, a25's template."""
    if _fold_on() and target is not None and not _ring_embeds(qc, target, max_error):
        tpl = fold_template(qc, target, seed, max_error)
        if tpl is not None:
            A22_STATS["templates_ring"] += 1
            return tpl
    elif not _fold_on():
        A26_STATS["fold_off"] += 1
    return _A25_MODEL_TEMPLATE(qc, target, basis_gates, seed, max_error)


def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,  # noqa: F811
                              commutation_start=True, do_polish=True, layout_search=True, return_info=False,
                              target=None, **kwargs):
    """Item 50 (a26): as model_template -- a ring classifier that does not embed and that item 50's rule folds is
    built by the FoldTemplate and bound; everything else, and PSF_RING_FOLD=0, as a25."""
    seed = kwargs.get("seed_transpiler")
    seed = seed if seed is not None else (seeds[0] if seeds else 0)
    max_error = kwargs.get("prune_max_error", 0.5)
    if _fold_on() and target is not None and basis_gates is not None and not _ring_embeds(qc, target, max_error):
        tpl = fold_template(qc, target, seed, max_error)
        if tpl is not None:
            out = tpl.bind(qc)
            A19_STATS["formula"] += 1
            info = {"path": "formula", "chosen": "fold_" + tpl.schedule, "candidates": ["fold_" + tpl.schedule],
                    "version": AI_COMPILE_VERSION}
            return (out, info) if return_info else out
    return _A25_COMPILE(qc, coupling_map, basis_gates, entangling_basis=entangling_basis, seeds=seeds,
                        commutation_start=commutation_start, do_polish=do_polish, layout_search=layout_search,
                        return_info=return_info, target=target, **kwargs)
