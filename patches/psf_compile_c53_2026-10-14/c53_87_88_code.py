

# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c51 (2026-10-14): item 87, a chain Hamiltonian built by formula as a brickwork (N1);
# item 88, on a Target with rzz and rx, its terms emitted as rzz / rx with the signs carried in a Pauli X frame
# ---------------------------------------------------------------------------------------------------------------------

import inspect as _inspect  # noqa: E402  (item 87)

CHAIN_BRICKWORK = True  # item 87 (proposal): chain Pauli evolutions built as a brickwork on a path of the device
CHAIN_FRAME = True  # item 88 (proposal): with rzz and rx in the Target, the chain's terms as rzz / rx, signs in a frame
CHAIN_STATS = {"tried": 0, "not_chain": 0, "not_cz": 0, "no_path": 0, "snake": 0, "vf2_path": 0,
               "common_length": 0, "used": 0, "failed_check": 0, "kept": 0, "kept_default": 0,
               "frame": 0, "frame_x_end": 0, "frame_folded": 0}
_CHAIN_DEPTH = [0]  # item 87: set while c51's own output is compiled for the comparison (no item 87 inside it)
_CHAIN_VF2_CALLS = 5_000_000


def _chain_support(lb):
    n = len(lb)
    return [(n - 1 - i, ch) for i, ch in enumerate(lb) if ch != "I"]


def _chain_commute(a, b):
    sa, sb = dict(_chain_support(a)), dict(_chain_support(b))
    return sum(1 for q in set(sa) & set(sb) if sa[q] != sb[q]) % 2 == 0


def _chain_read(qc):
    """Item 87: (prefix, meas, terms [(label, angle)], phase, suffix, perm, path) when every term (read by items
    72-73, merged where exact by `_st_merge`) acts on one or two qubits and the two-qubit supports form one path
    through all the circuit's qubits; else None."""
    ev = _st_parse(qc)
    ev = ev + ([], None) if ev is not None else _st_parse_gates(qc)
    if ev is None:
        return None
    prefix, meas, triples, phase, suffix, perm = ev
    terms = [(lb, a * t) for lb, a, t in _st_merge(triples)]
    n = qc.num_qubits
    adj = {q: set() for q in range(n)}
    for lb, _ in terms:
        sup = [q for q, _ in _chain_support(lb)]
        if len(sup) > 2:
            return None
        if len(sup) == 2:
            adj[sup[0]].add(sup[1])
            adj[sup[1]].add(sup[0])
    if any(len(v) > 2 for v in adj.values()):
        return None
    ends = [q for q in range(n) if len(adj[q]) == 1]
    if n < 2 or len(ends) != 2:
        return None
    path, prev, cur = [min(ends)], None, min(ends)
    while True:
        nxt = [v for v in adj[cur] if v != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)
    if len(path) != n:
        return None
    return prefix, meas, terms, phase, suffix, perm or list(range(n)), path


def _chain_lattice(target, max_error):
    """Item 87: (K, its adjacency, the adjacency restricted to couplers of the device's most common two-qubit gate
    length, the snake path) -- W5's cut: qubits with sx or readout error >= max_error and couplers with error >=
    max_error removed, the largest connected component kept (ties: the smallest qubit)."""
    N = target.num_qubits
    err, dur = {}, {}
    for qa, pr in target["cz"].items():
        if qa is None:
            continue
        k = (min(qa), max(qa))
        err[k] = max(err.get(k, 0.0), pr.error if pr is not None and pr.error is not None else 0.0)
        dur[k] = max(dur.get(k, 0.0), pr.duration if pr is not None and pr.duration is not None else 0.0)
    adj = {q: set() for q in range(N)}
    for a, b in err:
        adj[a].add(b)
        adj[b].add(a)
    bad = set()
    for name in ("sx", "measure"):
        if name in target.operation_names:
            for qa, pr in target[name].items():
                if qa is not None and pr is not None and pr.error is not None and pr.error >= max_error:
                    bad.add(qa[0])
    gadj = {q: set() for q in range(N) if q not in bad}
    for (a, b), e in err.items():
        if e < max_error and a not in bad and b not in bad:
            gadj[a].add(b)
            gadj[b].add(a)
    seen, comps = set(), []
    for q in sorted(gadj):
        if q in seen:
            continue
        comp, fr = {q}, [q]
        while fr:
            u = fr.pop()
            for v in gadj[u]:
                if v not in comp:
                    comp.add(v)
                    fr.append(v)
        seen |= comp
        comps.append(comp)
    if not comps:
        return None
    K = max(comps, key=lambda c: (len(c), -min(c)))
    kadj = {q: {v for v in gadj[q] if v in K} for q in K}
    lens = {}
    for a in K:
        for b in kadj[a]:
            if a < b:
                k = round(dur[(a, b)] * 1e12)
                lens[k] = lens.get(k, 0) + 1
    modal = max(lens, key=lambda x: (lens[x], -x)) if lens else None
    fadj = {q: {v for v in kadj[q] if round(dur[(min(q, v), max(q, v))] * 1e12) == modal} for q in K}
    return K, kadj, fadj, _chain_snake(N, adj)


def _chain_snake(N, adj):
    """Item 87: Heron's snake (W5): rows of consecutive qubits joined by couplers, row 0 left to right, the bridge
    at the far end down, the next row back, and so on. A lattice that is not read as rows gives a shorter path."""
    rows, cur = [], [0]
    for q in range(1, N):
        if q in adj[q - 1]:
            cur.append(q)
        else:
            rows.append(cur)
            cur = [q]
    rows.append(cur)
    rows = [r for r in rows if len(r) > 2]
    inrow = {q: (i, j) for i, r in enumerate(rows) for j, q in enumerate(r)}
    path, col = [], 0
    for i, r in enumerate(rows):
        leftward = i % 2 == 1
        if i + 1 < len(rows):
            br = [(inrow[a][1], b) for b in range(N) if b not in inrow for a in adj[b]
                  if a in inrow and inrow[a][0] == i and any(c in inrow and inrow[c][0] == i + 1 for c in adj[b])]
            if not br:
                return path + list(r)
            exit_col, bq = (min(br) if leftward else max(br))
        else:
            exit_col, bq = (0 if leftward else len(r) - 1), None
        step = -1 if leftward else 1
        start = col if i else 0
        path += [r[j] for j in range(start, exit_col + step, step)]
        if bq is not None:
            path.append(bq)
            below = next(c for c in adj[bq] if c in inrow and inrow[c][0] == i + 1)
            col = inrow[below][1]
    return path


def _chain_place(n, K, kadj, snake):
    """Item 87: n physical qubits forming a path in K: the longest surviving snake piece, centred; else the first
    path rustworkx's VF2 finds with nodes in index order (no score). (path, how) or (None, None)."""
    pieces, cur = [], []
    for q in snake:
        if q in K and (not cur or q in kadj[cur[-1]]):
            cur.append(q)
        else:
            if cur:
                pieces.append(cur)
            cur = [q] if q in K else []
    if cur:
        pieces.append(cur)
    if pieces:
        best = max(pieces, key=lambda p: (len(p), -snake.index(p[0])))
        if len(best) >= n:
            s = (len(best) - n) // 2
            return best[s:s + n], "snake"
    import rustworkx as rx
    kn = sorted(K)
    ki = {q: i for i, q in enumerate(kn)}
    G = rx.PyGraph()
    G.add_nodes_from(kn)
    G.add_edges_from_no_data([(ki[a], ki[b]) for a in kn for b in kadj[a] if a < b])
    P = rx.PyGraph()
    P.add_nodes_from(range(n))
    P.add_edges_from_no_data([(i, i + 1) for i in range(n - 1)])
    m = next(rx.vf2_mapping(G, P, subgraph=True, induced=False, id_order=True, call_limit=_CHAIN_VF2_CALLS), None)
    if m is None:
        return None, None
    inv = {h: kn[g] for g, h in m.items()}
    return [inv[i] for i in range(n)], "vf2_path"





def _chain_rotation(qc, lb, a, frame=False):
    """Item 87: exp(-i a P) on the circuit's own qubits; XX, YY, ZZ as Qiskit's rxx, ryy, rzz. Item 88 (`frame`):
    every two-qubit term as rzz between single-qubit basis changes (X: h; Y: rx(pi/2) before, rx(-pi/2) after), so
    that the device's one-pulse rzz carries it; single-qubit terms as rx / ry / rz."""
    sup = _chain_support(lb)
    if len(sup) == 1:
        q, ch = sup[0]
        getattr(qc, {"X": "rx", "Y": "ry", "Z": "rz"}[ch])(2 * a, q)
        return
    (q1, c1), (q2, c2) = sup
    if c1 == c2 and not frame:
        getattr(qc, {"X": "rxx", "Y": "ryy", "Z": "rzz"}[c1])(2 * a, q1, q2)
        return
    for q, ch in ((q1, c1), (q2, c2)):
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            qc.rx(math.pi / 2, q)
    if frame:
        qc.rzz(2 * a, q1, q2)
    else:
        qc.cx(q1, q2)
        qc.rz(2 * a, q2)
        qc.cx(q1, q2)
    for q, ch in ((q1, c1), (q2, c2)):
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            qc.rx(-math.pi / 2, q)


def _chain_frame(log):
    """Item 88: `log` (single-qubit gates, rzz with any angle, measurements) rewritten so that every rzz angle lies in
    IBM's executable range 0 < theta <= pi/2, with the same action (unitary, and outcome distribution when measured).

    Each angle is reduced to (-pi, pi]; an angle of size above pi/2 is moved by pi with Z x Z (rz(pi) on both qubits,
    the phase kept). A negative angle (after the frame's sign, below) is made positive by one x on the rzz's first
    qubit before it, merged into the single-qubit run before it; the x that would follow it is not placed but kept in
    a Pauli X frame: a single-qubit gate G on a qubit in the frame is placed as X G X (merged into one pulse with its
    run), an rzz with exactly one of its qubits in the frame has its sign flipped, and the frame is undone inside the
    circuit -- one x before the qubit's measurement, or at the end of an unmeasured qubit, merged into the last
    single-qubit gate where there is one. No measured bit is flipped afterwards: the circuit itself outputs the
    input's distribution, and acts as the input's unitary when it is not measured."""
    n = log.num_qubits
    out = QuantumCircuit(*log.qregs, *log.cregs, global_phase=log.global_phase)
    f = [0] * n
    for ins in log.data:
        nm = ins.operation.name
        qs = [log.find_bit(q).index for q in ins.qubits]
        if nm == "barrier":
            out.append(ins.operation, qs)
            continue
        if nm == "measure":
            q = qs[0]
            if f[q]:
                out.x(q)
                f[q] = 0
            out.measure(q, log.find_bit(ins.clbits[0]).index)
            continue
        if nm == "rzz":
            a, b = qs
            th = float(ins.operation.params[0])
            th = math.pi - (math.pi - th) % (2 * math.pi)  # (-pi, pi]
            if abs(th) > math.pi / 2:  # rzz(th) = rzz(th - sign(th) pi) (rz(pi) x rz(pi)) e^{i pi/2} for th > 0
                out.rz(math.pi, a)
                out.rz(math.pi, b)
                out.global_phase += math.copysign(math.pi / 2, th) + math.pi * (f[a] + f[b])  # X rz(pi) X = -rz(pi)
                th -= math.copysign(math.pi, th)
                CHAIN_STATS["frame_folded"] += 1
            if abs(th) < 1e-12:
                continue
            eff = -th if f[a] != f[b] else th
            if eff < 0:
                out.x(a)
                f[a] ^= 1
                eff = -eff
            out.rzz(eff, a, b)
            continue
        if len(qs) != 1:
            raise TranspilerError(f"item 88: unexpected {nm} on {len(qs)} qubits")
        q = qs[0]
        if f[q]:
            out.x(q)
        out.append(ins.operation, [q])
        if f[q]:
            out.x(q)
    for q in range(n):
        if f[q]:
            out.x(q)
            CHAIN_STATS["frame_x_end"] += 1
    return out


def _chain_brickwork(qc, target, max_error, seed):
    """Item 87 (workplace proposal, N1): a Pauli evolution whose two-qubit terms form one chain, compiled by formula.
    The chain goes on a path of the device (`_chain_place`: first on couplers of the device's most common two-qubit
    gate length, then on all of the cut device); every term gets a level, 1 + the largest level of an earlier term
    that shares a qubit and does not commute with it; terms are emitted by level, two-qubit terms on even edges of the
    chain, then odd edges, then single-qubit terms (every non-commuting pair keeps its order: exact). Without rzz in
    the Target, Qiskit's level-3 preset then translates on that fixed layout without routing, given the coupling map
    and basis only (no error value is read). Item 88: with rzz and rx in the Target (`CHAIN_FRAME`), the terms are
    emitted as rzz between basis changes, diagonal gates directly before measurements are dropped (Qiskit's
    RemoveDiagonalGatesBeforeMeasure), the signs are carried in a Pauli X frame (`_chain_frame`), Qiskit's level-0
    preset places it on the fixed layout, and every single-qubit run is merged into rz rx rz (one pulse); an rzz angle
    outside 0 < theta <= pi/2 would go through item 86's `_fold`. None when the input is not such a chain, the
    device's two-qubit gate is not cz, no path of n qubits exists, or the output would use a failed element."""
    CHAIN_STATS["tried"] += 1
    if "cz" not in target.operation_names:
        CHAIN_STATS["not_cz"] += 1
        return None
    got = _chain_read(qc)
    if got is None:
        CHAIN_STATS["not_chain"] += 1
        return None
    prefix, meas, terms, phase, suffix, perm, path = got
    n, N = qc.num_qubits, target.num_qubits
    lat = _chain_lattice(target, max_error)
    if lat is None:
        CHAIN_STATS["no_path"] += 1
        return None
    K, kadj, fadj, snake = lat
    phys, how = _chain_place(n, K, fadj, snake)
    if phys is not None:
        CHAIN_STATS["common_length"] += 1
    else:
        phys, how = _chain_place(n, K, kadj, snake)
    if phys is None:
        CHAIN_STATS["no_path"] += 1
        return None
    CHAIN_STATS[how] += 1
    rank = {path[i]: i for i in range(n)}
    level, last = [], {}
    for k, (lb, _) in enumerate(terms):
        lv = 0
        for q, _ in _chain_support(lb):
            for j, lb2 in last.get(q, ()):
                if not _chain_commute(lb, lb2):
                    lv = max(lv, level[j] + 1)
        level.append(lv)
        for q, _ in _chain_support(lb):
            last.setdefault(q, []).append((k, lb))

    def key(k):
        sup = [q for q, _ in _chain_support(terms[k][0])]
        if len(sup) == 2:
            lo = min(rank[q] for q in sup)
            return (level[k], 0, lo % 2, lo, k)
        return (level[k], 1, 0, rank[sup[0]], k)
    frame = CHAIN_FRAME and "rzz" in target.operation_names and "rx" in target.operation_names
    log = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=phase)
    for op, q in prefix:
        log.append(op, [q])
    for k in sorted(range(len(terms)), key=key):
        _chain_rotation(log, *terms[k], frame=frame)
    for op, q in suffix:
        log.append(op, [q])
    for q, cb in meas:
        log.measure(perm[q], cb)  # item 76: logical q's state is on wire perm[q]
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for a in sorted(K):
        for b in sorted(kadj[a]):
            cm.add_edge(a, b)
    layout = [phys[rank[q]] for q in range(n)]
    if frame:
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveDiagonalGatesBeforeMeasure
        log = PassManager([RemoveDiagonalGatesBeforeMeasure()]).run(log)
        low = _chain_frame(log)
        pm = generate_preset_pass_manager(0, coupling_map=cm, basis_gates=["rzz", "rx", "rz", "sx", "x", "cz"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        placed = pm.run(low)
        out = PassManager([Optimize1qGatesDecomposition(basis=["rz", "rx"])]).run(placed)
        out._layout = placed._layout
        if not _rzz_in_range(out):
            out = _fold(out, target)
        CHAIN_STATS["frame"] += 1
    else:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        out = pm.run(log)
    edges, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges, bad):
        CHAIN_STATS["failed_check"] += 1
        return None
    CHAIN_STATS["used"] += 1
    return out


_C49_SIGNATURE = _inspect.signature(_compile_for_hardware_c49)


def compile_for_hardware(qc, *args, **kwargs):
    """Items 87-88 (workplace proposal on prototype c51): with the device given (`steiner` "auto" true) and no
    `coupling_map`, `initial_layout`, `final_resynthesis`, `compare_level3` or `compare_floor` of the caller's, a chain
    Pauli evolution is built by formula (`_chain_brickwork`) and compared with c51's own output for the same call
    (`_compile_for_hardware_c51`: c49, then item 86) by item 84's rule (`_better`; a tie keeps the brickwork). Any
    other input, and every nested call, is compiled exactly as by c51.

    """
    if _CHAIN_DEPTH[0] or _FRAC_DEPTH[0] or not CHAIN_BRICKWORK:
        return _compile_for_hardware_c51(qc, *args, **kwargs)
    try:
        bound = _C49_SIGNATURE.bind(qc, *args, **kwargs)
    except TypeError:
        return _compile_for_hardware_c51(qc, *args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    target = a["target"] if a["target"] is not None else getattr(a["backend"], "target", None)
    steiner = a["steiner"]
    if steiner == "auto":  # item 68's "auto"
        steiner = (a["backend"] is not None or a["target"] is not None) and a["coupling_map"] is None and \
            a["initial_layout"] is None
    out87 = None
    if (steiner and not a["_inner"] and target is not None and a["coupling_map"] is None
            and a["initial_layout"] is None
            and not (a["final_resynthesis"] or a["compare_level3"] or a["compare_floor"])):
        out87 = _chain_brickwork(qc, target, a["prune_max_error"], a["seed_transpiler"])
    if out87 is None:
        return _compile_for_hardware_c51(qc, *args, **kwargs)
    _CHAIN_DEPTH[0] += 1
    try:
        own = _compile_for_hardware_c51(qc, *args, **kwargs)
    except FailedElementsError:
        own = None
    finally:
        _CHAIN_DEPTH[0] -= 1
    if own is not None and _better(own, out87, target):
        CHAIN_STATS["kept_default"] += 1
        return own
    CHAIN_STATS["kept"] += 1
    return out87


compile_for_hardware.__doc__ += _compile_for_hardware_c49.__doc__ or ""
