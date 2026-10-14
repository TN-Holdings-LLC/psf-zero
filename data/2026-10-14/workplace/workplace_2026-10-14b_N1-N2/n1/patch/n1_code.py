# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on c49 (N1, 2026-10-14): item 86, a chain Hamiltonian built by formula as a brickwork
# ---------------------------------------------------------------------------------------------------------------------

CHAIN_BRICKWORK = True  # item 86 (proposal): chain Pauli evolutions built as a brickwork on a path of the device
CHAIN_STATS = {"tried": 0, "not_chain": 0, "not_cz": 0, "no_path": 0, "snake": 0, "vf2_path": 0,
               "common_length": 0, "used": 0, "failed_check": 0, "kept": 0, "kept_c49": 0}
_CHAIN_BUSY = [0]  # item 86: set while c49's own path is compiled for the comparison (no item 86 inside it)
_CHAIN_VF2_CALLS = 5_000_000


def _chain_support(lb):
    n = len(lb)
    return [(n - 1 - i, ch) for i, ch in enumerate(lb) if ch != "I"]


def _chain_commute(a, b):
    sa, sb = dict(_chain_support(a)), dict(_chain_support(b))
    return sum(1 for q in set(sa) & set(sb) if sa[q] != sb[q]) % 2 == 0


def _chain_read(qc):
    """Item 86: (prefix, meas, terms [(label, angle)], phase, suffix, perm, path) when every term (read by items
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
    """Item 86: (K, its adjacency, the adjacency restricted to couplers of the device's most common two-qubit gate
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
    """Item 86: Heron's snake (W5): rows of consecutive qubits joined by couplers, row 0 left to right, the bridge
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
    """Item 86: n physical qubits forming a path in K: the longest surviving snake piece, centred; else the first
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


def _chain_rotation(qc, lb, a):
    """Item 86: exp(-i a P) on the circuit's own qubits; XX, YY, ZZ as Qiskit's rxx, ryy, rzz."""
    sup = _chain_support(lb)
    if len(sup) == 1:
        q, ch = sup[0]
        getattr(qc, {"X": "rx", "Y": "ry", "Z": "rz"}[ch])(2 * a, q)
        return
    (q1, c1), (q2, c2) = sup
    if c1 == c2:
        getattr(qc, {"X": "rxx", "Y": "ryy", "Z": "rzz"}[c1])(2 * a, q1, q2)
        return
    for q, ch in ((q1, c1), (q2, c2)):
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            qc.rx(math.pi / 2, q)
    qc.cx(q1, q2)
    qc.rz(2 * a, q2)
    qc.cx(q1, q2)
    for q, ch in ((q1, c1), (q2, c2)):
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            qc.rx(-math.pi / 2, q)


def _chain_brickwork(qc, target, max_error, seed):
    """Item 86 (workplace proposal, N1): a Pauli evolution whose two-qubit terms form one chain, compiled by formula.
    The chain goes on a path of the device (`_chain_place`: first on couplers of the device's most common two-qubit
    gate length, then on all of the cut device); every term gets a level, 1 + the largest level of an earlier term
    that shares a qubit and does not commute with it; terms are emitted by level, two-qubit terms on even edges of the
    chain, then odd edges, then single-qubit terms (every non-commuting pair keeps its order: exact). Qiskit's level-3
    preset then translates on that fixed layout without routing, given the coupling map and basis only (no error
    value is read). None when the input is not such a chain, the device's two-qubit gate is not cz, no path of n
    qubits exists, or the output would use a failed element."""
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
    log = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=phase)
    for op, q in prefix:
        log.append(op, [q])
    for k in sorted(range(len(terms)), key=key):
        _chain_rotation(log, *terms[k])
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
    pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                      initial_layout=[phys[rank[q]] for q in range(n)], routing_method="none",
                                      seed_transpiler=seed)
    out = pm.run(log)
    edges, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges, bad):
        CHAIN_STATS["failed_check"] += 1
        return None
    CHAIN_STATS["used"] += 1
    return out
