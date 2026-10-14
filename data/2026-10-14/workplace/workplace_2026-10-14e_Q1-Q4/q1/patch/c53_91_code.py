

# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on c53 (2026-10-14): item 91, a Pauli evolution whose interaction graph embeds in the device,
# built by formula: first VF2 embedding (no score), edges coloured into at most 3 matchings (Koenig), one layer per
# colour
# ---------------------------------------------------------------------------------------------------------------------

GRAPH_BRICKWORK = True  # item 91 (proposal): embeddable interaction graphs built by edge colouring
GRAPH_STATS = {"tried": 0, "not_read": 0, "degree": 0, "not_cz": 0, "no_embedding": 0, "common_length": 0,
               "used": 0, "colours": {}, "failed_check": 0, "frame": 0}
_GRAPH_VF2_CALLS = 2_000_000


def _graph_read(qc):
    """Item 91: (prefix, meas, terms [(label, angle)], phase, suffix, perm, edges) when every term (items 72-73's
    reader, merged where exact) acts on one or two qubits; edges: the sorted two-qubit supports. Else None."""
    ev = _st_parse(qc)
    ev = ev + ([], None) if ev is not None else _st_parse_gates(qc)
    if ev is None:
        return None
    prefix, meas, triples, phase, suffix, perm = ev
    terms = [(lb, a * t) for lb, a, t in _st_merge(triples)]
    edges = set()
    for lb, _ in terms:
        sup = sorted(q for q, _ in _chain_support(lb))
        if len(sup) > 2:
            return None
        if len(sup) == 2:
            edges.add((sup[0], sup[1]))
    if not edges:
        return None
    return prefix, meas, terms, phase, suffix, perm or list(range(qc.num_qubits)), sorted(edges)


def _graph_colours(edges):
    """Item 91: a proper edge colouring of a bipartite graph with max-degree colours (Koenig), deterministic: edges in
    sorted order; an edge (u, v) takes the smallest colour a free at u; if a is used at v, the a/b path from v (b the
    smallest colour free at v) is swapped first. {edge: colour} or None when the graph is not bipartite."""
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    D = max(deg.values())
    at = {x: {} for x in deg}  # node -> {colour: neighbour}
    for u, v in edges:
        a = min(c for c in range(D) if c not in at[u])
        if a in at[v]:
            b = min(c for c in range(D) if c not in at[v])
            path, x, c, seen = [], v, a, {v}
            while c in at[x]:
                y = at[x][c]
                path.append((x, y, c))
                if y in seen or y == u:
                    return None  # an odd cycle: not bipartite
                seen.add(y)
                x, c = y, (b if c == a else a)
            for x, y, c in path:
                del at[x][c]
                del at[y][c]
            for x, y, c in path:
                d = b if c == a else a
                at[x][d] = y
                at[y][d] = x
        at[u][a] = v
        at[v][a] = u
    col = {}
    for x, m in at.items():
        for c, y in m.items():
            col[(min(x, y), max(x, y))] = c
    return col


def _graph_place(n, edges, K, kadj):
    """Item 91: the first VF2 embedding (rustworkx, nodes in index order, no score) of the interaction graph into the
    cut device K: {logical qubit: physical qubit} or None."""
    import rustworkx as rx
    kn = sorted(K)
    ki = {q: i for i, q in enumerate(kn)}
    G = rx.PyGraph()
    G.add_nodes_from(kn)
    G.add_edges_from_no_data([(ki[a], ki[b]) for a in kn for b in kadj[a] if a < b])
    P = rx.PyGraph()
    P.add_nodes_from(range(n))
    P.add_edges_from_no_data(edges)
    if n > len(kn):
        return None
    m = next(rx.vf2_mapping(G, P, subgraph=True, induced=False, id_order=True, call_limit=_GRAPH_VF2_CALLS), None)
    if m is None:
        return None
    return {h: kn[g] for g, h in m.items()}


def _graph_brickwork(qc, target, max_error, seed):
    """Item 91 (workplace proposal on c53): a Pauli evolution whose terms act on one or two qubits and whose
    interaction graph embeds in the cut device (max degree <= 3, bipartite), compiled by formula. Placement: the
    first VF2 embedding, tried first on the couplers of the device's most common two-qubit gate length, then on all
    of the cut device (item 87's lattice). Order: every term at level 1 + the largest level of an earlier term that
    shares a qubit and does not commute with it (exact); within a level the two-qubit terms by colour of a proper
    edge colouring with max-degree colours (each colour a matching: one layer), then the single-qubit terms.
    Emission as item 87 (Qiskit level 3 on the fixed layout, CZ) or, with rzz and rx in the Target, item 88 (rzz,
    rx, Pauli frame undone inside the circuit). None when not applicable."""
    GRAPH_STATS["tried"] += 1
    if "cz" not in target.operation_names:
        GRAPH_STATS["not_cz"] += 1
        return None
    got = _graph_read(qc)
    if got is None:
        GRAPH_STATS["not_read"] += 1
        return None
    prefix, meas, terms, phase, suffix, perm, edges = got
    n, N = qc.num_qubits, target.num_qubits
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    if max(deg.values()) > 3:
        GRAPH_STATS["degree"] += 1
        return None
    col = _graph_colours(edges)
    if col is None:
        GRAPH_STATS["no_embedding"] += 1
        return None
    lat = _chain_lattice(target, max_error)
    if lat is None:
        GRAPH_STATS["no_embedding"] += 1
        return None
    K, kadj, fadj, _ = lat
    place = _graph_place(n, edges, K, fadj)
    if place is not None:
        GRAPH_STATS["common_length"] += 1
    else:
        place = _graph_place(n, edges, K, kadj)
    if place is None:
        GRAPH_STATS["no_embedding"] += 1
        return None
    ncol = max(col.values()) + 1
    GRAPH_STATS["colours"][ncol] = GRAPH_STATS["colours"].get(ncol, 0) + 1
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
        sup = sorted(q for q, _ in _chain_support(terms[k][0]))
        if len(sup) == 2:
            return (level[k], 0, col[(sup[0], sup[1])], k)
        return (level[k], 1, 0, k)
    frame = CHAIN_FRAME and "rzz" in target.operation_names and "rx" in target.operation_names
    log = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=phase)
    for op, q in prefix:
        log.append(op, [q])
    for k in sorted(range(len(terms)), key=key):
        _chain_rotation(log, *terms[k], frame=frame)
    for op, q in suffix:
        log.append(op, [q])
    for q, cb in meas:
        log.measure(perm[q], cb)  # item 76
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for a in sorted(K):
        for b in sorted(kadj[a]):
            cm.add_edge(a, b)
    used = set(place.values())
    free = [q for q in sorted(K) if q not in used]
    layout = [place[q] if q in place else free.pop(0) for q in range(n)]  # qubits without a term: unused places
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
        GRAPH_STATS["frame"] += 1
    else:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        out = pm.run(log)
    edges_f, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges_f, bad):
        GRAPH_STATS["failed_check"] += 1
        return None
    GRAPH_STATS["used"] += 1
    return out
