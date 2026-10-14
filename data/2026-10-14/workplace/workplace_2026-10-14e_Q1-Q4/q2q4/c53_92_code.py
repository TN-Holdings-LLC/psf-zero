

# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on c53 (2026-10-14): item 92, a periodic chain (ring Hamiltonian) built by N2's formula
# ---------------------------------------------------------------------------------------------------------------------

RING_BRICKWORK = True  # item 92 (proposal): ring Pauli evolutions by N2's folded line and mirrored rounds
RING_MIN, RING_MAX = 4, 16  # item 92: ring lengths built (a 12-ring embeds in a hexagon: item 91 is tried first)
RING_STATS = {"tried": 0, "not_ring": 0, "size": 0, "no_path": 0, "odd_unmeasured": 0, "used": 0, "frame": 0,
              "failed_check": 0}


def _ring_rounds92(n):
    """Item 92 (N2, schedule of Q4): (folded order of ring indices, rounds [(ring edges made adjacent, SWAPs
    (position p, p+1))]). Positions hold 0, n-1, 1, n-2, ...: ring indices 0, 1, 2, ... on the even positions and
    n-1, n-2, ... on the odd ones, so every ring edge is at distance at most 2. Round 1: the ring edges already
    adjacent (the two ends and the middle), then SWAPs SA = (1,2), (5,6), (9,10), ...; round 2: the edges made
    adjacent, then SWAPs SA (undone) together with SB = (3,4), (7,8), ... (disjoint: one layer); round 3: the rest.
    Every ring of n >= 4 is covered in three rounds and two SWAP layers: CZ-depth 9 and about 3n CZ per layer for
    any n (N2's odd-even transposition rounds grew with n: 55 CZ at n = 7, 203 at n = 11)."""
    order, lo, hi = [], 0, n - 1
    while lo <= hi:
        order.append(lo)
        if hi != lo:
            order.append(hi)
        lo, hi = lo + 1, hi - 1
    edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    occ, rounds = list(order), []

    def adjacent():
        done = []
        for p in range(n - 1):
            e = (min(occ[p], occ[p + 1]), max(occ[p], occ[p + 1]))
            if e in edges:
                done.append(e)
                edges.discard(e)
        return done
    SA = [(p, p + 1) for p in range(1, n - 1, 4)]
    SB = [(p, p + 1) for p in range(3, n - 1, 4)]
    done = adjacent()
    for sw in (SA, SA + SB):
        if not edges:
            break
        rounds.append((done, sw))
        for p, p1 in sw:
            occ[p], occ[p1] = occ[p1], occ[p]
        done = adjacent()
    rounds.append((done, []))
    if edges:
        raise TranspilerError("item 92: ring schedule incomplete")
    return order, rounds


def _ring_cycle(n, edges):
    """Item 92: the qubits in cycle order (from 0, toward its smaller neighbour) when `edges` form one cycle through
    all n qubits; else None."""
    adj = {q: [] for q in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    if any(len(v) != 2 for v in adj.values()):
        return None
    cyc = [0]
    nxt = min(adj[0])
    while nxt != 0:
        cyc.append(nxt)
        nxt = [v for v in adj[nxt] if v != cyc[-2]][0]
        if len(cyc) > n:
            return None
    return cyc if len(cyc) == n else None


def _ring_brickwork(qc, target, max_error, seed):
    """Item 92 (workplace proposal on c53): a Pauli evolution whose terms act on one or two qubits, the two-qubit
    supports forming one cycle through all its RING_MIN..RING_MAX qubits (periodic TFIM, XXZ, XYZ chains), compiled
    by N2's formula. The cycle goes in folded order (c0, c[n-1], c1, c[n-2], ...) on a line placed by item 87's rule.
    Terms get item 87's levels (exact); for each level with two-qubit terms the ring's rounds run -- the level's
    two-qubit terms on the edges made adjacent, then the round's SWAPs -- and the next such level runs the rounds
    backwards (mirror: undo the SWAPs, then the terms), so every level costs the same; the level's single-qubit terms
    follow on wherever their qubits are. Emitted on wires bound to line positions (a SWAP moves states), then as item
    87 (Qiskit level 3 on the fixed layout, CZ) or item 88 (rzz / rx with the Pauli frame; a SWAP carries the frame
    bits with the states). An unmeasured input that ends permuted (an odd number of two-qubit levels) is not built.
    None when not applicable."""
    RING_STATS["tried"] += 1
    if "cz" not in target.operation_names:
        return None
    got = _graph_read(qc)
    if got is None:
        RING_STATS["not_ring"] += 1
        return None
    prefix, meas, terms, phase, suffix, perm, edges = got
    n, N = qc.num_qubits, target.num_qubits
    cyc = _ring_cycle(n, edges)
    if cyc is None:
        RING_STATS["not_ring"] += 1
        return None
    if not RING_MIN <= n <= RING_MAX:
        RING_STATS["size"] += 1
        return None
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
    levels = sorted(set(level))
    two_levels = [L for L in levels if any(level[k] == L and len(_chain_support(terms[k][0])) == 2
                                           for k in range(len(terms)))]
    if not meas and len(two_levels) % 2 == 1:
        RING_STATS["odd_unmeasured"] += 1
        return None
    lat = _chain_lattice(target, max_error)
    if lat is None:
        RING_STATS["no_path"] += 1
        return None
    K, kadj, fadj, snake = lat
    line, _ = _chain_place(n, K, fadj, snake)
    if line is None:
        line, _ = _chain_place(n, K, kadj, snake)
    if line is None:
        RING_STATS["no_path"] += 1
        return None
    order, rounds = _ring_rounds92(n)
    ridx = {q: i for i, q in enumerate(cyc)}  # input qubit -> ring index
    wire_at = [cyc[order[p]] for p in range(n)]  # position -> wire (fixed for the whole circuit)
    p_of_wire = {w: p for p, w in enumerate(wire_at)}
    pos = {order[p]: p for p in range(n)}  # ring index -> current position
    occ = list(order)
    frame = CHAIN_FRAME and "rzz" in target.operation_names and "rx" in target.operation_names
    log = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=phase)

    def here(q):
        return wire_at[pos[ridx[q]]]

    def rot(k):
        lb, a = terms[k]
        sup = _chain_support(lb)
        # the term written on the wires now holding its qubits: relabel the Pauli string's qubits
        lab = ["I"] * n
        for q, ch in sup:
            lab[n - 1 - here(q)] = ch
        _chain_rotation(log, "".join(lab), a, frame=frame)

    def swaps(sw):
        for p, p1 in sw:
            log.swap(wire_at[p], wire_at[p1])
            a, b = occ[p], occ[p1]
            occ[p], occ[p1] = b, a
            pos[a], pos[b] = p1, p

    by_edge = {}
    for op, q in prefix:
        log.append(op, [here(q)])
    layer = 0
    for L in levels:
        ks = [k for k in range(len(terms)) if level[k] == L]
        two = [k for k in ks if len(_chain_support(terms[k][0])) == 2]
        one = [k for k in ks if len(_chain_support(terms[k][0])) == 1]
        if two:
            by_edge = {}
            for k in two:
                a, b = sorted(ridx[q] for q, _ in _chain_support(terms[k][0]))
                by_edge.setdefault((a, b), []).append(k)
            if layer % 2 == 0:
                for done, sw in rounds:
                    for e in done:
                        for k in by_edge.get(e, ()):
                            rot(k)
                    swaps(sw)
            else:
                for done, sw in reversed(rounds):
                    swaps(list(reversed(sw)))
                    for e in done:
                        for k in by_edge.get(e, ()):
                            rot(k)
            layer += 1
        for k in one:
            rot(k)
    for op, q in suffix:
        log.append(op, [here(q)])
    for q, cb in meas:
        log.measure(here(perm[q]), cb)  # item 76: logical q's state is on wire perm[q]; that wire's state is here
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for i in range(n - 1):
        cm.add_edge(line[i], line[i + 1])
        cm.add_edge(line[i + 1], line[i])
    layout = [line[p_of_wire[w]] for w in range(n)]
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
        RING_STATS["frame"] += 1
    else:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        out = pm.run(log)
    edges_f, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges_f, bad):
        RING_STATS["failed_check"] += 1
        return None
    RING_STATS["used"] += 1
    return out
