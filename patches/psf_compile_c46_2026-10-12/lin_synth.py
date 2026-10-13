def _lin_rank_insert(basis, v):
    """Item 81: reduce bitmask `v` by the echelon `basis` {pivot bit: row}; insert it if independent. True if it was."""
    while v:
        top = v.bit_length() - 1
        if top in basis:
            v ^= basis[top]
        else:
            basis[top] = v
            return True
    return False


def _lin_solve(rows, target):
    """Item 81: a set of indices S with XOR of rows[S] == target, or None. rows: {index: bitmask}."""
    basis = {}  # pivot -> (row bitmask, index set as bitmask over indices)
    for idx in sorted(rows):
        v, comb = rows[idx], 1 << idx
        while v:
            top = v.bit_length() - 1
            if top in basis:
                v ^= basis[top][0]
                comb ^= basis[top][1]
            else:
                basis[top] = (v, comb)
                break
    v, comb = target, 0
    while v:
        top = v.bit_length() - 1
        if top not in basis:
            return None
        v ^= basis[top][0]
        comb ^= basis[top][1]
    return [i for i in sorted(rows) if comb >> i & 1]


def _lin_targets(n, spec):
    """Item 81: the matrix to synthesise, as rows[w] (bitmask over the n wires' values before the network): the measured
    parities `spec` [(bitmask, sign)] placed on distinct wires within their support where possible (a bipartite
    matching), the other wires completed by unit rows so that the matrix is invertible. (rows, place) with place[k] the
    wire of spec[k]; None if no completion exists."""
    match = {}  # wire -> spec index

    def augment(k, seen):
        for w in range(n):
            if spec[k][0] >> w & 1 and w not in seen:
                seen.add(w)
                if w not in match or augment(match[w], seen):
                    match[w] = k
                    return True
        return False
    for k in range(len(spec)):
        if not augment(k, set()):
            return None
    place = {k: w for w, k in match.items()}
    rows = [None] * n
    basis = {}
    for k, (mask, _) in enumerate(spec):
        if not _lin_rank_insert(basis, mask):
            return None
        rows[place[k]] = mask
    free = [w for w in range(n) if rows[w] is None]
    units = list(range(n))
    for w in free:  # its own unit row if independent, else the first independent unit row
        for u in [w] + units:
            if _lin_rank_insert(dict(basis), 1 << u):
                _lin_rank_insert(basis, 1 << u)
                rows[w] = 1 << u
                break
        else:
            return None
    return rows, [place[k] for k in range(len(spec))]


def _lin_dijkstra(alive, adj, w):
    """Item 81: all-pairs Dijkstra on the coupling graph restricted to the vertex set `alive`."""
    dist, par = {}, {}
    for s in alive:
        d, p, heap = {s: 0.0}, {s: None}, [(0.0, s)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > d[u]:
                continue
            for v in sorted(adj[u]):
                if v not in alive:
                    continue
                nd = du + w.get((min(u, v), max(u, v)), 1.0)
                if nd < d.get(v, math.inf):
                    d[v], p[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        dist[s], par[s] = d, p
    return dist, par


def _lin_cut_vertices(alive, adj):
    """Item 81: the articulation points of the graph on `alive` (iterative Tarjan)."""
    alive = set(alive)
    disc, low, cut = {}, {}, set()
    t = 0
    for root in sorted(alive):
        if root in disc:
            continue
        disc[root] = low[root] = t
        t += 1
        children = 0
        stack = [(root, None, iter(sorted(v for v in adj[root] if v in alive)))]
        while stack:
            u, parent, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                if parent is not None:
                    low[parent] = min(low[parent], low[u])
                    if stack[-1][1] is not None and low[u] >= disc[parent]:
                        cut.add(parent)
                continue
            if nxt == parent:
                continue
            if nxt in disc:
                low[u] = min(low[u], disc[nxt])
            else:
                disc[nxt] = low[nxt] = t
                t += 1
                if u == root:
                    children += 1
                stack.append((nxt, u, iter(sorted(v for v in adj[nxt] if v in alive))))
        if children > 1:
            cut.add(root)
    return cut


def _lin_post_order(root, parent):
    """Item 81: the tree's (child, parent) edges, children before parents (deepest first)."""
    kids = {}
    for c, p in parent.items():
        kids.setdefault(p, []).append(c)
    order, stack = [], [(root, 0)]
    depth = {root: 0}
    while stack:
        u, d = stack.pop()
        for c in sorted(kids.get(u, [])):
            depth[c] = d + 1
            stack.append((c, d + 1))
    for c in sorted(parent, key=lambda c: (-depth[c], c)):
        order.append((c, parent[c]))
    return order


def _lin_synth(rows, phys, adj, w):
    """Item 81: CNOTs on the coupling graph, as [(control, target)] physical pairs in circuit order, implementing the
    linear map whose rows are `rows` (rows[k]: bitmask over wires 0..n-1; wire k sits on physical qubit phys[k]),
    by RowCol elimination (Wu et al. 2023; Steiner-Gauss, Kissinger and Meijer-van de Griend 2019) along error-weighted
    Steiner trees: a vertex that does not disconnect the rest is eliminated at each step (its column, then its row).
    None if the wires' qubits are not connected."""
    n = len(rows)
    A = {phys[k]: 0 for k in range(n)}
    for k in range(n):  # rows and columns indexed by physical qubits
        v = 0
        for j in range(n):
            if rows[k] >> j & 1:
                v |= 1 << phys[j]
        A[phys[k]] = v
    alive = set(phys)
    sub = {p: {q for q in adj.get(p, ()) if q in alive} for p in alive}
    seen, dq = {phys[0]}, deque([phys[0]])
    while dq:
        u = dq.popleft()
        for v in sub[u]:
            if v not in seen:
                seen.add(v)
                dq.append(v)
    if seen != alive:
        return None
    ops = []

    def op(c, t):
        A[t] ^= A[c]
        ops.append((c, t))
    while alive:
        cut = _lin_cut_vertices(alive, sub)
        cands = [p for p in alive if p not in cut]
        i = min(cands, key=lambda p: (len([q for q in sub[p] if q in alive]), p))
        dist, par = _lin_dijkstra(alive, sub, w)
        bit = 1 << i
        # the column: every other remaining row loses bit i
        terms = [i] + sorted(p for p in alive if p != i and A[p] & bit)
        if len(terms) > 1 or not A[i] & bit:
            if not A[i] & bit and len(terms) == 1:
                return None  # singular: cannot happen for an invertible matrix
            root, parent = _st_steiner_tree(terms, dist, par)
            order = _lin_post_order(root, parent)
            for c, p in order:  # fill: every tree vertex gets bit i
                if not A[p] & bit:
                    op(c, p)
            for c, p in order:  # clear: children first
                op(p, c)
        # the row: row i becomes its unit row
        rest = {p: A[p] for p in alive if p != i}
        need = A[i] ^ bit
        if need:
            sel = _lin_solve(rest, need)
            if sel is None:
                return None
            root, parent = _st_steiner_tree([i] + sel, dist, par)
            order = _lin_post_order(root, parent)
            chosen = set(sel)
            kids = {}
            for c, p in parent.items():
                kids.setdefault(p, []).append(c)
            for c, p in order:  # Steiner vertices, deepest first: their own rows are cancelled
                if c not in chosen and c != i:
                    op(c, min(kids[c]))
            for c, p in order:  # accumulate towards i
                op(c, p)
        if A[i] != bit:
            return None
        alive.discard(i)
    return ops[::-1]
