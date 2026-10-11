def _st_working_graph(target):
    """(adjacency of the working couplers, failed qubits): couplers and qubits with error >= _ST_FAILED, and qubits whose
    measurement fails, are left out."""
    g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    bad_q = set()
    for name in ("sx", "measure"):
        if name in target.operation_names:
            for qa, p in target[name].items():
                if qa is not None and p is not None and p.error is not None and p.error >= _ST_FAILED:
                    bad_q.add(qa[0])
    adj = {q: set() for q in range(target.num_qubits) if q not in bad_q}
    for qa, p in target[g2].items():
        if qa is None or (p is not None and p.error is not None and p.error >= _ST_FAILED):
            continue
        a, b = qa
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return adj, bad_q


def _st_bfs_all(adj):
    """Hop distances and BFS parents from every working qubit."""
    dist, par = {}, {}
    for s in adj:
        d, p, dq = {s: 0}, {s: None}, deque([s])
        while dq:
            u = dq.popleft()
            for w in sorted(adj[u]):
                if w not in d:
                    d[w], p[w] = d[u] + 1, u
                    dq.append(w)
        dist[s], par[s] = d, p
    return dist, par


def _st_steiner_tree(terminals, dist, par):
    """An approximate Steiner tree (edges as a parent map rooted at terminals[0]) spanning the terminals: Prim's tree on
    their hop distances, each edge expanded into a shortest path, then a BFS tree of the union with non-terminal
    leaves pruned."""
    ts = list(dict.fromkeys(terminals))
    if len(ts) == 1:
        return ts[0], {}
    inside, nodes, edges = {ts[0]}, {ts[0]}, set()
    while len(inside) < len(ts):
        best = min(((dist[u][v], u, v) for u in inside for v in ts if v not in inside), key=lambda x: x[:3])
        _, u, v = best
        x = v
        while x != u:  # walk v -> u along u's BFS tree
            y = par[u][x]
            edges.add((min(x, y), max(x, y)))
            nodes.update((x, y))
            x = y
        inside.add(v)
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    root, parent, dq = ts[0], {ts[0]: None}, deque([ts[0]])
    while dq:
        u = dq.popleft()
        for w in sorted(adj[u]):
            if w not in parent:
                parent[w] = u
                dq.append(w)
    term = set(ts)
    changed = True
    while changed:  # prune non-terminal leaves
        changed = False
        kids = {}
        for c, p in parent.items():
            if p is not None:
                kids.setdefault(p, []).append(c)
        for c in list(parent):
            if c not in term and not kids.get(c) and parent[c] is not None:
                del parent[c]
                changed = True
    return root, {c: p for c, p in parent.items() if p is not None}


def _st_tree_cost(root, parent, terminals):
    steiner = sum(1 for c in parent if c not in terminals)
    return 2 * (len(parent) + steiner)


def _st_placement(n, terms, adj, dist, start):
    """Greedy: logical qubits in order of their co-occurrence with those already placed, each on the free working
    qubit closest (by co-occurrence-weighted hop distance) to its placed partners."""
    w = {}
    for lb, _ in terms:
        sup = [j for j, ch in enumerate(reversed(lb)) if ch != "I"]
        for i in range(len(sup)):
            for k in range(i + 1, len(sup)):
                a, b = sup[i], sup[k]
                w[(a, b)] = w.get((a, b), 0) + 1
                w[(b, a)] = w.get((b, a), 0) + 1
    deg = [sum(w.get((i, j), 0) for j in range(n)) for i in range(n)]
    order = [max(range(n), key=lambda i: (deg[i], -i))]
    phi, free = {order[0]: start}, set(adj) - {start}
    while len(phi) < n:
        cand = [i for i in range(n) if i not in phi]
        nxt = max(cand, key=lambda i: (sum(w.get((i, j), 0) for j in phi), deg[i], -i))
        best = min(free, key=lambda v: (sum(w.get((nxt, j), 0) * dist[phi[j]].get(v, 10 ** 6) for j in phi),
                                        -len(adj[v]), v))
        phi[nxt] = best
        free.discard(best)
    return [phi[i] for i in range(n)]


def _st_gadget_ops(label, coeff, t, phys, dist, par):
    """The gates of exp(-i t c P) on the device: [(name, qubits, params)]."""
    sup = [(j, ch) for j, ch in enumerate(reversed(label)) if ch != "I"]
    if not sup:
        return [("gphase", (), (-t * coeff,))]
    pre, post = [], []
    for j, ch in sup:
        q = phys[j]
        if ch == "X":
            pre.append(("h", (q,), ()))
            post.append(("h", (q,), ()))
        elif ch == "Y":
            pre += [("sdg", (q,), ()), ("h", (q,), ())]
            post += [("h", (q,), ()), ("s", (q,), ())]
    terms_q = [phys[j] for j, _ in sup]
    root, parent = _st_steiner_tree(terms_q, dist, par)
    kids = {}
    for c, p in parent.items():
        kids.setdefault(p, []).append(c)
    term = set(terms_q)
    collect = []

    def visit(v):
        p = parent.get(v)
        if p is not None and v not in term:
            collect.append(("cx", (v, p), ()))  # a Steiner qubit's own bit is added twice to its parent: it cancels
        for c in sorted(kids.get(v, [])):
            visit(c)
        if p is not None:
            collect.append(("cx", (v, p), ()))
    for c in sorted(kids.get(root, [])):
        visit(c)
    return pre + collect + [("rz", (root,), (2 * t * coeff,))] + collect[::-1] + post


def _st_error_distances(adj, target):
    """All-pairs Dijkstra on the working couplers, a coupler costing -ln(1 - its two-qubit error) (the larger of the
    two directions); (dist, par) with STEINER3's interface (par[s][x] = x's parent in the tree from s)."""
    g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    err = {}
    for qa, p in target[g2].items():
        if qa is None or p is None or p.error is None:
            continue
        k = (min(qa), max(qa))
        err[k] = max(err.get(k, 0.0), p.error)
    w = {k: -math.log(1.0 - min(e, 0.999)) + 1e-9 for k, e in err.items()}
    dist, par = {}, {}
    for s in adj:
        d, p, heap = {s: 0.0}, {s: None}, [(0.0, s)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > d[u]:
                continue
            for v in sorted(adj[u]):
                nd = du + w.get((min(u, v), max(u, v)), 1.0)
                if nd < d.get(v, math.inf):
                    d[v], p[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        dist[s], par[s] = d, p
    return dist, par, w


def _st_interaction(n, terms):
    w = {}
    for lb, _ in terms:
        sup = [j for j, ch in enumerate(reversed(lb)) if ch != "I"]
        for i in range(len(sup)):
            for k in range(i + 1, len(sup)):
                e = (sup[i], sup[k])
                w[e] = w.get(e, 0) + 1
    return w


def _st_embeddings(n, terms, adj, dist):
    """Up to _ST_KEEP placements from VF2 _st_embeddings of the heaviest embeddable part of the _st_interaction graph."""
    import rustworkx as rx
    nodes = sorted(adj)
    idx = {q: i for i, q in enumerate(nodes)}
    dev = rx.PyGraph()
    dev.add_nodes_from(nodes)
    dev.add_edges_from_no_data(sorted({(idx[a], idx[b]) for a in adj for b in adj[a] if a < b}))
    w = _st_interaction(n, terms)
    keep = []
    for e in sorted(w, key=lambda e: (-w[e], e)):
        pat = rx.PyGraph()
        pat.add_nodes_from(range(n))
        pat.add_edges_from_no_data(keep + [e])
        if max(pat.degree(i) for i in range(n)) > max(dev.degree(i) for i in range(len(nodes))):
            continue
        if rx.is_subgraph_isomorphic(dev, pat, id_order=False, induced=False, call_limit=_ST_VF2_CHECK):
            keep.append(e)
    pat = rx.PyGraph()
    pat.add_nodes_from(range(n))
    pat.add_edges_from_no_data(keep)
    scored = []
    try:
        it = rx.vf2_mapping(dev, pat, id_order=False, subgraph=True, induced=False, call_limit=_ST_VF2_CHECK * 10)
        for k, m in enumerate(it):
            if k >= _ST_VF2_MAPS:
                break
            phys = [None] * n
            for dnode, lnode in m.items():
                phys[lnode] = nodes[dnode]
            if any(p is None for p in phys):
                continue
            score = sum(c * dist[phys[a]].get(phys[b], 1e6) for (a, b), c in w.items())
            scored.append((score, phys))
    except Exception:  # noqa: BLE001 - no embedding found within the call limit: the greedy _st_placement remains
        pass
    scored.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, phys in scored:
        if tuple(phys) not in seen:
            seen.add(tuple(phys))
            out.append(phys)
        if len(out) >= _ST_KEEP:
            break
    return out, len(keep), len(w)


def _st_placement_cost(phys, terms, dist, par, edge_w):
    """Hop arm (edge_w None): STEINER3's tree cost summed. Error arm: the trees' couplers' costs x CNOTs on them."""
    cost = 0.0
    for lb, _ in terms:
        sup = [phys[j] for j, ch in enumerate(reversed(lb)) if ch != "I"]
        if len(sup) < 2:
            continue
        if any(dist[sup[0]].get(v) is None for v in sup):
            return math.inf
        r, p = _st_steiner_tree(sup, dist, par)
        if edge_w is None:
            cost += _st_tree_cost(r, p, set(sup))
        else:
            term = set(sup)
            cost += sum(edge_w.get((min(c, q), max(c, q)), 1.0) * (2 if c in term else 4) for c, q in p.items())
    return cost


def _st_term_masks(label):
    """(x mask, z mask, support) of a label; character j from the right acts on logical qubit j."""
    x = z = 0
    sup = []
    for j, ch in enumerate(reversed(label)):
        if ch == "I":
            continue
        sup.append(j)
        if ch in "XY":
            x |= 1 << j
        if ch in "ZY":
            z |= 1 << j
    return x, z, tuple(sup)


def _st_commute(a, b):
    return (bin((a[0] & b[1]) ^ (a[1] & b[0])).count("1") & 1) == 0


class _StMover:
    def __init__(self, n, triples, adj, dist, par, pi0, w=64, f=12, e=24, err=None):
        """err: None (hop counts only) or (error distances, their parents, coupler weight -ln(1 - e) by sorted pair):
        bridged terms then take their trees on the error distances, and a SWAP's cost is scaled by its coupler's
        weight over the median coupler's."""
        self.n, self.adj, self.dist, self.par = n, adj, dist, par
        self.err = err
        self._memo = {}
        self._pmemo = {}
        if err is not None:
            ws = sorted(err[2].values())
            self.w_med = ws[len(ws) // 2] if ws else 1.0
        self.triples = triples
        self.masks = [_st_term_masks(lb) for lb, _, _ in triples]
        self.pi = list(pi0)
        self.inv = {p: i for i, p in enumerate(self.pi)}
        self.w, self.f, self.e = w, f, e
        self.ops = []  # (name, physical qubits, params)
        self.gphase = 0.0
        self.stats = dict(swaps=0, merged_swaps=0, bridged_terms=0, bridge_excess=0, terms=0)
        self.order = []
        self.last2q = {}  # physical qubit -> the pair of the last two-qubit gate on it (single-qubit gates join blocks)

    # costs ---------------------------------------------------------
    def cost(self, j, pi=None):
        pi = self.pi if pi is None else pi
        sup = self.masks[j][2]
        if len(sup) <= 1:
            return 0
        key = tuple(sorted(pi[q] for q in sup))
        c = self._memo.get(key)
        if c is None:
            r, p = _st_steiner_tree(list(key), self.dist, self.par)
            c = self._memo[key] = _st_tree_cost(r, p, set(key))
        return c

    def excess(self, j, pi=None):
        k = len(self.masks[j][2])
        return 0 if k <= 1 else self.cost(j, pi) - 2 * (k - 1)

    def proxy_excess(self, j, pi):
        """A cheap stand-in for the excess when scoring SWAPs: for two qubits, 2 (d - 1) x 2 (a path of d hops costs
        2 d + 2 (d - 1) CNOTs); for more, the same on the minimum spanning tree of their hop distances."""
        sup = self.masks[j][2]
        k = len(sup)
        if k <= 1:
            return 0
        phys = [pi[q] for q in sup]
        if k == 2:
            d = self.dist[phys[0]][phys[1]]
            return 4 * (d - 1)
        key = tuple(sorted(phys))
        c = self._pmemo.get(key)
        if c is not None:
            return c
        c = self._pmemo[key] = self._mst_excess(list(key))
        return c

    def _mst_excess(self, phys):
        inside = {phys[0]}
        rest = set(phys[1:])
        tot = 0
        best = {v: self.dist[phys[0]][v] for v in rest}
        while rest:
            v = min(rest, key=lambda u: (best[u], u))
            tot += best[v] - 1
            rest.discard(v)
            inside.add(v)
            for u in rest:
                du = self.dist[v][u]
                if du < best[u]:
                    best[u] = du
        return 4 * tot

    # emission ------------------------------------------------------
    def run_term(self, j):
        lb, c, t = self.triples[j]
        ex = self.excess(j)
        dist, par = (self.dist, self.par) if (self.err is None or ex == 0) else (self.err[0], self.err[1])
        for name, q, prm in _st_gadget_ops(lb, c, t, self.pi, dist, par):
            if name == "gphase":
                self.gphase += prm[0]
            else:
                self.ops.append((name, q, prm))
                if len(q) == 2:
                    pair = frozenset(q)
                    self.last2q[q[0]] = self.last2q[q[1]] = pair
        if ex > 0:
            self.stats["bridged_terms"] += 1
            self.stats["bridge_excess"] += ex
        self.stats["terms"] += 1
        self.order.append(j)

    def mergeable(self, p, q):
        """A SWAP on (p, q) joins the block of the last two-qubit gates there: Qiskit's level 3 fuses them into one
        two-qubit unitary of at most 3 CNOTs, so the SWAP adds at most 1 CNOT to a block that had 2 (a ZZ term)."""
        pair = frozenset((p, q))
        return self.last2q.get(p) == pair and self.last2q.get(q) == pair

    def swap(self, p, q):
        if self.mergeable(p, q):
            self.stats["merged_swaps"] += 1
        self.ops.append(("swap", (p, q), ()))
        self.last2q[p] = self.last2q[q] = frozenset((p, q))
        a, b = self.inv.get(p), self.inv.get(q)
        if a is not None:
            self.pi[a] = q
        if b is not None:
            self.pi[b] = p
        self.inv = {}
        for i, v in enumerate(self.pi):
            self.inv[v] = i
        self.stats["swaps"] += 1

    # scheduling ----------------------------------------------------
    def build(self):
        T = len(self.triples)
        unrun = list(range(T))  # in order
        pos = 0
        guard = 0
        stuck_swaps = 0
        while unrun:
            guard += 1
            if guard > 50 * T + 10000:
                raise RuntimeError("no progress")
            window = unrun[: self.w]
            elig = []
            for idx, j in enumerate(window):
                mj = self.masks[j]
                if all(_st_commute(self.masks[i], mj) for i in window[:idx]):
                    elig.append(j)
            # run every eligible term that needs no bridge
            free = [j for j in elig if self.excess(j) == 0]
            if free:
                for j in free:
                    self.run_term(j)
                done = set(free)
                unrun = [j for j in unrun if j not in done]
                stuck_swaps = 0
                continue
            front = elig[: self.f]
            ext = elig[self.f: self.f + self.e]
            # beyond the window: the next terms in order, as extended set when the window is small
            h0 = self.score(front, ext, self.pi)
            best = None
            cand = set()
            for j in front:
                for q in self.masks[j][2]:
                    p = self.pi[q]
                    for v in self.adj[p]:
                        cand.add((min(p, v), max(p, v)))
            for (p, v) in sorted(cand):
                pi2 = list(self.pi)
                a, b = self.inv.get(p), self.inv.get(v)
                if a is not None:
                    pi2[a] = v
                if b is not None:
                    pi2[b] = p
                merged = self.mergeable(p, v)
                c_swap = 1 if merged else 3
                if self.err is not None:
                    c_swap *= self.err[2].get((p, v), self.w_med) / self.w_med
                gain = h0 - self.score(front, ext, pi2)
                key = (gain - c_swap, merged, -p, -v)
                if best is None or key > best[0]:
                    best = (key, p, v, merged, gain, c_swap)
            min_ex = min((self.excess(j), j) for j in front)
            if best is not None and best[4] > best[5] and stuck_swaps < 4 * self.n:
                _, p, v, merged, _, _ = best
                self.swap(p, v)
                stuck_swaps += 1
                continue
            # no SWAP pays: run the cheapest eligible term with its bridges
            j = min_ex[1]
            self.run_term(j)
            unrun = [x for x in unrun if x != j]
            stuck_swaps = 0
        return self

    def score(self, front, ext, pi):
        h = sum(self.proxy_excess(j, pi) for j in front)
        if ext:
            h += 0.5 * sum(self.proxy_excess(j, pi) for j in ext) / len(ext) * len(front) / max(1, len(front))
        return h


class _StMover2(_StMover):
    """STEINER6 v4: every unrun term that commutes with all earlier unrun terms is eligible (not only those in a
    window); the anticommutation between terms is computed once. A SWAP is scored on every eligible term that
    contains one of the two logical qubits it moves (weight 1) and on the blocked ones (weight LAMBDA), by the change
    of their proxy excess; it is made if that gain exceeds its cost (3 CNOTs, 1 when it joins the last block on the
    pair, scaled by the coupler's error weight in the error mode). Otherwise the eligible term of the least excess
    (earliest on a tie) runs with its bridges."""
    LAMBDA = 0.1
    LOOK = 200
    MU = 0.03

    def build(self):
        import numpy as np
        T = len(self.triples)
        n = self.n
        X = np.zeros((T, n), dtype=np.uint8)
        Z = np.zeros((T, n), dtype=np.uint8)
        for j, (x, z, sup) in enumerate(self.masks):
            for q in sup:
                X[j, q] = (x >> q) & 1
                Z[j, q] = (z >> q) & 1
        succ = [[] for _ in range(T)]
        blocked = [0] * T
        B = 512
        for i0 in range(0, T, B):
            a = ((X[i0:i0 + B].astype(np.int32) @ Z.T.astype(np.int32)) + (Z[i0:i0 + B].astype(np.int32)
                 @ X.T.astype(np.int32))) & 1
            for ii in range(a.shape[0]):
                i = i0 + ii
                later = np.nonzero(a[ii, i + 1:])[0] + i + 1
                succ[i] = later.tolist()
                for j in succ[i]:
                    blocked[j] += 1
        terms_of = [[] for _ in range(n)]
        for j, (_, _, sup) in enumerate(self.masks):
            for q in sup:
                terms_of[q].append(j)
        unrun = [True] * T
        elig = set(j for j in range(T) if blocked[j] == 0)
        ex = {j: self.excess(j) for j in elig}
        left = T
        stuck = 0
        guard = 0
        ptr = 0
        while left:
            guard += 1
            if guard > 20 * T + 100000:
                raise RuntimeError("no progress")
            free = sorted(j for j in elig if ex[j] == 0)
            if free:
                small = [k for k in free if len(self.masks[k][2]) <= 2]
                if small:  # terms on one or two qubits: in the given order, all at once
                    for k in small:
                        self._run(k, unrun, elig, ex, succ, blocked)
                        left -= 1
                else:  # terms on 3 or more qubits commute pairwise: the most similar to the last one next
                    j = max(free, key=lambda k: (self.similar(self.last_term, k), -k))
                    self._run(j, unrun, elig, ex, succ, blocked)
                    left -= 1
                stuck = 0
                continue
            # the blocked terms looked at: the next LOOK unrun ones in the given order
            while ptr < T and not unrun[ptr]:
                ptr += 1
            look = set()
            k = ptr
            while k < T and len(look) < self.LOOK:
                if unrun[k] and k not in elig:
                    look.add(k)
                k += 1
            # SWAP candidates: couplers next to the qubits of the eligible terms of least excess
            order = sorted(elig, key=lambda j: (ex[j], j))
            focus = order[: self.f]
            cand = set()
            for j in focus:
                for q in self.masks[j][2]:
                    p = self.pi[q]
                    for v in self.adj[p]:
                        cand.add((min(p, v), max(p, v)))
            best = None
            for (p, v) in sorted(cand):
                a, b = self.inv.get(p), self.inv.get(v)
                pi2 = list(self.pi)
                if a is not None:
                    pi2[a] = v
                if b is not None:
                    pi2[b] = p
                touched = set()
                for lq in (a, b):
                    if lq is not None:
                        touched.update(terms_of[lq])
                gain = 0.0
                fset = set(focus)
                for j in touched:
                    if not unrun[j] or not (j in elig or j in look):
                        continue
                    wgt = 1.0 if j in fset else (self.MU if j in elig else self.LAMBDA)
                    gain += wgt * (self.proxy_excess(j, self.pi) - self.proxy_excess(j, pi2))
                merged = self.mergeable(p, v)
                c_swap = 1 if merged else 3
                if self.err is not None:
                    c_swap *= self.err[2].get((p, v), self.w_med) / self.w_med
                key = (gain - c_swap, merged, -p, -v)
                if best is None or key > best[0]:
                    best = (key, p, v, a, b)
            if best is not None and best[0][0] > 0 and stuck < 4 * n:
                _, p, v, a, b = best
                self.swap(p, v)
                for lq in (a, b):
                    if lq is not None:
                        for j in terms_of[lq]:
                            if j in elig:
                                ex[j] = self.excess(j)
                stuck += 1
                continue
            j = order[0]
            self._run(j, unrun, elig, ex, succ, blocked)
            left -= 1
            stuck = 0
        return self

    last_term = None

    def similar(self, i, k):
        """Qubits on which terms i and k carry the same Pauli, minus half those where they differ (0 for no i)."""
        if i is None:
            return 0
        xi, zi, si = self.masks[i]
        xk, zk, sk = self.masks[k]
        both = set(si) & set(sk)
        same = sum(1 for q in both if ((xi >> q) & 1) == ((xk >> q) & 1) and ((zi >> q) & 1) == ((zk >> q) & 1))
        return same - 0.5 * (len(both) - same)

    def _run(self, j, unrun, elig, ex, succ, blocked):
        self.last_term = j
        self.run_term(j)
        unrun[j] = False
        elig.discard(j)
        ex.pop(j, None)
        for k in succ[j]:
            blocked[k] -= 1
            if blocked[k] == 0 and unrun[k]:
                elig.add(k)
                ex[k] = self.excess(k)
