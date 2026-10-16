"""Prototype B: formula before VF2 in c63a's placement sites (patched in place; c63a's file is not changed).

B-a  _scope96_embeds: (1) the device's working graph is bipartite and the interaction graph is not -> False;
     (2) the interaction graph has a cycle shorter than the device's shortest cycle -> False;
     (3) the interaction graph is a path and the device's snake has a working piece at least as long -> True.
     Otherwise VF2 as before. (1)-(2) are necessary conditions, so a False is the answer VF2 must also give.
B-b  _graph_place: an interaction graph that is a path is placed on the snake's longest piece in the given
     adjacency, centred (item 87's rule); otherwise VF2 as before. This can change the placement.
"""
import collections


def _bfs_girth(nodes, adj):
    best = None
    for r in nodes:
        dist, par = {r: 0}, {r: None}
        dq = collections.deque([r])
        while dq:
            u = dq.popleft()
            if best is not None and 2 * dist[u] + 1 >= best:
                break
            for v in adj[u]:
                if v not in dist:
                    dist[v], par[v] = dist[u] + 1, u
                    dq.append(v)
                elif par[u] != v:
                    c = dist[u] + dist[v] + 1
                    if best is None or c < best:
                        best = c
    return best  # None: a forest


def _bipartite(nodes, adj):
    col = {}
    for r in nodes:
        if r in col:
            continue
        col[r] = 0
        dq = collections.deque([r])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in col:
                    col[v] = 1 - col[u]
                    dq.append(v)
                elif col[v] == col[u]:
                    return False
    return True


def _path_order(edges):
    """The nodes of `edges` in path order, or None when the graph is not a simple path."""
    adj = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    nodes = list(adj)
    if len(edges) != len(nodes) - 1 or any(len(adj[q]) > 2 for q in nodes):
        return None
    ends = [q for q in nodes if len(adj[q]) == 1]
    if len(ends) != 2:
        return None
    order, prev = [min(ends)], None
    while len(order) < len(nodes):
        nxt = [v for v in adj[order[-1]] if v != prev]
        if not nxt:
            return None
        prev = order[-1]
        order.append(nxt[0])
    return order


def _pieces(snake, ok, adj):
    pieces, cur = [], []
    for q in snake:
        if ok(q) and (not cur or q in adj[cur[-1]]):
            cur.append(q)
        else:
            if cur:
                pieces.append(cur)
            cur = [q] if ok(q) else []
    if cur:
        pieces.append(cur)
    return pieces


STATS = collections.Counter()


def install(pc):
    orig_embeds, orig_place, orig_lattice = pc._scope96_embeds, pc._graph_place, pc._chain_lattice
    cache = {}
    last = {}

    def lattice(target, max_error):
        r = orig_lattice(target, max_error)
        last["snake"] = r[3] if r else None
        return r

    def embeds(edges, n, target, max_error):
        cm = target.build_coupling_map()
        if cm is None:
            return orig_embeds(edges, n, target, max_error)
        fedges, fq = pc._failed_elements(target, max_error)
        und = set()
        for a, b in cm.get_edges():
            if a in fq or b in fq or (a, b) in fedges or (b, a) in fedges:
                continue
            und.add((min(a, b), max(a, b)))
        key = (frozenset(und), target.num_qubits)
        if key not in cache:
            gadj = collections.defaultdict(set)
            for a, b in und:
                gadj[a].add(b)
                gadj[b].add(a)
            full = {q: set() for q in range(target.num_qubits)}
            for a, b in cm.get_edges():
                full[a].add(b)
                full[b].add(a)
            snake = pc._chain_snake(target.num_qubits, full)
            longest = max((len(p) for p in _pieces(snake, lambda q: q in gadj, gadj)), default=0)
            cache[key] = (_bipartite(list(gadj), gadj), _bfs_girth(sorted(gadj), gadj), longest)
        g_bip, g_girth, longest = cache[key]
        padj = collections.defaultdict(set)
        for a, b in edges:
            padj[a].add(b)
            padj[b].add(a)
        if g_bip and not _bipartite(list(padj), padj):
            STATS["a_not_bipartite"] += 1
            return False
        p_girth = _bfs_girth(sorted(padj), padj)
        if p_girth is not None and (g_girth is None or p_girth < g_girth):
            STATS["a_short_cycle"] += 1
            return False
        if _path_order(edges) is not None and longest >= len(padj):
            STATS["a_path_snake"] += 1
            return True
        STATS["a_vf2"] += 1
        return orig_embeds(edges, n, target, max_error)

    def place(n, edges, K, kadj):
        order = _path_order(edges)
        snake = last.get("snake")
        if order is not None and snake and len(order) == n:
            pieces = _pieces(snake, lambda q: q in K, kadj)
            if pieces:
                best = max(pieces, key=lambda p: (len(p), -snake.index(p[0])))
                if len(best) >= n:
                    s = (len(best) - n) // 2
                    STATS["b_path_snake"] += 1
                    return {order[i]: best[s + i] for i in range(n)}
        STATS["b_vf2"] += 1
        return orig_place(n, edges, K, kadj)

    pc._chain_lattice = lattice
    pc._scope96_embeds = embeds
    pc._graph_place = place
