"""make_c65.py -- c65 = the base psf_compile + item 102: item 96's embedding test decides by formula before VF2.

    python patches/psf_compile_c65_2026-10-16/make_c65.py <base psf_compile.py> <output psf_compile.py>

The base is c63a (the workplace's 2026-10-15c delivery) until c64 (c63a with the unreached code removed) is
delivered; then c65 is made again from c64 with this same script. Only `_scope96_embeds` is replaced, and the version
line rewritten. Prototype B (2026-10-16) measured it: the same output, bit for bit, on 924 calls.
"""
import re
import sys

ITEM = '''

# ---------------------------------------------------------------------------------------------------------------------
# Item 102 (c65, home, 2026-10-16): item 96's embedding test decides by formula before VF2.
# ---------------------------------------------------------------------------------------------------------------------
_S96_GRAPH_CACHE = {}  # item 102: per working graph, (bipartite, shortest cycle, longest working piece of the snake)
_S96_GRAPH_CACHE_MAX = 64
S96_FORMULA_STATS = {"not_bipartite": 0, "short_cycle": 0, "path_on_snake": 0, "vf2": 0}


def _s96_bipartite(adj):
    """Item 102: True when the graph {node: neighbours} has no odd cycle (two-colouring by breadth-first search)."""
    import collections
    col = {}
    for r in sorted(adj):
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


def _s96_girth(adj):
    """Item 102: the length of the graph's shortest cycle (breadth-first search from every node), None for a forest."""
    import collections
    best = None
    for r in sorted(adj):
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
    return best


def _s96_is_path(adj, n_edges):
    """Item 102: True when the graph {node: neighbours} with n_edges edges is one simple path."""
    if n_edges != len(adj) - 1 or any(len(v) > 2 for v in adj.values()):
        return False
    start = next((q for q in sorted(adj) if len(adj[q]) == 1), None)
    if start is None:
        return False
    seen, prev, cur = {start}, None, start
    while True:
        nxt = [v for v in adj[cur] if v != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur in seen:
            return False
        seen.add(cur)
    return len(seen) == len(adj)


def _scope96_embeds(edges, n, target, max_error):
    """Item 96: True when the interaction graph `edges` (of an n-qubit circuit) embeds in the device's working
    coupling graph -- couplers and qubits the Target reports failed (error >= max_error, either direction) removed --
    by the first match of rustworkx's VF2 (subgraph monomorphism, nodes in index order, no score). Not finding one
    within _S96_VF2_CALLS states counts as no embedding.

    Item 102: decided by formula first, VF2 only when the formula does not decide.
      - The working graph is bipartite (heavy-hex is) and the interaction graph is not: no embedding (an embedding
        maps an odd cycle to an odd cycle).
      - The interaction graph has a cycle shorter than the working graph's shortest cycle (12 on heavy-hex): no
        embedding (an embedding maps a cycle to a cycle of the same length).
      - The interaction graph is a path, and the device's snake (item 87) has a working piece at least as long:
        it embeds.
    The first two are necessary conditions, so their answer is the one VF2 gives. Why: VF2 proving that a ring of 4-11
    qubits, or of odd length, does not embed took up to 70% of a compile of TOQB's ring circuits (prototype B,
    2026-10-16)."""
    import rustworkx as rx
    cm = target.build_coupling_map()
    if cm is None:  # all-to-all: every graph embeds
        return True
    fedges, fq = _failed_elements(target, max_error)
    und = set()
    for a, b in cm.get_edges():
        if a in fq or b in fq or (a, b) in fedges or (b, a) in fedges:
            continue
        und.add((min(a, b), max(a, b)))
    working = target.num_qubits - len(fq)
    if n > working or not und:
        return False
    key = (target.num_qubits, frozenset(und), frozenset(tuple(e) for e in cm.get_edges()))
    facts = _S96_GRAPH_CACHE.get(key)
    if facts is None:
        gadj = {}
        for a, b in und:
            gadj.setdefault(a, set()).add(b)
            gadj.setdefault(b, set()).add(a)
        full = {q: set() for q in range(target.num_qubits)}
        for a, b in cm.get_edges():
            full[a].add(b)
            full[b].add(a)
        longest, cur = 0, []
        for q in _chain_snake(target.num_qubits, full):
            if q in gadj and (not cur or q in gadj[cur[-1]]):
                cur.append(q)
            else:
                cur = [q] if q in gadj else []
            longest = max(longest, len(cur))
        facts = (_s96_bipartite(gadj), _s96_girth(gadj), longest)
        if len(_S96_GRAPH_CACHE) >= _S96_GRAPH_CACHE_MAX:
            _S96_GRAPH_CACHE.clear()
        _S96_GRAPH_CACHE[key] = facts
    g_bipartite, g_girth, longest = facts
    padj = {}
    for a, b in edges:
        padj.setdefault(a, set()).add(b)
        padj.setdefault(b, set()).add(a)
    if g_bipartite and not _s96_bipartite(padj):
        S96_FORMULA_STATS["not_bipartite"] += 1
        return False
    p_girth = _s96_girth(padj)
    if p_girth is not None and (g_girth is None or p_girth < g_girth):
        S96_FORMULA_STATS["short_cycle"] += 1
        return False
    if _s96_is_path(padj, len(set(edges))) and longest >= len(padj):
        S96_FORMULA_STATS["path_on_snake"] += 1
        return True
    S96_FORMULA_STATS["vf2"] += 1
    gn = sorted({q for e in und for q in e})
    gi = {q: i for i, q in enumerate(gn)}
    G = rx.PyGraph()
    G.add_nodes_from(gn)
    G.add_edges_from_no_data([(gi[a], gi[b]) for a, b in sorted(und)])
    pn = sorted({q for e in edges for q in e})
    pi = {q: i for i, q in enumerate(pn)}
    P = rx.PyGraph()
    P.add_nodes_from(pn)
    P.add_edges_from_no_data([(pi[a], pi[b]) for a, b in edges])
    if len(pn) > len(gn) or max(P.degree(i) for i in P.node_indices()) > max(G.degree(i) for i in G.node_indices()):
        return False
    m = next(rx.vf2_mapping(G, P, subgraph=True, induced=False, id_order=True, call_limit=_S96_VF2_CALLS), None)
    return m is not None
'''

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()
starts = [m.start() for m in re.finditer(r"^def _scope96_embeds\(", s, flags=re.M)]
if len(starts) != 1:
    raise SystemExit(f"STOP: expected one _scope96_embeds, found {len(starts)}")
st = starts[0]
m = re.compile(r"^(def |class |[A-Za-z_][A-Za-z_0-9]* *=|# ---)", re.M).search(s, st + 1)
end = m.start() if m else len(s)
s = s[:st].rstrip("\n") + ITEM + "\n\n" + s[end:].lstrip("\n")
v = re.search(r'^VERSION = "(2026-10-1[56]\.c6[34]a?)"  # ', s, flags=re.M)
if not v:
    raise SystemExit("STOP: version line not found")
s = s[:v.start()] + f'VERSION = "2026-10-16.c65"  # item 102 (item 96 by formula before VF2) on {v.group(1)}; {v.group(1)} = ' \
    + s[v.end():]
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst, len(s.splitlines()), "lines")
