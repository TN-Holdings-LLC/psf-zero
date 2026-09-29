"""Exploratory (not pre-registered): coupling-graph facts of FakeKingston and whether a
vertex-disjoint cover by 2- and 3-qubit paths (a {P2,P3}-factor) exists."""
import statistics as st
import rustworkx as rx
import networkx as nx
from qiskit_ibm_runtime.fake_provider import FakeKingston

b = FakeKingston()
cmap = b.target.build_coupling_map()
nq = cmap.size()
edges = sorted({tuple(sorted(e)) for e in cmap.get_edges()})
g = nx.Graph(); g.add_nodes_from(range(nq)); g.add_edges_from(edges)
deg = sorted(d for _, d in g.degree())
m = nx.max_weight_matching(g, maxcardinality=True)
print("qubits", nq, "edges", len(edges), "degree min/median/max", deg[0], st.median(deg), deg[-1],
      "deg hist", {d: deg.count(d) for d in set(deg)})
print("max matching", len(m), "unmatched", nq - 2 * len(m), "bipartite", nx.is_bipartite(g), "connected", nx.is_connected(g))
print("native", [x for x in b.target.operation_names])
# {P2,P3}-factor: attach each unmatched vertex to a distinct adjacent matched pair
def try_factor(m):
    mate = {}
    for u, v in m:
        mate[u] = v; mate[v] = u
    un = [x for x in range(nq) if x not in mate]
    pid = {}
    for k, (u, v) in enumerate(sorted(tuple(sorted(e)) for e in m)):
        pid[u] = k; pid[v] = k
    bg = nx.Graph()
    bg.add_nodes_from(("u", x) for x in un)
    for x in un:
        for y in g.neighbors(x):
            bg.add_edge(("u", x), ("p", pid[y]))
    mm = nx.bipartite.maximum_matching(bg, top_nodes=[("u", x) for x in un])
    got = sum(1 for k in mm if k[0] == "u")
    return got, len(un)
print("SDR attach (first matching):", try_factor(m))
import random
best = 0
for s in range(20):
    rng = random.Random(s)
    h = nx.Graph(); nodes = list(range(nq)); rng.shuffle(nodes); h.add_nodes_from(nodes)
    ee = edges[:]; rng.shuffle(ee); h.add_edges_from(ee)
    mm = nx.max_weight_matching(h, maxcardinality=True)
    got, need = try_factor(mm)
    best = max(best, got)
    if got == need:
        print("factor found with seed", s, got, need); break
print("best", best)
