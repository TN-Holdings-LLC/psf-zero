"""q1_common.py -- circuits for Q1 (item 91): trees, heavy-hex Ising patches, rings, C52-VAL's heis2d."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa


def evo(n, terms, t, steps, order=2, measure=True):
    return pauli_chain(n, steps, terms, t=t, measure=measure, order=order)


def tree(n, seed):
    """a random tree on n nodes with max degree 3, labels shuffled"""
    rng = random.Random(seed)
    edges, deg = [], {0: 0}
    for v in range(1, n):
        cand = [u for u in range(v) if deg[u] < 3]
        u = rng.choice(cand)
        edges.append((u, v)); deg[u] += 1; deg[v] = 1
    perm = list(range(n)); rng.shuffle(perm)
    return [(perm[a], perm[b]) for a, b in edges]


def _adj(target, adj=None):
    if adj is not None:
        return adj
    cm = target.build_coupling_map()
    adj = {}
    for a, b in cm.get_edges():
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    return adj


def hh_patch(target, n, start=0, seed=0, adj=None):
    """n qubits of the device's coupling graph (or of `adj`, e.g. the cut lattice) grown by BFS from `start`, all
    couplers among them (heavy-hex patch), labels shuffled"""
    adj = _adj(target, adj)
    order, seen, fr = [], {start}, [start]
    while fr and len(order) < n:
        u = fr.pop(0); order.append(u)
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v); fr.append(v)
    S = set(order[:n])
    rng = random.Random(seed); perm = list(range(n)); rng.shuffle(perm)
    idx = {q: perm[i] for i, q in enumerate(order[:n])}
    return sorted({(min(idx[a], idx[b]), max(idx[a], idx[b])) for a in S for b in adj[a] if b in S})


def ising(n, edges, steps, seed=5, t=1.0, signs=False):
    rng = random.Random(seed)
    J = (lambda: rng.choice([-1, 1]) * rng.uniform(0.3, 2.0)) if signs else (lambda: 1.0)
    terms = [(lab(n, {a: "Z", b: "Z"}), J()) for a, b in edges] + [(lab(n, {q: "X"}), 0.7) for q in range(n)]
    return evo(n, terms, t, steps)


def heis_graph(n, edges, steps, seed=6, t=0.6):
    rng = random.Random(seed)
    terms = [(lab(n, {a: p, b: p}), rng.uniform(-1, 1)) for a, b in edges for p in "XYZ"]
    terms += [(lab(n, {q: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
    return evo(n, terms, t, steps)


def ring(n):
    return [(q, (q + 1) % n) for q in range(n)]


def heis2d(n=16, reps=1):
    sys.path.insert(0, os.path.join(REPO, "data", "2026-10-12", "c48_dev"))
    import gens
    return gens.gen(f"heis2d_n{n}_r{reps}")


def bfs_tree(target, n, start=0, seed=0, adj=None):
    """a spanning tree (BFS from `start`) of an n-qubit heavy-hex patch: embeddable by construction, labels shuffled"""
    adj = _adj(target, adj)
    order, par, seen, fr = [], {}, {start}, [start]
    while fr and len(order) < n:
        u = fr.pop(0); order.append(u)
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v); par[v] = u; fr.append(v)
    S = order[:n]
    rng = random.Random(seed); perm = list(range(n)); rng.shuffle(perm)
    idx = {q: perm[i] for i, q in enumerate(S)}
    return sorted((min(idx[v], idx[par[v]]), max(idx[v], idx[par[v]])) for v in S[1:])


def heis_xyz(n, edges, steps, seed=7, t=0.6):
    """XYZ model on a graph with fields, terms listed all XX, then all YY, then all ZZ, then Z fields"""
    rng = random.Random(seed)
    terms = [(lab(n, {a: p, b: p}), rng.uniform(0.5, 1.5) * rng.choice([-1, 1])) for p in "XYZ" for a, b in edges]
    terms += [(lab(n, {q: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
    return evo(n, terms, t, steps, order=1)
