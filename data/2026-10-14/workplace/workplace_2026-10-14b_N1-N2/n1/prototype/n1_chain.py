"""n1_chain.py -- N1 (2026-10-14): a Pauli evolution whose two-qubit terms form one chain, built by formula as a
brickwork on the device. Prototype for a c49 patch: it uses c49's own reader (items 72-73, `_st_parse`,
`_st_parse_gates`, `_st_merge`) and nothing that scores by the device's errors.

    chain_brickwork(qc, target, max_error=0.5) -> compiled circuit (all of the device's qubits, measurements kept,
                                                  layout attached) or None when the input is not a chain

Steps (all fixed; no search, no estimate):
  1. READ. Terms [(label, angle)] in order, through c49's reader, merged where exact (`_st_merge`). Applies when
     every term acts on one or two qubits, and the two-qubit supports form one path through all n qubits.
  2. CUT AND PLACE (W5, variant B). Qubits with sx or readout error >= max_error and couplers with error >= max_error
     are removed; the largest connected component K is kept. The chain goes, in path order, on the longest piece of
     Heron's snake that survives in K, centred; if no piece is long enough, on the first path of n qubits that
     rustworkx's VF2 finds in K with nodes in index order (no score). Both are tried first on K's couplers whose
     two-qubit gate has the device's most common length (gate lengths are fixed per device), then on all of K.
     Otherwise: not applicable.
  3. ORDER (formula). Every term gets a level: 1 + the largest level of an earlier term that shares a qubit with it
     and does not commute with it (0 if none). Terms are emitted by level; within a level, two-qubit terms on the
     chain's even edges, then odd edges, then single-qubit terms. Every non-commuting pair keeps its order, so the
     action is the input's.
  4. EMIT. exp(-i a P) on two adjacent qubits as basis change, cx, rz(2a), cx, basis change back; single-qubit terms
     as rotations. Qiskit's level-2 pass manager then translates to the device's gates on the fixed layout with no
     routing (`routing_method="none"`), coupling map and basis only (no Target, so nothing is re-placed by errors).
"""
import math

import numpy as np

_X, _Z = 1, 2
LEVEL = 3   # Qiskit's preset level used only to translate on the fixed layout (no routing, no Target)


def _label_support(lb):
    n = len(lb)
    return [(n - 1 - i, ch) for i, ch in enumerate(lb) if ch != "I"]


def _commute(a, b):
    sa = dict(_label_support(a))
    sb = dict(_label_support(b))
    anti = 0
    for q in set(sa) & set(sb):
        if sa[q] != sb[q]:
            anti += 1
    return anti % 2 == 0


def read_chain(C, qc):
    """(prefix, meas, terms [(label, angle)], phase, suffix, perm, path [logical qubits in chain order]) or None"""
    ev = C._st_parse(qc)
    ev = ev + ([], None) if ev is not None else C._st_parse_gates(qc)
    if ev is None:
        return None
    prefix, meas, triples, phase, suffix, perm = ev
    terms = [(lb, a * t) for lb, a, t in C._st_merge(triples)]
    n = qc.num_qubits
    adj = {q: set() for q in range(n)}
    for lb, _ in terms:
        sup = [q for q, _ in _label_support(lb)]
        if len(sup) > 2:
            return None
        if len(sup) == 2:
            a, b = sup
            adj[a].add(b)
            adj[b].add(a)
    if any(len(v) > 2 for v in adj.values()):
        return None
    ends = [q for q in range(n) if len(adj[q]) == 1]
    if n == 1 or len(ends) != 2:
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


def lattice_from_target(target, max_error):
    """(K, kadj, snake path) from a Target: W5's cut and snake, read from the Target's coupling and errors."""
    N = target.num_qubits
    two = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    err, dur = {}, {}
    for qa, pr in target[two].items():
        if qa is None:
            continue
        k = (min(qa), max(qa))
        e = pr.error if pr is not None and pr.error is not None else 0.0
        err[k] = max(err.get(k, 0.0), e)
        d = pr.duration if pr is not None and pr.duration is not None else 0.0
        dur[k] = max(dur.get(k, 0.0), d)
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
    K = max(comps, key=lambda c: (len(c), -min(c)))
    kadj = {q: {v for v in gadj[q] if v in K} for q in K}
    # gate lengths (fixed per device, allowed): the couplers whose two-qubit gate has the device's most common length
    from collections import Counter
    lens = Counter(round(dur[(min(a, b), max(a, b))] * 1e12) for a in K for b in kadj[a] if a < b)
    modal = max(lens, key=lambda x: (lens[x], -x)) if lens else None
    fadj = {q: {v for v in kadj[q] if round(dur[(min(q, v), max(q, v))] * 1e12) == modal} for q in K}
    return K, kadj, _snake(N, adj), fadj


def _snake(N, adj):
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


def place_chain(n, K, kadj, snake):
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
    Pg = rx.PyGraph()
    Pg.add_nodes_from(range(n))
    Pg.add_edges_from_no_data([(i, i + 1) for i in range(n - 1)])
    m = next(rx.vf2_mapping(G, Pg, subgraph=True, induced=False, id_order=True, call_limit=5_000_000), None)
    if m is None:
        return None, "none"
    inv = {h: kn[g] for g, h in m.items()}
    return [inv[i] for i in range(n)], "vf2 path"


def _rot_ops(lb, a, phys_of):
    """gates for exp(-i a P), P on <= 2 qubits: list of (name, qubits, params) on physical qubits"""
    sup = _label_support(lb)
    if len(sup) == 1:
        q, ch = sup[0]
        return [({"X": "rx", "Y": "ry", "Z": "rz"}[ch], [phys_of[q]], [2 * a])]
    (q1, c1), (q2, c2) = sup
    p1, p2 = phys_of[q1], phys_of[q2]
    if c1 == c2:                                  # XX, YY, ZZ: Qiskit's own two-qubit rotation gate
        return [({"X": "rxx", "Y": "ryy", "Z": "rzz"}[c1], [p1, p2], [2 * a])]
    pre, post = [], []
    for p, ch in ((p1, c1), (p2, c2)):
        if ch == "X":
            pre.append(("h", [p], []))
            post.append(("h", [p], []))
        elif ch == "Y":
            pre.append(("rx", [p], [math.pi / 2]))
            post.append(("rx", [p], [-math.pi / 2]))
    return pre + [("cx", [p1, p2], []), ("rz", [p2], [2 * a]), ("cx", [p1, p2], [])] + post


def chain_brickwork(qc, target, C, max_error=0.5, seed=0):
    from qiskit import QuantumCircuit
    from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
    got = read_chain(C, qc)
    if got is None:
        return None, "not a chain"
    prefix, meas, terms, phase, suffix, perm, path = got
    n, N = qc.num_qubits, target.num_qubits
    K, kadj, snake, fadj = lattice_from_target(target, max_error)
    phys_path, how = place_chain(n, K, fadj, snake)        # first on the couplers of the common gate length
    if phys_path is not None:
        how += " (common gate length)"
    else:
        phys_path, how = place_chain(n, K, kadj, snake)
    if phys_path is None:
        return None, "no path of n qubits in K"
    phys_of = {path[i]: phys_path[i] for i in range(n)}
    rank = {path[i]: i for i in range(n)}
    level = []
    last = {}                                     # qubit -> list of (index, label) of earlier terms on it
    for k, (lb, a) in enumerate(terms):
        lv = 0
        for q, _ in _label_support(lb):
            for j, lb2 in last.get(q, []):
                if not _commute(lb, lb2):
                    lv = max(lv, level[j] + 1)
        level.append(lv)
        for q, _ in _label_support(lb):
            last.setdefault(q, []).append((k, lb))

    def key(k):
        lb, _ = terms[k]
        sup = [q for q, _ in _label_support(lb)]
        if len(sup) == 2:
            lo = min(rank[q] for q in sup)
            return (level[k], 0, lo % 2, lo, k)
        return (level[k], 1, 0, rank[sup[0]], k)
    order = sorted(range(len(terms)), key=key)
    phys = QuantumCircuit(N, qc.num_clbits, global_phase=phase)
    for op, q in prefix:
        phys.append(op, [phys_of[q]])
    for k in order:
        for name, qs, prm in _rot_ops(terms[k][0], terms[k][1], phys_of):
            getattr(phys, name)(*prm, *qs)
    for op, q in suffix:
        phys.append(op, [phys_of[q]])
    for q, cb in meas:
        phys.measure(phys_of[perm[q]], cb)
    edges = sorted({(a, b) for a in K for b in kadj[a]})
    basis = [g for g in ("cz", "ecr", "cx") if g in target.operation_names][:1] + ["rz", "sx", "x"]
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for a, b in edges:
        cm.add_edge(a, b)
    pm = generate_preset_pass_manager(LEVEL, coupling_map=cm, basis_gates=basis,
                                      initial_layout=list(range(N)), routing_method="none", seed_transpiler=seed)
    out = pm.run(phys)
    return out, how
