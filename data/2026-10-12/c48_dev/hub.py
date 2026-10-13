"""Prototype (c48 development): copies of a hub's Z value on spare qubits ("holders"), inside its diagonal segments.

For a logical qubit h, a diagonal segment is a maximal run of its terms (in the term order) that are all Z-type. Before
the segment's first term a CNOT h -> c copies h's computational value onto a holder c (a spare qubit in |0>); after
its last term the same CNOT returns c to |0>. In between Z_h Z_c = +1 holds, so any term Z_h Z_p ... may be applied as
Z_c Z_p ... instead. Every term between the copy and the return that touches h or c is diagonal, so the relation is
kept. The CNOT is written as three commuting Pauli rotations (Z_h X_c, Z_h, X_c; global phase aside), so that
STEINER6's mover places, routes and orders everything with its own machinery."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import math


def _set(label, n_new, changes):
    """label on n qubits -> label on n_new qubits with {logical: char} changes (char j from the right is logical j)."""
    lst = list(reversed(label)) + ["I"] * (n_new - len(label))
    for q, ch in changes.items():
        lst[q] = ch
    return "".join(reversed(lst))


def transform(triples, n, hubs=1, k=2, min_deg=4):
    """(new triples, new n, description). hubs: how many logical qubits (by their number of diagonal two-qubit terms)
    get holders; k: holders per hub; a hub's partners are split in k + 1 groups (group 0 stays on the hub)."""
    diag = [set(lb) <= {"I", "Z"} for lb, _, _ in triples]
    sup = [[j for j, ch in enumerate(reversed(lb)) if ch != "I"] for lb, _, _ in triples]
    deg = [0] * n
    nbr = [dict() for _ in range(n)]
    for i, s in enumerate(sup):
        if diag[i] and len(s) == 2:
            a, b = s
            deg[a] += 1
            deg[b] += 1
            nbr[a][b] = nbr[a].get(b, 0) + 1
            nbr[b][a] = nbr[b].get(a, 0) + 1
    chosen = [h for h in sorted(range(n), key=lambda q: (-deg[q], q)) if deg[h] >= min_deg][:hubs]
    if not chosen:
        return triples, n, "no hub"
    holder = {}  # (hub, group) -> logical index of the holder
    n_new = n
    group = {}  # (hub, partner) -> group
    for h in chosen:
        for g in range(1, k + 1):
            holder[(h, g)] = n_new
            n_new += 1
        parts = sorted(nbr[h], key=lambda p: (-deg[p], p))
        groups = [[] for _ in range(k + 1)]
        for p in parts:  # the group with most edges to p among its members; then the smallest; then the lowest
            g = max(range(k + 1), key=lambda g: (sum(nbr[p].get(x, 0) for x in groups[g]), -len(groups[g]), -g))
            groups[g].append(p)
            group[(h, p)] = g
    # segments: per hub, runs of its terms that are diagonal
    out = []
    open_seg = {h: False for h in chosen}
    hub_terms = {h: [i for i, s in enumerate(sup) if h in s] for h in chosen}
    last_in_seg = {}
    for h in chosen:
        idx = hub_terms[h]
        for a, i in enumerate(idx):
            nxt = idx[a + 1] if a + 1 < len(idx) else None
            if diag[i] and (nxt is None or not diag[nxt]):
                last_in_seg[(h, i)] = True

    def copy_terms(h):
        res = []
        for g in range(1, k + 1):
            c = holder[(h, g)]
            # CNOT h -> c = e^{i pi/4} exp(-i pi/4 Z_h) exp(-i pi/4 X_c) exp(+i pi/4 Z_h X_c); a triple (P, c, t) is
            # exp(-i t c P)
            res.append((_set("I" * n, n_new, {h: "Z", c: "X"}), -math.pi / 4, 1.0))
            res.append((_set("I" * n, n_new, {h: "Z"}), math.pi / 4, 1.0))
            res.append((_set("I" * n, n_new, {c: "X"}), math.pi / 4, 1.0))
        return res

    for i, (lb, cf, t) in enumerate(triples):
        changes = {}
        for h in chosen:
            if h in sup[i] and diag[i]:
                if not open_seg[h]:
                    out.extend(copy_terms(h))
                    open_seg[h] = True
                others = [q for q in sup[i] if q != h]
                if len(others) == 1 and (h, others[0]) in group and group[(h, others[0])] > 0:
                    changes[h] = "I"
                    changes[holder[(h, group[(h, others[0])])]] = "Z"
        out.append((_set(lb, n_new, changes), cf, t))
        for h in chosen:
            if open_seg[h] and last_in_seg.get((h, i)):
                out.extend(copy_terms(h))
                open_seg[h] = False
    desc = ", ".join(f"hub L{h} (deg {deg[h]}) -> {k} holders" for h in chosen)
    return out, n_new, desc
