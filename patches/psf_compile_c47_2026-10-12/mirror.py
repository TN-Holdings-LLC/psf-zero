class _StMirror:
    """Item 82: repeated layers of commuting terms (QAOA's cost layers, an Ising or ZZ Hamiltonian's Trotter steps):
    the term sequence is C_1 L_1 C_2 L_2 ... where every C_k is the same set of pairwise commuting terms on two or more
    qubits (angles may differ) and every L_k holds single-qubit terms. C_1 is scheduled once by STEINER6's mover; C_2
    replays C_1's events (terms and SWAPs) in reverse order, C_3 forwards again, and so on, so every layer costs what
    the first does and every second layer ends where the first began. L_k's terms run where their qubits are. A fixed
    construction, not a search."""

    def __init__(self, n, triples, adj, dist, par, pi0, err, blocks):
        self.n, self.adj, self.dist, self.par, self.err = n, adj, dist, par, err
        self.triples = triples
        self.blocks = blocks  # [("C", [term indices]) | ("L", [term indices])]
        self.pi = list(pi0)
        self.ops, self.gphase = [], 0.0
        self.stats = dict(swaps=0, merged_swaps=0, bridged_terms=0, bridge_excess=0, terms=0)

    def _term(self, j):
        lb, c, t = self.triples[j]
        sup = [k for k, ch in enumerate(reversed(lb)) if ch != "I"]
        phys = [self.pi[k] for k in sup]
        ex = 0
        if len(phys) > 1:
            r, p = _st_steiner_tree(phys, self.dist, self.par)
            ex = _st_tree_cost(r, p, set(phys)) - 2 * (len(phys) - 1)
        dist, par = (self.dist, self.par) if (self.err is None or ex == 0) else (self.err[0], self.err[1])
        for name, q, prm in _st_gadget_ops(lb, c, t, self.pi, dist, par):
            if name == "gphase":
                self.gphase += prm[0]
            else:
                self.ops.append((name, q, prm))
        self.stats["terms"] += 1
        if ex > 0:
            self.stats["bridged_terms"] += 1

    def _swap(self, p, q):
        self.ops.append(("swap", (p, q), ()))
        a = self.pi.index(p) if p in self.pi else None
        b = self.pi.index(q) if q in self.pi else None
        if a is not None:
            self.pi[a] = q
        if b is not None:
            self.pi[b] = p
        self.stats["swaps"] += 1

    def build(self, first, c1):
        """first: STEINER6's mover run on C_1's terms alone (c1: their indices here) from the same initial placement;
        its emission gives the events (a term per rz, SWAPs) that every layer replays."""
        events = []
        term_iter = iter(first.order)
        for name, q, prm in first.ops:
            if name == "swap":
                events.append(("s", q[0], q[1]))
            elif name == "rz":
                events.append(("t", self.triples[c1[next(term_iter)]][0]))
        c_index = 0
        for kind, idx in self.blocks:
            if kind == "L":
                for j in idx:
                    self._term(j)
                continue
            by_label = {self.triples[j][0]: j for j in idx}
            for ev in (events if c_index % 2 == 0 else events[::-1]):
                if ev[0] == "s":
                    self._swap(ev[1], ev[2])
                else:
                    self._term(by_label[ev[1]])
            c_index += 1
        return self


def _st_mirror_blocks(triples, limit=4000):
    """Item 82: [("L", idx) | ("C", idx)] when the terms are p >= 2 layers of the same commuting set: the k-th
    occurrence of each multi-qubit label goes to C_k (every multi-qubit label occurs p times; C_1's terms commute),
    and each single-qubit term to the gap right after the last C layer it must follow (that of the latest earlier
    term it anticommutes with; before C_1 if none); the order of the blocks keeps every pair of anticommuting terms in
    its original order, so the reordering is exact. None otherwise."""
    T = len(triples)
    if T > limit:
        return None
    multi = [sum(ch != "I" for ch in lb) >= 2 for lb, _, _ in triples]
    seen, layer = {}, [None] * T
    for j, (lb, _, _) in enumerate(triples):
        if multi[j]:
            layer[j] = seen.get(lb, 0)
            seen[lb] = layer[j] + 1
    if not seen:
        return None
    p = max(seen.values())
    if p < 2 or any(c != p for c in seen.values()):
        return None
    masks = [_st_term_masks(lb) for lb, _, _ in triples]
    slot = [None] * T
    for j in range(T):
        if not multi[j]:
            before = [layer[i] for i in range(j) if multi[i] and not _st_commute(masks[i], masks[j])]
            slot[j] = max(before) if before else -1
    blocks = []
    lead = [j for j in range(T) if not multi[j] and slot[j] == -1]
    if lead:
        blocks.append(("L", lead))
    for k in range(p):
        blocks.append(("C", [j for j in range(T) if multi[j] and layer[j] == k]))
        ls = [j for j in range(T) if not multi[j] and slot[j] == k]
        if ls:
            blocks.append(("L", ls))
    c1 = blocks[1][1] if blocks[0][0] == "L" else blocks[0][1]
    for a in range(len(c1)):
        for b in range(a + 1, len(c1)):
            if not _st_commute(masks[c1[a]], masks[c1[b]]):
                return None
    pos = {}
    for kind, idx in blocks:
        for j in idx:
            pos[j] = len(pos)
    for i in range(T):  # anticommuting pairs keep their order
        for j in range(i + 1, T):
            if pos[j] < pos[i] and not _st_commute(masks[i], masks[j]):
                return None
    return blocks
