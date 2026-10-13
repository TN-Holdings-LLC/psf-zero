"""Prototype (c48 development): time-aware reordering of STEINER6's output.

STEINER6's mover emits one block per Pauli term (a Steiner-tree gadget, exactly exp(-i t c P) on its terminals and the
identity on its Steiner qubits) or per SWAP. The emission order fixes the as-soon-as-possible schedule of the circuit,
and so its duration. Blocks that commute as unitaries may be emitted in any order without changing the circuit's
action or its gates; this module records the blocks and re-emits them by list scheduling on the device's qubits.

A SWAP that the mover counted as merged (it joins the last two-qubit gates on its pair, which Qiskit fuses into one
two-qubit unitary) stays fused to that block, so that the fusion survives the reordering."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).


# --------------------------------------------------------------------------------------------- block recording


def install(M):
    """Wrap the mover's emission so that every mover records its blocks as [kind, op indices, data]: kind 'g' (a
    gadget; data: its Pauli on physical qubits {qubit: 'X'|'Y'|'Z'}), 's' (a SWAP; data: the pair) or 'f' (a gadget
    and the merged SWAP after it; data: the pair)."""
    if getattr(M, "_c48_installed", False):
        return
    orig_run, orig_swap = M._StMover.run_term, M._StMover.swap

    def _rec(self):
        d = self.__dict__
        if "blocks" not in d:
            d["blocks"], d["_lastb"] = [], {}
        return d["blocks"], d["_lastb"]

    def run_term(self, j):
        blocks, lastb = _rec(self)
        a = len(self.ops)
        lb = self.triples[j][0]
        pauli = {self.pi[q]: ch for q, ch in enumerate(reversed(lb)) if ch != "I"}
        orig_run(self, j)
        idx = list(range(a, len(self.ops)))
        blocks.append(["g", idx, pauli])
        for k in idx:
            for x in self.ops[k][1]:
                lastb[x] = len(blocks) - 1

    def swap(self, p, q):
        blocks, lastb = _rec(self)
        merged = self.mergeable(p, q)
        a = len(self.ops)
        orig_swap(self, p, q)
        idx = list(range(a, len(self.ops)))
        k = lastb.get(p)
        if merged and k is not None and lastb.get(q) == k and blocks[k][0] == "g":
            touched = {x for i in blocks[k][1] for x in self.ops[i][1]}
            if touched <= {p, q}:
                blocks[k] = ["f", blocks[k][1] + idx, (p, q)]
                return
        blocks.append(["s", idx, (p, q)])
        lastb[p] = lastb[q] = len(blocks) - 1

    M._StMover.run_term, M._StMover.swap = run_term, swap
    M._c48_installed = True


# --------------------------------------------------------------------------------------------- commutation


def _commute(a, b):
    ka, kb = a[0], b[0]
    if ka == "f" or kb == "f":  # conservative: a fused block commutes only with what is disjoint from it (checked
        return False            # by the caller, which only asks about blocks that share a qubit)
    if ka == "g" and kb == "g":
        pa, pb = a[2], b[2]
        anti = 0
        for q, ch in pa.items():
            o = pb.get(q)
            if o is not None and o != ch:
                anti ^= 1
        return anti == 0
    if ka == "s" and kb == "s":
        return set(a[2]) == set(b[2])
    g, s = (a, b) if ka == "g" else (b, a)
    p, q = s[2]
    return g[2].get(p, "I") == g[2].get(q, "I")


# --------------------------------------------------------------------------------------------- scheduling

DUR = {"cx": 1.0, "swap": 3.0, "rz": 0.0}
DUR_1Q = 0.3


def _dur(name):
    return DUR.get(name, DUR_1Q)


def asap(ops):
    """Makespan of an op list in the order-based ASAP model, in CX units."""
    clock = {}
    for name, q, _ in ops:
        s = max(clock.get(x, 0.0) for x in q)
        for x in q:
            clock[x] = s + _dur(name)
    return max(clock.values()) if clock else 0.0


def reorder(ops, blocks, rule="start"):
    """The ops re-emitted block by block by list scheduling, or None if the blocks do not cover the ops. Rules
    (each picks one ready block at a time; a block is ready when every earlier block it does not commute with and
    shares a qubit with has been emitted):
    'start': the earliest start; ties by the larger bottom level (longest chain of dependent work after it).
    'cp':    the largest bottom level among the blocks that can start no later than the earliest start + slack 1."""
    B = len(blocks)
    seen = sorted(i for b in blocks for i in b[1])
    if seen != list(range(len(ops))):
        return None
    touched = [tuple(sorted({x for i in b[1] for x in ops[i][1]})) for b in blocks]
    dur = [sum(_dur(ops[i][0]) for i in b[1]) for b in blocks]
    first = [min(b[1]) for b in blocks]
    order0 = sorted(range(B), key=lambda i: first[i])  # the mover's order
    on = {}
    for i in order0:
        for x in touched[i]:
            on.setdefault(x, []).append(i)
    preds = [set() for _ in range(B)]
    for x, lst in on.items():
        for k, i in enumerate(lst):
            for h in range(k - 1, -1, -1):
                j = lst[h]
                if j not in preds[i] and not _commute(blocks[j], blocks[i]):
                    preds[i].add(j)
    succ = [[] for _ in range(B)]
    indeg = [0] * B
    for i in range(B):
        for j in preds[i]:
            succ[j].append(i)
        indeg[i] = len(preds[i])
    blevel = [0.0] * B
    for i in reversed(order0):  # predecessors come earlier in the mover's order
        blevel[i] = dur[i] + max((blevel[k] for k in succ[i]), default=0.0)
    clock = {}
    ready = {i for i in range(B) if indeg[i] == 0}
    out = []
    while ready:
        starts = {i: max(clock.get(x, 0.0) for x in touched[i]) for i in ready}
        if rule == "start":
            i = min(ready, key=lambda k: (starts[k], -blevel[k], first[k]))
        else:
            s0 = min(starts.values())
            i = min((k for k in ready if starts[k] <= s0 + 1.0), key=lambda k: (-blevel[k], starts[k], first[k]))
        ready.discard(i)
        for k in sorted(blocks[i][1]):
            name, q, prm = ops[k]
            s = max(clock.get(x, 0.0) for x in q)
            for x in q:
                clock[x] = s + _dur(name)
            out.append(ops[k])
        for k in succ[i]:
            indeg[k] -= 1
            if indeg[k] == 0:
                ready.add(k)
    if len(out) != len(ops):
        raise RuntimeError("reorder lost operations")
    return out
