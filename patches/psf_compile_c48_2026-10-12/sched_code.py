
# ------------------------------------------------------------------ item 83: time-aware order of STEINER6's blocks

ST_SCHEDULE = True  # item 83: re-emit STEINER6's blocks by list scheduling when that shortens the schedule
_ST_DUR = {"cx": 1.0, "swap": 3.0, "rz": 0.0}  # item 83: durations in CX units for the ordering (rz is virtual)
_ST_DUR_1Q = 0.3
_ST_REORDER_PAIRS = 5_000_000  # item 83: commutation checks allowed per circuit; beyond it the mover's order is kept


def _st_record_term(mv, first_op, pauli):
    """Item 83: record the gadget emitted at mv.ops[first_op:] as a block; pauli: its term on physical qubits."""
    idx = list(range(first_op, len(mv.ops)))
    mv.st_blocks.append(["g", idx, pauli])
    k = len(mv.st_blocks) - 1
    for i in idx:
        for x in mv.ops[i][1]:
            mv._lastb[x] = k


def _st_record_swap(mv, first_op, p, q):
    """Item 83: record the SWAP emitted at mv.ops[first_op:]. When the last block on both of its qubits is one gadget
    on exactly that pair, the SWAP is fused to it (Qiskit merges the two into one two-qubit unitary; the fusion keeps
    them together)."""
    idx = list(range(first_op, len(mv.ops)))
    k = mv._lastb.get(p)
    if k is not None and mv._lastb.get(q) == k and mv.st_blocks[k][0] == "g":
        touched = {x for i in mv.st_blocks[k][1] for x in mv.ops[i][1]}
        if touched <= {p, q}:
            mv.st_blocks[k] = ["f", mv.st_blocks[k][1] + idx, (p, q)]
            return
    mv.st_blocks.append(["s", idx, (p, q)])
    mv._lastb[p] = mv._lastb[q] = len(mv.st_blocks) - 1


def _st_blocks_commute(a, b):
    """Item 83: True if blocks a and b (sharing a qubit) commute as unitaries. A gadget is exp(-i t c P) on its
    terminals and the identity on its Steiner qubits; a fused block is taken to commute with nothing it touches."""
    ka, kb = a[0], b[0]
    if ka == "f" or kb == "f":
        return False
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


def _st_op_dur(name):
    return _ST_DUR.get(name, _ST_DUR_1Q)


def _st_asap(ops):
    """Item 83: the makespan of `ops` (in order, each as soon as its qubits are free) in CX units."""
    clock = {}
    for name, q, _ in ops:
        s = max(clock.get(x, 0.0) for x in q)
        for x in q:
            clock[x] = s + _st_op_dur(name)
    return max(clock.values()) if clock else 0.0


def _st_reorder(ops, blocks):
    """Item 83: `ops` re-emitted block by block by list scheduling, or None when the blocks do not cover the ops
    exactly once. A block is ready once every earlier block (in the mover's order) that shares a qubit with it and does
    not commute with it has been emitted; of the ready blocks, the one that can start earliest goes next (ties: the
    longer chain of dependent work after it, then the mover's order). The blocks and their gates are unchanged, and
    every pair that does not commute keeps its order, so the circuit's action is unchanged."""
    if not blocks:
        return None
    B = len(blocks)
    if sorted(i for b in blocks for i in b[1]) != list(range(len(ops))):
        return None
    touched = [tuple(sorted({x for i in b[1] for x in ops[i][1]})) for b in blocks]
    dur = [sum(_st_op_dur(ops[i][0]) for i in b[1]) for b in blocks]
    first = [min(b[1]) for b in blocks]
    order0 = sorted(range(B), key=lambda i: first[i])
    on = {}
    for i in order0:
        for x in touched[i]:
            on.setdefault(x, []).append(i)
    if sum(len(lst) * (len(lst) - 1) // 2 for lst in on.values()) > _ST_REORDER_PAIRS:
        return None
    preds = [set() for _ in range(B)]
    for lst in on.values():
        for k, i in enumerate(lst):
            for h in range(k - 1, -1, -1):
                j = lst[h]
                if j not in preds[i] and not _st_blocks_commute(blocks[j], blocks[i]):
                    preds[i].add(j)
    succ = [[] for _ in range(B)]
    indeg = [len(p) for p in preds]
    for i in range(B):
        for j in preds[i]:
            succ[j].append(i)
    blevel = [0.0] * B
    for i in reversed(order0):
        blevel[i] = dur[i] + max((blevel[k] for k in succ[i]), default=0.0)
    clock = {}
    ready = {i for i in range(B) if indeg[i] == 0}
    out = []
    while ready:
        i = min(ready, key=lambda k: (max(clock.get(x, 0.0) for x in touched[k]), -blevel[k], first[k]))
        ready.discard(i)
        for k in sorted(blocks[i][1]):
            name, q, _ = ops[k]
            s = max(clock.get(x, 0.0) for x in q)
            for x in q:
                clock[x] = s + _st_op_dur(name)
            out.append(ops[k])
        for k in succ[i]:
            indeg[k] -= 1
            if indeg[k] == 0:
                ready.add(k)
    if len(out) != len(ops):  # a cycle cannot arise (edges go forwards in the mover's order); kept as a guard
        return None
    return out

