FINAL_STATS = {"push_calls": 0, "push_applied": 0, "relabel_calls": 0, "relabel_applied": 0}  # items 65-66

_PAULI = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Y": np.array([[0, -1j], [1j, 0]], dtype=complex), "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
_CONJ_CACHE = {}


def _nlf(error):
    """-ln(1 - error) for an error in [0, 1); None for an error >= 1."""
    import math
    e = 0.0 if error is None or error != error else float(error)
    return None if e >= 1.0 else -math.log1p(-max(e, 0.0))


def _op_cost(target, name, qargs, max_error):
    """Items 65-66: -ln(1 - error) of `name` on `qargs` in `target`; None if the device does not offer it there or
    reports it failed (error >= max_error)."""
    if name not in target.operation_names:
        return None
    entries = target[name]
    if qargs in entries:
        props = entries[qargs]
    elif None in entries:  # a global instruction (control flow): no error reported
        return 0.0
    else:
        return None
    e = props.error if props is not None and props.error is not None else 0.0
    if e >= max_error:
        return None
    return _nlf(e)


def _conjugate_2q(op, pa, pb):
    """Item 66: G (Pa on the first qubit, Pb on the second) G^dagger = phase * (Pa' on the first, Pb' on the second),
    as (Pa', Pb', phase); None if `op` is not a matrix gate or the result is not a Pauli. Qiskit's order: the first
    qubit is the least significant."""
    try:
        key = (op.name, tuple(float(p) for p in op.params), pa, pb)
    except (TypeError, ValueError):
        return None
    if key in _CONJ_CACHE:
        return _CONJ_CACHE[key]
    out = None
    try:
        g = np.asarray(op.to_matrix(), dtype=complex)
    except Exception:  # noqa: BLE001 - no matrix: the Paulis are not carried through
        g = None
    if g is not None and g.shape == (4, 4):
        m = g @ np.kron(_PAULI[pb], _PAULI[pa]) @ g.conj().T
        for la in _PAULI:
            for lb in _PAULI:
                p = np.kron(_PAULI[lb], _PAULI[la])
                c = np.trace(p.conj().T @ m) / 4
                if abs(abs(c) - 1.0) < 1e-9 and np.allclose(m, c * p, atol=1e-9):
                    out = (la, lb, c)
                    break
            if out is not None:
                break
    _CONJ_CACHE[key] = out
    return out


class _PushPaulis(TransformationPass):
    """Item 66: carries the Pauli part of single-qubit gate runs forward through two-qubit Clifford gates (a CZ maps
    X on one qubit to X on it and Z on the other; Z costs nothing, it joins an rz) until a run absorbs it at no extra
    cost. Exact, global phase included. A run is re-synthesised (rz, sx and x, as Qiskit's Optimize1qGatesDecomposition
    writes them) only when it takes in or gives out a Pauli; the result is kept only if its single-qubit gates' summed
    -ln(1 - error) on the device is lower than the input's, and two-qubit gates and measurements are untouched."""

    def __init__(self, target, max_error):
        super().__init__()
        self.target = target
        self.max_error = max_error

    def _cost_1q(self, dag):
        total = 0.0
        for node in dag.op_nodes():
            if len(node.qargs) != 1 or node.op.name in ("measure", "reset", "delay", "barrier"):
                continue
            q = dag.find_bit(node.qargs[0]).index
            c = _op_cost(self.target, node.op.name, (q,), self.max_error)
            if c is None:
                return None
            total += c
        return total

    def run(self, dag):
        from qiskit.synthesis import OneQubitEulerDecomposer
        FINAL_STATS["push_calls"] += 1
        names = set(self.target.operation_names)
        if {"rz", "sx", "x"} <= names:
            dec = OneQubitEulerDecomposer("ZSXX")
        elif {"rz", "sx"} <= names:
            dec = OneQubitEulerDecomposer("ZSX")
        else:
            return dag
        new = dag.copy_empty_like()
        factor = 1.0 + 0.0j  # the global phase gained, as a factor, applied at the end
        run = {}      # qubit -> (matrix of the pending run, list of its nodes)
        pending = {}  # qubit -> Pauli label carried forward
        changed = False

        has_x = "x" in names

        def cost(v):
            """Non-rz gates the decomposer writes for v, from its middle Euler angle (atol as the decomposer's)."""
            import math
            theta = 2.0 * math.atan2(abs(v[1, 0]), abs(v[0, 0]))
            if abs(theta) < 1e-12:
                return 0
            if abs(theta - math.pi / 2) < 1e-12:
                return 1
            if abs(theta - math.pi) < 1e-12:
                return 1 if has_x else 2
            return 2

        def emit_matrix(q, m):
            nonlocal factor
            c = dec(m)
            factor *= np.exp(1j * c.global_phase)
            for ins in c.data:
                new.apply_operation_back(ins.operation, (q,), ())

        def close(q):
            """Finish q's run, taking in its pending Pauli and possibly giving one out."""
            nonlocal changed, factor
            m, nodes = run.pop(q, (None, []))
            p = pending.pop(q, "I")
            if m is None:
                if p != "I":
                    pending[q] = p
                return
            full = m @ _PAULI[p]
            best = None
            for out in ("I", "X", "Y", "Z"):  # I first: on a tie the run absorbs the Pauli
                v = _PAULI[out] @ full
                k = cost(v)
                if best is None or k < best[0]:
                    best = (k, out, v)
            if p == "I" and best[1] == "I":  # nothing taken in or given out: the run is kept as written
                for node in nodes:
                    new.apply_operation_back(node.op, node.qargs, node.cargs)
                return
            changed = True
            emit_matrix(q, best[2])
            if best[1] != "I":
                pending[q] = best[1]

        def flush(q):
            """Write q's run and pending Pauli out before an operation they cannot pass."""
            close(q)
            p = pending.pop(q, "I")
            if p != "I":
                emit_matrix(q, _PAULI[p])

        for node in dag.topological_op_nodes():
            qs = node.qargs
            op = node.op
            is_1q_gate = (len(qs) == 1 and not node.cargs and op.name not in ("measure", "reset", "delay", "barrier")
                          and getattr(op, "_directive", False) is False)
            if is_1q_gate:
                try:
                    m = np.asarray(op.to_matrix(), dtype=complex)
                except Exception:  # noqa: BLE001 - no matrix
                    m = None
                if m is not None and m.shape == (2, 2):
                    prev, nodes = run.get(qs[0], (np.eye(2, dtype=complex), []))
                    run[qs[0]] = (m @ prev, nodes + [node])
                    continue
            if len(qs) == 2 and not node.cargs:
                for q in qs:
                    close(q)
                pa, pb = pending.get(qs[0], "I"), pending.get(qs[1], "I")
                if pa == "I" and pb == "I":
                    new.apply_operation_back(op, qs, node.cargs)
                    continue
                conj = _conjugate_2q(op, pa, pb)
                if conj is not None:
                    pending.pop(qs[0], None)
                    pending.pop(qs[1], None)
                    new.apply_operation_back(op, qs, node.cargs)
                    if conj[0] != "I":
                        pending[qs[0]] = conj[0]
                    if conj[1] != "I":
                        pending[qs[1]] = conj[1]
                    factor *= conj[2]
                    continue
                for q in qs:
                    flush(q)
                new.apply_operation_back(op, qs, node.cargs)
                continue
            for q in qs:
                flush(q)
            new.apply_operation_back(op, qs, node.cargs)
        for q in list(run) + list(pending):
            flush(q)
        if not changed:
            return dag
        import cmath
        new.global_phase = dag.global_phase + cmath.phase(factor)
        before, after = self._cost_1q(dag), self._cost_1q(new)
        if before is None or after is None or after >= before - 1e-12:
            return dag
        FINAL_STATS["push_applied"] += 1
        return new


class _FinalRelabel(AnalysisPass):
    """Item 65: a last relabelling of the physical qubits, scored on every operation's -ln(1 - error) on the device,
    readout and single-qubit gates included. Qiskit's exact VF2PostLayout (item 33) gives up when two or more qubits
    carry no two-qubit gate, as in a circuit of single-qubit gates and measurements; here the qubits with two-qubit
    gates are matched by VF2PostLayout without those qubits, and those qubits are then assigned to the remaining
    physical qubits at the least cost (an assignment problem, solved exactly). Gates are not changed. The relabelling
    is kept only if it lowers the summed cost and uses no failed element (error >= max_error)."""

    def __init__(self, target, max_error, call_limit, max_trials):
        super().__init__()
        self.target = target
        self.max_error = max_error
        self.call_limit = call_limit
        self.max_trials = max_trials

    def _add(self, op, qa, one, two):
        """Counts `op` on physical qubits `qa`; the gates in a control-flow block count on the outer qubits."""
        from qiskit.circuit import ControlFlowOp
        if op.name in ("barrier", "delay"):
            return True
        if isinstance(op, ControlFlowOp):
            for block in op.blocks:
                for ins in block.data:
                    inner = tuple(qa[block.find_bit(q).index] for q in ins.qubits)
                    if not self._add(ins.operation, inner, one, two):
                        return False
            return True
        if len(qa) == 1:
            d = one.setdefault(qa[0], {})
        elif len(qa) == 2:
            d = two.setdefault(qa, {})
        else:
            return len(qa) == 0
        d[op.name] = d.get(op.name, 0) + 1
        return True

    def _usage(self, dag):
        one, two = {}, {}
        for node in dag.op_nodes():
            if not self._add(node.op, tuple(dag.find_bit(q).index for q in node.qargs), one, two):
                return None
        return one, two

    def _score(self, one, two, mapping):
        total = 0.0
        for q, ops in one.items():
            for name, n in ops.items():
                c = _op_cost(self.target, name, (mapping[q],), self.max_error)
                if c is None:
                    return None
                total += n * c
        for (a, b), ops in two.items():
            for name, n in ops.items():
                c = _op_cost(self.target, name, (mapping[a], mapping[b]), self.max_error)
                if c is None:
                    return None
                if name in _SYMMETRIC_2Q:  # item 41: a symmetric gate must not sit on a coupler failed either way
                    try:
                        rev = self.target[name][(mapping[b], mapping[a])]
                    except KeyError:
                        rev = None
                    if rev is not None and rev.error is not None and rev.error >= self.max_error:
                        return None
                total += n * c
        return total

    def run(self, dag):
        from qiskit.transpiler import Layout
        from scipy.optimize import linear_sum_assignment
        FINAL_STATS["relabel_calls"] += 1
        self.property_set["psf_final_relabel"] = False
        nphys = dag.num_qubits()
        if self.target.num_qubits is None or nphys != self.target.num_qubits:
            return
        use = self._usage(dag)
        if use is None:
            return
        one, two = use
        coupled = sorted({q for pair in two for q in pair})
        loose = sorted(q for q in one if q not in set(coupled))
        ident = list(range(nphys))
        base = self._score(one, two, ident)
        failed_q = _failed_elements(self.target, self.max_error)[1]
        # 1. the coupled qubits: item 33's match, made without the loose qubits when there are two or more of them
        cmap = {q: q for q in coupled}
        if coupled and len(loose) >= 2:
            sub = dag.copy_empty_like()
            drop = {dag.qubits[q] for q in loose}
            for node in dag.topological_op_nodes():
                if not drop.intersection(node.qargs):
                    sub.apply_operation_back(node.op, node.qargs, node.cargs)
            vf2 = VF2PostLayout(target=self.target, seed=-1, call_limit=self.call_limit, max_trials=self.max_trials,
                                strict_direction=True)
            vf2.run(sub)
            if vf2.property_set["VF2PostLayout_stop_reason"] == VF2PostLayoutStopReason.SOLUTION_FOUND:
                post = vf2.property_set["post_layout"]
                cmap = {q: post[dag.qubits[q]] for q in coupled}
        # 2. the loose qubits: the cheapest physical qubits left, by an exact assignment
        lmap = {}
        if loose:
            taken = set(cmap.values())
            free = [p for p in range(nphys) if p not in taken and p not in failed_q]
            if len(free) < len(loose):
                return
            big = 1e18
            cost = np.full((len(loose), len(free)), big)
            for i, q in enumerate(loose):
                for j, p in enumerate(free):
                    s = 0.0
                    for name, n in one[q].items():
                        c = _op_cost(self.target, name, (p,), self.max_error)
                        if c is None:
                            s = big
                            break
                        s += n * c
                    cost[i, j] = s
            rows, cols = linear_sum_assignment(cost)
            if any(cost[r, c] >= big for r, c in zip(rows, cols)):
                return
            lmap = {loose[r]: free[c] for r, c in zip(rows, cols)}
        mapping = {}
        mapping.update(cmap)
        mapping.update(lmap)
        rest = [p for p in range(nphys) if p not in set(mapping.values())]
        for q in range(nphys):
            if q not in mapping:
                mapping[q] = rest.pop(0)
        if mapping == {q: q for q in range(nphys)}:
            return
        if any(mapping[q] in failed_q for q in set(one) | set(coupled)):
            return
        new = self._score(one, two, mapping)
        if new is None or (base is not None and new >= base - 1e-12):
            return
        self.property_set["post_layout"] = Layout({dag.qubits[q]: p for q, p in mapping.items()})
        self.property_set["psf_final_relabel"] = True
        FINAL_STATS["relabel_applied"] += 1


def _relabel_found(property_set):
    return bool(property_set["psf_final_relabel"])


TRIES_STATS = {"calls": 0, "tries": 0, "failed_discarded": 0, "errors": 0, "replaced": 0, "skipped_large": 0}  # item 67
ROUTE_TRIES_MAX_2Q = 5000  # item 67: above this many two-qubit gates in the first output, one compile (ESP ~ 0 there)


def _esp_cost(circ, target):
    """Item 67: the summed -ln(1 - error) of every operation of `circ` on the device (-ln ESP); inf if one is not
    offered there or has error >= 1. Control flow counts 0, as the device reports no error for it."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    total = 0.0
    for ins in circ.data:
        name = ins.operation.name
        if name in ("barrier", "delay"):
            continue
        c = _op_cost(target, name, tuple(pos[q] for q in ins.qubits), 2.0)
        if c is None:
            return float("inf")
        total += c
    return total


def _route_tries(args, out, target, full_map, edges, qubits, max_error, tries):
    """Item 67: compiles again with seeds seed, seed + 1, ..., seed + tries - 1, each on the map without the failed
    elements and on the full map, and returns the output of the lowest -ln ESP among those that use no failed
    element; `out` (the first seed on the map without the failed elements, as item 61 compiles it) is one of them.
    `out` is returned as it is when it has more than ROUTE_TRIES_MAX_2Q two-qubit gates: the time would be spent on
    circuits whose ESP is near 0 on today's devices (the largest of BP-FINAL's runnable tests had 2362)."""
    TRIES_STATS["calls"] += 1
    if _count_2q(out) > ROUTE_TRIES_MAX_2Q:
        TRIES_STATS["skipped_large"] += 1
        return out
    best, best_cost = out, (float("inf") if _uses_failed(out, edges, qubits) else _esp_cost(out, target))
    pruned = prune_coupling_map(full_map, target, max_error) if (edges or qubits) else None
    maps = [full_map] if pruned is None else [pruned, full_map]
    seed = args.get("seed_transpiler")
    for i in range(tries):
        for k, cm in enumerate(maps):
            if i == 0 and k == 0:
                continue  # `out`
            TRIES_STATS["tries"] += 1
            try:
                o = compile_for_hardware(**dict(args, coupling_map=cm, seed_transpiler=None if seed is None
                                                else seed + i))
            except QiskitError:
                TRIES_STATS["errors"] += 1
                continue
            if _uses_failed(o, edges, qubits):
                TRIES_STATS["failed_discarded"] += 1
                continue
            c = _esp_cost(o, target)
            if c < best_cost - 1e-12:
                best, best_cost = o, c
    if best is not out:
        TRIES_STATS["replaced"] += 1
    return best


