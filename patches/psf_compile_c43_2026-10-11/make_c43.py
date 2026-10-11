"""Builds prototype 2026-10-11.c43 from prototype 2026-10-11.c42 by exact substitutions:
  - item 74, revised: only the near-failed elements the first output used are avoided in the second compile;
  - item 75: a deterministic exact re-placement in place of Qiskit's VF2PostLayout, whose result varies from call to
    call with a fixed seed (C42-VAL, Addendum 450);
  - item 76: item 73 also reads circuits whose Clifford gates left at the end are a qubit permutation (with
    single-qubit Cliffords), such as QFT with its final swaps; the permutation goes into the output's layout.
    python make_c43.py <c42 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

DET = '''

DET_STATS = {"calls": 0, "applied": 0, "trials": 0, "skipped": 0}  # item 75


class _DeterministicPostLayout(AnalysisPass):
    """Item 75: the exact re-placement of a routed circuit by the device's errors, in a fixed order: subgraph matchings
    of its two-qubit interaction graph into the device's coupling graph from rustworkx's VF2 with `id_order=True`, each
    scored by the summed -ln(1 - error) of every operation as placed; qubits with no two-qubit gate then go, one by one
    in index order, to the free physical qubit where their own operations cost least. The lowest score is applied if it
    is below the current placement's (the first found on a tie). It sets `post_layout` and
    `VF2PostLayout_stop_reason` as Qiskit's VF2PostLayout does, so the passes after it are unchanged. Qiskit's pass
    returns different layouts from call to call with a fixed seed (Addendum 450); this one does not."""

    def __init__(self, target, call_limit=300_000, max_trials=2_500, strict_direction=True):
        super().__init__()
        self.target = target
        self.call_limit = call_limit
        self.max_trials = max_trials
        self.strict = strict_direction

    def run(self, dag):
        import rustworkx as rx
        from qiskit.transpiler import Layout
        DET_STATS["calls"] += 1
        self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.NO_SOLUTION_FOUND
        t = self.target
        N = t.num_qubits
        if len(dag.qubits) != N:
            DET_STATS["skipped"] += 1
            return
        idx = {q: i for i, q in enumerate(dag.qubits)}
        one, two = {}, {}
        for node in dag.op_nodes():
            name = node.op.name
            if name in ("barrier", "delay"):
                continue
            if getattr(node.op, "blocks", None) is not None or len(node.qargs) > 2 or node.op.num_clbits and \\
                    name != "measure":
                DET_STATS["skipped"] += 1
                return
            qs = tuple(idx[q] for q in node.qargs)
            if len(qs) == 2:
                two[(name, qs)] = two.get((name, qs), 0) + 1
            elif len(qs) == 1:
                one[(name, qs[0])] = one.get((name, qs[0]), 0) + 1
        weights = {}
        for name in {n for n, _ in one} | {n for n, _ in two}:
            if name not in t.operation_names:
                DET_STATS["skipped"] += 1
                return
            table = {}
            for qargs, props in t[name].items():
                e = props.error if props is not None and props.error is not None else 0.0
                table[None if qargs is None else tuple(qargs)] = math.inf if e >= 1.0 else -math.log1p(-max(e, 0.0))
            weights[name] = table

        def w(name, qargs):
            table = weights[name]
            v = table.get(qargs)
            if v is None and not self.strict and len(qargs) == 2:
                v = table.get(qargs[::-1])
            if v is None:
                v = 0.0 if None in table else math.inf
            return v

        active = sorted({q for _, q in one} | {q for _, qs in two for q in qs})
        if not active:
            return
        cost1 = {}
        for (name, q), c in one.items():
            row = cost1.setdefault(q, [0.0] * N)
            for p in range(N):
                row[p] += c * w(name, (p,))
        zero = [0.0] * N
        conn = sorted({q for _, qs in two for q in qs})
        iso = [q for q in active if q not in set(conn)]
        pos = {q: i for i, q in enumerate(conn)}
        names2 = sorted({n for n, _ in two})

        def place_isolated(taken, total):
            m = {}
            for q in iso:
                row = cost1.get(q, zero)
                best_p, best_c = None, math.inf
                for p in range(N):
                    if p not in taken and row[p] < best_c - 1e-15:
                        best_p, best_c = p, row[p]
                if best_p is None:
                    return None, math.inf
                m[q] = best_p
                taken.add(best_p)
                total += best_c
            return m, total

        def score(mc):  # mc: logical connected qubit -> physical
            total = 0.0
            for q, p in mc.items():
                total += cost1.get(q, zero)[p]
            for (name, (a, b)), c in two.items():
                total += c * w(name, (mc[a], mc[b]))
                if total == math.inf:
                    return total
            return total

        current = sum(cost1.get(q, zero)[q] for q in active) + sum(c * w(n, qs) for (n, qs), c in two.items())
        best, best_map = current, None
        trials = 0
        if conn:
            dev_edges = None
            for name in names2:
                ok = {tuple(q) for q in t[name] if q is not None}
                if not self.strict:
                    ok |= {(b, a) for a, b in ok}
                dev_edges = ok if dev_edges is None else dev_edges & ok
            dg, cg = rx.PyDiGraph(), rx.PyDiGraph()
            dg.add_nodes_from(range(N))
            dg.add_edges_from_no_data(sorted(dev_edges))
            cg.add_nodes_from(conn)
            cedges = {(pos[a], pos[b]) for _, (a, b) in two}
            if not self.strict:
                cedges |= {(b, a) for a, b in cedges}
            cg.add_edges_from_no_data(sorted(cedges))
            for mapping in rx.vf2_mapping(dg, cg, subgraph=True, induced=False, id_order=True,
                                          call_limit=self.call_limit):
                trials += 1
                mc = {conn[cn]: dn for dn, cn in mapping.items()}
                sc = score(mc)
                if sc < best:
                    mi, total = place_isolated(set(mc.values()), sc)
                    if total < best - 1e-12:
                        best, best_map = total, {**mc, **mi}
                if trials >= self.max_trials:
                    break
        else:
            mi, total = place_isolated(set(), 0.0)
            if total < best - 1e-12:
                best, best_map = total, mi
        DET_STATS["trials"] += trials
        if best_map is None:
            self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND
            return
        used = set(best_map.values())
        free = iter([p for p in range(N) if p not in used])
        post = Layout()
        for i, q in enumerate(dag.qubits):
            post[q] = best_map[i] if i in best_map else next(free)
        self.property_set["post_layout"] = post
        self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.SOLUTION_FOUND
        DET_STATS["applied"] += 1


class _NoPostLayout(AnalysisPass):
    """Item 75: in place of Qiskit's VF2PostLayout when the call has no device Target: there are no errors to place
    by, and Qiskit's pass then chooses among equal layouts differently from call to call."""

    def run(self, dag):
        self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.NO_SOLUTION_FOUND


class _DeterministicVF2:
    """Item 75: while active, Qiskit's preset pass managers built in this block get item 75's re-placement (scored on
    `target`, the device's own Target) or, without a Target, none, in place of VF2PostLayout."""

    def __init__(self, target):
        self.target = target

    def __enter__(self):
        import qiskit.transpiler.preset_passmanagers.builtin_plugins as plugins
        import qiskit.transpiler.preset_passmanagers.common as common
        self.mods = (plugins, common)
        self.saved = [m.VF2PostLayout for m in self.mods]
        target = self.target

        def make(*args, call_limit=300_000, max_trials=2_500, strict_direction=False, **kw):
            if target is None:
                return _NoPostLayout()
            return _DeterministicPostLayout(target, call_limit or 300_000, max_trials or 2_500, strict_direction)
        for m in self.mods:
            m.VF2PostLayout = make
        return self

    def __exit__(self, *exc):
        for m, v in zip(self.mods, self.saved):
            m.VF2PostLayout = v
        return False
'''

PERM = '''    suffix = []
    table = None
    perm = list(range(n))  # item 76: perm[q] is the wire whose state logical qubit q holds at the end
    seen = set()
    for q in range(n):
        if frame[q] == [(1 << q, 0, 0), (0, 1 << q, 0)]:
            seen.add(q)
            continue
        key = []
        wire = None
        for x, z, k in frame[q]:
            both = x | z
            if not both or both & (both - 1):
                return None  # the Clifford left at the end acts on more than one qubit here
            if wire is None:
                wire = both.bit_length() - 1
            elif both != 1 << wire:
                return None
            ch = "Y" if x and z else ("X" if x else "Z")
            sg = (k - (1 if ch == "Y" else 0)) % 4
            if sg not in (0, 2):
                return None
            key.append((ch, 1 if sg == 0 else -1))
        if wire in seen:
            return None
        seen.add(wire)
        perm[q] = wire
        table = table or _st_cliff1q()
        word = table.get(tuple(key))
        if word is None:
            return None
        suffix += [(g, wire) for g in word]
    if len(seen) != n:
        return None
    STEINER_STATS["gate_level"] += 1
    if perm != list(range(n)):
        STEINER_STATS["permuted"] += 1
    return prefix, meas, terms, phase, suffix, perm'''

NEAR_NEW = '''def _near_alternative(out, call_args, target, near, max_error):
    """Item 74 (revised in prototype c43): `out`, or the input compiled with the near-failed elements `out` used
    avoided as well, when that output is better on the device (lower summed -ln(1 - error), item 69's measure). The
    second compile sees a copy of the Target in which those elements, and only those, are marked failed; the choice is
    made on the real Target. Two fixed constructions compared by the device's own error table, not a search."""
    NEAR_STATS["checked"] += 1
    if near is None or near >= max_error:
        return out
    used = _near_used_elements(out, target, near, max_error)
    if not used:
        return out
    NEAR_STATS["tried"] += 1
    import copy
    from qiskit.transpiler import InstructionProperties
    marked = copy.deepcopy(target)
    for name, qargs in used:
        old = marked[name][qargs]
        marked.update_instruction_properties(name, qargs, InstructionProperties(
            duration=getattr(old, "duration", None), error=max_error))
    try:
        alt = compile_for_hardware(**dict(call_args, backend=None, target=marked, prune_max_error=max_error,
                                          near_failed_error=None, steiner=False, on_failed_elements="raise"))
    except (FailedElementsError, TranspilerError, ValueError):
        alt = None
    if alt is None:
        NEAR_STATS["no_alternative"] += 1
        return out
    if _st_cost(alt, target) < _st_cost(out, target):
        NEAR_STATS["kept_alternative"] += 1
        return alt
    NEAR_STATS["kept_first"] += 1
    return out


def _near_used_elements(circ, target, near, max_error):
    """Item 74 (prototype c43): the near-failed elements (error in [near, max_error)) that `circ` uses, as
    [(instruction name, qargs)] of the Target: the coupler of each two-qubit gate on one (both directions), the sx
    (and x) of a qubit with any gate on it whose sx error is in range, the measurement of a measured qubit whose readout
    error is."""
    name2 = next((g for g in ("cz", "ecr", "cx") if g in target.operation_names), None)
    edges = set()
    if name2 is not None:
        for qargs, props in target[name2].items():
            if qargs is not None and props is not None and props.error is not None and near <= props.error < max_error:
                edges.add(tuple(qargs))
    gate_q = {q for q, e in qubit_errors_from_target(target, "sx").items() if near <= e < max_error}
    read_q = {q for q, e in readout_errors_from_target(target).items() if near <= e < max_error}
    if not (edges or gate_q or read_q):
        return []
    pos = {q: i for i, q in enumerate(circ.qubits)}
    out = set()
    for inst in circ.data:
        idx = tuple(pos[q] for q in inst.qubits)
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        if name == "measure":
            if idx[0] in read_q:
                out.add(("measure", idx))
        for q in idx:
            if q in gate_q:
                for g in ("sx", "x"):
                    if g in target.operation_names and (q,) in target[g]:
                        out.add((g, (q,)))
        if len(idx) == 2:
            for e in (idx, idx[::-1]):
                if e in edges:
                    out.add((name2, e))
                    if e[::-1] in target[name2]:
                        out.add((name2, e[::-1]))
    return sorted(out)


def _near_alternative_c42(out, call_args, target, near, max_error):  # item 74 as in c42, kept for the record'''

for old, new in [
    ("VERSION: 2026-10-11.c42 -- prototype (from c41, c40, c39, release 2026-10-11.2): items 69-74",
     "VERSION: 2026-10-11.c43 -- prototype (from c42; release 2026-10-11.3 = c41): items 69-76"),
    ('VERSION = "2026-10-11.c42"  # prototype: release 2026-10-11.2 + items 69-74 (STEINER6 kept where its ESP is '
     'higher; a size limit; non-commuting, multi-step, QAOA-type and gate-level inputs; near-failed elements)',
     'VERSION = "2026-10-11.c43"  # prototype: release 2026-10-11.3 + items 74-76 (near-failed elements the output '
     'used; a deterministic re-placement; permutations left at the end)'),
    ('''    0.001-0.004, a fifth of the failed threshold); None turns it off. Counts in `NEAR_STATS`.
"""''', '''    0.001-0.004, a fifth of the failed threshold); None turns it off. Counts in `NEAR_STATS`.
    **Revised in prototype c43:** the second compile avoids only the near-failed elements the first output used (it
    sees a copy of the Target with those marked failed). In C42-VAL, avoiding every element of error >= 0.1 at once left
    no placement in 100 of 198 attempts (FakeTorino has 19 qubits with readout error above 0.1).
75. **REPRODUCIBILITY: a deterministic exact re-placement (prototype c43).** Qiskit's VF2PostLayout, with a fixed seed,
    returns different layouts from call to call (C42-VAL, Addendum 450): with the routing call's own Target, which has
    no errors, it chooses among equal layouts by an order that changes between calls. In its place -- in Qiskit's preset
    pass managers built here, in item 33's re-placement and in item 68's relabel -- `_DeterministicPostLayout` scores
    subgraph matchings found by rustworkx's VF2 in a fixed order (`id_order=True`) by the summed -ln(1 - error) of
    every operation on the device's own Target, places qubits without two-qubit gates one by one on their cheapest free
    qubit, and applies the lowest score if it is below the current placement's. Without a device Target nothing is
    re-placed (there are no errors to place by). The same input then gives the same output. Counts in `DET_STATS`.
76. **ACCURACY: permutations left at the end (prototype c43).** Item 73 required the Clifford gates left at the end to
    act on each qubit alone. They may now also permute the qubits (as QFT's final swaps do): the permutation costs no
    gate, it is carried into the measurements and the output's final layout. Counts in `STEINER_STATS` ("permuted").
"""'''),
    ('''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0}  # changelog items 68-73''',
     '''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0, "permuted": 0}  # changelog items 68-76'''),
    ("\n\nNEAR_STATS = {", DET + "\n\nNEAR_STATS = {"),
    ("def _near_alternative(out, call_args, target, near, max_error):", NEAR_NEW),
    ('''    suffix = []
    table = None
    for q in range(n):
        if frame[q] == [(1 << q, 0, 0), (0, 1 << q, 0)]:
            continue
        key = []
        for x, z, k in frame[q]:
            if (x | z) & ~(1 << q) or not (x | z):
                return None  # the Clifford left at the end acts on more than this qubit
            ch = "Y" if x and z else ("X" if x else "Z")
            sg = (k - (1 if ch == "Y" else 0)) % 4
            if sg not in (0, 2):
                return None
            key.append((ch, 1 if sg == 0 else -1))
        table = table or _st_cliff1q()
        word = table.get(tuple(key))
        if word is None:
            return None
        suffix += [(g, q) for g in word]
    STEINER_STATS["gate_level"] += 1
    return prefix, meas, terms, phase, suffix''', "@@PERM@@"),
    ('''    ev = ev + ([],) if ev is not None else _st_parse_gates(qc)  # item 73
    if ev is None:
        STEINER_STATS["not_applicable"] += 1
        return None
    prefix, meas, triples, extra_phase, suffix = ev''', '''    ev = ev + ([], None) if ev is not None else _st_parse_gates(qc)  # items 73, 76
    if ev is None:
        STEINER_STATS["not_applicable"] += 1
        return None
    prefix, meas, triples, extra_phase, suffix, perm = ev
    perm = perm or list(range(qc.num_qubits))'''),
    ('''    for q, cb in meas:
        phys.measure(m.pi[q], cb)
    out = transpile(phys, target=target, initial_layout=list(range(N)), routing_method="none", optimization_level=3,
                    seed_transpiler=seed)''', '''    for q, cb in meas:
        phys.measure(m.pi[perm[q]], cb)  # item 76: logical q's state is on wire perm[q]'s place
    with _DeterministicVF2(target):  # item 75
        out = transpile(phys, target=target, initial_layout=list(range(N)), routing_method="none",
                        optimization_level=3, seed_transpiler=seed)'''),
    ('''    vf2 = VF2PostLayout(target=target, seed=-1, call_limit=call_limit, max_trials=max_trials, strict_direction=True)''',
     '''    vf2 = _DeterministicPostLayout(target, call_limit, max_trials, True)  # item 75'''),
    ('''    pos = [0] * N  # pos[p]: where the state that started at physical p ends
    for p_now, p_start in enumerate(at):
        pos[p_start] = p_now''', '''    pos = [0] * N  # pos[p]: where the state that started at physical p ends
    for p_now, p_start in enumerate(at):
        pos[p_start] = p_now
    if perm != list(range(n)):  # item 76: logical q started at pi0[q] and ends where wire perm[q]'s state ends
        ends = [pos[pi0[w]] for w in range(n)]
        for q in range(n):
            pos[pi0[q]] = ends[perm[q]]'''),
    ('''    if permutation is None and not post_routing_resynthesis and _refine_target is None:
        return transpile(
            qc_compressed,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=routing_optimization_level,
            seed_transpiler=seed_transpiler,
            initial_layout=initial_layout,
            callback=callback,
        )
    pm = generate_preset_pass_manager(
        optimization_level=routing_optimization_level,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        seed_transpiler=seed_transpiler,
        initial_layout=initial_layout,
    )''', '''    if permutation is None and not post_routing_resynthesis and _refine_target is None:
        with _DeterministicVF2(None):  # item 75: no Target, nothing to re-place by
            return transpile(
                qc_compressed,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                optimization_level=routing_optimization_level,
                seed_transpiler=seed_transpiler,
                initial_layout=initial_layout,
                callback=callback,
            )
    with _DeterministicVF2(_refine_target):  # item 75
        pm = generate_preset_pass_manager(
            optimization_level=routing_optimization_level,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            seed_transpiler=seed_transpiler,
            initial_layout=initial_layout,
        )'''),
    ('''        refine = [VF2PostLayout(target=_refine_target, seed=-1, call_limit=placement_call_limit,
                                max_trials=placement_max_trials, strict_direction=True),''',
     '''        refine = [_DeterministicPostLayout(_refine_target, placement_call_limit, placement_max_trials, True),  # 75'''),
]:
    assert s.count(old) == 1, old[:90]
    s = s.replace(old, new)
s = s.replace("@@PERM@@", PERM)
# the measurement-placement test of item 73 uses m.pi[q] elsewhere only for measurements; nothing else to change
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
