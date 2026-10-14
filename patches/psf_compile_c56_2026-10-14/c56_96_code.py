

# ---------------------------------------------------------------------------------------------------------------------
# Prototype c56: item 96, the scope gate (home, after the owner's scope declaration of 2026-10-14, Addendum 488):
# PSF-Zero compiles a circuit only when its two-qubit interaction graph embeds in the device's working couplers;
# every other circuit is handed to Qiskit's level 3 unchanged
# ---------------------------------------------------------------------------------------------------------------------

SCOPE_GATE = True  # item 96
SCOPE96_STATS = {"calls": 0, "in_scope": 0, "handed": 0, "handed_working": 0, "handed_failed_fallback": 0,
                 "reasons": {}, "handoff_exact": 0, "handoff_unchecked": 0, "handoff_inexact": 0}
_S96_DEPTH = [0]
_S96_VF2_CALLS = 2_000_000  # item 96: VF2 states tried before an embedding counts as not found (a count, not a time)


def _scope96_edges(qc):
    """Item 96: (sorted two-qubit interaction edges, None), or (None, reason) when the circuit cannot be in scope.
    A circuit items 72-73's reader reads as Pauli terms on one or two qubits gives the terms' two-qubit supports
    (item 91's `_graph_read`); otherwise every two-qubit operation as written gives an edge, and an operation on more
    than two qubits puts the circuit out of scope."""
    got = _graph_read(qc)
    if got is not None:
        return sorted(got[-1]), None
    edges = set()
    for ins in qc.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if len(qs) > 2:
            return None, "multi_qubit_operation"
        if len(qs) == 2:
            edges.add((min(qs), max(qs)))
    return sorted(edges), None


def _scope96_embeds(edges, n, target, max_error):
    """Item 96: True when the interaction graph `edges` (of an n-qubit circuit) embeds in the device's working
    coupling graph -- couplers and qubits the Target reports failed (error >= max_error, either direction) removed --
    by the first match of rustworkx's VF2 (subgraph monomorphism, nodes in index order, no score). Not finding one
    within _S96_VF2_CALLS states counts as no embedding."""
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


def _hand_off96(qc, target, seed, max_error):
    """Item 96: (Qiskit level 3's output on the device, how), with the device's own Target; if that output uses a
    failed element, Qiskit level 3 on the device without its failed qubits (item 85's `_working_target`), written back
    on the device's indices; (None, "failed") if that too uses a failed element."""
    fe, fq = _failed_elements(target, max_error)
    out = generate_preset_pass_manager(3, target=target, seed_transpiler=seed).run(qc)
    if not _uses_failed(out, fe, fq):
        return out, "qiskit"
    if fq and target.num_qubits - len(fq) >= qc.num_qubits:
        sub, keep = _working_target(target, set(fq))
        o = generate_preset_pass_manager(3, target=sub, seed_transpiler=seed).run(qc)
        o = _embed_working(o, keep, target.num_qubits)
        if not _uses_failed(o, fe, fq):
            return o, "qiskit_working"
    return None, "failed"


def compile_for_hardware(qc, *args, **kwargs):
    """Item 96 (prototype c56): the scope gate. With the device's Target (`target=` or `backend=`), the input's
    two-qubit interaction graph (`_scope96_edges`) is tested against the device's working couplers
    (`_scope96_embeds`): if it embeds -- or the input has no two-qubit operation -- the call is
    `_compile_for_hardware_c55` (items 1-95 as before, item 94 off); otherwise the input is handed to Qiskit's level 3
    on the device unchanged (seed `seed_transpiler`, 0 when none is given; on the working qubits if Qiskit's output
    uses a failed element; PSF-Zero's own path only if that fails too, counted), checked by item 93's test where it
    can be, and returned with a warning if inexact. Without a Target, or with positional arguments, nothing changes.
    `SCOPE_GATE = False` restores c55 with item 94 off. Counts in `SCOPE96_STATS`.

    """
    target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
    if _S96_DEPTH[0] or not SCOPE_GATE or args or target is None:
        return _compile_for_hardware_c55(qc, *args, **kwargs)
    SCOPE96_STATS["calls"] += 1
    max_error = kwargs.get("prune_max_error", 0.5)
    edges, reason = _scope96_edges(qc)
    if reason is None:
        in_scope = not edges or _scope96_embeds(edges, qc.num_qubits, target, max_error)
        reason = ("no_two_qubit" if not edges else "embeds") if in_scope else "no_embedding"
    else:
        in_scope = False
    SCOPE96_STATS["reasons"][reason] = SCOPE96_STATS["reasons"].get(reason, 0) + 1
    if in_scope:
        SCOPE96_STATS["in_scope"] += 1
        _S96_DEPTH[0] += 1
        try:
            return _compile_for_hardware_c55(qc, **kwargs)
        finally:
            _S96_DEPTH[0] -= 1
    seed = kwargs.get("seed_transpiler")
    seed = 0 if seed is None else seed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, how = _hand_off96(qc, target, seed, max_error)
    if out is None:
        SCOPE96_STATS["handed_failed_fallback"] += 1
        _S96_DEPTH[0] += 1
        try:
            return _compile_for_hardware_c55(qc, **kwargs)
        finally:
            _S96_DEPTH[0] -= 1
    SCOPE96_STATS["handed"] += 1
    if how == "qiskit_working":
        SCOPE96_STATS["handed_working"] += 1
    verdict = _exact93(qc, out)
    if verdict is None:
        SCOPE96_STATS["handoff_unchecked"] += 1
    elif verdict:
        SCOPE96_STATS["handoff_exact"] += 1
    else:
        SCOPE96_STATS["handoff_inexact"] += 1
        warnings.warn("psf_compile item 96: Qiskit level 3's output, handed on for a circuit outside PSF-Zero's scope, "
                      f"is not exact within EXACT_CHECK_TOL = {EXACT_CHECK_TOL}; returned unchanged", RuntimeWarning,
                      stacklevel=2)
    return out


compile_for_hardware.__doc__ += _compile_for_hardware_c55.__doc__ or ""
