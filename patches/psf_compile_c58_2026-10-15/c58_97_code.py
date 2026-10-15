

# ---------------------------------------------------------------------------------------------------------------------
# Prototype c58 (home, 2026-10-15): item 97, formula only inside the scope -- an in-scope circuit is built by item 87's
# chain brickwork or item 91's edge colouring and returned as it is; nothing else is built or compared, and no output
# is simulated to check it. A circuit the formulas do not build goes to Qiskit's level 3 as item 96 hands it on.
# ---------------------------------------------------------------------------------------------------------------------

FORMULA_ONLY = True  # item 97
FORMULA_CHECK = False  # item 97: True checks each formula output by item 93's test (slow above 16 qubits)
FORMULA97_STATS = {"calls": 0, "chain": 0, "graph": 0, "not_built": 0, "checked": 0, "check_failed": 0}


def compile_for_hardware(qc, *args, **kwargs):
    """Item 97 (prototype c58): with the device's Target and an input inside item 96's scope, the output of item 87's
    `_chain_brickwork` or else item 91's `_graph_brickwork`, returned without building or comparing anything else and
    without simulating it (`FORMULA_CHECK = True` adds item 93's check, and the usual path if it fails). If neither
    builds, the input is handed to Qiskit's level 3 as item 96 does (rzz folded on a fractional Target). Out of scope,
    without a Target, with positional arguments or `FORMULA_ONLY = False`: `_compile_for_hardware_c57` unchanged.
    Counts in `FORMULA97_STATS`.

    """
    target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
    if not FORMULA_ONLY or _S96_DEPTH[0] or args or target is None:
        return _compile_for_hardware_c57(qc, *args, **kwargs)
    max_error = kwargs.get("prune_max_error", 0.5)
    edges, reason = _scope96_edges(qc)
    in_scope = reason is None and (not edges or _scope96_embeds(edges, qc.num_qubits, target, max_error))
    if not in_scope:
        return _compile_for_hardware_c57(qc, **kwargs)
    FORMULA97_STATS["calls"] += 1
    seed = kwargs.get("seed_transpiler")
    seed = 0 if seed is None else seed
    out = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "cz" in target.operation_names:
            out = _chain_brickwork(qc, target, max_error, seed)
            kind = "chain"
            if out is None:
                out = _graph_brickwork(qc, target, max_error, seed)
                kind = "graph"
    if out is not None and "rzz" in target.operation_names and not _rzz_in_range(out):
        try:
            out = _fold(out, target)
        except TranspilerError:
            out = None
    if out is not None and FORMULA_CHECK:
        FORMULA97_STATS["checked"] += 1
        if _exact93(qc, out) is False:
            FORMULA97_STATS["check_failed"] += 1
            out = None
    if out is not None:
        FORMULA97_STATS[kind] += 1
        return out
    FORMULA97_STATS["not_built"] += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        handed, how = _hand_off96(qc, target, seed, max_error)
    if handed is not None and "rzz" in target.operation_names and not _rzz_in_range(handed):
        try:
            handed = _fold(handed, target)
        except TranspilerError:
            handed = None
    if handed is not None:
        return handed
    _S96_DEPTH[0] += 1
    try:
        return _compile_for_hardware_c55(qc, **kwargs)
    finally:
        _S96_DEPTH[0] -= 1


compile_for_hardware.__doc__ += _compile_for_hardware_c57.__doc__ or ""
