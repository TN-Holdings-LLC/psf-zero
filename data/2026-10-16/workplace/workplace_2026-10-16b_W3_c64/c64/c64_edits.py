# c64_edits.py -- the edits make_c64.py applies to c63a: exact substitutions (`sub`) and whole top-level definitions
# (`REPLACE`). Every edit fixes a switch at c63a's default and removes what that value cannot reach; nothing else.
REPLACE = {}

# --- compile_for_hardware: GATE_CLOSED = True, FORMULA_ONLY = True, FORMULA_CHECK = False, NATIVE_BRICKWORK = True;
# _S96_DEPTH stays 0 (it was raised only around _compile_for_hardware_c55 / c57, which are removed)
REPLACE["compile_for_hardware"] = '''def compile_for_hardware(qc, *args, **kwargs):
    """Item 97 (prototype c58), items 98-99, c64: with the device's Target (`target=` or `backend=`) and an input
    inside item 96's scope, the output of item 87's `_chain_brickwork`, else item 91's `_graph_brickwork`, else item
    99's `_native_brickwork`, returned without building or comparing anything else and without simulating it (an rzz
    outside IBM's range goes through item 86's `_fold`). Without a Target, with positional arguments, out of scope or
    with no formula reading the input: OutOfScopeError (item 98). Counts in `FORMULA97_STATS` and `GATE98_STATS`."""
    target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
    if target is None or args:  # item 98
        GATE98_STATS["no_target"] += 1
        raise OutOfScopeError("PSF-Zero compiles only with the device's Target given as target= or backend= "
                              "(changelog item 98); compile this call with Qiskit yourself")
    max_error = kwargs.get("prune_max_error", 0.5)
    edges, reason = _scope96_edges(qc)
    in_scope = reason is None and (not edges or _scope96_embeds(edges, qc.num_qubits, target, max_error))
    if not in_scope:  # item 98
        GATE98_STATS["out_of_scope"] += 1
        why = ("an operation acts on more than two qubits" if reason == "multi_qubit_operation" else
               "its two-qubit interaction graph does not embed in the device's working couplers")
        raise OutOfScopeError(f"outside PSF-Zero's scope: {why} (changelog item 98); compile it with Qiskit yourself")
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
            if out is None:  # item 99
                out = _native_brickwork(qc, target, max_error, seed)
                kind = "native"
    if out is not None and "rzz" in target.operation_names and not _rzz_in_range(out):
        try:
            out = _fold(out, target)
        except TranspilerError:
            out = None
    if out is not None:
        FORMULA97_STATS[kind] += 1
        return out
    FORMULA97_STATS["not_built"] += 1
    GATE98_STATS["not_built"] += 1
    raise OutOfScopeError("inside the scope, but no formula (item 87's chain, item 91's graph, item 99's native "
                          "gates) reads this circuit (changelog items 98-99); compile it with Qiskit yourself")'''

# --- items 87 / 91: CHAIN_FRAME = True, CZ_OWN_EMIT = True (the level-3 branch is removed)
sub('    frame = CHAIN_FRAME and "rzz" in target.operation_names and "rx" in target.operation_names\n',
    '    frame = "rzz" in target.operation_names and "rx" in target.operation_names\n', count=2)
sub('''    elif CZ_OWN_EMIT:  # item 101: PSF-Zero emits the CZ path itself
        out = _cz101_emit(qc, target, dict(prefix=prefix, terms=terms, order=sorted(range(len(terms)), key=key),
                                           suffix=suffix, meas=meas, perm=perm, layout=layout, n=n))
    else:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        out = pm.run(log)
''', '''    else:  # item 101: PSF-Zero emits the CZ path itself
        out = _cz101_emit(qc, target, dict(prefix=prefix, terms=terms, order=sorted(range(len(terms)), key=key),
                                           suffix=suffix, meas=meas, perm=perm, layout=layout, n=n))
''', count=2)

# --- item 89: DROP_DIAGONAL_BEFORE_MEASURE = True
sub("    if DROP_DIAGONAL_BEFORE_MEASURE and meas and {q for q, _ in meas} == set(range(n)):  # item 89\n",
    "    if meas and {q for q, _ in meas} == set(range(n)):  # item 89\n")

# --- item 84: SELECT_BY = "duration" (the "esp" branch is removed)
sub('''    ca, cb = _st_cost(a, target), _st_cost(b, target)
    if SELECT_BY == "esp":
        return ca < cb
    fa, fb''', '''    ca, cb = _st_cost(a, target), _st_cost(b, target)
    fa, fb''')

# --- item 100 (c63): TEMPLATE_BATCH = True
sub('        self.batch = gen.kind != "native" and TEMPLATE_BATCH\n', '        self.batch = gen.kind != "native"\n')

# --- item 99 in compile_template: NATIVE_BRICKWORK = True (the constant stays for a23; nothing branches on it)
sub('''    if NATIVE_BRICKWORK:
        try:
            tpl = CompileTemplate(_T100Native(qc, target, max_error), target, seed_transpiler)
        except _T100NotHere:
            tpl = None
        if tpl is not None and not _uses_failed(tpl.circuit, fe, fq):
            return tpl
''', '''    try:
        tpl = CompileTemplate(_T100Native(qc, target, max_error), target, seed_transpiler)
    except _T100NotHere:
        tpl = None
    if tpl is not None and not _uses_failed(tpl.circuit, fe, fq):
        return tpl
''')

# --- W3 option A (c63a): TEMPLATE_FRAME_FIXED = True. The frame-flipping `_T100Formula._frame_items` is never used
# (c63a replaced it after the class); c63a's function becomes the method itself, and the replacement is removed.
_a = s.index("    def _frame_items(self, values):\n", s.index("class _T100Formula"))
_b = s.index("\n    # -- the CZ path", _a)
_old_method = s[_a:_b + 1]
_c = s.index("def _t63a_frame_items(self, values):\n")
_d = s.index("\n\n\n_T100Formula._frame_items = _t63a_frame_items\n")
_new_body = s[_c:_d].replace('''def _t63a_frame_items(self, values):
    if not TEMPLATE_FRAME_FIXED:
        yield from _t63a_orig_frame_items(self, values)
        return
''', '''def _frame_items(self, values):
    """W3 option A (c63a): item 88's frame fixed at build. A negative rzz angle becomes X RZZ(|theta|) X on its first
    qubit, locally; a fold (|theta| crossing pi/2) is Rz(pi) on both qubits; the items on both sides of every rzz are
    pinned (`_t63_emit`), so a sign change or a fold alters angles only, never the shape."""
''')
assert _new_body.startswith("def _frame_items")
_new_method = "\n".join(("    " + x) if x else x for x in _new_body.split("\n")) + "\n"
sub(_old_method, _new_method)
sub(s[s.index("# ======================================================================================================================\n# W3 option A"):] if False else
    s[s.index("# ======================================================================================================================\n# W3 option A"):_c],
    "_T63A_I = (1 + 0j, 0j, 0j, 1 + 0j)  # W3 option A (c63a): the identity, copied into new tuples by _frame_items\n\n\n")
sub(s[_c:_d] + "\n\n\n_T100Formula._frame_items = _t63a_frame_items\n", "")

# --- version
sub('VERSION = "2026-10-15.c63a"  # W3 option A (frame fixed in the template) on c63; c63 =',
    'VERSION = "2026-10-16.c64"  # workplace proposal: c63a without unreachable code (make_c64.py); c63a = option A on c63; c63 =')

# --- the first, overwritten definition of _n99_kron_steps (c61's faster form replaces it further down)
sub('''def _n99_kron_steps(steps, q0, q1):
    """Item 99: the 4 x 4 matrix (qubit q0 least significant) of a list of steps on (q0, q1), for the check."""
    U = np.eye(4, dtype=complex)
    for st in steps:
        if st[0] == "1":
            a, b, c, d = st[2]
            m = np.array([[a, b], [c, d]])
            U = (np.kron(np.eye(2), m) if st[1] == q0 else np.kron(m, np.eye(2))) @ U
        elif st[0] == "cz":
            U = np.diag([1, 1, 1, -1]).astype(complex) @ U
        else:
            t = st[3]
            U = np.diag(np.exp(-0.5j * t * np.array([1, -1, -1, 1]))) @ U
    return U


''', "")

# --- pyflakes' existing finding in kept code: the unused local N in _T100Native.__init__
sub('''        ops, meas = got
        n, N = qc.num_qubits, target.num_qubits
        edges = sorted({(min(qs), max(qs)) for _, qs, _, _ in ops if len(qs) == 2})''', '''        ops, meas = got
        n = qc.num_qubits
        edges = sorted({(min(qs), max(qs)) for _, qs, _, _ in ops if len(qs) == 2})''')

# names whose every kept use is a local import inside the function (the module-level import is then unused)
LOCAL_ONLY = {"TwoQubitBasisDecomposer", "TwoQubitWeylDecomposition", "Optimize1qGatesDecomposition"}
