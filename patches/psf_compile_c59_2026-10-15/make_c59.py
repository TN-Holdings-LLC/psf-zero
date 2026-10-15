"""make_c59.py -- prototype 2026-10-15.c59 from c58 by exact substitutions: item 98, the gate closed -- no input is
handed to Qiskit's level 3. With the device's Target, an in-scope input is built by formula (item 97) or refused; an
out-of-scope input is refused; without a Target the call is refused. OutOfScopeError (a TranspilerError) says why.
    python make_c59.py <c58 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub("VERSION: 2026-10-15.c58 -- prototype (from release 2026-10-14.1 = c57: item 97 added; c57's items unchanged)",
    "VERSION: 2026-10-15.c59 -- prototype (from c58: item 98 added; release 2026-10-14.1 = c57)")
sub('VERSION = "2026-10-15.c58"  # prototype: item 97 (formula only inside the scope) +',
    'VERSION = "2026-10-15.c59"  # prototype: item 98 (gate closed: nothing handed to Qiskit) + item 97 (formula only inside the scope) +')
i = s.index("97. **SPEED: formula only inside the scope (prototype c58).**")
j = s.index('\n"""', i)
s = s[:j] + '''
98. **SCOPE: the gate closed (prototype c59).** The owner's decision of 2026-10-15 (Addendum 494): PSF-Zero stops
    handing circuits to Qiskit. With `GATE_CLOSED = True` (default), `compile_for_hardware` builds an in-scope input by
    formula (item 97) and refuses every other call with `OutOfScopeError` (a TranspilerError) saying why: no device
    Target; an operation on more than two qubits; an interaction graph that does not embed in the device's working
    couplers; or an in-scope input neither formula reads. The caller compiles those with Qiskit (or another compiler)
    themselves. `GATE_CLOSED = False` restores c58. Counts in `GATE98_STATS`.''' + s[j:]
sub('''    target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
    if not FORMULA_ONLY or _S96_DEPTH[0] or args or target is None:
        return _compile_for_hardware_c57(qc, *args, **kwargs)
    max_error = kwargs.get("prune_max_error", 0.5)
    edges, reason = _scope96_edges(qc)
    in_scope = reason is None and (not edges or _scope96_embeds(edges, qc.num_qubits, target, max_error))
    if not in_scope:
        return _compile_for_hardware_c57(qc, **kwargs)''', '''    target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
    if GATE_CLOSED and not _S96_DEPTH[0] and (target is None or args):  # item 98
        GATE98_STATS["no_target"] += 1
        raise OutOfScopeError("PSF-Zero compiles only with the device's Target given as target= or backend= "
                              "(changelog item 98); compile this call with Qiskit yourself")
    if not FORMULA_ONLY or _S96_DEPTH[0] or args or target is None:
        return _compile_for_hardware_c57(qc, *args, **kwargs)
    max_error = kwargs.get("prune_max_error", 0.5)
    edges, reason = _scope96_edges(qc)
    in_scope = reason is None and (not edges or _scope96_embeds(edges, qc.num_qubits, target, max_error))
    if not in_scope:
        if GATE_CLOSED:  # item 98
            GATE98_STATS["out_of_scope"] += 1
            why = ("an operation acts on more than two qubits" if reason == "multi_qubit_operation" else
                   "its two-qubit interaction graph does not embed in the device's working couplers")
            raise OutOfScopeError(f"outside PSF-Zero's scope: {why} (changelog item 98); compile it with Qiskit yourself")
        return _compile_for_hardware_c57(qc, **kwargs)''')
sub('''    FORMULA97_STATS["not_built"] += 1
    with warnings.catch_warnings():''', '''    FORMULA97_STATS["not_built"] += 1
    if GATE_CLOSED:  # item 98
        GATE98_STATS["not_built"] += 1
        raise OutOfScopeError("inside the scope, but neither formula (item 87's chain, item 91's graph) reads this "
                              "circuit (changelog item 98); compile it with Qiskit yourself")
    with warnings.catch_warnings():''')
sub('FORMULA97_STATS = {"calls": 0, "chain": 0, "graph": 0, "not_built": 0, "checked": 0, "check_failed": 0}\n',
    'FORMULA97_STATS = {"calls": 0, "chain": 0, "graph": 0, "not_built": 0, "checked": 0, "check_failed": 0}\n'
    'GATE_CLOSED = True  # item 98\nGATE98_STATS = {"no_target": 0, "out_of_scope": 0, "not_built": 0}\n\n\n'
    'class OutOfScopeError(TranspilerError):\n'
    '    """Item 98: the call is outside PSF-Zero\'s scope; compile it with Qiskit (or another compiler) yourself."""\n')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
