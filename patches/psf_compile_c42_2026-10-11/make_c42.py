"""Builds prototype 2026-10-11.c42 from prototype 2026-10-11.c41 by exact substitutions (changelog item 74: when the
output uses a coupler, a qubit's gates or a measured qubit's readout whose error is at least `near_failed_error`
(default 0.1) but below `prune_max_error`, the input is compiled again with those elements avoided as well, and the
output with the higher ESP on the device is returned).
    python make_c42.py <c41 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

NEAR = '''

NEAR_STATS = {"checked": 0, "tried": 0, "kept_alternative": 0, "kept_first": 0, "no_alternative": 0}  # item 74


def _near_used(circ, target, near, max_error):
    """Item 74: True if `circ` uses an element whose error is in [near, max_error): a two-qubit gate on such a coupler
    (either direction for symmetric gates), any gate on a qubit whose sx error is in that range, or a measurement of
    a qubit whose readout error is."""
    name = next((g for g in ("cz", "ecr", "cx") if g in target.operation_names), None)
    edges = set()
    if name is not None:
        for qargs, props in target[name].items():
            if qargs is not None and props is not None and props.error is not None and near <= props.error < max_error:
                edges.add(tuple(qargs))
    gate_q = {q for q, e in qubit_errors_from_target(target, "sx").items() if near <= e < max_error}
    read_q = {q for q, e in readout_errors_from_target(target).items() if near <= e < max_error}
    if not (edges or gate_q or read_q):
        return False
    pos = {q: i for i, q in enumerate(circ.qubits)}
    for inst in circ.data:
        idx = tuple(pos[q] for q in inst.qubits)
        if inst.operation.name == "measure":
            if idx[0] in read_q or idx[0] in gate_q:
                return True
            continue
        if inst.operation.name in ("barrier", "delay"):
            continue
        if any(i in gate_q for i in idx):
            return True
        if len(idx) == 2 and (idx in edges or idx[::-1] in edges):
            return True
    return False


def _near_alternative(out, call_args, target, near, max_error):
    """Item 74: `out`, or the input compiled with every element of error >= `near` avoided when that output is
    better on the device (lower summed -ln(1 - error), item 69's measure). Two fixed constructions compared by the
    device's own error table, not a search."""
    NEAR_STATS["checked"] += 1
    if near is None or near >= max_error or not _near_used(out, target, near, max_error):
        return out
    NEAR_STATS["tried"] += 1
    try:
        alt = compile_for_hardware(**dict(call_args, prune_max_error=near, near_failed_error=None, steiner=False,
                                          on_failed_elements="raise"))
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
'''

for old, new in [
    ("VERSION: 2026-10-11.c41 -- prototype (from c40, c39, release 2026-10-11.2): items 69-73",
     "VERSION: 2026-10-11.c42 -- prototype (from c41, c40, c39, release 2026-10-11.2): items 69-74"),
    ('VERSION = "2026-10-11.c41"  # prototype: release 2026-10-11.2 + items 69-73 (STEINER6 kept where its ESP is higher; '
     'a size limit; non-commuting, multi-step, QAOA-type and gate-level inputs)',
     'VERSION = "2026-10-11.c42"  # prototype: release 2026-10-11.2 + items 69-74 (STEINER6 kept where its ESP is higher; '
     'a size limit; non-commuting, multi-step, QAOA-type and gate-level inputs; near-failed elements)'),
    ('''    Clifford gates' phases are not followed). QASM files of QAOA and Trotter circuits (cx-rz-cx) are of this kind.
"""''', '''    Clifford gates' phases are not followed). QASM files of QAOA and Trotter circuits (cx-rz-cx) are of this kind.
74. **ACCURACY: near-failed elements (prototype c42).** Item 31 avoids elements with error >= `prune_max_error`
    (0.5). An element just below it -- a coupler with 15% error, a qubit whose readout is wrong one time in five -- was
    used like any other, as often as routing chose (C41-VAL: on FakeBoston one Heisenberg circuit crossed a 15% coupler
    hundreds of times, ESP 2e-6 where an error-aware placement got 0.04). With the device's Target, when the output uses
    an element with error in [`near_failed_error`, `prune_max_error`) -- a two-qubit gate on such a coupler, any gate
    on such a qubit (sx error), or a measurement of a qubit with such a readout error -- the input is compiled again
    with every element of error >= `near_failed_error` avoided, and the output with the lower summed -ln(1 - error)
    (item 69's measure) is returned; on a tie, or when the second compile cannot avoid them, the first. Default 0.1
    (fixed before any measurement of it: an order of magnitude above these devices' median two-qubit errors of
    0.001-0.004, a fifth of the failed threshold); None turns it off. Counts in `NEAR_STATS`.
"""'''),
    ('''    on_failed_elements: str = "raise",
    _inner: bool = False,
) -> QuantumCircuit:''', '''    on_failed_elements: str = "raise",
    near_failed_error: float | None = 0.1,
    _inner: bool = False,
) -> QuantumCircuit:'''),
    ('''    compiled by STEINER6's construction. False compiles every input as the release does.
    """''', '''    compiled by STEINER6's construction. False compiles every input as the release does.
    `near_failed_error` (prototype 2026-10-11.c42, item 74): with a Target, an output that uses an element with error
    in [`near_failed_error`, `prune_max_error`) is compared with the input compiled with those elements avoided too,
    and the better by ESP is returned. None turns it off.
    """'''),
    ('''        out = _resynthesise(out, final_resynthesis, target, prune_max_error)
        if not compare_floor and candidate_score == "excitation":''',
     '''        out = _resynthesise(out, final_resynthesis, target, prune_max_error)
        if not _inner:  # item 74
            out = _near_alternative(out, call_args, target, near_failed_error, prune_max_error)
        if not compare_floor and candidate_score == "excitation":'''),
    ("\n\ndef _st_term_masks(label):", NEAR + "\n\ndef _st_term_masks(label):"),
]:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
