"""Builds candidate 2026-10-11.c39 from release 2026-10-11.2 by exact substitutions (changelog items 69-70: the STEINER6
output is kept only if its ESP on the device is above the usual path's; inputs of more than 2000 terms take the usual
path).
    python make_c39.py <release 2026-10-11.2 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
for old, new in [
    ("VERSION: 2026-10-11.2 -- release, adopted on 2026-10-11 from candidate 2026-10-11.c38, accepted in C38-VAL "
     "(Addendum 445) (previous release: 2026-10-11.1)",
     "VERSION: 2026-10-11.c39 -- candidate (from release 2026-10-11.2): items 69-70"),
    ('VERSION = "2026-10-11.2"  # release (from candidate 2026-10-11.c38, C38-VAL): 2026-10-11.1 + item 68 (commuting '
     'Pauli evolutions by STEINER6)',
     'VERSION = "2026-10-11.c39"  # candidate: release 2026-10-11.2 + items 69-70 (STEINER6 kept only where its ESP is '
     'higher; a size limit)'),
    ('''    data only (Addendum 444: on all-commuting tests it beat the release on 12 of 13 runnable FakeTorino/FakeKingston
    cases; on the others it was even). Counts in `STEINER_STATS`.
"""''', '''    data only (Addendum 444: on all-commuting tests it beat the release on 12 of 13 runnable FakeTorino/FakeKingston
    cases; on the others it was even). Counts in `STEINER_STATS`.
69. **ACCURACY: STEINER6's output is kept only where it is better (candidate 2026-10-11.c39).** When item 68 applies,
    the input is also compiled by the usual path (`steiner=False`), and the output with the lower summed
    -ln(1 - error) over all its operations on the device (the higher ESP) is returned; on a tie, the usual path's.
    C38-VAL (Addendum 445) found item 68 worse than the usual path on the dense two-local diagonal type (HamLib tsp
    in unary encoding: 0.87 of its ESP), which no structural rule from the data separated from types it wins on (a
    dense two-local gnp instance won by 2-3 times). Two fixed constructions compared by the device's own error table,
    not a search over seeds. Counts in `STEINER_STATS` ("chosen", "kept_usual").
70. **SPEED: a size limit for item 68 (candidate 2026-10-11.c39).** An evolution of more than 2000 terms takes the
    usual path. The largest commuting test runnable on a device so far had 449 terms (Addenda 444-445); C38-VAL's
    slowest call was a 5335-term, 144-qubit evolution whose ESP is near 0 either way (129 s against the usual path's
    10 s). Counts in `STEINER_STATS` ("too_large").
"""'''),
    ('STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0}  # changelog item 68',
     'STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,\n'
     '                 "kept_usual": 0}  # changelog items 68-70\n'
     '_ST_MAX_TERMS = 2000  # item 70\n\n\n'
     'def _st_cost(circ, target):\n'
     '    """Item 69: the summed -ln(1 - error) of every operation of `circ` on the device (-ln ESP); inf if one is not\n'
     '    offered there or has error >= 1. Barriers and delays count 0, and so does control flow (no error reported)."""\n'
     '    pos = {q: i for i, q in enumerate(circ.qubits)}\n'
     '    total = 0.0\n'
     '    for ins in circ.data:\n'
     '        name = ins.operation.name\n'
     '        if name in ("barrier", "delay"):\n'
     '            continue\n'
     '        if name not in target.operation_names:\n'
     '            return math.inf\n'
     '        entries = target[name]\n'
     '        qargs = tuple(pos[q] for q in ins.qubits)\n'
     '        if qargs in entries:\n'
     '            props = entries[qargs]\n'
     '        elif None in entries:\n'
     '            continue\n'
     '        else:\n'
     '            return math.inf\n'
     '        e = props.error if props is not None and props.error is not None else 0.0\n'
     '        if e >= 1.0:\n'
     '            return math.inf\n'
     '        total += -math.log1p(-max(e, 0.0))\n'
     '    return total'),
    ('''    if not triples or not _st_all_commute(triples):
        STEINER_STATS["not_commuting"] += 1
        return None
''', '''    if len(triples) > _ST_MAX_TERMS:  # item 70
        STEINER_STATS["too_large"] += 1
        return None
    if not triples or not _st_all_commute(triples):
        STEINER_STATS["not_commuting"] += 1
        return None
'''),
    ('''        st = _st_compile(qc, target, prune_max_error, seed_transpiler, placement_call_limit, placement_max_trials)
        if st is not None:
            return st
''', '''        st = _st_compile(qc, target, prune_max_error, seed_transpiler, placement_call_limit, placement_max_trials)
        if st is not None:  # item 69: kept only if its ESP on the device is above the usual path's
            try:
                usual = compile_for_hardware(**dict(call_args, steiner=False))
            except FailedElementsError:
                usual = None
            if usual is None or _st_cost(st, target) < _st_cost(usual, target):
                STEINER_STATS["chosen"] += 1
                return st
            STEINER_STATS["kept_usual"] += 1
            return usual
'''),
]:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
