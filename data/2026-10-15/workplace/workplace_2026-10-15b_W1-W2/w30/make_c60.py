"""make_c60.py -- workplace proposal, 2026-10-15: prototype 2026-10-15.c60 from c59 by exact substitutions: item 99,
native two-qubit gate circuits on an embeddable graph built by formula as written. An in-scope input that neither
item 87's chain nor item 91's graph formula reads (c59: OutOfScopeError "neither formula reads") is built by
`_native_brickwork`; everything the two formulas build is returned exactly as in c59.
    python make_c60.py <c59 psf_compile.py> <c60_99_code.py> <output psf_compile.py>"""
import sys

src, code, dst = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
add = open(code, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub("VERSION: 2026-10-15.c59 -- prototype (from c58: item 98 added; release 2026-10-14.1 = c57)",
    "VERSION: 2026-10-15.c60 -- workplace proposal (from c59: item 99 added; release 2026-10-14.1 = c57)")
sub('VERSION = "2026-10-15.c59"  # prototype: item 98',
    'VERSION = "2026-10-15.c60"  # workplace proposal: item 99 (native gate circuits on an embeddable graph by formula) + item 98')
i = s.index("98. **SCOPE: the gate closed (prototype c59).**")
j = s.index('\n"""', i)
s = s[:j] + '''
99. **SCOPE: native gate circuits on an embeddable graph, by formula (workplace proposal c60).** CZ / CX chains and
    lattices (hardware-efficient ansaetze, `real_amplitudes`, `efficient_su2` expanded, CZ-ring classifiers) are
    inside item 96's scope but neither item 87 nor item 91 reads them, so c59 refused them. `_native_brickwork`
    builds them as written in one fixed pass: item 91's placement (first VF2 embedding), commuting diagonal gates
    ordered by item 91's edge colouring, each two-qubit gate in a fixed native form (cz; rzz native and folded, or
    cz rz cz; others through RZZ or the closed-form CZ decomposition, each multiplied out and checked), single-qubit
    runs merged in closed form (ZYZ). No pass manager, no comparison, no simulation. Outputs of items 87 and 91 are
    unchanged. `NATIVE_BRICKWORK = False` restores c59. Counts in `NATIVE99_STATS`.''' + s[j:]
sub('''            if out is None:
                out = _graph_brickwork(qc, target, max_error, seed)
                kind = "graph"
''', '''            if out is None:
                out = _graph_brickwork(qc, target, max_error, seed)
                kind = "graph"
            if out is None and NATIVE_BRICKWORK:  # item 99
                out = _native_brickwork(qc, target, max_error, seed)
                kind = "native"
''')
sub('FORMULA97_STATS = {"calls": 0, "chain": 0, "graph": 0, "not_built": 0, "checked": 0, "check_failed": 0}\n',
    'FORMULA97_STATS = {"calls": 0, "chain": 0, "graph": 0, "native": 0, "not_built": 0, "checked": 0, "check_failed": 0}\n')
sub('''        raise OutOfScopeError("inside the scope, but neither formula (item 87's chain, item 91's graph) reads this "
                              "circuit (changelog item 98); compile it with Qiskit yourself")''',
    '''        raise OutOfScopeError("inside the scope, but no formula (item 87's chain, item 91's graph, item 99's native "
                              "gates) reads this circuit (changelog items 98-99); compile it with Qiskit yourself")''')
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
