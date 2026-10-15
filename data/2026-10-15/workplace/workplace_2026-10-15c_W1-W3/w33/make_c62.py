"""make_c62.py -- workplace proposal, 2026-10-15: prototype 2026-10-15.c62 from c60 by exact substitutions: item 101,
items 87 and 91's CZ path (no rzz frame) emitted by PSF-Zero itself instead of Qiskit's level-3 preset on the fixed
layout. CZ_OWN_EMIT = False restores c60.
    python make_c62.py <c60 psf_compile.py> <c62_101_code.py> <output psf_compile.py>"""
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


sub("VERSION: 2026-10-15.c60 -- workplace proposal (from c59: item 99 added; release 2026-10-14.1 = c57)",
    "VERSION: 2026-10-15.c62 -- workplace proposal (from c60: item 101 added; release 2026-10-14.1 = c57)")
sub('VERSION = "2026-10-15.c60"  # workplace proposal: item 99',
    'VERSION = "2026-10-15.c62"  # workplace proposal: item 101 (items 87/91\'s CZ path emitted by PSF-Zero) + item 99')
i = s.index("99. **SCOPE: native gate circuits on an embeddable graph, by formula (workplace proposal c60).**")
j = s.index('\n"""', i)
s = s[:j] + '''
101. **SCOPE: items 87/91's CZ path emitted by PSF-Zero (workplace proposal c62).** Without item 88's frame (no rzz
    and rx in the Target), items 87 and 91 handed their logical circuit to Qiskit's level-3 preset on the fixed layout:
    the last outside stage inside the scope. `_cz101_emit` emits it instead, by block kind: one two-qubit term (a
    single interaction) as a local Clifford around RZZ in closed form; several terms on one pair (with the
    single-qubit operations between them) by the Weyl decomposition with closed-form circuits; diagonal operations
    directly before the measurements left out; runs merged in closed form (ZYZ, theta = pi as one x pulse). Placement
    and order are items 87/91's. Measured in C62-VAL (0.76-0.83 x c60 on Ising models in C61-VAL). `CZ_OWN_EMIT = False`
    restores c60. Counts in `CZ101_STATS`.''' + s[j:]
sub('''    else:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        out = pm.run(log)''', '''    elif CZ_OWN_EMIT:  # item 101: PSF-Zero emits the CZ path itself
        out = _cz101_emit(qc, target, dict(prefix=prefix, terms=terms, order=sorted(range(len(terms)), key=key),
                                           suffix=suffix, meas=meas, perm=perm, layout=layout, n=n))
    else:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        out = pm.run(log)''', count=2)
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
