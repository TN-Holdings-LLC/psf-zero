"""make_c63.py -- workplace proposal, 2026-10-15: prototype 2026-10-15.c63 from c62 by exact substitutions: item 100,
the compile template of c61 (`compile_template`, `CompileTemplate`), with a planned binding (each shape keeps its
emission plan). c62's default call is unchanged; items 87 and 91 gain c61's hook (unset by default).
    python make_c63.py <c62 psf_compile.py> <c63_100_code.py> <output psf_compile.py>"""
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


sub("VERSION: 2026-10-15.c62 -- workplace proposal (from c60: item 101 added; release 2026-10-14.1 = c57)",
    "VERSION: 2026-10-15.c63 -- workplace proposal (from c62: item 100 added, planned binding; release 2026-10-14.1 = c57)")
sub('VERSION = "2026-10-15.c62"  # workplace proposal: item 101',
    'VERSION = "2026-10-15.c63"  # workplace proposal: item 100 (compile template, planned binding) + item 101')
i = s.index("101. **SCOPE: items 87/91's CZ path emitted by PSF-Zero (workplace proposal c62).**")
j = s.index('\n"""', i)
s = s[:j] + '''
100. **SPEED: the compile template (workplace proposal c61; planned binding c63).** `compile_template(qc, target)`
    builds an in-scope circuit once, by the formula path (item 87, 91 or 99), into a parametrised circuit on the
    device; `CompileTemplate.bind_values` / `.bind(qc)` give the compiled circuit for new angles in closed form only
    (no pass manager, no simulation). A shape signature covers every value-dependent branch; a new shape is built and
    counted, a shape seen before is taken from the template's cache. c63: each shape keeps its emission plan (every
    run's factors, class and slots), so a bind multiplies and merges the runs without re-emitting -- the same
    arithmetic, the same output. Item 99's outputs are c62's value for value; items 87/91's frame path equals c62's in
    duration and two-qubit count; their CZ path uses c61's fast closed form for XX + YY + ZZ blocks (not item 101's
    Weyl form); a run with a factor of a general Weyl block is pinned to the full form, so the decomposition's
    discrete branches do not change the shape. c62's default call is unchanged. Counts in `TEMPLATE100_STATS`.''' + s[j:]
sub('''    if frame:
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveDiagonalGatesBeforeMeasure
        log = PassManager([RemoveDiagonalGatesBeforeMeasure()]).run(log)''', '''    if _T100_HOOK[0] is not None:  # item 100: compile_template reads the structure here (nothing is compiled)
        _T100_HOOK[0](dict(prefix=prefix, terms=terms, order=sorted(range(len(terms)), key=key), suffix=suffix,
                           meas=meas, perm=perm, layout=layout, frame=frame, phase=phase, n=n,
                           extra=locals().get("path", locals().get("edges"))))
    if frame:
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveDiagonalGatesBeforeMeasure
        log = PassManager([RemoveDiagonalGatesBeforeMeasure()]).run(log)''', count=2)
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
