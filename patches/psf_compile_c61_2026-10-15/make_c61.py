"""make_c61.py -- workplace proposal, 2026-10-15: prototype 2026-10-15.c61 from c60 by exact substitutions: item 100,
the compile template (`compile_template`, `CompileTemplate`). c60's default call is unchanged; items 87 and 91 gain a
hook (unset by default) through which compile_template reads the structure they build.
    python make_c61.py <c60 psf_compile.py> <c61_100_code.py> <output psf_compile.py>"""
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
    "VERSION: 2026-10-15.c61 -- workplace proposal (from c60: item 100 added; release 2026-10-14.1 = c57)")
sub('VERSION = "2026-10-15.c60"  # workplace proposal: item 99',
    'VERSION = "2026-10-15.c61"  # workplace proposal: item 100 (compile template) + item 99')
i = s.index("99. **SCOPE: native gate circuits on an embeddable graph, by formula (workplace proposal c60).**")
j = s.index('\n"""', i)
s = s[:j] + '''
100. **SPEED: the compile template (workplace proposal c61).** In a variational loop or a Trotter sweep the shape
    stays and only the angles change. `compile_template(qc, target)` builds an in-scope circuit once, by the formula
    path c60 takes (item 87, 91 or 99), into a parametrised circuit on the device; `CompileTemplate.bind_values` /
    `.bind(qc)` give the compiled circuit for new angles in closed form only (single-qubit runs by ZYZ, two-qubit
    forms as at build, assign_parameters): no pass manager, no simulation. A shape signature covers every
    value-dependent branch (single-qubit run classes, rzz folding and dropping, item 88's frame, block CZ counts);
    when it changes, the template is rebuilt and counted. Item 99's outputs are reproduced value for value; items
    87/91's are emitted by PSF-Zero itself (their last step is Qiskit's), exact, see the C61-VAL record. c60's
    default call is unchanged. Counts in `TEMPLATE100_STATS`.''' + s[j:]
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
