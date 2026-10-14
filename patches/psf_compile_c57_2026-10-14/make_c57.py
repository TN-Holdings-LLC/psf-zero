"""make_c57.py -- prototype 2026-10-14.c57 from c56 by exact substitutions: item 96 revised. C56-VAL refuted its V6:
on a Target with IBM's fractional gates, Qiskit's level 3 emits rzz angles outside the executable range
0 < theta <= pi/2 (79 of 80 handed-on outputs), since only a qiskit-ibm-runtime backend object folds them by itself.
The handed-on output is now folded by item 86's `_fold` (FoldRzzAngle, then single-qubit runs merged), as IBM's
backend would; if folding fails, c55's own path is used (counted).
    python make_c57.py <c56 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub("VERSION: 2026-10-14.c56 -- prototype (from c55: item 94 off, item 96 added; c43 = release 2026-10-11.4): "
    "items 75 revised, 77-80, 82-91, 93, 95-96",
    "VERSION: 2026-10-14.c57 -- prototype (from c56: item 96 revised; c43 = release 2026-10-11.4): "
    "items 75 revised, 77-80, 82-91, 93, 95-96")
sub('VERSION = "2026-10-14.c56"  # prototype: item 96 (scope gate:',
    'VERSION = "2026-10-14.c57"  # prototype: item 96 revised (handed-on rzz folded into IBM\'s range) + item 96 (scope gate:')
i = s.index("96. **SCOPE: a circuit PSF-Zero does not build by formula is handed to Qiskit's level 3 unchanged (prototype c56).**")
j = s.index('\n"""', i)
s = s[:j] + '''
    Revised in c57: C56-VAL (Addendum 490) refuted its V6. On a Target with IBM's fractional gates, Qiskit's level 3
    emitted rzz angles outside the executable range 0 < theta <= pi/2 in 79 of the 80 handed-on outputs, because
    only a qiskit-ibm-runtime backend object folds them by itself, not a Target. With rzz in the Target the handed-on
    output is now folded by item 86's `_fold` (qiskit-ibm-runtime's FoldRzzAngle, then single-qubit runs merged), as
    IBM's backend would do; if folding fails, c55's own path is used (counted as "fold_failed_fallback"). On a Target
    without rzz nothing changes.''' + s[j:]
sub('''    SCOPE96_STATS["handed"] += 1
    if how == "qiskit_working":''', '''    if "rzz" in target.operation_names and not _rzz_in_range(out):  # item 96 revised (c57)
        try:
            out = _fold(out, target)
            SCOPE96_STATS["folded"] = SCOPE96_STATS.get("folded", 0) + 1
        except TranspilerError:
            SCOPE96_STATS["fold_failed_fallback"] = SCOPE96_STATS.get("fold_failed_fallback", 0) + 1
            _S96_DEPTH[0] += 1
            try:
                return _compile_for_hardware_c55(qc, **kwargs)
            finally:
                _S96_DEPTH[0] -= 1
    SCOPE96_STATS["handed"] += 1
    if how == "qiskit_working":''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
