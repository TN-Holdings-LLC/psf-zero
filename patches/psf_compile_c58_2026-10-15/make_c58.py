"""make_c58.py -- prototype 2026-10-15.c58 from release 2026-10-14.1 (= c57) by exact substitutions: item 97, formula
only inside the scope (no other candidate, no comparison, no simulation check by default).
    python make_c58.py <release psf_compile.py> <c58_97_code.py> <output psf_compile.py>"""
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


sub("VERSION: 2026-10-14.1 -- release, adopted on 2026-10-14 from prototype 2026-10-14.c57, accepted in C57-VAL "
    "(Addendum 491) (previous release: 2026-10-11.4; items 75 revised, 77-80, 82-91, 93, 95-96 since)",
    "VERSION: 2026-10-15.c58 -- prototype (from release 2026-10-14.1 = c57: item 97 added; c57's items unchanged)")
sub('VERSION = "2026-10-14.1"  # release (from prototype 2026-10-14.c57, C57-VAL):',
    'VERSION = "2026-10-15.c58"  # prototype: item 97 (formula only inside the scope) +')
sub('''def compile_for_hardware(qc, *args, **kwargs):
    """Item 96 (prototype c56): the scope gate.''', '''def _compile_for_hardware_c57(qc, *args, **kwargs):
    """Item 96 (prototype c56): the scope gate.''')
sub('compile_for_hardware.__doc__ += _compile_for_hardware_c55.__doc__ or ""\n',
    '_compile_for_hardware_c57.__doc__ += _compile_for_hardware_c55.__doc__ or ""\n')
i = s.index("96. **SCOPE: a circuit PSF-Zero does not build by formula is handed to Qiskit's level 3 unchanged")
j = s.index('\n"""', i)
s = s[:j] + '''
97. **SPEED: formula only inside the scope (prototype c58).** Measured on 2026-10-15 (Addendum 493): inside the scope
    the release spent 0.6-37 s per call, most of it on item 93/95's state-vector checks (17-30 s at 20 qubits) and on
    candidates that lose, while item 87's or item 91's formula alone took 27-73 ms. With the device's Target and an
    input inside item 96's scope, the output of `_chain_brickwork` or else `_graph_brickwork` is returned as it is:
    nothing else is built or compared, and nothing is simulated (`FORMULA_CHECK = True` adds item 93's check). An
    in-scope input the formulas do not build is handed to Qiskit's level 3 as item 96 does. `FORMULA_ONLY = False`
    restores 2026-10-14.1. Counts in `FORMULA97_STATS`.''' + s[j:]
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
