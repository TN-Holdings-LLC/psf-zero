"""make_c56.py -- prototype 2026-10-14.c56 from c55 by exact substitutions (home, after the owner's scope declaration
of 2026-10-14, Addendum 488): item 94 off by default (dropped); item 96, the scope gate -- a circuit whose two-qubit
interaction graph does not embed in the device's working couplers is handed to Qiskit's level 3 unchanged.
    python make_c56.py <c55 psf_compile.py> <c56_96_code.py> <output psf_compile.py>"""
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


sub("VERSION: 2026-10-14.c55 -- prototype (from c54 + the workplace's item 94 + item 95; c43 = release 2026-10-11.4): "
    "items 75 revised, 77-80, 82-91, 93-95",
    "VERSION: 2026-10-14.c56 -- prototype (from c55: item 94 off, item 96 added; c43 = release 2026-10-11.4): "
    "items 75 revised, 77-80, 82-91, 93, 95-96")
sub('VERSION = "2026-10-14.c55"  # prototype: item 95',
    'VERSION = "2026-10-14.c56"  # prototype: item 96 (scope gate: a circuit that does not embed in the device handed to '
    'Qiskit level 3 unchanged) + item 94 off (dropped) + item 95')
sub("ZZ_RELAYER = True  # item 94 (proposal)\n", "ZZ_RELAYER = False  # item 94 (proposal); off since c56: dropped (Addendum 488)\n")
sub('''def compile_for_hardware(qc, *args, **kwargs):
    """Item 94 (workplace proposal on c54):''', '''def _compile_for_hardware_c55(qc, *args, **kwargs):
    """Item 94 (workplace proposal on c54; off by default since c56):''')
sub('compile_for_hardware.__doc__ += _compile_for_hardware_c54.__doc__ or ""\n',
    '_compile_for_hardware_c55.__doc__ += _compile_for_hardware_c54.__doc__ or ""\n')
sub("94. **DURATION: runs of commuting ZZ terms re-layered by edge colouring (workplace proposal R1, on c54).**",
    "94. **DURATION: runs of commuting ZZ terms re-layered by edge colouring (workplace proposal R1, on c54; off by "
    "default since c56, dropped: Addendum 488).**")
i = s.index("95. **CORRECTNESS: item 93's distribution check up to 20 active qubits (prototype c55).**")
j = s.index('\n"""', i)
s = s[:j] + '''
96. **SCOPE: a circuit PSF-Zero does not build by formula is handed to Qiskit's level 3 unchanged (prototype c56).**
    On real devices (HW1, HW2, TIME1, FRAC1; 54 pairs pooled after the fact, Addendum 488) PSF-Zero's construction
    won 19 of 23 pairs on Ising chains (F 1.21 x) but its compiler choices alone won 9 of 25 on QAOA and
    Barabasi-Albert circuits (F 0.78 x); TIME1 (Addendum 487) traced the losses to placement where routing is needed,
    which needs search or calibration, which PSF-Zero does not use by design. The owner declared the scope the same
    day: PSF-Zero competes only where it builds by formula. With the device's Target, the input's two-qubit
    interaction graph -- the terms' supports when items 72-73's reader reads it as Pauli terms on one or two qubits
    (item 91's `_graph_read`), else every two-qubit operation as written; an operation on more than two qubits is out
    of scope -- is tested for an embedding in the device's working coupling graph (failed couplers and qubits
    removed; rustworkx's first VF2 match, nodes in index order, no score; not found within 2,000,000 states counts as
    none). Embeds, or no two-qubit operation: the call is c55's (item 94 off). Otherwise the input goes to Qiskit's
    `generate_preset_pass_manager(3, target, seed_transpiler)` unchanged (seed 0 when none is given, so the output is
    reproducible); if that output uses a failed element, Qiskit's level 3 on the device without its failed qubits
    (item 85's working Target), written back on the device's indices; only if that fails too, c55's own path
    (counted). The handed-on output is checked by item 93's test where it can be and returned with a warning if
    inexact. Arguments such as `steiner=` or `level3_candidate=` do not apply to a handed-on circuit. Without a
    Target nothing changes. `SCOPE_GATE = False` restores c55 with item 94 off. Counts in `SCOPE96_STATS`.''' + s[j:]
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
