"""make_c55_94.py -- step 1 of prototype 2026-10-14.c55 (the workplace's make_c54_94.py with the version strings
changed; its code is unchanged): the workplace's item 94 on prototype 2026-10-14.c54 by exact substitutions:
runs of commuting ZZ terms re-layered by a first-fit edge colouring before the compile, the rewrite's output kept when
it is shorter (item 84) and exact (item 93). The code is appended; c54's wrapper is renamed `_compile_for_hardware_c54`.
    python make_c55_94.py <c54 psf_compile.py> <c55_94_code.py> <output psf_compile.py>"""
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


sub("VERSION: 2026-10-14.c54 -- prototype (from c53 + the workplace's item 91 + item 93;",
    "VERSION: 2026-10-14.c55 -- prototype (from c54 + the workplace's item 94 + item 95;")
sub('VERSION = "2026-10-14.c54"  # prototype:',
    'VERSION = "2026-10-14.c55"  # prototype:')
sub('''    `EXACT_CHECK = False` restores c53 + item 91. Counts in `EXACT93_STATS`.
"""''', '''    `EXACT_CHECK = False` restores c53 + item 91. Counts in `EXACT93_STATS`.
94. **DURATION: runs of commuting ZZ terms re-layered by edge colouring (workplace proposal R1, on c54).** The family
    map (Addendum 484) found the user's best ahead on 3-regular QAOA, always from the hand-layered circuit. The
    workplace's R1 separated the two effects with one fixed seed: the hand-layered circuit through Qiskit level 3 at
    seed 0 alone beat c54 on qaoa_n12_p2 (5.53 against 5.98 us, F 4.07 against 4.64) but not on qaoa_n20_p2 or
    qaoa3reg_n20_r3 (6.48 / 13.38 against 6.31 / 11.08), where the user's margin comes from trying seeds (placement).
    No fixed placement tried by formula came near (a first Hamiltonian path on item 87's line, reverse Cuthill-McKee
    on the line or on a compact patch: 1.5-3 times c54). On the outermost call, the input's maximal runs of pairwise
    commuting ZZ terms (items 72-73's reader, in their order) are regrouped into layers of disjoint pairs by a
    first-fit edge colouring in that order, written as rzz (other terms unchanged, as their own PauliEvolutionGate):
    exact by construction (only commuting terms move). c54 compiles the rewrite too, and its output is returned when
    item 84's rule prefers it and item 93's check against the input does not refuse it. No seed, placement or score
    is searched: two fixed inputs, chosen by duration. `ZZ_RELAYER = False` restores c54. Counts in `ZZ94_STATS`.
"""''')
sub('''def compile_for_hardware(qc, *args, **kwargs):
    """Item 93 (prototype c54):''', '''def _compile_for_hardware_c54(qc, *args, **kwargs):
    """Item 93 (prototype c54):''')
sub('compile_for_hardware.__doc__ += _compile_for_hardware_c53g.__doc__ or ""\n',
    '_compile_for_hardware_c54.__doc__ += _compile_for_hardware_c53g.__doc__ or ""\n')
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
