"""make_c54.py -- step 2 of prototype 2026-10-14.c54 (step 1: make_c54_91.py, the workplace's item 91 on c53): item 93,
every output checked exact at a tight tolerance where it can be, with fallbacks, never an inexact circuit returned
silently.
    python make_c54.py <step-1 psf_compile.py> <c54_93_code.py> <output psf_compile.py>"""
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


sub('''def compile_for_hardware(qc, *args, **kwargs):
    """Items 87-88 (workplace proposal on prototype c51):''', '''def _compile_for_hardware_c53g(qc, *args, **kwargs):
    """Items 87-88 (workplace proposal on prototype c51):''')
sub('compile_for_hardware.__doc__ += _compile_for_hardware_c49.__doc__ or ""\n',
    '_compile_for_hardware_c53g.__doc__ += _compile_for_hardware_c49.__doc__ or ""\n')
i = s.index("91. **DURATION: an interaction graph that embeds in the device")
j = s.index('\n"""', i)
s = s[:j] + '''
93. **CORRECTNESS: every output checked exact where it can be, with fallbacks (prototype c54).** The workplace's Q3
    (Addendum 485) traced a total variation of 1.4e-5 in one AI-front-end output to Qiskit's two-qubit synthesis: its
    Weyl decomposition snaps a nearly degenerate block when the snap costs at most 1e-9 in gate fidelity, even at
    approximation_degree 1.0, here inside STEINER6's final level-3 transpile. Nothing checked whole outputs before.
    On the outermost call, the output is now checked (`_exact93`): a measured circuit by its exact outcome
    distribution against the input's own synthesis (largest probability difference at most EXACT_CHECK_TOL = 1e-6,
    the tolerance every validation since C48-VAL has used), an unmeasured one by item 39's check at the same
    tolerance against the input's own synthesis (a Pauli evolution's product formula, not its exact exponential); only with at most RESYNTH_MAX_QUBITS active qubits and final measurements, otherwise not checked
    (counted). An output that fails is replaced by the usual path's (`steiner=False, level3_candidate=False`) or else
    Qiskit's level-1 preset on the device (refused if it uses a failed element), whichever is first exact; if none is,
    TranspilerError is raised. At 1e-9 the check refused ordinary outputs: the same synthesis leaves differences of
    about 1e-7 (6.1e-7 on c48's 12-qubit TFIM ring on FakeTorino) in many outputs, which only a synthesis without the
    snap would remove; the largest difference accepted is recorded (`EXACT93_STATS["largest_passed"]`).
    `EXACT_CHECK = False` restores c53 + item 91. Counts in `EXACT93_STATS`.''' + s[j:]
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8").write(s)
print("written", dst)
