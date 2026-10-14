"""make_c55.py -- step 2 of prototype 2026-10-14.c55 (step 1: make_c55_94.py, the workplace's item 94 on c54): item 95,
item 93's distribution check extended from RESYNTH_MAX_QUBITS (16) to EXACT_CHECK_MAX_QUBITS = 20 active qubits.
    python make_c55.py <step-1 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub("items 75 revised, 77-80, 82-91, 93\n", "items 75 revised, 77-80, 82-91, 93-95\n")
sub('VERSION = "2026-10-14.c55"  # prototype: item 93',
    'VERSION = "2026-10-14.c55"  # prototype: item 95 (item 93\'s check up to 20 active qubits) + item 94 (workplace: '
    'commuting ZZ runs re-layered by edge colouring) + item 93')
i = s.index("94. **DURATION: runs of commuting ZZ terms re-layered by edge colouring")
j = s.index('\n"""', i)
s = s[:j] + '''
95. **CORRECTNESS: item 93's distribution check up to 20 active qubits (prototype c55).** The workplace's R2
    (Addendum 486) scanned C52-VAL's 240 cells on c54: every output is within 5.1e-14 of its input's distribution
    except tfim_n20_r3's six (1.5e-8 on the Target without fractional gates on every device, 3.2e-7 with them on
    FakeTorino), left by Qiskit's two-qubit synthesis and CommutativeCancellation inside STEINER6's final level-3
    transpile. They have 20 active qubits, so item 93, stopping at RESYNTH_MAX_QUBITS = 16, never saw them. A
    measured circuit's check now goes up to EXACT_CHECK_MAX_QUBITS = 20 (a statevector of 16 MB); an unmeasured one
    stays at RESYNTH_MAX_QUBITS (it compares unitaries). The tolerance stays 1e-6: at 1e-9 these six outputs would
    fall back to the usual path, 17-21 times longer (R2), and the exact synthesis that removes them costs about 10%
    and is not built into the compiler yet. `EXACT_CHECK_MAX_QUBITS = RESYNTH_MAX_QUBITS` restores c54's reach.''' + s[j:]
sub('''def _dist93(circ, limit=RESYNTH_MAX_QUBITS):''', '''EXACT_CHECK_MAX_QUBITS = 20  # item 95: active qubits up to which item 93 checks a measured circuit


def _dist93(circ, limit=None):''')
sub('''    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    if not meas or len(active) > limit:''', '''    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    if limit is None:  # item 95
        limit = EXACT_CHECK_MAX_QUBITS
    if not meas or len(active) > limit:''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
