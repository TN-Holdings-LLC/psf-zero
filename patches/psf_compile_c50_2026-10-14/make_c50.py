"""Builds prototype 2026-10-14.c50 from prototype 2026-10-14.c49 by exact substitutions:
  - item 86: with a Target that offers IBM Heron's native fractional gates (rzz, rx), the finished circuit is
    re-synthesised on its own physical qubits with them, and kept when it is shorter (item 84) and exact.
    python make_c50.py <c49 psf_compile.py> <c50_code.py> <output psf_compile.py>"""
import sys

src, code, dst = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
c50 = open(code, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"expected {count} occurrence(s), found {n}: {old[:80]!r}")
    s = s.replace(old, new)


sub("VERSION: 2026-10-14.c49 -- prototype (from c48; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82-85",
    "VERSION: 2026-10-14.c50 -- prototype (from c49; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82-86")
sub('VERSION = "2026-10-14.c49"  # prototype: items 84',
    'VERSION = "2026-10-14.c50"  # prototype: item 86 (native fractional gates) + items 84')
sub('''    is compiled exactly as by c48. `CUT_FAILED_QUBITS = False` restores c48. Counts in `CUT_STATS`.
"""''', '''    is compiled exactly as by c48. `CUT_FAILED_QUBITS = False` restores c48. Counts in `CUT_STATS`.
86. **DURATION: the device's native fractional gates (prototype c50).** IBM's Heron devices offer rzz(theta), a ZZ
    rotation in one two-qubit pulse of a CZ's length (68 ns on ibm_marrakesh, 2026-10-13), and rx(theta) in one
    single-qubit pulse (36 ns), when the backend is loaded with `use_fractional_gates=True`. A ZZ rotation built from
    CZ gates takes two CZ and the single-qubit gates between them. When the Target offers rzz (`FRACTIONAL = "auto"`),
    the finished circuit is re-synthesised on its own physical qubits and layout: two-qubit blocks consolidated and
    re-synthesised (Qiskit's ConsolidateBlocks and UnitarySynthesis on a copy of the Target with every error removed,
    so no reported error chooses), single-qubit runs merged, rzz angles folded into the executable range
    0 < theta <= pi/2 (qiskit-ibm-runtime's FoldRzzAngle). The result is kept when it is shorter (item 84) and, where
    item 39's check can be made (at most RESYNTH_MAX_QUBITS touched qubits), acts as the circuit before it; above that
    size it is kept unverified (counted), as item 80 keeps Qiskit's circuit. The re-synthesis runs once, on the
    outermost call (`compile_for_hardware` wraps c49's function, now `_compile_for_hardware_c49`). Without rzz in the
    Target, or with `FRACTIONAL = False`, the output is c49's. Counts in `FRAC_STATS`.
"""''')
sub("def compile_for_hardware(\n    qc: QuantumCircuit,", "def _compile_for_hardware_c49(\n    qc: QuantumCircuit,")
s = s.rstrip("\n") + "\n" + c50
open(dst, "w", encoding="utf-8").write(s)
print("written", dst)
