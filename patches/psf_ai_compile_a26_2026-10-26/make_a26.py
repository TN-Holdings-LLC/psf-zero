"""Builds psf_ai_compile a26 from a25 (run from the repository root):
    python patches/psf_ai_compile_a26_2026-10-26/make_a26.py

a26 = a25 + one item (owner, 2026-10-26: "release it, in the direction of a regular release"):
  50. The ring folded onto one of the device's own cycles, the default (PSF_RING_FOLD=0 gives a25's output). A ring
      classifier that does not embed is laid on a heavy-hex cycle (12, 20, 24, 28, ... qubits, the outline of a fixed
      shape of cells read off the coupling graph), with its extra nodes hung off the cycle ("pendants", n + (n - m)
      two-qubit gates per layer) or the cycle's spare qubits moved as |0> "holes" (n + 2 (m - n)); of those and the
      zipper the fewest extra gates decides. Ported from TOQB toqb/fold.py (fold:rule, 145f38b).
The new code is a26_section.py, appended to a25's text; the edits below change only a25's docstring and version.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "psf_ai_compile_a25_2026-10-25", "psf_ai_compile.py")
DST = os.path.join(HERE, "psf_ai_compile.py")


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


s = open(SRC, encoding="utf-8").read()

s = rep(s, '''"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, 2026-10-25.a25
(a24 + item 49:''', '''"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, 2026-10-26.a26
(a25 + item 50: the ring folded onto a device cycle, the default -- see patches/psf_ai_compile_a26_2026-10-26/
make_a26.py). a25's description:
(a24 + item 49:''')

s = rep(s, '''      not run on a device. Even n: the rounds are a24's, unchanged. Needs psfbind ring r3 or later (item 46's check).
Placement is psf_compile's''', '''      not run on a device. Even n: the rounds are a24's, unchanged. Needs psfbind ring r3 or later (item 46's check).
  50. (a26) The ring folded onto one of the device's own cycles, the default (owner, 2026-10-26; PSF_RING_FOLD=0
      gives a25's output). Heavy-hex has cycles of 12, 20, 24, 28, ... qubits and none between (TOQB
      results/fold-holes-2026-10-25). A ring classifier (item 28's reading, any one cycle through the qubits) that
      does not embed (item 42) is laid on the outline of a fixed shape of cells, read off the coupling graph by
      formula (bridges, rows, cells; the first working shape in a fixed order; no search, no error value decides;
      a coupler or qubit at error >= FAILED_MAX_ERROR is never used, every listed direction of a coupler must work):
      - pendants: the longest such cycle of m <= n qubits, the n - m extra ring nodes hung off cycle nodes that have
        a working outside neighbour (spread in cycle order); each layer every pendant node trades places with its
        cycle node by one CZ + SWAP block: n + (n - m) two-qubit gates per layer, on exactly n qubits;
      - holes: the shortest such cycle of m >= n qubits, its m - n spare qubits starting in |0> and moved across by
        two CX each: n + 2 (m - n) per layer;
      - the rule: of pendants n - m, holes 2 (m - n) and the zipper's extra gates per layer, the fewest decides
        (ties: pendants, then holes); where the zipper's are fewest (small n), or no cycle fits, a25's path.
      Built once into a FoldTemplate (Qiskit's level-1 preset on the fixed layout, no routing, as item 28's
      translation), angles bound per call; the default call binds the same template. Evidence (TOQB, models of a
      2026-10-13 calibration of ibm_marrakesh, not a device): flow E (pendants against holes and the zipper, n = 13-
      18, L = 8, 16) and flow F, a declared match (the rule against the zipper, n = 13-18, L = 16, four random models:
      WIN at every n, 24 of 24 models; the rule's lam 0.51-0.76, the zipper's 0.09-0.46). Not run on a device.
Placement is psf_compile's''')

s = rep(s, '''AI_COMPILE_VERSION = "2026-10-25.a25"  # item 49''', '''AI_COMPILE_VERSION = "2026-10-26.a26"  # item 50 (the ring folded onto a device cycle, default) + 2026-10-25.a25: item 49''')

s += open(os.path.join(HERE, "a26_section.py"), encoding="utf-8").read()
open(DST, "w", encoding="utf-8").write(s)
print("wrote", DST)
