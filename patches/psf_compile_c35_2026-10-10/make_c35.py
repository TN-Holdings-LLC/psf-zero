"""Builds candidate 2026-10-10.c35 from candidate 2026-10-10.c34 by exact substitutions."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
for old, new in [
    ("VERSION: 2026-10-10.c34 -- candidate (from candidate 2026-10-10.c33): items 58, 59a, 60, 61 and 62",
     "VERSION: 2026-10-10.c35 -- candidate (from candidate 2026-10-10.c34): items 58, 59a, 60, 61, 62 and 63"),
    ('VERSION = "2026-10-10.c34"  # candidate: c33 (release 2026-10-10.2 + items 58, 59a, 60, 61) + item 62 (speed: _uses_failed; switches to measure what the compression and the SWAP absorption are worth)',
     'VERSION = "2026-10-10.c35"  # candidate: c34 (release 2026-10-10.2 + items 58, 59a, 60-62) + item 63 (the SWAP absorption synthesised by Qiskit\'s Rust decomposer)'),
    ("""    (the size of a maximum matching does not depend on the algorithm), in patches/psf_compile_c34_2026-10-10.
\"\"\"""", """    (the size of a maximum matching does not depend on the algorithm), in patches/psf_compile_c34_2026-10-10.
63. **SPEED: the SWAP absorption's blocks synthesised by Qiskit's decomposer (candidate 2026-10-10.c35).**
    `ABSORB_SYNTH` is "qiskit" by default. On BP-FINAL's 106 FakeTorino tests (ABLATE-C34, arm C34Q) every output had
    c34's two-qubit count, ESP was 1.005 of c34's, and the summed time 0.89 of it; the outputs differ in their
    single-qubit gates only. Both of PSF-Zero's own steps paid for themselves there: without the absorption 10% more
    two-qubit gates (ESP 0.918), without the compression 11% more (ESP 0.956). "psf" restores c34's synthesis.
\"\"\""""),
    ('''ABSORB_SYNTH = "psf"  # item 62: "psf" (default) or "qiskit": how item 30's absorbed blocks are synthesised''',
     '''ABSORB_SYNTH = "qiskit"  # items 62-63: "qiskit" (default) or "psf": how item 30's absorbed blocks are synthesised'''),
]:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
