"""make_release.py -- release 2026-10-18.1 = prototype c68 (item 105, chains and rings by psfz r4).

    python patches/psf_compile_c68_2026-10-18/make_release.py <c68 psf_compile.py> <output>

Three changes to c68's text, nothing else: the module docstring, the version line, and item 105's heading comment.
"""
import re
import sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()

DOC = '''"""PSF-Zero compiler -- VERSION: 2026-10-18.1 -- release, adopted on 2026-10-18 from prototype 2026-10-18.c68
(previous release: 2026-10-17.1, kept in patches/psf_compile_release_2026-10-17.1/).

Two entry points: `compile_for_hardware(qc, target=...)` and `compile_template(qc, target)`. A circuit whose
two-qubit interaction graph embeds in the device's working couplers is built by formula (items 87-101), placed by the
first VF2 match with no score; every other circuit, and every call without the device's Target, is refused with
`OutOfScopeError`: nothing is handed to a search (item 98). Item 96's embedding test decides by formula before VF2
(item 102). In a template on a fractional-gate target, item 88's frame is applied by segments, solved over GF(2)
(item 103). On a device whose two-qubit gate is ECR (IBM Eagle), the default call builds on a CZ view of the device and
writes every CZ as the coupler's ECR (item 104); templates on ECR devices are refused. The default call first hands
the circuit to psfz r4, PSF-Zero's compiler in Rust (rust/psfz/, a separate program), and returns its output when psfz
compiles it -- chains and even rings on heavy-hex CZ and ECR devices (item 105); without the program, or when psfz
refuses, the call is 2026-10-17.1's. 2026-10-18.1 is 2026-10-17.1 plus item 105; validated in C68-VAL (Addendum 524),
psfz r4 in PSFZ-VAL2 (Addendum 522).

The changelog of items 1-105 is docs/psf_compile_CHANGELOG.md. The functions of earlier releases that the default
calls never used (`compile()`, `prune_coupling_map`, `edge_errors_from_target`, the cost models, ...) are not in this
release; they remain in patches/psf_compile_release_2026-10-14.1/psf_compile.py."""'''

if not s.startswith('"""PSF-Zero compiler -- VERSION: 2026-10-17.1'):
    raise SystemExit("STOP: docstring not found")
s = DOC + s[s.index('"""', 3) + 3:]
m = re.search(r'^VERSION = "2026-10-18\.c68"  # .*$', s, flags=re.M)
if not m:
    raise SystemExit("STOP: version line not found")
s = (s[:m.start()] + 'VERSION = "2026-10-18.1"  # release, adopted on 2026-10-18 from prototype 2026-10-18.c68 '
     '(item 105 on release 2026-10-17.1)' + s[m.end():])
old = "# Item 105 (prototype c68, 2026-10-18): "
if s.count(old) != 1:
    raise SystemExit("STOP: item 105's heading not found once")
s = s.replace(old, "# Item 105 (prototype c68, 2026-10-18; release 2026-10-18.1): ")
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
