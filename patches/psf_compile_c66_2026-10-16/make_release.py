"""make_release.py -- release 2026-10-16.2 = prototype c66 (item 103), with its prototype switch removed.

    python patches/psf_compile_c66_2026-10-16/make_release.py <c66 psf_compile.py> <output>

Three changes to c66's text, nothing else: the module docstring, the version line, and the switch block at the end
(`TEMPLATE_FRAME_SEGMENTS`, which C66-VAL ran at True) replaced by the assignment it made. Option A's
`_T100Formula._frame_items` stays defined but is no longer called.
"""
import re
import sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()

DOC = '''"""PSF-Zero compiler -- VERSION: 2026-10-16.2 -- release, adopted on 2026-10-17 from prototype 2026-10-16.c66
(previous release: 2026-10-16.1, kept in patches/psf_compile_release_2026-10-16.1/).

Two entry points: `compile_for_hardware(qc, target=...)` and `compile_template(qc, target)`. A circuit whose
two-qubit interaction graph embeds in the device's working couplers is built by formula (items 87-101), placed by the
first VF2 match with no score; every other circuit, and every call without the device's Target, is refused with
`OutOfScopeError`: nothing is handed to a search (item 98). Item 96's embedding test decides by formula before VF2
(item 102). In a template on a fractional-gate target, item 88's frame is applied by segments, solved over GF(2)
(item 103): the template's shape never changes with the values, without option A's extra pulses. 2026-10-16.2 is
2026-10-16.1 plus item 103; validated in C66-VAL (Addendum 513): its default calls are 2026-10-16.1's bit for bit.

The changelog of items 1-103 is docs/psf_compile_CHANGELOG.md. The functions of earlier releases that the default
calls never used (`compile()`, `prune_coupling_map`, `edge_errors_from_target`, the cost models, ...) are not in this
release; they remain in patches/psf_compile_release_2026-10-14.1/psf_compile.py."""'''

if not s.startswith('"""PSF-Zero compiler -- VERSION: 2026-10-16.1'):
    raise SystemExit("STOP: docstring not found")
s = DOC + s[s.index('"""', 3) + 3:]

m = re.search(r'^VERSION = "2026-10-16\.c66"  # .*$', s, flags=re.M)
if not m:
    raise SystemExit("STOP: version line not found")
s = (s[:m.start()] + 'VERSION = "2026-10-16.2"  # release, adopted on 2026-10-17 from prototype 2026-10-16.c66 '
     '(item 103 on release 2026-10-16.1)' + s[m.end():])

SWITCH = '''TEMPLATE_FRAME_SEGMENTS = True  # item 103 (prototype switch for the comparison with option A; removed on release)
_t103_option_a = _T100Formula._frame_items


def _t103_items(self, values):
    if TEMPLATE_FRAME_SEGMENTS:
        return _t103_frame_items(self, values)
    return _t103_option_a(self, values)


_T100Formula._frame_items = _t103_items
'''
if s.count(SWITCH) != 1 or not s.endswith(SWITCH):
    raise SystemExit("STOP: switch block not found at the end")
s = s[:-len(SWITCH)] + "_T100Formula._frame_items = _t103_frame_items  # item 103 (release 2026-10-16.2)\n"
s = s.replace("# Item 103 (prototype c66, home, 2026-10-16): ", "# Item 103 (prototype c66, home, 2026-10-16; release 2026-10-16.2): ")
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
