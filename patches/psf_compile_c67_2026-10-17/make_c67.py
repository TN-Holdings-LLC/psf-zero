"""make_c67.py -- prototype c67 = release 2026-10-16.2 (c66) + item 104 (ECR devices by a CZ view), with option A's
`_T100Formula._frame_items` and its constant `_T63A_I` removed (unreachable since 2026-10-16.2, Addendum 514).

    python patches/psf_compile_c67_2026-10-17/make_c67.py <release psf_compile.py> <item104_ecr_code.py> <output>
"""
import re
import sys

src, code, dst = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
m = re.search(r'^VERSION = "2026-10-16\.2"  # .*$', s, flags=re.M)
if not m:
    raise SystemExit("STOP: version line not found")
s = s[:m.start()] + 'VERSION = "2026-10-17.c67"  # prototype: item 104 (ECR devices by a CZ view) on release 2026-10-16.2' + s[m.end():]
start = s.index('    def _frame_items(self, values):\n        """W3 option A (c63a)')
end = s.index("    # -- the CZ path: PSF-Zero's own blocks", start)
if s.count('    def _frame_items(self, values):\n') != 1:
    raise SystemExit("STOP: option A's _frame_items not found once")
s = s[:start] + s[end:]
line = "_T63A_I = (1 + 0j, 0j, 0j, 1 + 0j)  # W3 option A (c63a): the identity, copied into new tuples by _frame_items\n"
if s.count(line) != 1 or "_T63A_I" in s.replace(line, ""):
    raise SystemExit("STOP: _T63A_I not found once, or still used")
s = s.replace(line, "")
s = s.rstrip("\n") + "\n" + open(code, encoding="utf-8").read()
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
