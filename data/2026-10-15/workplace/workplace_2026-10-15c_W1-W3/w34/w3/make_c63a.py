"""make_c63a.py -- W3 measurement variant (workplace, 2026-10-15): c63 with item 88's frame fixed at build in the
template (option A). Not a proposal by itself; home decides between options A and B.
    python make_c63a.py <c63 psf_compile.py> <c63a_frame_code.py> <output psf_compile.py>"""
import sys

src, code, dst = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
old = 'VERSION = "2026-10-15.c63"  # workplace proposal:'
if s.count(old) != 1:
    raise SystemExit("version line not found")
s = s.replace(old, 'VERSION = "2026-10-15.c63a"  # W3 option A (frame fixed in the template) on c63; c63 =')
s = s.rstrip("\n") + "\n" + open(code, encoding="utf-8").read()
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
