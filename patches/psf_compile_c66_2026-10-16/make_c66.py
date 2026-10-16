"""make_c66.py -- prototype c66 = release 2026-10-16.1 (c65) + item 103 (item 88's frame in a template by segments).

    python patches/psf_compile_c66_2026-10-16/make_c66.py <release psf_compile.py> <item103_frame_code.py> <output>
"""
import re
import sys

src, code, dst = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
m = re.search(r'^VERSION = "2026-10-16\.1"  # .*$', s, flags=re.M)
if not m:
    raise SystemExit("STOP: version line not found")
s = s[:m.start()] + 'VERSION = "2026-10-16.c66"  # prototype: item 103 (frame by segments in templates) on release 2026-10-16.1' + s[m.end():]
s = s.rstrip("\n") + "\n" + open(code, encoding="utf-8").read()
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
