"""make_c68.py -- prototype c68 = release 2026-10-17.1 + item 105 (chains and rings by psfz r4).

    python patches/psf_compile_c68_2026-10-18/make_c68.py <release psf_compile.py> <item105_psfz_code.py> <output>
"""
import re
import sys

src, code, dst = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
m = re.search(r'^VERSION = "2026-10-17\.1"  # .*$', s, flags=re.M)
if not m:
    raise SystemExit("STOP: version line not found")
s = s[:m.start()] + 'VERSION = "2026-10-18.c68"  # prototype: item 105 (chains and rings by psfz r4) on release 2026-10-17.1' + s[m.end():]
s = s.rstrip("\n") + "\n" + open(code, encoding="utf-8").read()
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
