"""Builds candidate 2026-10-10.c36 from release 2026-10-10.3 by exact substitutions (changelog item 64: routing at
Qiskit's level 3 whenever the device's Target is given)."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
for old, new in [
    ("VERSION: 2026-10-10.3 -- release, adopted on 2026-10-10 from candidate 2026-10-10.c35 of Addenda 431-432, "
     "accepted in C35-VAL (Addenda 436-437) (previous release: 2026-10-10.2)",
     "VERSION: 2026-10-10.c36 -- candidate (from release 2026-10-10.3): item 64"),
    ('VERSION = "2026-10-10.3"  # release (from candidate 2026-10-10.c35, C35-VAL): 2026-10-10.2 + items 58, 59a, '
     '60-63 (the device given by backend= or target=: no failed element used; placement by the errors; speed)',
     'VERSION = "2026-10-10.c36"  # candidate: release 2026-10-10.3 + item 64 (routing at level 3 when the device is '
     'given)'),
    ('''    single-qubit gates only. Both of PSF-Zero's own steps paid for themselves there: without the absorption 10% more
    two-qubit gates (ESP 0.918), without the compression 11% more (ESP 0.956). "psf" restores c34's synthesis.
"""''', '''    single-qubit gates only. Both of PSF-Zero's own steps paid for themselves there: without the absorption 10% more
    two-qubit gates (ESP 0.918), without the compression 11% more (ESP 0.956). "psf" restores c34's synthesis.
64. **ACCURACY: routing at level 3 when the device is given (candidate 2026-10-10.c36).** `routing_optimization_level`
    is "auto": 3 when a Target is given (`backend=` or `target=`), 1 otherwise. In ROUTE-X (Addendum 439; release
    2026-10-10.3's plain call with backend= on BP-FINAL's 106 FakeTorino tests) level 3 raised ESP by 5% (FakeTorino)
    and 10% (FakeKingston) over level 1, to 0.975 and 0.985 of Qiskit level 3's, at 1.15-1.2 times the time. Without
    a Target nothing changes. An explicit integer keeps its meaning.
"""'''),
    ("    routing_optimization_level: int = 1,\n",
     '    routing_optimization_level: Union[int, str] = "auto",\n'),
    ("    `routing_optimization_level` defaults to 1. It used to default to 2, on the",
     '    `routing_optimization_level` defaults to "auto" (item 64): 3 when the device\'s Target is given, 1\n'
     "    otherwise; what follows is about the call without a Target. It defaulted to 1, and earlier to 2, on the"),
    ('''    if on_failed_elements not in ("raise", "keep"):''',
     '''    if routing_optimization_level == "auto":  # item 64: level 3 with the device's Target, level 1 without
        routing_optimization_level = 3 if target is not None else 1
    if on_failed_elements not in ("raise", "keep"):'''),
]:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
