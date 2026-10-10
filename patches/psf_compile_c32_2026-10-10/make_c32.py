"""Builds candidate 2026-10-10.c32 from candidate 2026-10-10.c31 by exact substitutions."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


sub("VERSION: 2026-10-10.c31 -- candidate (from release 2026-10-10.2): items 58 and 59a",
    "VERSION: 2026-10-10.c32 -- candidate (from candidate 2026-10-10.c31): items 58, 59a and 60")
sub('VERSION = "2026-10-10.c31"  # candidate: release 2026-10-10.2 + item 58 (failed elements avoided whenever the device is known, never returned silently) + item 59a (exactness margins recorded)',
    'VERSION = "2026-10-10.c32"  # candidate: c31 (release 2026-10-10.2 + items 58, 59a) + item 60 (the default call places by the device\'s errors when it has a Target)')
sub("""    where (`_same_action`, `_implements`), up to `EXACT_SEEN_MAX` entries. No decision changes.
\"\"\"""", """    where (`_same_action`, `_implements`), up to `EXACT_SEEN_MAX` entries. No decision changes.
60. **QUALITY ON THE DEVICE: with a Target, the default call places by the device's errors (candidate
    2026-10-10.c32).** With item 58 the default call avoids failed elements, but it chose among the working qubits
    without looking at their errors: on BP-FINAL's 106 FakeTorino tests its estimated success probability was 0.66
    (FakeTorino) and 0.84 (FakeKingston) of Qiskit level 2's (ESP-C31). `placement_refine` (item 33, Qiskit level 3's
    exact re-placement by the Target's errors) is now "auto" by default: on whenever a Target is given (by `target`
    or `backend`), off otherwise. `placement_refine=False` gives c31's default call with a Target; calls without a
    Target, and calls that pass `placement_refine` explicitly, are unchanged.
\"\"\"""")
sub("""    placement_refine: bool = False,""", """    placement_refine: Union[bool, str] = "auto",""")
sub("""    if placement_refine and target is None:
        raise ValueError("placement_refine=True needs the device `target` (changelog item 33)")""",
    """    if placement_refine == "auto":  # item 60: on with a Target, off without one
        placement_refine = target is not None
    if placement_refine and target is None:
        raise ValueError("placement_refine=True needs the device `target` (changelog item 33)")""")
sub("""    Without a Target, the first call in a process warns that they cannot be avoided.
    \"\"\"""", """    Without a Target, the first call in a process warns that they cannot be avoided.
    `placement_refine` (candidate 2026-10-10.c32, item 60): "auto" (default) is True when a Target is given and False
    otherwise.
    \"\"\"""")
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
