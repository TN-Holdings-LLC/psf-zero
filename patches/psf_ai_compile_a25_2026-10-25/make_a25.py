"""Builds psf_ai_compile a25 from a24 (run from the repository root):
    python patches/psf_ai_compile_a25_2026-10-25/make_a25.py

a25 = a24 + one item (owner, 2026-10-25: "a change of specification is fine"):
  49. The zipper also for odd n (TOQB results/zipper-odd-2026-10-25): with PSF_RING_ZIP=1, a ring of n >= 5 odd also
      gets the zipper schedule, by formula: the same start order; one CZ + SWAP block at the left end only; (n - 3)/2
      interior SWAPs; (5n - 7)/2 CZ per layer, the least found by search. Needs psfbind ring r3. Even n: unchanged.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "psf_ai_compile_a24_2026-10-22", "psf_ai_compile.py")
DST = os.path.join(HERE, "psf_ai_compile.py")


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


s = open(SRC, encoding="utf-8").read()

s = rep(s, '''"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, 2026-10-22.a24
(a23 + items 46-48:''', '''"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, 2026-10-25.a25
(a24 + item 49: the zipper schedule also for odd n -- see patches/psf_ai_compile_a25_2026-10-25/make_a25.py). a24's
description:
(a23 + items 46-48:''')

s = rep(s, '''      shrink, which at 16 layers is stronger than item 31's (results/lam-ablate-2026-10-22); not run on a device.
Placement is psf_compile's''', '''      shrink, which at 16 layers is stronger than item 31's (results/lam-ablate-2026-10-22); not run on a device.
  49. (a25) The zipper also for odd n (owner, 2026-10-25): with PSF_RING_ZIP=1, a ring of odd n >= 5 also gets the
      zipper, by formula: the same start order (0, 1, 2, n-1, n-2, 3, 4, ...); round A CZ on positions (1,2), ...,
      (n-2,n-1); round B CZ then SWAP on (0,1) only; round C SWAP on (2,3), ..., (n-3,n-2); round D as round A;
      (5n - 7)/2 CZ per layer once the CZ + SWAP is merged (9, 14, 19 at n = 5, 7, 9; N2's and item 31's counts are
      higher). Chosen as in item 48. Evidence (TOQB results/zipper-odd-2026-10-25, analysis): one layer equals the
      ring's CZ layer and a permutation, and forward then backward is the identity, for n = 4 to 41; (5n - 7)/2 is
      the least found by exhaustive search (every state for n = 5, 7, 9; every minimum-length start for n = 5-11);
      not run on a device. Even n: the rounds are a24's, unchanged. Needs psfbind ring r3 or later (item 46's check).
Placement is psf_compile's''')

s = rep(s, '''AI_COMPILE_VERSION = "2026-10-22.a24"  # items 46-48''', '''AI_COMPILE_VERSION = "2026-10-25.a25"  # item 49 (the zipper for odd n) + 2026-10-22.a24: items 46-48''')

s = rep(s, '''PSFBIND_RING_MIN = 2  # item 46: "psfbind ring r2" or later (r2 has the zipper)''',
        '''PSFBIND_RING_MIN = 3  # item 46: "psfbind ring r3" or later (r2 has the zipper, r3 also for odd n: item 49)''')

s = rep(s, '''    """Item 48: the zipper is a candidate only with PSF_RING_ZIP=1, for an even ring of n >= 4."""
    return os.environ.get("PSF_RING_ZIP", "") == "1" and n >= 4 and n % 2 == 0''',
        '''    """Items 48-49: the zipper is a candidate only with PSF_RING_ZIP=1, for a ring of n >= 4 (odd n: item 49)."""
    return os.environ.get("PSF_RING_ZIP", "") == "1" and n >= 4''')

s = rep(s, '''    if n < 4 or n % 2:
        raise RuntimeError("zipper: n must be even and >= 4")
    top = list(range(1, n))''', '''    if n < 4:
        raise RuntimeError("zipper: n must be >= 4")
    top = list(range(1, n))''')

s = rep(s, '''    odd = list(range(1, n - 2, 2))
    rounds.append(([take(p) for p in odd], []))
    ends = [0, n - 2]''', '''    odd = list(range(1, n - 1, 2))  # item 49: even n up to (n-3, n-2) as a24, odd n up to (n-2, n-1)
    rounds.append(([take(p) for p in odd], []))
    ends = [0, n - 2] if n % 2 == 0 else [0]  # item 49: odd n, one end block''')

s = rep(s, '''    mid = [(p, p + 1) for p in range(2, n - 3, 2)]
    if mid:
        rounds.append(([], mid))''', '''    mid = [(p, p + 1) for p in range(2, n - 3 + n % 2, 2)]  # item 49: odd n up to (n-3, n-2)
    if mid:
        rounds.append(([], mid))''')

s = rep(s, '''"psfbind ring r2" or later; RuntimeError otherwise (no fallback).''',
        '''"psfbind ring r3" or later (a25); RuntimeError otherwise (no fallback).''')

open(DST, "w", encoding="utf-8").write(s)
print("wrote", DST)
