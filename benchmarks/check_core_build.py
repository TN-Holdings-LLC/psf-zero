"""check_core_build.py -- task 4: is the compiled `psf_zero_core` binary
actually built from the current `lib.rs`?

Every source-level comparison this session (addenda 96-99) read `lib.rs`.
What runs is the compiled extension built from it. If the binary predates
the source, a source diff says nothing about runtime behaviour. This
script checks that, and says exactly what to do if it's stale.

Checks:
  1. Locate the imported `psf_zero_core` extension file (.pyd on Windows,
     .so elsewhere) via its `__file__`.
  2. Compare its modification time to `lib.rs`. Binary older than source
     means STALE -- rebuild required.
  3. Confirm the expected functions are exported (`geometric_decompose`,
     and `geometric_decompose_checked`, which the current `psf_compile.py`
     uses for `verify=True`). A missing `_checked` export means the binary
     is from an older `lib.rs` regardless of timestamps.

Usage:
    python check_core_build.py
If it reports STALE or a missing export, run:
    maturin develop --release
then run this script again.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> int:
    src = Path("lib.rs")
    if not src.exists():
        # maturin projects usually keep it under src/
        alt = Path("src") / "lib.rs"
        if alt.exists():
            src = alt
        else:
            print("ERROR: lib.rs not found (looked in ./ and ./src/).")
            return 1

    try:
        import psf_zero_core
    except ImportError as exc:
        print(f"ERROR: cannot import psf_zero_core: {exc!r}")
        print("  -> run: maturin develop --release")
        return 1

    binary = Path(psf_zero_core.__file__)
    src_mtime = src.stat().st_mtime
    bin_mtime = binary.stat().st_mtime
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))

    print(f"source : {src}  (modified {fmt(src_mtime)})")
    print(f"binary : {binary}  (modified {fmt(bin_mtime)})")

    stale = bin_mtime < src_mtime
    has_plain = hasattr(psf_zero_core, "geometric_decompose")
    has_checked = hasattr(psf_zero_core, "geometric_decompose_checked")
    print(f"exports geometric_decompose         : {has_plain}")
    print(f"exports geometric_decompose_checked : {has_checked}")

    if stale:
        print("\nRESULT: STALE -- the binary is OLDER than lib.rs.")
        print("  Every source-level finding about lib.rs may not reflect what")
        print("  actually ran. Rebuild, then re-run this check:")
        print("      maturin develop --release")
        return 2
    if not (has_plain and has_checked):
        print("\nRESULT: MISSING EXPORT -- the binary was built from an older lib.rs")
        print("  that lacks a function the current psf_compile.py calls.")
        print("      maturin develop --release")
        return 2

    print("\nRESULT: OK -- binary is newer than lib.rs and exports both functions.")
    print("  Source-level findings about lib.rs apply to the running binary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
