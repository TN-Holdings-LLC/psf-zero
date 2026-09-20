"""apply_readme.py -- task 5: replace the repository's README.md with the
corrected version, safely.

The current README's "Update (2026-09-19)" block overstated what was
established ("definitively isolated," "the solution is a single flag,"
"PSF-Zero's own layout search sidesteps... completely") -- each claim
contradicted by the project's own addenda 84-87. The corrected block states
the same real results without claiming more certainty than the record
supports, and adds the four verified fixes (addenda 88-95) that followed.

Steps: verify the release file's SHA256 -> back up the current README.md
-> install. No smoke check is possible for prose, so the hash check is the
only gate; it refuses to install anything but the exact reviewed file.

Usage:
    python apply_readme.py
"""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

EXPECTED_SHA256 = "0d84fcf722a9ea9ec57e56348a38272af75bc1a704a825ccc221e0858afb7f73"
RELEASE = Path("README_release.md")
TARGET = Path("README.md")


def main() -> int:
    if not RELEASE.exists():
        print(f"ERROR: {RELEASE} not found in the current directory.")
        return 1
    got = hashlib.sha256(RELEASE.read_bytes()).hexdigest()
    if got != EXPECTED_SHA256:
        print("ERROR: release file hash mismatch -- refusing to install.")
        print(f"  expected {EXPECTED_SHA256}")
        print(f"  got      {got}")
        return 1
    print(f"OK: {RELEASE} hash matches the reviewed release.")

    if TARGET.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = Path(f"README.BACKUP-{stamp}.md")
        shutil.copy2(TARGET, backup)
        print(f"OK: backed up existing {TARGET} -> {backup}")
    else:
        print(f"NOTE: no existing {TARGET}; nothing to back up.")

    shutil.copy2(RELEASE, TARGET)
    print(f"DONE: installed {RELEASE} as {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
