"""Replace benchmarks/psf_smart_layout.py with the candidate 2026-09-29.c1 (corrected
feasibility check + short-path shortcut) and add benchmarks/test_short_path_layout.py,
only if the current module is exactly the release 2026-09-26.m1 (normalized SHA-256
a639efde...d875).

    python apply_layout_update_2026-09-29.py <this folder> <repository root>

The same change is in psf_smart_layout_c1_2026-09-29.patch (git diff against f4b4a6c).
After replacing: run the six release test files and test_short_path_layout.py
(83 + 10 tests). psf_compile.py and the Rust core are not touched.
"""
import hashlib
import os
import shutil
import sys

BASE = "a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875"
NEW = "e25952a33bacfbb8de7a439b53892e9bab88329d8c47c6de867c29fc660145fa"
TEST = "3ce3bdf1dca71e1e6af7265c76b0ed2f11c40aec9ceccf33b56c6fd73abf84a1"


def norm_sha(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


here, repo = sys.argv[1], sys.argv[2]
target = os.path.join(repo, "benchmarks", "psf_smart_layout.py")
new = os.path.join(here, "psf_smart_layout.py")
test_src = os.path.join(here, "test_short_path_layout.py")
test_dst = os.path.join(repo, "benchmarks", "test_short_path_layout.py")
if norm_sha(new) != NEW or norm_sha(test_src) != TEST:
    sys.exit("STOP: the files in this folder are not the expected candidate files.")
cur = norm_sha(target)
if cur == NEW:
    sys.exit("Nothing to do: benchmarks/psf_smart_layout.py is already 2026-09-29.c1.")
if cur != BASE:
    sys.exit("STOP: benchmarks/psf_smart_layout.py is not the release 2026-09-26.m1; not replaced. hash " + cur)
if os.path.exists(test_dst):
    sys.exit("STOP: benchmarks/test_short_path_layout.py already exists; not replaced.")
shutil.copyfile(target, target + ".bak_2026-09-26.m1")
shutil.copyfile(new, target)
shutil.copyfile(test_src, test_dst)
print("benchmarks/psf_smart_layout.py replaced (backup: .bak_2026-09-26.m1); "
      "benchmarks/test_short_path_layout.py added. Now run the tests.")
