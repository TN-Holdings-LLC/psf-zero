"""Replace src/lib.rs with the candidate core 2026-09-29.1 (core changelog item 12:
eigen-route fallback), only if the current file is exactly the release core
2026-09-28.1 (normalized SHA-256 bf3bf537...d234, repository 5a0cba8 and later).

    python apply_update_2026-09-29.py <this folder> <repository root>

The same change is in lib_rs_eigen_route_2026-09-29.patch (unified diff against the
release src/lib.rs). After replacing: cargo test --release --lib (9 tests), then
maturin develop --release, then check that psf_zero_core.CORE_VERSION is 2026-09-29.1.
"""
import hashlib
import os
import shutil
import sys

BASE = "bf3bf537df3eb3e81444d2d2ba6fe723770048f2c80c5c6bb9da48884148d234"
NEW = "5364630ec4e3648fe944b4103d76e8caa440e722c27ee6621b0ac51fdb26ad8f"


def norm_sha(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    return hashlib.sha256("\n".join(l.rstrip() for l in text.splitlines()).encode()).hexdigest()


here, repo = sys.argv[1], sys.argv[2]
target = os.path.join(repo, "src", "lib.rs")
new = os.path.join(here, "lib.rs")
if norm_sha(new) != NEW:
    sys.exit("STOP: the new lib.rs in this folder is not the expected file.")
cur = norm_sha(target)
if cur == NEW:
    sys.exit("Nothing to do: src/lib.rs is already core 2026-09-29.1.")
if cur != BASE:
    sys.exit("STOP: src/lib.rs is not the release core 2026-09-28.1; not replaced. hash " + cur)
shutil.copyfile(target, target + ".bak_2026-09-28.1")
shutil.copyfile(new, target)
print("src/lib.rs replaced (backup: src/lib.rs.bak_2026-09-28.1). Now: cargo test --release --lib, "
      "then maturin develop --release")
