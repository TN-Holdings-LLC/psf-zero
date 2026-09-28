"""Replace src/lib.rs with core 2026-09-28.1 (changelog item 11, CORE_VERSION), only if
the current file is exactly the expected base (repository 100e768).

    python apply_update.py <this folder> <repository root>
"""
import hashlib, os, shutil, sys

BASE = "b11f35b5ee9dbe4c0245675c22ec0eca9be19baa35d7158d3070dba4e2889077"
NEW = "bf3bf537df3eb3e81444d2d2ba6fe723770048f2c80c5c6bb9da48884148d234"


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
    sys.exit("Nothing to do: src/lib.rs is already core 2026-09-28.1.")
if cur != BASE:
    sys.exit("STOP: src/lib.rs is not the expected base (repository 100e768); not replaced. hash " + cur)
shutil.copyfile(target, target + ".bak_2026-09-28")
shutil.copyfile(new, target)
print("src/lib.rs replaced (backup: src/lib.rs.bak_2026-09-28). Now: cargo test --release --lib, then maturin develop --release")
