"""release_2026-10-01.py -- make the adopted candidates of 2026-10-01 the release, in one checked step.

    python3 release_2026-10-01.py <c2 candidate folder> <repository root>            (check only)
    python3 release_2026-10-01.py <c2 candidate folder> <repository root> --apply    (change files)

What it does (owner's decision of 2026-10-01: adopt c2 together with core 2026-09-29.1):
  1. src/lib.rs                       release core 2026-09-28.1 -> core 2026-09-29.1
     (patches/core_eigen_route_2026-09-29.1/, already in the repository)
  2. psf_compile.py                   release 2026-09-28.1      -> 2026-10-01.c2 -> release 2026-10-01.1
     benchmarks/psf_smart_layout.py   release 2026-09-26.m1     -> 2026-10-01.c2 -> release 2026-10-01.1
     benchmarks/test_core_fix_c2.py, benchmarks/core_fix_c2_eval.py added
     (the workplace apply script apply_core_fix_c2_2026-10-01.py does the copy and the backups)
  3. Version strings: the candidate labels become the release labels (exact, checked replacements).
  4. Tests: test_core_fix_c2.py and test_release_2026_09_28.py expect the new release strings;
     the c1 test test_short_path_layout.py (patches/psf_smart_layout_c1_2026-09-29/) is added to
     benchmarks/ with its version check updated.
  5. The .bak_* copies the two apply scripts leave in the repository are moved to
     ~/psf_release_backup_2026-10-01/ so the working tree stays clean.

Every precondition is checked before anything is written: repository at 9131cee with no tracked
changes, and the three base files at their release hashes. Nothing is committed; the script prints
the files to `git add` after the tests pass.
"""
import hashlib
import os
import shutil
import subprocess
import sys

HEAD = "9131cee"
BASES = {
    "src/lib.rs": "bf3bf537df3eb3e81444d2d2ba6fe723770048f2c80c5c6bb9da48884148d234",
    "psf_compile.py": "3616efc8b8a7bea184d6170fb7379d509d0bfec9828f4eb2f703dd4d26d8c60b",
    "benchmarks/psf_smart_layout.py": "a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875",
}
CORE_DIR = "patches/core_eigen_route_2026-09-29.1"
C1_TEST = "patches/psf_smart_layout_c1_2026-09-29/test_short_path_layout.py"
NEW = "2026-10-01.1"

EDITS = {
    "psf_compile.py": [
        ("VERSION: 2026-10-01.c2 -- CANDIDATE, not a release (base: release 2026-09-28.1; previous revision: 2026-09-27.7)",
         "VERSION: 2026-10-01.1 -- release, adopted on 2026-10-01 from candidate 2026-10-01.c2 (previous release: 2026-09-28.1)"),
        ("Candidate 2026-10-01.c2 (workplace; not a release until its pre-registered evaluation is scored and adopted):",
         "2026-10-01.1 (release; candidate 2026-10-01.c2, adopted on 2026-10-01 after its pre-registered evaluation):"),
        ('VERSION = "2026-10-01.c2"  # candidate (workplace): c1 + permutation elision + post-routing re-synthesis',
         'VERSION = "2026-10-01.1"  # release (from candidate 2026-10-01.c2): c1 + permutation elision + post-routing re-synthesis'),
    ],
    "benchmarks/psf_smart_layout.py": [
        ('LAYOUT_VERSION = "2026-10-01.c2"  # candidate (workplace): c1 + exact packing search for disjoint 2-/3-qubit paths',
         'LAYOUT_VERSION = "2026-10-01.1"  # release (from candidate 2026-10-01.c2): c1 + exact packing search for disjoint 2-/3-qubit paths'),
    ],
    "benchmarks/test_core_fix_c2.py": [
        ('    assert pc.VERSION == "2026-10-01.c2"\n    assert psl.LAYOUT_VERSION.startswith("2026-10-01.c2")',
         '    assert pc.VERSION == "2026-10-01.1"  # release of candidate 2026-10-01.c2\n'
         '    assert psl.LAYOUT_VERSION == "2026-10-01.1"'),
    ],
    "benchmarks/test_release_2026_09_28.py": [
        ('    assert pc.VERSION == "2026-09-28.1"',
         '    assert pc.VERSION == "2026-10-01.1"  # current release (this file was written for 2026-09-28.1)'),
    ],
}
C1_EDIT = ('    assert psl.LAYOUT_VERSION == "2026-09-29.c1"',
           '    assert psl.LAYOUT_VERSION == "2026-10-01.1"  # c1 is part of release 2026-10-01.1')


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout.rstrip("\n")


def replace_exact(path, pairs):
    with open(path, encoding="utf-8", newline="") as f:
        t = f.read()
    nl = "\r\n" if "\r\n" in t else "\n"
    u = t.replace(nl, "\n")
    for old, new in pairs:
        if u.count(old) != 1:
            sys.exit(f"ERROR: expected text not found exactly once in {path}: {old[:70]!r}")
        u = u.replace(old, new)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(u.replace("\n", nl))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cand, repo, apply = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), "--apply" in sys.argv[3:]
    c2_apply = os.path.join(cand, "apply_core_fix_c2_2026-10-01.py")
    problems = []
    if not os.path.exists(c2_apply):
        problems.append(f"{c2_apply} not found (pass the candidate/ folder of work_2026-10-01_core_fix_c2)")
    head = git(repo, "rev-parse", "--short=7", "HEAD").strip()
    if head != HEAD:
        problems.append(f"repository HEAD is {head}, expected {HEAD}")
    dirty = [ln for ln in git(repo, "status", "--porcelain").splitlines() if not ln.startswith("??")]
    if dirty:
        problems.append("tracked files have changes: " + ", ".join(ln[3:] for ln in dirty[:5]))
    for rel, want in BASES.items():
        p = os.path.join(repo, rel)
        if not os.path.exists(p) or norm_sha(p) != want:
            problems.append(f"{rel} is not the expected release file")
    for rel in (os.path.join(CORE_DIR, "apply_update_2026-09-29.py"), os.path.join(CORE_DIR, "lib.rs"), C1_TEST):
        if not os.path.exists(os.path.join(repo, rel)):
            problems.append(f"{rel} missing in the repository")
    if os.path.exists(os.path.join(repo, "benchmarks", "test_short_path_layout.py")):
        problems.append("benchmarks/test_short_path_layout.py already exists")
    if problems:
        print("STOP -- nothing was changed:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(f"checks passed: HEAD {head}, clean tree, release lib.rs / psf_compile.py / psf_smart_layout.py")
    r = subprocess.run([sys.executable, c2_apply, cand, repo], capture_output=True, text=True)
    print("c2 apply script (check only):\n  " + r.stdout.strip().replace("\n", "\n  "))
    if r.returncode != 0:
        sys.exit("STOP: the c2 apply script refused; nothing was changed")
    if not apply:
        print("check only; run again with --apply.")
        return

    # 1. core
    r = subprocess.run([sys.executable, os.path.join(repo, CORE_DIR, "apply_update_2026-09-29.py"),
                        os.path.join(repo, CORE_DIR), repo], capture_output=True, text=True)
    print("1. core: " + (r.stdout.strip() or r.stderr.strip()))
    if r.returncode != 0:
        sys.exit("STOP after step 0: the core apply script refused")
    # 2. c2
    r = subprocess.run([sys.executable, c2_apply, cand, repo, "--apply"], capture_output=True, text=True)
    print("2. c2:\n  " + r.stdout.strip().replace("\n", "\n  "))
    if r.returncode != 0:
        sys.exit("ERROR: the c2 apply script failed after the core was replaced; see git status")
    # 3./4. version strings and tests
    for rel, pairs in EDITS.items():
        replace_exact(os.path.join(repo, rel), pairs)
    dst = os.path.join(repo, "benchmarks", "test_short_path_layout.py")
    shutil.copyfile(os.path.join(repo, C1_TEST), dst)
    replace_exact(dst, [C1_EDIT])
    print(f"3. version strings -> {NEW}; tests updated; benchmarks/test_short_path_layout.py added")
    # 5. backups out of the tree
    bdir = os.path.expanduser("~/psf_release_backup_2026-10-01")
    os.makedirs(bdir, exist_ok=True)
    moved = []
    for rel in ("src/lib.rs.bak_2026-09-28.1", "psf_compile.py.bak_2026-09-28.1",
                "benchmarks/psf_smart_layout.py.bak_2026-09-26.m1"):
        p = os.path.join(repo, rel)
        if os.path.exists(p):
            shutil.move(p, os.path.join(bdir, os.path.basename(rel)))
            moved.append(rel)
    print(f"4. backups moved to {bdir}: {', '.join(moved)}")
    print("\nNext: cargo test --release --lib; maturin develop --release; check CORE_VERSION; pytest.")
    print("Files to add after the tests pass:")
    for rel in ("src/lib.rs", "psf_compile.py", "benchmarks/psf_smart_layout.py", "benchmarks/test_core_fix_c2.py",
                "benchmarks/core_fix_c2_eval.py", "benchmarks/test_release_2026_09_28.py",
                "benchmarks/test_short_path_layout.py"):
        print("  " + rel)


if __name__ == "__main__":
    main()
