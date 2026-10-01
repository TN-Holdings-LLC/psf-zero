"""Install the PSF-Zero candidates of 2026-10-01 into a repository checkout, only from known bases.

    python apply_core_fix_c2_2026-10-01.py <this folder> <repository root>          (check only)
    python apply_core_fix_c2_2026-10-01.py <this folder> <repository root> --apply  (replace / add)

Replaces:
  psf_compile.py                  release 2026-09-28.1      -> candidate 2026-10-01.c2
  benchmarks/psf_smart_layout.py  release 2026-09-26.m1, or candidate 2026-09-29.c1 -> candidate 2026-10-01.c2
Adds (refuses if they already exist):
  benchmarks/test_core_fix_c2.py, benchmarks/core_fix_c2_eval.py
Each replaced file is first copied to <name>.bak_<old version> (never overwritten if the backup exists).
The same change is in core_fix_c2_2026-10-01.patch (git diff against 9131cee).
Hashes are normalized SHA-256 (lines right-stripped, CRLF -> LF, trailing blank lines dropped).
The Rust core is not touched. After --apply: run the test files under benchmarks/ (see the README).
"""
import hashlib
import os
import shutil
import sys

FILES = {
    # name in this folder: (destination, expected normalized SHA-256 of the candidate, {base hash: base label})
    "psf_compile.py": ("psf_compile.py",
                       "e22dc6ddf4bf717d722a3f7ce01e23ecbe096d41730f7fe54c0d29eeaa4a3bd1",
                       {"3616efc8b8a7bea184d6170fb7379d509d0bfec9828f4eb2f703dd4d26d8c60b": "2026-09-28.1"}),
    "psf_smart_layout.py": (os.path.join("benchmarks", "psf_smart_layout.py"),
                            "f0d38519adad04864de42c456564a8321584ae3feadd744101a4cb719070fc14",
                            {"a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875": "2026-09-26.m1",
                             "e25952a33bacfbb8de7a439b53892e9bab88329d8c47c6de867c29fc660145fa": "2026-09-29.c1"}),
    "test_core_fix_c2.py": (os.path.join("benchmarks", "test_core_fix_c2.py"),
                            "d2a188d694585588f6950176daf6d9ff256956f44c4a24e0f51d795a0d06e8fb", None),
    "core_fix_c2_eval.py": (os.path.join("benchmarks", "core_fix_c2_eval.py"),
                            "9f4e124f996bbf83a15768148631279c5e1ce73541d929d0701dbe91da871bac", None),
}


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    here, repo, apply = sys.argv[1], sys.argv[2], "--apply" in sys.argv[3:]
    plan, problems = [], []
    for name, (dest, want, bases) in FILES.items():
        src = os.path.join(here, name)
        dst = os.path.join(repo, dest)
        if not os.path.exists(src) or norm_sha(src) != want:
            problems.append(f"{name}: missing or not the expected candidate file in {here}")
            continue
        if bases is None:
            if os.path.exists(dst):
                if norm_sha(dst) == want:
                    print(f"already present: {dest}")
                else:
                    problems.append(f"{dest} already exists and differs; not replaced")
                continue
            plan.append((src, dst, None))
            continue
        if not os.path.exists(dst):
            problems.append(f"{dest} not found in the repository")
            continue
        cur = norm_sha(dst)
        if cur == want:
            print(f"already the candidate: {dest}")
        elif cur in bases:
            bak = dst + ".bak_" + bases[cur]
            if os.path.exists(bak):
                problems.append(f"backup {bak} already exists; not replaced")
            else:
                plan.append((src, dst, bak))
        else:
            problems.append(f"{dest} is not a known base (hash {cur}); not replaced")
    if problems:
        print("STOP -- nothing was changed:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    for src, dst, bak in plan:
        print(("replace " if bak else "add     ") + os.path.relpath(dst, repo) + (f"  (backup {os.path.basename(bak)})" if bak else ""))
    if not apply:
        print("check only; run again with --apply to make these changes.")
        return
    for src, dst, bak in plan:
        if bak:
            shutil.copyfile(dst, bak)
        shutil.copyfile(src, dst)
    for src, dst, _ in plan:
        if norm_sha(dst) != norm_sha(src):
            sys.exit(f"ERROR: {dst} does not match after copying")
    print("done. Now run the tests (README).")


if __name__ == "__main__":
    main()
