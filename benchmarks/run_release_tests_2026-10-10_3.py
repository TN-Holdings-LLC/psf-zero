"""run_release_tests_2026-10-10_3.py -- the checks before the commit of release 2026-10-10.3 (Addendum 438): the
files of run_release_tests_2026-10-10_2.py, the new release test and candidates c31-c35's tests, each in its own
pytest session.

    python benchmarks/run_release_tests_2026-10-10_3.py --out data/2026-10-10/release_2026-10-10.3

Writes tests_log.txt (each session's last lines) and summary.json into --out, and prints the summary.
"""
import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FILES = [
    "benchmarks/test_release_2026_10_10_3.py",
    "benchmarks/test_release_2026_10_10_2.py",
    "benchmarks/test_release_2026_10_10_1.py",
    "benchmarks/test_release_2026_10_07_1.py",
    "benchmarks/test_release_2026_10_06_4.py",
    "patches/psf_compile_c19_2026-10-07/test_c19.py",
    "benchmarks/test_release_2026_09_28.py",
    "benchmarks/test_core_fix_c2.py",
    "benchmarks/test_release_2026_10_02.py",
    "benchmarks/test_release_2026_10_02_2.py",
    "benchmarks/test_release_2026_10_03.py",
    "benchmarks/test_release_2026_10_03_2.py",
    "benchmarks/test_release_2026_10_03_3.py",
    "benchmarks/test_release_2026_10_04_1.py",
    "benchmarks/test_release_2026_10_05_1.py",
    "benchmarks/test_ai_compile_a7.py",
    "benchmarks/test_ai_compile_a8.py",
    "benchmarks/test_ai_compile_a9.py",
    "benchmarks/test_ai_compile_a11.py",
    "benchmarks/test_ai_compile_a12.py",
    "patches/psf_ai_compile_a6_2026-10-02/test_ai6.py",
    "patches/psf_ai_compile_a7_2026-10-02/test_a7.py",
    "patches/psf_ai_compile_a8_2026-10-04/test_a8.py",
    "patches/psf_ai_compile_a12_2026-10-06/test_a12.py",
    "patches/psf_ai_compile_a13_2026-10-06/test_a13.py",
    "patches/psf_compile_c3_2026-10-02/test_c3_prune.py",
    "patches/psf_compile_c4_2026-10-02/test_c4_layout.py",
    "patches/psf_compile_c5_2026-10-02/test_c5_placement.py",
    "patches/psf_compile_c6_2026-10-03/test_c6_floor.py",
    "patches/psf_compile_c8_2026-10-03/test_c8_resynth.py",
    "patches/psf_compile_c9_2026-10-03/test_c9_compare.py",
    "patches/psf_compile_c10_2026-10-03/test_c10_floor_pauli.py",
    "patches/psf_compile_c11_2026-10-04/test_c11_hybrid.py",
    "patches/psf_compile_c12_2026-10-05/test_c12_exact.py",
    "patches/psf_compile_c24_2026-10-07/test_c24.py",
    "patches/psf_compile_c26_2026-10-09/test_c26.py",
    "patches/psf_compile_c29_2026-10-10/test_c29.py",
    "patches/psf_compile_c30_2026-10-10/test_c30.py",
    "patches/psf_compile_c31_2026-10-10/test_c31.py",
    "patches/psf_compile_c32_2026-10-10/test_c32.py",
    "patches/psf_compile_c33_2026-10-10/test_c33.py",
    "patches/psf_compile_c34_2026-10-10/test_c34.py",
    "patches/psf_compile_c35_2026-10-10/test_c35.py",
]


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import numpy
    import qiskit
    for d in (os.path.join(REPO, "benchmarks"), REPO):
        sys.path.insert(0, d)
    import psf_compile
    import psf_zero_core
    env = dict(python=platform.python_version(), qiskit=qiskit.__version__, numpy=numpy.__version__,
               psf_compile=psf_compile.VERSION, core=getattr(psf_zero_core, "CORE_VERSION", "?"),
               core57=getattr(psf_compile, "CORE57_VERSION", None),
               git_head=subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                       text=True).stdout.strip(),
               dirty=subprocess.run(["git", "status", "--short", "--untracked-files=no"], cwd=REPO,
                                    capture_output=True, text=True).stdout.splitlines())
    log = [f"environment: {json.dumps(env)}", f"start {now()}"]
    print(log[0], flush=True)
    results = []
    for f in FILES:
        t0 = now()
        p = subprocess.run([sys.executable, "-m", "pytest", f, "-q", "-p", "no:cacheprovider"], cwd=REPO,
                           capture_output=True, text=True)
        tail = (p.stdout + p.stderr).strip().splitlines()[-15:]
        last = tail[-1] if tail else ""
        n = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", last)}
        results.append(dict(file=f, returncode=p.returncode, start=t0, passed=n.get("passed", 0),
                            failed=n.get("failed", 0), errors=n.get("error", 0) + n.get("errors", 0),
                            skipped=n.get("skipped", 0), last=last))
        line = f"{f}: {last} (exit {p.returncode})"
        print(line, flush=True)
        log += ["", f"== {f} ({t0})"] + tail
    tot = {k: sum(r[k] for r in results) for k in ("passed", "failed", "errors", "skipped")}
    bad = [r["file"] for r in results if r["returncode"] != 0]
    summary = f"files {len(FILES)}; {tot}; files not passing: {bad}; end {now()}"
    log += ["", summary]
    print(summary)
    with open(os.path.join(args.out, "tests_log.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(env=env, results=results, totals=tot, not_passing=bad), fh, indent=1)


if __name__ == "__main__":
    main()
