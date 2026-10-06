"""run_skip_2026-10-06.py -- runs the SKIP test (skip_eval.py, Addendum 379) on any platform: one job per device
(6), then scores and runs the independent check (skip_verify.py). run_a12_2026-10-06.py (Addendum 364) with SKIP's
jobs, plus the check.

    python benchmarks/run_skip_2026-10-06.py skip <out dir> [--par N] [--smoke]

Run from the repository root. --par: parallel jobs (default: half the logical CPUs, at most 6); it changes only the
wall time. Progress: <out dir>/progress.txt has one "   done" line per finished job (6).
"""
import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TESTS = {
    "skip": dict(script="benchmarks/skip_eval.py",
                 files=["benchmarks/skip_eval.py", "benchmarks/skip_verify.py",
                        "patches/psf_compile_c17_2026-10-06/psf_compile.py", "psf_compile.py"],
                 jobs=[[d] for d in ("FakeKingston", "FakeTorino", "FakeBrussels", "FakeOsaka", "FakeHanoiV2",
                                     "FakeAuckland")],
                 args=lambda j: ["--device", j[0]], done="SKIP DONE"),
}
THREAD_ENV = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1",
                  QISKIT_PARALLEL="FALSE", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=sorted(TESTS))
    ap.add_argument("out")
    ap.add_argument("--par", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    T = TESTS[a.test]
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, **THREAD_ENV)
    py = sys.executable
    vers = subprocess.run([py, "-c", "import sys, numpy, qiskit, qiskit_aer; print('python', sys.version.split()[0], "
                                     "'numpy', numpy.__version__, 'qiskit', qiskit.__version__, 'aer', "
                                     "qiskit_aer.__version__)"], capture_output=True, text=True, encoding="utf-8", env=env).stdout.strip()
    head = subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
    info = [f"date_utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            f"platform {platform.platform()}", f"processor {platform.processor()}", f"CPU(s) {os.cpu_count()}", vers]
    info += [f"{sha256(os.path.join(REPO, p))}  {p}" for p in T["files"] + ["benchmarks/run_skip_2026-10-06.py"]]
    info += [head, f"PAR {a.par} SMOKE {int(a.smoke)}"]
    open(os.path.join(out, "env.txt"), "w", encoding="utf-8").write("\n".join(info) + "\n")
    print("\n".join(info), flush=True)
    t0 = time.time()
    prog = open(os.path.join(out, "progress.txt"), "w", encoding="utf-8")

    def run_one(job):
        name = "_".join(job)
        cmd = [py, "-u", os.path.join(REPO, T["script"]), "run"] + T["args"](job) + ["--out", out] + \
            (["--smoke"] if a.smoke else [])
        with open(os.path.join(out, f"log_{name}.txt"), "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
        line = f"   done {' '.join(job)} ({int(time.time() - t0)} s){'' if rc == 0 else f' EXIT {rc}'}"
        prog.write(line + "\n")
        prog.flush()
        print(line, flush=True)

    with ThreadPoolExecutor(max_workers=a.par) as ex:
        list(ex.map(run_one, T["jobs"]))
    prog.close()
    print(f"== all jobs finished after {int(time.time() - t0)} s ({len(T['jobs'])} jobs)", flush=True)
    for f in sorted(os.listdir(out)):
        if f.startswith("log_"):
            txt = open(os.path.join(out, f), encoding="utf-8", errors="replace").read()
            if "Traceback" in txt or "STOP" in txt:
                print("problem in", f)
    sc = subprocess.run([py, os.path.join(REPO, T["script"]), "score", "--out", out], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", env=env, cwd=REPO)
    open(os.path.join(out, "score_log.txt"), "w", encoding="utf-8").write(sc.stdout + sc.stderr)
    print("\n".join((sc.stdout + sc.stderr).rstrip("\n").split("\n")[-16:]))
    if not a.smoke:
        vf = subprocess.run([py, os.path.join(REPO, "benchmarks", "skip_verify.py"), out], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", env=env, cwd=REPO)
        open(os.path.join(out, "verify_log.txt"), "w", encoding="utf-8").write(vf.stdout + vf.stderr)
        print("\n".join((vf.stdout + vf.stderr).rstrip("\n").split("\n")[-8:]))
    print(T["done"])


if __name__ == "__main__":
    main()
