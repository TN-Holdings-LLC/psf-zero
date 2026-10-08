"""run_margin.py -- the same runs as run_margin.sh (MARGIN, Addendum 405), for machines without a POSIX shell
(Windows PowerShell). Train (4 jobs), then 96 deployment jobs, then the score. Run from the repository root.

    python benchmarks/run_margin.py <out dir> [--dry] [--par 6]
"""
import argparse
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "margin_eval.py")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1",
           QISKIT_PARALLEL="FALSE", PYTHONIOENCODING="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--par", type=int, default=6)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    git = lambda *c: subprocess.run(["git", "-C", os.path.join(HERE, ".."), *c], capture_output=True,  # noqa: E731
                                    text=True).stdout.strip()
    ver = subprocess.run([sys.executable, "-c", "import sys,qiskit,qiskit_aer,qiskit_ibm_runtime,sklearn,numpy;"
                          "print(sys.version.split()[0],qiskit.__version__,qiskit_aer.__version__,"
                          "qiskit_ibm_runtime.__version__,sklearn.__version__,numpy.__version__)"],
                         capture_output=True, text=True)
    with open(os.path.join(out, "env.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), str(os.cpu_count()),
                           git("rev-parse", "--short", "HEAD"), git("status", "--short", "--untracked-files=no"),
                           (ver.stdout + ver.stderr).strip(), platform.platform(),
                           f"PAR {a.par} DRY {int(a.dry)} (run_margin.py)"]) + "\n")
    t0 = time.time()
    prog = open(os.path.join(out, "progress.txt"), "w", encoding="utf-8", newline="\n")

    def job(args):
        log = os.path.join(out, "log_" + "_".join(args).replace("-", "_") + ".txt")
        cmd = [sys.executable, "-u", S, *args, "--out", out] + (["--dry"] if a.dry else [])
        with open(log, "w", encoding="utf-8", newline="\n") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=ENV)
        line = f"   done {' '.join(args)} ({int(time.time() - t0)} s)"
        prog.write(line + "\n")
        prog.flush()
        print(line, flush=True)

    train = [["train", "--dataset", ds, "--n", str(n)] for ds in ("BC", "D38") for n in (4, 6)]
    deploy = [["deploy", "--dataset", ds, "--n", str(n), "--device", dev, "--arm", arm, "--cal", c]
              for dev in ("FakeTorino", "FakeAuckland") for ds in ("BC", "D38") for n in (6, 4)
              for c in ("t", "0", "1", "2") for arm in ("REC", "C24", "RPSF")]
    for batch in (train, deploy):
        with ThreadPoolExecutor(a.par) as ex:
            list(ex.map(job, batch))
    prog.close()
    done = sum(1 for ln in open(os.path.join(out, "progress.txt"), encoding="utf-8") if ln.startswith("   done"))
    print(f"== all jobs finished after {int(time.time() - t0)} s ({done} done)")
    bad = [n for n in sorted(os.listdir(out)) if n.startswith("log_")
           and any(k in open(os.path.join(out, n), encoding="utf-8", errors="replace").read() for k in ("Traceback", "STOP"))]
    for n in bad[:5]:
        print("   ERROR in", n)
    sc = subprocess.run([sys.executable, S, "score", "--out", out], capture_output=True, text=True, env=ENV)
    with open(os.path.join(out, "score_log.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(sc.stdout + sc.stderr)
    print("\n".join((sc.stdout + sc.stderr).strip().split("\n")[-45:]))


if __name__ == "__main__":
    main()
