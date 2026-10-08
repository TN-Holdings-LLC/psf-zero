"""bp_final_rerun.py -- BP-FINAL: re-run, once, every job that ended in a timeout, because the workplace machine went
to sleep during the scored run (2026-10-08, lunch break) and a job in flight during the sleep can reach the 1,500 s
limit on the wall clock without having used the time. Rule fixed before any result was seen (2026-10-08, 12:19 JST):
  - only jobs whose record is "timeout 1500 s" are re-run; errors of any other kind are kept as they are;
  - each is re-run once, with the locked bp_final.py ("one" mode), the same limit, 12 at a time;
  - the re-run's record replaces the first and carries "rerun": true and the first record under "first";
  - a job that times out again stays a timeout;
  - the first file is kept as bp_final_first.json.

    python benchmarks/bp_final_rerun.py --bp <benchpress clone> --out DIR [--par 12]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
S = os.path.join(HERE, "bp_final.py")
TIMEOUT = 1500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--par", type=int, default=12)
    a = ap.parse_args()
    fn = os.path.join(a.out, "bp_final.json")
    first = os.path.join(a.out, "bp_final_first.json")
    if os.path.exists(first):
        raise SystemExit("STOP: bp_final_first.json exists; the re-run is made once")
    with open(fn, encoding="utf-8") as f:
        d = json.load(f)
    idx = [i for i, r in enumerate(d["rows"]) if str(r.get("error", "")).startswith("timeout")]
    print(f"{len(idx)} timed-out jobs to re-run", flush=True)
    shutil.copyfile(fn, first)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1",
               QISKIT_PARALLEL="FALSE", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t00 = time.time()

    sys.path[:0] = [HERE, REPO]
    import bp_final as F
    pop = {t[1]: t for t in F.population(a.bp)}  # kind and builder argument, as the run had them

    def run(i):
        r = d["rows"][i]
        t = pop[r["test"]]
        cmd = [sys.executable, S, "one", "--bp", a.bp, "--arm", r["arm"], "--job", json.dumps(t)]
        w0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=REPO, env=env,
                               encoding="utf-8", errors="replace")
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            new = json.loads(lines[-1]) if lines else dict(stratum=t[0], test=t[1], arm=r["arm"],
                                                           error=(p.stderr or "")[-400:])
        except subprocess.TimeoutExpired:
            new = dict(stratum=t[0], test=t[1], arm=r["arm"], error=f"timeout {TIMEOUT} s")
        new["wall"] = round(time.perf_counter() - w0, 2)
        new["rerun"] = True
        new["first"] = r
        print(f"{int(time.time() - t00):6d} s  {t[1][:58]:58s} {r['arm']:4s} "
              f"{ {k: new[k] for k in ('q2', 'valid', 'error') if k in new} }", flush=True)
        return i, new

    with ThreadPoolExecutor(max_workers=a.par) as ex:
        for i, new in ex.map(run, idx):
            d["rows"][i] = new
    d["meta"]["rerun"] = dict(jobs=len(idx), wall_s=round(time.time() - t00, 1),
                              start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t00)))
    with open(fn, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1)
    still = sum(1 for i in idx if "error" in d["rows"][i])
    print(f"re-ran {len(idx)} jobs; {still} still failed; first file kept as bp_final_first.json")


if __name__ == "__main__":
    main()
