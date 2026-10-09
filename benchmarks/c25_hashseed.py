"""c25_hashseed.py -- exploratory follow-up of C25-ID2 (2026-10-09; not pre-registered, nothing predicted).

C25-ID2 (Addendum 411) made the layout search independent of time, yet REL2 differed from REL on 8 of 185 tests and
calls, and C25 on the same 8. A remaining suspect is Python's hash randomization: each process gets its own
PYTHONHASHSEED, which changes the iteration order of sets and dicts keyed by strings.

1. For each of the 8, prints a short form of the three arms' signatures from C25-ID2's records.
2. Runs the 8 again in each arm with PYTHONHASHSEED=0, and REL once more with PYTHONHASHSEED=1, through
   c25_identity2.py's own "one" mode (virtual clock), 4 at a time, and prints the signatures.

    python benchmarks/c25_hashseed.py --bp <benchpress clone> [--par 4]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RECORDS = os.path.join(REPO, "data", "2026-10-09", "c25_identity2", "c25_identity2.jsonl")
SCRIPT = os.path.join(HERE, "c25_identity2.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True)
    ap.add_argument("--par", type=int, default=4)
    a = ap.parse_args()
    lines = [json.loads(x) for x in open(RECORDS, encoding="utf-8")]
    by = {}
    for r in lines[1:]:
        by.setdefault((r["test"], r["call"]), {})[r["arm"]] = r
    differ = sorted(k for k, v in by.items() if all(x in v and "error" not in v[x] for x in ("REL", "REL2", "C25"))
                    and len({v[x]["sig"] for x in ("REL", "REL2", "C25")}) > 1)
    print(f"C25-ID2: {len(differ)} tests and calls where the three arms do not all agree")
    for k in differ:
        v = by[k]
        print(f"  {k[0][:58]} ({k[1]}): REL {v['REL']['sig'][:8]} REL2 {v['REL2']['sig'][:8]} C25 {v['C25']['sig'][:8]}"
              f"  q2 {v['REL']['q2']}/{v['REL2']['q2']}/{v['C25']['q2']}")
    runs = [(arm, "0") for arm in ("REL", "REL2", "C25")] + [("REL", "1")]
    jobs = []
    for test, call in differ:
        base = {x: by[(test, call)]["REL"][x] for x in ("stratum", "test", "kind", "arg", "call")}
        jobs += [(dict(base, arm=arm), seed) for arm, seed in runs]

    def go(job_seed):
        job, seed = job_seed
        env = dict(os.environ, PYTHONHASHSEED=seed)
        p = subprocess.run([sys.executable, SCRIPT, "one", "--bp", a.bp, "--job", json.dumps(job)], capture_output=True,
                           text=True, env=env, timeout=3600, encoding="utf-8", errors="replace")
        out = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        return job, seed, (json.loads(out[-1]) if out else {"error": (p.stderr or "")[-200:]})

    t0 = time.perf_counter()
    res = {}
    with ThreadPoolExecutor(a.par) as ex:
        for job, seed, rec in ex.map(go, jobs):
            res[(job["test"], job["call"], job["arm"], seed)] = rec
            print(f"[{len(res)}/{len(jobs)} {time.perf_counter() - t0:5.0f} s] {job['arm']} hashseed {seed} "
                  f"{job['test'][:50]} ({job['call']}): {rec.get('sig', 'ERROR')[:8]}", flush=True)
    print("\nwith PYTHONHASHSEED fixed:")
    same0 = same01 = 0
    for test, call in differ:
        s = {f"{arm}/{seed}": res[(test, call, arm, seed)].get("sig", "ERROR")[:8] for arm, seed in runs}
        all0 = len({s["REL/0"], s["REL2/0"], s["C25/0"]}) == 1
        same0 += all0
        same01 += s["REL/0"] == s["REL/1"]
        print(f"  {test[:58]} ({call}): " + " ".join(f"{k} {x}" for k, x in s.items())
              + ("   three arms agree at seed 0" if all0 else "   DIFFER at seed 0"))
    print(f"\nSUMMARY three arms agree with PYTHONHASHSEED=0 on {same0} of {len(differ)}; "
          f"REL with seed 0 = REL with seed 1 on {same01} of {len(differ)}")


if __name__ == "__main__":
    main()
