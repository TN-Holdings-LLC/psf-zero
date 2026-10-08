"""c25_identity.py -- C25-ID (2026-10-09): candidate 2026-10-09.c25 (changelog item 52) against release 2026-10-07.1
on the 152 Benchpress tests this project's development used (BP-PROBE, BP-MOCK, BP-MOCK2). Item 52 changes only how
two matching sizes are computed, so every output should be the same; this checks it, and records the compile times.

Arms (each test, call and arm in its own process; one compile, timed; the module load is not timed):
  REL   psf_compile.py and benchmarks/psf_smart_layout.py (release 2026-10-07.1, layout 2026-10-01.1)
  C25   patches/psf_compile_c25_2026-10-09/{psf_compile.py, psf_smart_layout.py}
Calls, as BP-FINAL makes them: "default" (coupling map and basis, layout_search=True, seed_transpiler=0) on every
test, and "recommended" (the README's recommended call with the backend's target) on the FakeTorino tests. The two
arms of a test and call run next to each other in the queue, in an order set by the test's hash.

    python benchmarks/c25_identity.py list    --bp <benchpress clone>
    python benchmarks/c25_identity.py run     --bp <benchpress clone> --out DIR [--par N]
    python benchmarks/c25_identity.py compare --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PATCH = os.path.join(REPO, "patches", "psf_compile_c25_2026-10-09")
sys.path[:0] = [HERE, REPO]
import bp_mock as B    # noqa: E402  (BP-MOCK, unchanged)
import bp_mock2 as B2  # noqa: E402  (BP-MOCK2, unchanged)

ARMS = {"REL": (os.path.join(REPO, "psf_compile.py"), os.path.join(HERE, "psf_smart_layout.py")),
        "C25": (os.path.join(PATCH, "psf_compile.py"), os.path.join(PATCH, "psf_smart_layout.py"))}
VERSIONS = {"REL": ("2026-10-07.1", "2026-10-01.1"), "C25": ("2026-10-09.c25", "2026-10-09.c25")}
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
TIMEOUT = 1800
N_TESTS = 152


def population(bp):
    """[(stratum, test id, kind, arg)]: BP-PROBE's 12, BP-MOCK's 92 and BP-MOCK2's 48 tests."""
    index = {t[0]: (name,) + tuple(t) for name, tests in B.strata(bp).items() for t in tests}
    out = [index[t] for t in sorted(B.PROBED)] + list(B.sample(bp)) + list(B2.sample(bp))
    return out


def jobs(bp):
    out = []
    for stratum, tid, kind, arg in population(bp):
        calls = ("default", "recommended") if stratum.endswith("FakeTorino") or kind == "summit" else ("default",)
        h = hashlib.sha256(("C25-ID|" + tid).encode()).hexdigest()
        arms = ("REL", "C25") if int(h, 16) % 2 == 0 else ("C25", "REL")
        for call in calls:
            out += [(h, dict(stratum=stratum, test=tid, kind=kind, arg=arg, call=call, arm=a)) for a in arms]
    return [j for _, j in sorted(out, key=lambda x: x[0])]


def one(args):
    warnings.simplefilter("ignore")
    job = json.loads(args.job)
    qc, backend = B.build(args.bp, job["kind"], job["arg"])
    import core_fix_c2_eval as H
    compile_path, layout_path = ARMS[job["arm"]]
    lay = H.load_module(layout_path, "psf_smart_layout")
    pc = H.load_module(compile_path, "psf_compile")
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0)
    if job["call"] == "recommended":
        kw.update(target=backend.target, **RECOMMENDED)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    rec = dict(job, version=pc.VERSION, layout_version=lay.LAYOUT_VERSION, t=time.perf_counter() - t0,
               q2=int(out.count_ops().get(backend.two_q_gate_type, 0)), sig=B.sig_hash(out))
    print(json.dumps(rec), flush=True)


def run(args):
    js = jobs(args.bp)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "c25_identity.jsonl")
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout
    meta = dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_head=head.strip(),
                dirty_tracked=dirty.strip(), tests=len({j["test"] for j in js}), jobs=len(js), par=args.par,
                cpus=os.cpu_count(), python=sys.version.split()[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=meta)) + "\n")

    def go(j):
        cmd = [sys.executable, os.path.abspath(__file__), "one", "--bp", args.bp, "--job", json.dumps(j)]
        w0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, encoding="utf-8",
                               errors="replace")
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else dict(j, error=(p.stderr or "")[-400:])
        except subprocess.TimeoutExpired:
            rec = dict(j, error=f"timeout {TIMEOUT} s")
        rec["wall"] = round(time.perf_counter() - w0, 2)
        return rec

    n = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in js]):
            rec = f.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            print(f"[{n}/{len(js)} {time.perf_counter() - t0:6.0f} s] {rec['arm']} {rec['call']} {rec['test'][:60]}"
                  + (" ERROR" if "error" in rec else ""), flush=True)
    compare(args)


def compare(args):
    lines = [json.loads(x) for x in open(os.path.join(args.out, "c25_identity.jsonl"), encoding="utf-8")]
    meta = lines[0]["meta"]
    recs = [r for r in lines[1:]]
    by = {}
    for r in recs:
        by.setdefault((r["test"], r["call"]), {})[r["arm"]] = r
    both = {k: v for k, v in by.items() if all(a in v and "error" not in v[a] for a in ("REL", "C25"))}
    same = [k for k, v in both.items() if v["REL"]["sig"] == v["C25"]["sig"]]
    diff = [k for k in both if k not in same]
    fail = {a: sorted(k for k, v in by.items() if a in v and "error" in v[a]) for a in ("REL", "C25")}
    versions_ok = all(r.get("version") == VERSIONS[r["arm"]][0] and r.get("layout_version") == VERSIONS[r["arm"]][1]
                      for r in recs if "error" not in r)
    ratios = [v["C25"]["t"] / v["REL"]["t"] for v in both.values() if v["REL"]["t"] > 0]
    small = [v["C25"]["t"] / v["REL"]["t"] for v in both.values() if 0 < v["REL"]["t"] < 1.0]
    gm = lambda xs: (2.718281828459045 ** (sum(__import__("math").log(x) for x in xs) / len(xs))) if xs else None  # noqa: E731
    L = ["# C25-ID", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['tests']} tests (expected {N_TESTS}), {meta['jobs']} jobs, {meta['par']} at a time, "
         f"{meta['cpus']} CPUs, Python {meta['python']}",
         f"versions as expected in every record: {versions_ok}", "",
         f"- pairs both arms finished: {len(both)} of {len(by)}",
         f"- identical outputs: {len(same)}; different: {len(diff)}",
         f"- failures: REL {len(fail['REL'])}, C25 {len(fail['C25'])}; the same set: {fail['REL'] == fail['C25']}",
         f"- compile time C25 / REL: median {statistics.median(ratios):.3f}, geometric mean {gm(ratios):.3f}"
         if ratios else "- no time ratio",
         f"- on the {len(small)} pairs REL compiled in under 1 s: median {statistics.median(small):.3f}, geometric "
         f"mean {gm(small):.3f}" if small else "- no pair under 1 s"]
    L += [f"  - different: {t} ({c})" for t, c in diff[:30]]
    L += [f"  - failed ({a}): {t} ({c})" for a in fail for t, c in fail[a][:10]]
    ok = (meta["tests"] == N_TESTS and not meta["dirty_tracked"] and versions_ok and not diff
          and fail["REL"] == fail["C25"])
    L += ["", f"VERDICT {'IDENTICAL' if ok else 'NOT IDENTICAL (see above)'}"]
    txt = "\n".join(L) + "\n"
    open(os.path.join(args.out, "compare.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("list", "run", "one", "compare"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "list":
        js = jobs(a.bp)
        print(f"{len({j['test'] for j in js})} tests, {len(js)} jobs")
        for j in js[:6]:
            print(" ", j["arm"], j["call"], j["test"])
        return
    {"run": run, "one": one, "compare": compare}[a.mode](a)


if __name__ == "__main__":
    main()
