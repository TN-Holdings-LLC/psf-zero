"""c25_nondet.py -- exploratory (2026-10-09; not pre-registered, nothing predicted): what differs between two runs of
the release on the same input, and does running every numeric library on one thread remove it?

C25-ID2 (Addendum 411) and c25_hashseed.py found 8 tests and calls whose output signature changes from run to run,
even with the layout search's clock virtual and PYTHONHASHSEED fixed, while the two-qubit count does not. This
compiles each of them with the release (virtual clock, as C25-ID2) RUNS times in separate processes, under two
environments:
  default        as C25-ID2 ran
  one-thread     OMP, OpenBLAS, MKL and Rayon limited to one thread, Qiskit's parallelism off, PYTHONHASHSEED=0
and compares the outputs instruction by instruction: same instruction names and qubits? largest parameter
difference? global phase? layouts?

    python benchmarks/c25_nondet.py --bp <benchpress clone> [--runs 3] [--par 4]
"""
import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, REPO]
RECORDS = os.path.join(REPO, "data", "2026-10-09", "c25_identity2", "c25_identity2.jsonl")
ONE_THREAD = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1",
                  QISKIT_PARALLEL="FALSE", PYTHONHASHSEED="0")


def child(args):
    warnings.simplefilter("ignore")
    import bp_mock as B
    import c25_identity2 as C
    import core_fix_c2_eval as H
    job = json.loads(args.job)
    qc, backend = B.build(args.bp, job["kind"], job["arg"])
    compile_path, layout_path = C.ARMS["REL"]
    lay = H.load_module(layout_path, "psf_smart_layout")
    lay.time = C._VirtualTime()
    pc = H.load_module(compile_path, "psf_compile")
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0)
    if job["call"] == "recommended":
        kw.update(target=backend.target, **C.RECOMMENDED)
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    ops = [[i.operation.name, [out.find_bit(q).index for q in i.qubits],
            [float(p) if isinstance(p, (int, float)) or hasattr(p, "__float__") else repr(p)
             for p in i.operation.params]] for i in out.data]
    lay_ = getattr(out, "layout", None)
    print(json.dumps(dict(ops=ops, phase=float(out.global_phase), sig=B.sig_hash(out),
                          initial=list(lay_.initial_index_layout(filter_ancillas=True)) if lay_ else None,
                          final=list(lay_.final_index_layout(filter_ancillas=True)) if lay_ else None)))


def compare(a, b):
    """A short description of how output b differs from output a."""
    if a["sig"] == b["sig"]:
        return "identical"
    parts = []
    if a["initial"] != b["initial"] or a["final"] != b["final"]:
        parts.append("layouts differ")
    if len(a["ops"]) != len(b["ops"]):
        parts.append(f"{len(a['ops'])} vs {len(b['ops'])} instructions")
    struct = sum(1 for x, y in zip(a["ops"], b["ops"]) if x[0] != y[0] or x[1] != y[1])
    if struct:
        parts.append(f"{struct} instructions differ in name or qubits")
    dmax, nd = 0.0, 0
    for x, y in zip(a["ops"], b["ops"]):
        if x[0] == y[0] and x[1] == y[1]:
            for p, q in zip(x[2], y[2]):
                if isinstance(p, float) and isinstance(q, float) and p != q:
                    nd += 1
                    d = abs(math.remainder(p - q, 2 * math.pi))
                    dmax = max(dmax, d)
    if nd:
        parts.append(f"{nd} parameters differ, largest by {dmax:.2e} (mod 2 pi)")
    if a["phase"] != b["phase"]:
        parts.append(f"global phase differs by {abs(math.remainder(a['phase'] - b['phase'], 2 * math.pi)):.2e}")
    return "; ".join(parts) or "signature differs (no difference found here)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=("run", "child"))
    ap.add_argument("--bp", required=True)
    ap.add_argument("--job")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--par", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "child":
        return child(a)
    lines = [json.loads(x) for x in open(RECORDS, encoding="utf-8")]
    by = {}
    for r in lines[1:]:
        by.setdefault((r["test"], r["call"]), {})[r["arm"]] = r
    differ = sorted(k for k, v in by.items() if all(x in v and "error" not in v[x] for x in ("REL", "REL2", "C25"))
                    and len({v[x]["sig"] for x in ("REL", "REL2", "C25")}) > 1)
    envs = {"default": {}, "one-thread": ONE_THREAD}
    jobs = []
    for test, call in differ:
        base = {x: by[(test, call)]["REL"][x] for x in ("stratum", "test", "kind", "arg", "call")}
        jobs += [(base, e, k) for e in envs for k in range(a.runs)]

    def go(item):
        base, e, k = item
        env = dict(os.environ, **envs[e])
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", "--bp", a.bp, "--job", json.dumps(base)],
                           capture_output=True, text=True, env=env, timeout=3600, encoding="utf-8", errors="replace")
        out = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        return base, e, k, (json.loads(out[-1]) if out else {"error": (p.stderr or "")[-300:]})

    t0 = time.perf_counter()
    res = {}
    with ThreadPoolExecutor(a.par) as ex:
        for base, e, k, rec in ex.map(go, jobs):
            res[(base["test"], base["call"], e, k)] = rec
            print(f"[{len(res)}/{len(jobs)} {time.perf_counter() - t0:5.0f} s] {e} run {k} {base['test'][:50]} "
                  f"({base['call']}): {rec.get('sig', 'ERROR ' + rec.get('error', '')[-80:])[:8]}", flush=True)
    print()
    stable = {e: 0 for e in envs}
    for test, call in differ:
        print(f"{test[:60]} ({call})")
        for e in envs:
            recs = [res[(test, call, e, k)] for k in range(a.runs)]
            if any("error" in r for r in recs):
                print(f"  {e}: a run failed")
                continue
            notes = [compare(recs[0], r) for r in recs[1:]]
            same = all(n == "identical" for n in notes)
            stable[e] += same
            print(f"  {e}: " + ("all runs identical" if same else " | ".join(notes)))
    print("\nSUMMARY " + ", ".join(f"{e}: identical across runs on {n} of {len(differ)}" for e, n in stable.items()))


if __name__ == "__main__":
    main()
