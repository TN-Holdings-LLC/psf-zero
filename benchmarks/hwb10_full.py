"""hwb10_full.py -- H10-FULL (2026-10-09; pre-registered in Addendum 419): the release's recommended call on hwb10
(Feynman, FakeTorino), the development test it has never finished, given up to 3 hours. Does it finish, and what
does it choose?

Two processes, one after the other (nothing else runs on the machine):
  L3    Qiskit's level 3 alone, called as `_compare_level3` calls it (the backend's target, seed_transpiler=0,
        approximation_degree=1.0): its time and two-qubit gate count. Killed after 1,800 s.
  REL   release 2026-10-07.1 (psf_compile.py), the recommended call, with the layout search's clock virtual (as
        C25-ID2) and PYTHONHASHSEED=0. Killed after 10,800 s.
REL writes an event before and after every estimate, exactness check, Qiskit transpile and re-synthesis candidate,
flushed to disk, so a killed run still shows where it was. Each estimate and check is counted as candidate c27 (item
54) would count it (units of about a nanosecond on the workplace PC; 0 for a call over 16 qubits), without any budget.

    python benchmarks/hwb10_full.py run --bp <benchpress clone> --out DIR
    python benchmarks/hwb10_full.py report --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import os
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, REPO]
TEST = "test_feynman_transpile[hwb10.qasm]"
L3_S, REL_S = 1800, 10800
COUNTED = ("excitation_cost", "hybrid_cost", "pauli_cost", "kraus_cost", "_implements", "_same_action")
TIMED = ("transpile", "_resynthesis_candidate")
WORK_PER_OP = 20_000  # candidate c27's constants (Addendum 416)
WORK_PER_AMP = {"excitation_cost": 57, "hybrid_cost": 46, "pauli_cost": 57, "kraus_cost": 57, "_implements": 5,
                "_same_action": 5}
TOP = 16
C27_Q2 = 113_292  # C27's two-qubit gates on hwb10 (Addenda 416-417; the same at home on 2026-10-09)


def _job(bp):
    import c25_identity as C
    hit = [t for t in C.population(bp) if t[1] == TEST and t[0].endswith("FakeTorino")]
    if len(hit) != 1:
        sys.exit(f"STOP: {len(hit)} tests named {TEST}")
    return dict(stratum=hit[0][0], test=hit[0][1], kind=hit[0][2], arg=hit[0][3])


def _shape(c):
    ops = [tuple(c.find_bit(x).index for x in ins.qubits) for ins in c.data
           if ins.operation.name not in ("barrier", "measure", "delay")]
    return len(ops), sum(len(q) >= 2 for q in ops), {i for q in ops for i in q}


def work(fn, args):
    """The work candidate c27 counts for this call (0 if it would be over 16 qubits, so not made)."""
    if fn in ("excitation_cost", "hybrid_cost", "pauli_cost", "kraus_cost"):
        n_all, n_multi, t = _shape(args[0])
        if len(t) > TOP:
            return 0
        k = max(len(t), 1)
        amps = (n_multi if fn in ("excitation_cost", "hybrid_cost") else n_all) * 2 ** k
        return WORK_PER_OP * n_all + WORK_PER_AMP[fn] * amps
    if fn == "_same_action":
        a, b = _shape(args[0]), _shape(args[1])
        k = len(a[2] | b[2])
        if k > TOP:
            return 0
        ops = 2 * (a[0] + b[0])
        return WORK_PER_OP * ops + WORK_PER_AMP[fn] * ops * 2 ** k
    qc, out = args[0], args[1]
    a, b = _shape(qc), _shape(out)
    n = qc.num_qubits
    lay = getattr(out, "layout", None)
    init = set(lay.initial_index_layout(filter_ancillas=True)[:n]) if lay is not None else set(range(n))
    fin = set(lay.final_index_layout(filter_ancillas=True)[:n]) if lay is not None else set(range(n))
    k = len(b[2] | init | fin)
    if n > TOP or k > TOP:
        return 0
    return WORK_PER_OP * 2 * (a[0] + b[0]) + WORK_PER_AMP[fn] * 2 * (a[0] * 2 ** n + b[0] * 2 ** k)


def child(args):
    warnings.simplefilter("ignore")
    import bp_mock as B
    import c25_identity as C
    import c25_identity2 as C2
    import core_fix_c2_eval as H
    job = _job(args.bp)
    qc, backend = B.build(args.bp, job["kind"], job["arg"])
    log = open(args.log, "a", encoding="utf-8", newline="\n")
    t_start = time.perf_counter()

    def write(rec):
        rec["at_s"] = round(time.perf_counter() - t_start, 3)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        os.fsync(log.fileno())

    write(dict(ev="input", arm=args.arm, qubits=qc.num_qubits, size=len(qc.data)))
    if args.arm == "L3":
        from qiskit import transpile
        t0 = time.perf_counter()
        out = transpile(qc, target=backend.target, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)
        write(dict(ev="l3_end", t=round(time.perf_counter() - t0, 3),
                   q2=int(out.count_ops().get(backend.two_q_gate_type, 0)), sig=B.sig_hash(out)))
        return
    lay = H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    lay.time = C2._VirtualTime()
    pc = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    seq = [0]

    def wrap(name, f):
        @functools.wraps(f)
        def g(*a, **kw):
            seq[0] += 1
            s = seq[0]
            w = work(name, a) if name in COUNTED else None
            write(dict(ev="start", seq=s, fn=name, caller=sys._getframe(1).f_code.co_name, work=w))
            t0 = time.perf_counter()
            r = f(*a, **kw)
            res = None if name in TIMED else (r if r is None or isinstance(r, bool) else "value")
            write(dict(ev="end", seq=s, fn=name, t=round(time.perf_counter() - t0, 3), result=res))
            return r
        return g

    for name in COUNTED + TIMED:  # module globals: the module's own calls go through the wrappers
        setattr(pc, name, wrap(name, getattr(pc, name)))
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, **C.RECOMMENDED)
    write(dict(ev="compile_start", version=pc.VERSION, layout_version=lay.LAYOUT_VERSION,
               clock=type(lay.time).__name__))
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, **kw)
    write(dict(ev="compile_end", t=round(time.perf_counter() - t0, 3),
               q2=int(out.count_ops().get(backend.two_q_gate_type, 0)), sig=B.sig_hash(out),
               stats={s: {k: v for k, v in getattr(pc, s).items() if v}
                      for s in ("RESYNTH_STATS", "COMPARE_STATS", "EXACT_STATS", "SKIP_STATS")}))


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def run(args):
    if os.path.exists(args.out):
        sys.exit(f"STOP: {args.out} exists; a run never replaces an earlier run's files")
    os.makedirs(args.out)
    _job(args.bp)
    meta = dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_head=git("rev-parse", "--short",
                "HEAD"), dirty_tracked=git("status", "--porcelain", "--untracked-files=no"), cpus=os.cpu_count(),
                python=sys.version.split()[0], l3_limit_s=L3_S, rel_limit_s=REL_S)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    env = dict(os.environ, PYTHONHASHSEED="0")
    for arm, limit in (("L3", L3_S), ("REL", REL_S)):
        log = os.path.join(args.out, f"{arm}.jsonl")
        print(f"{time.strftime('%H:%M:%S')} {arm} starts (limit {limit} s)", flush=True)
        w0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", "--bp", args.bp, "--arm", arm,
                                "--log", log], capture_output=True, text=True, timeout=limit, env=env)
            status = "ok" if p.returncode == 0 else "error: " + (p.stderr or "")[-300:].replace("\n", " ")
        except subprocess.TimeoutExpired:
            status = f"killed after {limit} s"
        with open(log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(dict(ev="job_end", status=status, wall=round(time.perf_counter() - w0, 1))) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {arm}: {status[:80]} ({time.perf_counter() - w0:.0f} s)", flush=True)
    report(args)


def _events(path):
    out = []
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def report(args):
    meta = json.load(open(os.path.join(args.out, "meta.json"), encoding="utf-8"))
    l3, rel = _events(os.path.join(args.out, "L3.jsonl")), _events(os.path.join(args.out, "REL.jsonl"))
    l3_end = next((e for e in l3 if e["ev"] == "l3_end"), None)
    end = next((e for e in rel if e["ev"] == "compile_end"), None)
    starts = {e["seq"]: e for e in rel if e["ev"] == "start"}
    ends = {e["seq"]: e for e in rel if e["ev"] == "end"}
    counted = [(starts[s], ends.get(s)) for s in sorted(starts) if starts[s]["fn"] in COUNTED]
    made = [(a, b) for a, b in counted if b is not None and a["work"]]
    wall = sum(b["t"] for _, b in made)
    units = sum(a["work"] for a, _ in made)
    ratio = wall / (units / 1e9) if units else None
    chose_l3 = bool(end) and end["stats"].get("COMPARE_STATS", {}).get("level3") == 1
    verdict = {
        "H1": ("finishes within 10,800 s", end is not None),
        "H2": ("chooses Qiskit level 3's circuit", chose_l3 if end else None),
        "H3": (f"fewer two-qubit gates than C27's {C27_Q2:,}", (end["q2"] < C27_Q2) if end else None),
        "H4": ("estimates' and checks' wall time 0.2-1.0 x their counted work",
               (0.2 <= ratio <= 1.0) if ratio else None),
        "H5": ("the same two-qubit gate count as level 3 alone",
               (end["q2"] == l3_end["q2"]) if end and l3_end else None),
    }
    word = {True: "CONFIRMED", False: "REFUTED", None: "NOT DECIDED"}
    L = ["# H10-FULL", "",
         f"git head {meta['git_head']}, uncommitted tracked changes: {'none' if not meta['dirty_tracked'] else 'YES'}; "
         f"{meta['cpus']} CPUs, Python {meta['python']}", "",
         f"- L3 alone: " + (f"{l3_end['t']:.1f} s, {l3_end['q2']:,} two-qubit gates" if l3_end else "did not finish"),
         f"- REL: " + (f"{end['t']:.1f} s, {end['q2']:,} two-qubit gates; decisions {json.dumps(end['stats'])}"
                       if end else "did not finish"),
         f"- estimates and checks made: {len(made)}, {wall:.0f} s for {units / 1e9:.0f} s of counted work"
         + (f" (ratio {ratio:.2f})" if ratio else ""), "",
         "| ID | prediction | verdict |", "|---|---|---|"]
    L += [f"| {k} | {t} | **{word[v]}** |" for k, (t, v) in verdict.items()]
    L += ["", "| # | caller | function | counted work (s) | wall (s) |", "|---|---|---|---|---|"]
    for a, b in counted:
        L.append(f"| {a['seq']} | {a['caller']} | {a['fn']} | {(a['work'] or 0) / 1e9:.1f} | "
                 f"{'never ended' if b is None else b['t']} |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "hwb10_full.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "report", "child"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--log")
    a = ap.parse_args()
    {"run": run, "report": report, "child": child}[a.mode](a)


if __name__ == "__main__":
    main()
