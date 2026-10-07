"""dispatch_probe2.py -- DISPATCH-PROBE 2 (exploratory, not pre-registered; 2026-10-07), after DISPATCH-PROBE
(dispatch_probe.py). The first probe found the default call's gap to Qiskit level 2 coming from two places:
instructions on three or more qubits (unrolling the input with level 0 first closed most of it), and routing
(routing_optimization_level=2 closed most of the rest on QFT, BV and a Trotter circuit). Full unrolling also broke the
two-qubit structure of a flat QFT (100 qubits: 13,456 two-qubit gates against 17,544). This probe measures the
combinations on the same circuits (same builders and seeds, imported from dispatch_probe.py):
  PSF    candidate c19's default call (as before; repeated to check that the numbers reproduce)
  L2     qiskit.transpile(coupling_map, basis_gates, optimization_level=2, seed_transpiler=0) (repeated)
  PSFU   Qiskit's Unroll3qOrMore first (only instructions on three or more qubits are expanded; two-qubit gates are
         kept as they are), then PSF
  PSFU2  PSFU with routing_optimization_level=2
  PSFH2  the first probe's PSFH (unrolled to the basis with level 0) with routing_optimization_level=2
Times include the unrolling (`t`); `t_pre` is the unrolling alone. Each (circuit, arm) in its own process, 600 s limit.

    cd <repo> && python <this file> run --bp <benchpress clone> --out DIR
"""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dispatch_probe as D  # noqa: E402  (same folder; jobs() and build() unchanged)

ARMS = ("PSF", "L2", "PSFU", "PSFU2", "PSFH2")


def one(args):
    warnings.simplefilter("ignore")
    job = json.loads(args.job)
    qc, cm, basis, two = D.build(args.bp, job)
    rec = dict(src=job[0], test=job[1], arm=args.arm)
    from qiskit import transpile
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Unroll3qOrMore
    t0 = time.perf_counter()
    if args.arm == "L2":
        out = transpile(qc, coupling_map=cm, basis_gates=basis, optimization_level=2, seed_transpiler=0)
    else:
        import core_fix_c2_eval as H
        H.load_module(os.path.join(D.REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
        pc = H.load_module(D.C19, "psf_compile")
        t0 = time.perf_counter()
        if args.arm in ("PSFU", "PSFU2"):
            circ = PassManager([Unroll3qOrMore(basis_gates=basis)]).run(qc)
        elif args.arm == "PSFH2":
            circ = transpile(qc, basis_gates=basis, optimization_level=0)
        else:
            circ = qc
        rec["t_pre"] = round(time.perf_counter() - t0, 4)
        kw = dict(coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True, seed_transpiler=0)
        if args.arm in ("PSFU2", "PSFH2"):
            kw["routing_optimization_level"] = 2
        with contextlib.redirect_stdout(io.StringIO()):
            out = pc.compile_for_hardware(circ, **kw)
        rec["version"] = pc.VERSION
    rec["t"] = round(time.perf_counter() - t0, 4)
    ops = out.count_ops()
    rec["q2"] = int(sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "swap")))
    rec["d2"] = out.depth(filter_function=lambda x: len(x.qubits) == 2 and x.operation.name != "barrier")
    edges = set(cm.get_edges())
    bad = [i.operation.name for i in out.data if i.operation.name not in basis + ["measure", "barrier", "delay"]]
    off = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier"
              and tuple(out.find_bit(q).index for q in i.qubits) not in edges)
    rec["valid"] = not bad and not off
    print(json.dumps(rec))


def run(args):
    os.makedirs(args.out, exist_ok=True)
    rows = []
    path = os.path.join(args.out, "dispatch_probe2.json")
    for job in D.jobs(args.bp):
        for arm in ARMS:
            t0 = time.perf_counter()
            cmd = [sys.executable, os.path.abspath(__file__), "one", "--bp", args.bp, "--arm", arm, "--job",
                   json.dumps(job)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, cwd=D.REPO)
                lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                rec = json.loads(lines[-1]) if lines else dict(src=job[0], test=job[1], arm=arm, error=p.stderr[-400:])
            except subprocess.TimeoutExpired:
                rec = dict(src=job[0], test=job[1], arm=arm, error=f"timeout {args.timeout} s")
            rec["wall"] = round(time.perf_counter() - t0, 2)
            rows.append(rec)
            json.dump(rows, open(path, "w"), indent=1)
            show = {k: rec[k] for k in ("q2", "d2", "t", "t_pre", "valid", "error") if k in rec}
            print(f"{job[1][:46]:46s} {arm:6s} {show}", flush=True)
    print("DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one"))
    ap.add_argument("--bp", required=True)
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--job")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    run(a) if a.mode == "run" else one(a)


if __name__ == "__main__":
    main()
