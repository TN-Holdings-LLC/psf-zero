"""dispatch_probe.py -- DISPATCH-PROBE (exploratory, not pre-registered; 2026-10-07): data for choosing a rule by which
the default call of compile_for_hardware() could hand a circuit to Qiskit level 2 instead of PSF-Zero's own path.
BP-PROBE (Addendum 377) found the default call behind Qiskit level 2 on general circuits and level on Quantum Volume;
PSF-Zero's own case (same-pair two-qubit chains, full occupancy) must not get worse. No rule is fixed here.

Circuits:
- BP-PROBE's 14 Benchpress tests, built as Benchpress builds them (benchmarks/bp_probe.py, unchanged);
- SKIP's four families (benchmarks/skip_eval.py, unchanged) at n = 12 and 40 on FakeTorino, seed 83,000,000 + k;
- PL-GPU-REDO's family T at spare 0 and 4 on FakeAuckland, and at spare 0 on FakeKingston (156 qubits).
Arms (each (circuit, arm) in its own process, 600 s limit), all without a target, as the default call is:
  PSF   candidate c19's default call (coupling map, basis, entangling_basis="cx", layout_search=True, seed 0)
  PSFH  the same after unrolling to the basis with Qiskit level 0
  L2    qiskit.transpile(coupling_map, basis_gates, optimization_level=2, seed_transpiler=0)
  PSF2  c19's default call with routing_optimization_level=2
Recorded per job: two-qubit count and depth, compile time, validity (gates in the basis, two-qubit gates on couplings).
Recorded per circuit: features of the input unrolled to cx (two-qubit count, how much of it sits in same-pair blocks
of at least 4 CX, the largest block), and its instructions on more than two qubits.

    cd <repo> && python dispatch_probe.py run --bp <benchpress clone> --out DIR
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

import numpy as np

REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
C19 = os.path.join(REPO, "patches", "psf_compile_c19_2026-10-07", "psf_compile.py")
ARMS = ("PSF", "PSFH", "L2", "PSF2")


def jobs(bp):
    import bp_probe
    cfg = bp_probe.bp_setup(bp)
    out = [("bp", tid, kind, arg) for tid, kind, arg in bp_probe.probe_tests(cfg)]
    k = 0
    for fam in ("ring", "brick", "pauli", "qft"):
        for n in (12, 40):
            out.append(("skip", f"{fam}{n}-FakeTorino", fam, [n, 83_000_000 + k]))
            k += 1
    for dev, spare in (("FakeAuckland", 0), ("FakeAuckland", 4), ("FakeKingston", 0)):
        out.append(("plT", f"T-spare{spare}-{dev}", dev, [spare, 84_000_000 + spare]))
    return out


def build(bp, job):
    """(circuit, coupling map, basis, two-qubit gate name)"""
    src, tid, kind, arg = job
    from qiskit_ibm_runtime import fake_provider
    if src == "bp":
        import bp_probe
        cfg = bp_probe.bp_setup(bp)
        qc, backend = bp_probe.build(cfg, kind, tuple(arg) if isinstance(arg, list) else arg)
        t = backend.target
    elif src == "skip":
        import skip_eval
        qc = skip_eval.family_circuit(kind, arg[0], np.random.default_rng(arg[1]))
        qc.measure_all()
        t = fake_provider.FakeTorino().target
    else:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import random_unitary
        import full_heavyhex_cliff as fh
        t = getattr(fake_provider, kind)().target
        f = fh.graph_facts(t)
        blocks = fh.layout_blocks("T", arg[0], f["nq"], f["matching"])
        rng = np.random.default_rng(arg[1])
        qc = QuantumCircuit(sum(len(b) for b in blocks))
        for b in blocks:
            for a, c in ([(b[1], b[0])] if len(b) == 2 else [(b[1], b[0]), (b[2], b[1])]):
                for _ in range(10):
                    qc.unitary(random_unitary(4, seed=int(rng.integers(2**31))).data, [a, c])
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    two = next(g for g in ("cz", "cx", "ecr") if g in basis)
    return qc, t.build_coupling_map(), basis, two


def features(qc):
    from qiskit import transpile
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler.passes import Collect2qBlocks
    wide = sorted({i.operation.name for i in qc.data if len(i.qubits) > 2 and i.operation.name != "barrier"})
    u = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=0)
    total = u.count_ops().get("cx", 0)
    dag = circuit_to_dag(u)
    pc = Collect2qBlocks()
    pc.run(dag)
    sizes = [sum(1 for nd in b if nd.op.name == "cx") for b in pc.property_set["block_list"]]
    deep = sum(s for s in sizes if s >= 4)
    return dict(n=qc.num_qubits, wide=wide, cx_unrolled=total, frac_deep=round(deep / total, 4) if total else 0.0,
                largest_block=max(sizes) if sizes else 0, blocks=len(sizes))


def one(args):
    warnings.simplefilter("ignore")
    job = json.loads(args.job)
    qc, cm, basis, two = build(args.bp, job)
    rec = dict(src=job[0], test=job[1], arm=args.arm)
    if args.arm == "FEATURES":
        rec.update(features(qc))
        print(json.dumps(rec))
        return
    from qiskit import transpile
    t0 = time.perf_counter()
    if args.arm == "L2":
        out = transpile(qc, coupling_map=cm, basis_gates=basis, optimization_level=2, seed_transpiler=0)
    else:
        import core_fix_c2_eval as H
        H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
        pc = H.load_module(C19, "psf_compile")
        circ = transpile(qc, basis_gates=basis, optimization_level=0) if args.arm == "PSFH" else qc
        t0 = time.perf_counter()
        kw = dict(coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True, seed_transpiler=0)
        if args.arm == "PSF2":
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
    path = os.path.join(args.out, "dispatch_probe.json")
    for job in jobs(args.bp):
        for arm in ("FEATURES",) + ARMS:
            t0 = time.perf_counter()
            cmd = [sys.executable, os.path.abspath(__file__), "one", "--bp", args.bp, "--arm", arm, "--job",
                   json.dumps(job)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, cwd=REPO)
                lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                rec = json.loads(lines[-1]) if lines else dict(src=job[0], test=job[1], arm=arm, error=p.stderr[-400:])
            except subprocess.TimeoutExpired:
                rec = dict(src=job[0], test=job[1], arm=arm, error=f"timeout {args.timeout} s")
            rec["wall"] = round(time.perf_counter() - t0, 2)
            rows.append(rec)
            json.dump(rows, open(path, "w"), indent=1)
            show = {k: rec[k] for k in ("q2", "d2", "t", "valid", "frac_deep", "cx_unrolled", "wide", "error") if k in rec}
            print(f"{job[1][:46]:46s} {arm:8s} {show}", flush=True)
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
