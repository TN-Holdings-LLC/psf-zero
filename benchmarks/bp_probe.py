"""bp_probe.py -- BP-PROBE (exploratory, not pre-registered; 2026-10-06): a handful of Benchpress transpilation tests
compiled by Qiskit's Benchpress call and by PSF-Zero, to learn how PSF-Zero handles Benchpress's inputs and how long it
takes, before BP-MOCK is designed in detail. The tests here are fixed by rule below and will be excluded from BP-MOCK's
sample.

Every test is built exactly as Benchpress's Qiskit gym builds it (Benchpress commit b695f30): the same input circuit,
the same backend (FakeTorino for device tests, FlexibleBackend for abstract topologies, basis id/sx/x/rz/cz), Qiskit at
optimization level 2, and Benchpress's own output metrics (2Q gate count and 2Q depth) and structural validator.

Arms:
  QK     generate_preset_pass_manager(2, backend).run(circuit)                     (Benchpress's Qiskit call)
  PSF    psf_compile.compile_for_hardware, default call (no target)
  PSFR   psf_compile.compile_for_hardware, README recommended call (target)        (device tests only)
  PSFH   PSF after unrolling high-level gates to the basis with Qiskit level 0      (Hamiltonian and 100-qubit tests)

Each (test, arm) runs in its own subprocess with a time limit.

    cd <psf-zero repo> && python bp_probe.py run --bp <benchpress clone> --ref published_ref.json --out DIR
        [--tests small,feynman] [--arms QK,PSF] [--timeout 600] [--psf patches/<candidate>/psf_compile.py]

v2 (2026-10-06): adds --psf (compile the PSF arms with another psf_compile.py, e.g. a candidate), records the file's
version and item 39's EXACT_STATS per job, and writes env.json (versions) next to bp_probe.json. Without --psf the
jobs are those of v1, which made the first two runs of Addendum 377.

Results are written to DIR/bp_probe.json after every job, so an interrupted run keeps what it has.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import warnings

REPO = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
TIMEOUT = 600


# ---------------------------------------------------------------- Benchpress test construction

def bp_setup(bp):
    for p in (bp, os.path.join(REPO, "benchmarks"), REPO):
        if p not in sys.path:
            sys.path.insert(0, p)
    from benchpress.config import Configuration
    Configuration.gym_name = "qiskit"
    return Configuration


def hamiltonians(cfg):
    return json.load(open(cfg.get_hamiltonian_dir("hamlib") + "100_representative.json"))


def hamiltonian_circuit(cfg, instance):
    """The circuit Benchpress's Qiskit gym builds for one HamLib instance: a single PauliEvolutionGate."""
    from qiskit.quantum_info import SparsePauliOp
    from benchpress.qiskit_gym.utils.io import qiskit_hamiltonian_circuit
    h = next(h for h in hamiltonians(cfg) if h["ham_instance"] == instance)
    return qiskit_hamiltonian_circuit(SparsePauliOp(h["ham_hamlib_hamiltonian_terms"],
                                                    h["ham_hamlib_hamiltonian_coefficients"]))


def probe_tests(cfg):
    """The fixed probe set: (test id as published, kind, builder args)."""
    from benchpress.utilities.io import get_qasmbench_circuits
    small = dict((n, p) for p, n in zip(*get_qasmbench_circuits(cfg.get_qasm_dir("qasmbench-small"))))
    tests = [(f"test_QASMBench_small[{n}-all-to-all]", "qasmbench", (small[n], "all-to-all"))
             for n in ("adder_n4", "basis_trotter_n4")]
    tests.append(("test_feynman_transpile[barenco_tof_10.qasm]", "feynman", "barenco_tof_10.qasm"))
    recs = sorted(hamiltonians(cfg), key=lambda h: h["ham_instance"])
    pick = [next(h for h in recs if h["ham_category"] == "chemistry" and 10 <= h["ham_qubits"] <= 20),
            next(h for h in recs if h["ham_category"] == "condensedmatter" and 40 <= h["ham_qubits"] <= 80)]
    for h in pick:
        tests.append((f"test_hamlib_hamiltonians_transpile[ham_{h['ham_instance'][1:-1]}]", "ham_device",
                      h["ham_instance"]))
    hb = next(h for h in recs if h["ham_category"] == "binaryoptimization" and 20 <= h["ham_qubits"] <= 40)
    tests.append((f"test_hamiltonians[ham_{hb['ham_instance'][1:-1]}-heavy-hex]", "ham_abstract",
                  (hb["ham_instance"], "heavy-hex")))
    med = sorted(zip(*get_qasmbench_circuits(cfg.get_qasm_dir("qasmbench-medium"))), key=lambda x: x[1])[:2]
    for path, name in med:
        tests.append((f"test_QASMBench_medium[{name}-square]", "qasmbench", (path, "square")))
    large = sorted(zip(*get_qasmbench_circuits(cfg.get_qasm_dir("qasmbench-large"))), key=lambda x: x[1])
    from qiskit import QuantumCircuit
    for path, name in large:
        if 28 <= QuantumCircuit.from_qasm_file(path).num_qubits <= 60:
            tests.append((f"test_QASMBench_large[{name}-linear]", "qasmbench", (path, "linear")))
            break
    tests += [("test_QV_100_transpile", "device100", "QV_100"),
              ("test_circSU2_100_transpile", "device100", "circSU2_100"),
              ("test_QFT_100_transpile", "device100", "QFT_100")]
    return tests


def build(cfg, kind, arg):
    """Return (circuit, backend) exactly as the Benchpress Qiskit gym builds them."""
    from qiskit import QuantumCircuit
    from benchpress.utilities.backends import FlexibleBackend
    if kind in ("device100", "feynman", "ham_device"):
        backend = cfg.backend()
        if kind == "feynman":
            return QuantumCircuit.from_qasm_file(cfg.get_qasm_dir("feynman") + arg), backend
        if kind == "ham_device":
            return hamiltonian_circuit(cfg, arg), backend
        from qiskit.circuit.library import EfficientSU2, QuantumVolume
        if arg == "QV_100":
            return QuantumVolume(100, 100, seed=12345), backend
        if arg == "circSU2_100":
            return EfficientSU2(100, reps=3, entanglement="circular"), backend
        if arg == "QFT_100":
            return QuantumCircuit.from_qasm_file(cfg.get_qasm_dir("qft") + "qft_N100.qasm"), backend
        raise ValueError(arg)
    if kind == "ham_abstract":
        inst, topo = arg
        qc = hamiltonian_circuit(cfg, inst)
        return qc, FlexibleBackend(qc.num_qubits, topo, control_flow=True)
    if kind == "qasmbench":
        path, topo = arg
        qc = QuantumCircuit.from_qasm_file(path)
        return qc, FlexibleBackend(qc.num_qubits, topo, control_flow=True)
    raise ValueError(kind)


# ---------------------------------------------------------------- one (test, arm) in a subprocess

def one(args):
    cfg = bp_setup(args.bp)
    warnings.simplefilter("ignore")
    tid, kind, arg = json.loads(args.job)
    if isinstance(arg, list):
        arg = tuple(arg)
    qc, backend = build(cfg, kind, arg)
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from benchpress.qiskit_gym.utils.validation import qiskit_circuit_validation
    two_q = backend.two_q_gate_type
    rec = dict(test=tid, kind=kind, arm=args.arm, input_qubits=qc.num_qubits,
               input_ops={k: int(v) for k, v in qc.count_ops().items()})
    out_text = io.StringIO()
    t0 = time.perf_counter()
    if args.arm == "QK":
        out = generate_preset_pass_manager(2, backend).run(qc)
    else:
        if args.psf:
            import importlib.util
            spec = importlib.util.spec_from_file_location("psf_compile", os.path.abspath(args.psf))
            pc = importlib.util.module_from_spec(spec)
            sys.modules["psf_compile"] = pc
            spec.loader.exec_module(pc)
        else:
            import psf_compile as pc
        rec["release"] = pc.VERSION
        basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
        circ = qc
        if args.arm == "PSFH":
            from qiskit import transpile
            circ = transpile(qc, basis_gates=basis, optimization_level=0)
            rec["unrolled_ops"] = {k: int(v) for k, v in circ.count_ops().items()}
        t0 = time.perf_counter()
        kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
                  seed_transpiler=0)
        if args.arm == "PSFR":
            kw.update(target=backend.target, **RECOMMENDED)
        with contextlib.redirect_stdout(out_text):
            out = pc.compile_for_hardware(circ, **kw)
    rec["compile_s"] = time.perf_counter() - t0
    if args.arm != "QK" and hasattr(pc, "EXACT_STATS"):
        rec["exact_stats"] = dict(pc.EXACT_STATS)
    rec["psf_stdout_tail"] = [ln for ln in out_text.getvalue().splitlines() if "block" in ln.lower()][-3:]
    ops = out.count_ops()
    rec["output_qubits"] = out.num_qubits
    rec["q2"] = int(ops.get(two_q, 0))
    rec["d2"] = out.depth(filter_function=lambda x: x.operation.name == two_q)
    rec["other_2q"] = {k: int(v) for k, v in ops.items() if k != two_q and k in ("cx", "cz", "ecr", "swap", "unitary")}
    try:
        qiskit_circuit_validation(out, backend)
        rec["valid"] = True
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["valid"] = f"{type(exc).__name__}: {exc}"[:200]
    if out.num_qubits <= 10:
        try:
            from qiskit.quantum_info import Operator
            a = qc.copy()
            a.remove_final_measurements(inplace=True)
            b = out.copy()
            b.remove_final_measurements(inplace=True)
            rec["equivalent"] = bool(Operator(a).equiv(Operator.from_circuit(b)))
        except Exception as exc:  # noqa: BLE001
            rec["equivalent"] = f"n/a: {type(exc).__name__}"[:80]
    print(json.dumps(rec))


def run(args):
    cfg = bp_setup(args.bp)
    ref = json.load(open(args.ref))
    tests = probe_tests(cfg)
    os.makedirs(args.out, exist_ok=True)
    env = {"python": sys.version.split()[0], "platform": sys.platform,
           "psf_file": os.path.basename(os.path.dirname(os.path.abspath(args.psf))) if args.psf else "repository"}
    for m in ("qiskit", "qiskit_ibm_runtime", "numpy", "scipy"):
        try:
            env[m] = __import__(m).__version__
        except Exception as exc:  # noqa: BLE001 - recorded
            env[m] = f"n/a ({type(exc).__name__})"
    json.dump(env, open(os.path.join(args.out, "env.json"), "w"), indent=1)
    rows = []
    if args.tests:
        tests = [t for t in tests if any(k in t[0] for k in args.tests.split(","))]
    print(f"{len(tests)} tests; out {args.out}", flush=True)
    for tid, kind, arg in tests:
        arms = ["QK", "PSF"] + (["PSFR"] if kind in ("device100", "feynman", "ham_device") else []) + \
            (["PSFH"] if kind.startswith("ham") or kind == "device100" else [])
        if args.arms:
            arms = [a for a in arms if a in args.arms.split(",")]
        for arm in arms:
            print(f"  ... {time.strftime('%H:%M:%S', time.gmtime())} UTC {tid[:60]} {arm}", flush=True)
            job = json.dumps([tid, kind, arg])
            t0 = time.perf_counter()
            try:
                cmd = [sys.executable, os.path.abspath(__file__), "one", "--bp", args.bp, "--arm", arm, "--job", job]
                if args.psf:
                    cmd += ["--psf", args.psf]
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, cwd=REPO)
                lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                rec = json.loads(lines[-1]) if lines else dict(test=tid, arm=arm, error=p.stderr[-600:])
            except subprocess.TimeoutExpired:
                rec = dict(test=tid, arm=arm, error=f"timeout {args.timeout} s")
            rec["wall_s"] = time.perf_counter() - t0
            rec["published"] = ref.get(tid)
            rows.append(rec)
            pub = ref.get(tid) or {}
            q = pub.get("qiskit_2.5.0rc1") or {}
            k = pub.get("tket_2.18.0") or {}
            print(f"{tid[:60]:60s} {arm:4s} " + (f"q2 {rec['q2']:6d} d2 {rec['d2']:6d} t {rec['compile_s']:8.2f}s "
                  f"valid {rec['valid']} eq {rec.get('equivalent', '-')}" if "q2" in rec else f"ERROR {rec['error'][-200:]}")
                  + f" | pub QK2.5 q2 {q.get('q2')} d2 {q.get('d2')} | Tket2.18 q2 {k.get('q2')} d2 {k.get('d2')}",
                  flush=True)
            json.dump(rows, open(os.path.join(args.out, "bp_probe.json"), "w"), indent=1)
    print("DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one"))
    ap.add_argument("--bp", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--job")
    ap.add_argument("--timeout", type=int, default=TIMEOUT)
    ap.add_argument("--tests", help="comma-separated substrings of test ids to keep (default: all)")
    ap.add_argument("--arms", help="comma-separated arms to keep, e.g. QK,PSF (default: all)")
    ap.add_argument("--psf", help="path of the psf_compile.py to use for the PSF arms (default: the repository's)")
    args = ap.parse_args()
    run(args) if args.mode == "run" else one(args)


if __name__ == "__main__":
    main()
