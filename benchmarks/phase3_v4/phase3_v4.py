# phase3_v4.py -- Real-device topology validation v4
#
# ===========================================================================
# v3 -> v4 (2026-09-08)
# ===========================================================================
# v3 measured hardware compilation using "circuits where PSF-Zero fundamentally 
# cannot do anything." Across 15 points (50-500 qubits, 3 seeds), it showed 
# PSF-Zero being 1.14 to 2.65 times slower than Qiskit, but the cause was the 
# circuit workload, not the implementation.
#
# [1] Incompatible workload (The core issue)
#     v3 used qiskit.circuit.random.random_circuit(). When measured:
#         random_circuit(100, depth=100)
#           Input  2q=160 / Total gates 3679 / depth 100
#           Output 2q=160 / Total gates 3679 / depth 100   <- Complete passthrough
#           Log "PSF-Zero Rust Core executed for 0/0 blocks"
#     In short, v3 was comparing "Qiskit processing everything on a circuit 
#     PSF passed through untouched" vs "Qiskit processing everything." The 
#     difference was pure overhead. It was impossible to win.
#     This was a fact already discovered by this project in the v1->v2 notes of 
#     test_scale_explosion_war2.py (random_circuit's 2q blocks average 3 gates, 
#     never exceeding block_gate_floor=12). It was applied to phase1/phase2/TKET 
#     tests but omitted in phase3.
#     v4 uses build_dense_pair_blocks_circuit() as the primary workload.
#     random_circuit is kept as a "passthrough" control to demonstrate behavior 
#     outside PSF-Zero's scope, not as a target for speed comparison.
#
# [2] basis_gates were not passed
#     v3 did not pass basis_gates to either arm, resulting in outputs that 
#     could not be submitted to real hardware (rxx/ryy/rzz or raw 2q unitaries 
#     remained). v4 passes basis_gates to all arms and compares ISA-compliant outputs.
#
# [3] Single sample per point -> Warm-up + REPS iterations, min/median reporting
#
# [4] Memory was not effectively measured
#     Polled from the parent at 0.5s intervals, but targets were 20-900ms processes.
#     v4 samples at 0.2ms intervals from within the child process (same as test1_v3.py).
#
# [5] "success" only meant "no exceptions were raised"
#     v4 verifies Operator equivalence at small scales before benchmarking. 
#     Since routing permutes qubits, Operator.from_circuit() is used to account 
#     for final layout.
#
# [6] Fixed Queue.empty() race condition, forced spawn
#     (To make Windows and Linux results comparable. Using fork passes the BLAS 
#     thread pool to the child, causing Qiskit's transpile to hang).
#
# [7] Record exactly how many blocks PSF processed every time
#     If this had been output, the v3 error would have been noticed on the first run.
# ===========================================================================
from __future__ import annotations

import argparse
import logging
import math
import multiprocessing
import platform
import statistics
import sys
import threading
import time
import tracemalloc

import numpy as np
import pandas as pd
import psutil
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, random_unitary
from qiskit.transpiler import CouplingMap

import psf_compile

QUBIT_SIZES = [50, 100, 156, 300]
GATES_PER_PAIR = 20
DEPTH = 100                      # For passthrough workload
SEEDS = [1, 2, 3]
REPS = 3
TIMEOUT_SECONDS = 900
CHECK_QUBITS = 6
MEM_WINDOW_S = 0.05
MEM_MAX_CALLS = 20
HW_BASIS = ["rz", "sx", "x", "cx"]
SEED_TRANSPILER = 7
OUTPUT_CSV = "phase3_v4_physical_topology_results.csv"


# --------------------------------------------------------------------------
def get_grid_cmap(num_qubits):
    cols = int(math.ceil(math.sqrt(num_qubits)))
    rows = int(math.ceil(num_qubits / cols))
    return CouplingMap.from_grid(rows, cols)


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed=1):
    """Deep 2-qubit interactions on the same pair. A structure PSF-Zero can compress."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for (a, b) in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def build_workload(kind, num_qubits, seed):
    if kind == "dense":
        return build_dense_pair_blocks_circuit(num_qubits, GATES_PER_PAIR, seed)
    if kind == "passthrough":
        return random_circuit(num_qubits=num_qubits, depth=DEPTH,
                              measure=False, seed=seed)
    raise ValueError(kind)


def count_coupling_violations(qc, cmap):
    edges = set()
    for a, b in cmap.get_edges():
        edges.add((a, b))
        edges.add((b, a))
    total_2q = violations = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            total_2q += 1
            qi = qc.find_bit(inst.qubits[0]).index
            qj = qc.find_bit(inst.qubits[1]).index
            if (qi, qj) not in edges:
                violations += 1
    return total_2q, violations


# --------------------------------------------------------------------------
# Arm definitions. qiskit opt=1 is the "fast but poor quality" side, opt=2/3 are baselines.
# --------------------------------------------------------------------------
def _qiskit_arm(level):
    def run(qc, cmap):
        return transpile(qc, coupling_map=cmap, basis_gates=HW_BASIS,
                         optimization_level=level, seed_transpiler=SEED_TRANSPILER)
    return run


def _psf_arm(level, basis):
    def run(qc, cmap):
        return psf_compile.compile_for_hardware(
            qc, cmap, basis_gates=HW_BASIS, routing_optimization_level=level,
            verify=False, entangling_basis=basis, seed_transpiler=SEED_TRANSPILER)
    return run


ARMS = {
    "qiskit_opt1": dict(fn=_qiskit_arm(1), family="Qiskit",
                        note="Fast but emits many 2q gates due to lack of consolidation"),
    "qiskit_opt2": dict(fn=_qiskit_arm(2), family="Qiskit",
                        note="Quality baseline (with consolidation)"),
    "qiskit_opt3": dict(fn=_qiskit_arm(3), family="Qiskit",
                        note="Quality baseline (maximum effort)"),
    "psf_rl1_cx": dict(fn=_psf_arm(1, "cx"), family="PSF-Zero",
                       note="Default in v4. Quality matches opt2 2q count, depth is +30-40%"),
    "psf_rl2_cx": dict(fn=_psf_arm(2, "cx"), family="PSF-Zero",
                       note="Old default. Output is bit-for-bit identical to plain transpile(opt=2)"),
}
DEFAULT_ARMS = list(ARMS)


# --------------------------------------------------------------------------
class RssSampler(threading.Thread):
    def __init__(self, interval=0.0002):
        super().__init__(daemon=True)
        self.interval = interval
        # Avoid naming conflict with internal threading.Thread._stop
        self._stop_evt = threading.Event()
        self.peak = 0.0
        self.samples = 0
        self._proc = psutil.Process()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                rss = self._proc.memory_info().rss / (1024 * 1024)
            except psutil.Error:
                break
            self.peak = max(self.peak, rss)
            self.samples += 1
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)


class _BlockLog(logging.Handler):
    """Catches how many blocks PSF-Zero processed."""

    def __init__(self):
        super().__init__()
        self.last = ""

    def emit(self, record):
        msg = record.getMessage()
        if "blocks" in msg:
            self.last = msg


def worker(spec, arm_name, reps, result_queue):
    try:
        kind, num_qubits, seed = spec
        fn = ARMS[arm_name]["fn"]
        qc = build_workload(kind, num_qubits, seed)
        cmap = get_grid_cmap(num_qubits)

        handler = _BlockLog()
        lg = logging.getLogger("psf_compile")
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)

        fn(qc, cmap)                                   # Warm-up (outside timer)
        block_note = handler.last

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn(qc, cmap)
            times.append(time.perf_counter() - t0)

        total_2q, violations = count_coupling_violations(out, cmap)
        ops = set(out.count_ops()) - {"barrier", "measure"}
        off_basis = sorted(ops - set(HW_BASIS))

        proc = psutil.Process()
        tracemalloc.start()
        fn(qc, cmap)
        _, tm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        baseline = proc.memory_info().rss / (1024 * 1024)
        sampler = RssSampler()
        sampler.start()
        t_end = time.perf_counter() + MEM_WINDOW_S
        mem_calls = 0
        main_peak = baseline
        while time.perf_counter() < t_end and mem_calls < MEM_MAX_CALLS:
            fn(qc, cmap)
            mem_calls += 1
            main_peak = max(main_peak, proc.memory_info().rss / (1024 * 1024))
        sampler.stop()
        peak_rss = max(sampler.peak, main_peak, baseline)

        status = "success" if violations == 0 else f"invalid ({violations} coupling violations)"
        if off_basis:
            status = f"invalid (off-basis gates: {off_basis})"

        result_queue.put({
            "status": status, "times": times,
            "final_2q_gates": total_2q, "final_depth": out.depth(),
            "coupling_violations": violations,
            "off_basis": ",".join(off_basis),
            "psf_blocks": block_note,
            "baseline_rss_mb": baseline, "peak_rss_mb": peak_rss,
            "rss_delta_mb": peak_rss - baseline,
            "rss_samples": sampler.samples + mem_calls,
            "tracemalloc_peak_mb": tm_peak / (1024 * 1024),
        })
    except Exception as e:  # noqa: BLE001
        result_queue.put({"status": f"error: {type(e).__name__}: {e}", "times": None})


def equivalence_worker(spec, arm_name, result_queue):
    """Routing permutes qubits, and if the grid is larger than the circuit, ancillas 
    increase. We pad the circuit and use Operator.from_circuit() for comparison."""
    try:
        kind, num_qubits, seed = spec
        qc = build_workload(kind, num_qubits, seed)
        cmap = get_grid_cmap(num_qubits)
        out = ARMS[arm_name]["fn"](qc, cmap)
        pad = QuantumCircuit(out.num_qubits)
        pad.compose(qc, range(qc.num_qubits), inplace=True)
        a = Operator(pad).data
        b = Operator.from_circuit(out).data
        d = a.shape[0]
        tr = np.trace(a.conj().T @ b)
        result_queue.put({"status": "success",
                          "infidelity": 1.0 - float(abs(tr) ** 2) / (d * d)})
    except Exception as e:  # noqa: BLE001
        result_queue.put({"status": f"error: {type(e).__name__}: {e}",
                          "infidelity": None})


def run_isolated(target, args, timeout=TIMEOUT_SECONDS):
    """v3 checked Queue.empty() after join() (a race condition, and a typical deadlock 
    scenario with a child holding a large result). We pull first, then join."""
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(*args, result_queue))
    process.start()
    result = None
    deadline = time.time() + timeout
    while True:
        try:
            result = result_queue.get(timeout=0.2)
            break
        except Exception:
            pass
        if not process.is_alive():
            break
        if time.time() > deadline:
            process.terminate()
            process.join(timeout=10)
            return {"status": "timeout (Dead Zone)", "times": None}
    process.join(timeout=30)
    return result if result is not None else {"status": "crash (OOM)", "times": None}


# --------------------------------------------------------------------------
def summarize(df):
    ok = df[df["Status"] == "success"]
    if ok.empty:
        print("\nNo successful results to summarize")
        return
    for kind, sub in ok.groupby("Workload"):
        print("\n" + "=" * 82)
        print(f"SUMMARY -- workload={kind}")
        if kind == "passthrough":
            print("  * PSF-Zero is designed to pass through with 0/0 blocks. This is not for speed comparison,")
            print("    but a control to demonstrate behavior out of scope.")
        print("=" * 82)
        order = [a for a in DEFAULT_ARMS if a in set(sub["Arm"])]
        t = sub.pivot_table(index="Qubits", columns="Arm",
                            values="Time_min_s", aggfunc="median")[order]
        g = sub.pivot_table(index="Qubits", columns="Arm",
                            values="Final_2Q_Gates", aggfunc="median")[order]
        d = sub.pivot_table(index="Qubits", columns="Arm",
                            values="Final_Depth", aggfunc="median")[order]
        print("\nCompile time min (ms)")
        print((t * 1000).round(1).to_string())
        print("\n2-Qubit gates")
        print(g.astype(int).to_string())
        print("\nDepth")
        print(d.astype(int).to_string())
        for base in ("qiskit_opt1", "qiskit_opt2"):
            if base not in t.columns:
                continue
            print(f"\nSpeedup vs {base} (>1 means PSF-Zero is faster)")
            for psf in ("psf_rl1_cx", "psf_rl2_cx"):
                if psf not in t.columns:
                    continue
                r = (t[base] / t[psf]).round(2)
                print(f"  {psf:12s}: " + "  ".join(f"{q}q={v}x" for q, v in r.items()))


def main():
    ap = argparse.ArgumentParser(description="Real-device topology validation v4")
    ap.add_argument("--qubits", type=int, nargs="+", default=QUBIT_SIZES)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    ap.add_argument("--workloads", nargs="+", default=["dense", "passthrough"],
                    choices=["dense", "passthrough"])
    ap.add_argument("--check-qubits", type=int, default=CHECK_QUBITS)
    ap.add_argument("--out", default=OUTPUT_CSV)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.qubits, args.seeds, args.reps = [50], 1, 2

    seeds = list(range(1, args.seeds + 1))
    env = dict(Platform=platform.platform(), Python=sys.version.split()[0],
               Qiskit=__import__("qiskit").__version__,
               StartMethod=multiprocessing.get_start_method(),
               CPU=platform.processor() or "unknown")
    print("environment:", env)
    print(f"arms={args.arms} workloads={args.workloads} qubits={args.qubits} "
          f"seeds={seeds} reps={args.reps}")

    equivalence = {}
    if args.check_qubits:
        print(f"\n--- Equivalence check @ {args.check_qubits} qubits (dense) ---")
        for arm in args.arms:
            r = run_isolated(equivalence_worker,
                             (("dense", args.check_qubits, 7), arm), timeout=600)
            infid = r.get("infidelity")
            equivalence[arm] = infid
            shown = "n/a" if infid is None else f"{infid:.3e}"
            ok = "OK" if infid is not None and abs(infid) < 1e-9 else "*** FAIL ***"
            print(f"   {arm:12s} 1-fidelity = {shown:12s} {ok}")
        if any(v is None or abs(v) >= 1e-9 for v in equivalence.values()):
            print("   Warning: Some arms did not reproduce the input unitary. "
                  "Their times are meaningless.")

    results = []
    for kind in args.workloads:
        for q in args.qubits:
            cmap = get_grid_cmap(q)
            print(f"\n{'='*74}")
            print(f"workload={kind}  {q} qubits (grid {cmap.size()}q)")
            print(f"{'='*74}", flush=True)
            for seed in seeds:
                for arm in args.arms:
                    print(f"[{arm}] seed {seed} ...", flush=True)
                    r = run_isolated(worker, ((kind, q, seed), arm, args.reps))
                    times = r.get("times")
                    results.append({
                        "Workload": kind, "Arm": arm, "Family": ARMS[arm]["family"],
                        "Qubits": q, "Seed": seed, "Reps": args.reps,
                        "Status": r["status"],
                        "Time_min_s": min(times) if times else None,
                        "Time_median_s": statistics.median(times) if times else None,
                        "Time_stdev_s": (statistics.stdev(times)
                                         if times and len(times) > 1 else None),
                        "Final_2Q_Gates": r.get("final_2q_gates"),
                        "Final_Depth": r.get("final_depth"),
                        "Coupling_Violations": r.get("coupling_violations"),
                        "Off_Basis": r.get("off_basis", ""),
                        "PSF_Blocks": r.get("psf_blocks", ""),
                        "Peak_RSS_MB": r.get("peak_rss_mb"),
                        "RSS_Delta_MB": r.get("rss_delta_mb"),
                        "TraceMalloc_Peak_MB": r.get("tracemalloc_peak_mb"),
                        "Equiv_Infidelity": equivalence.get(arm),
                        **env,
                    })
                    pd.DataFrame(results).to_csv(args.out, index=False)
                    if times:
                        print(f"  -> {r['status']}  min {min(times)*1000:.1f} ms"
                              f"  2q={r.get('final_2q_gates')}"
                              f"  depth={r.get('final_depth')}"
                              f"  viol={r.get('coupling_violations')}"
                              f"  dRSS={r.get('rss_delta_mb'):.1f}MB")
                        if r.get("psf_blocks"):
                            print(f"     {r['psf_blocks']}")
                    else:
                        print(f"  -> {r['status']}")

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    summarize(df)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Windows only supports spawn. If Linux uses its default fork, the child process
    # inherits a state where the parent has already touched NumPy/OpenBLAS, causing
    # Qiskit's transpile to deadlock during BLAS calls. We force spawn to align 
    # both OS environments and avoid this issue.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()