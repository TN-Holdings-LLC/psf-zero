# test1.py -- v3
#
# ===========================================================================
# Changes from v2 -> v3 (2026-09-08)
# ===========================================================================
# v2 fixed the *circuit* (dense adjacent-pair SU(4) blocks instead of
# random_circuit, which produced only 3-4 gate blocks). v3 fixes the
# *measurement*. Every item below was measured before it was changed; the
# numbers quoted are from a Linux run of the v2 script's own methodology.
#
# [1] NO WARM-UP -> the reported number was Qiskit's cold-start, not its
#     compile time. First call to transpile() in a fresh process builds the
#     preset pass managers and resolves plugin entry points:
#         qiskit  15q: call 1 = 67.8 ms, warm median = 3.8 ms  (18.0x)
#         psf     15q: call 1 =  2.5 ms, warm median = 1.4 ms  ( 1.9x)
#     Because the penalty is 18x on one side and 1.9x on the other, and
#     because it is a FIXED cost, it inflates the speed-up most at small
#     qubit counts. That, not an algorithmic property, is why v2's speed-up
#     appeared to shrink from 7.6x (15q) to 2.3x (156q). Warm, the same
#     comparison is a roughly flat 2.3-3.0x. Both workers now warm up
#     outside the timer.
#
# [2] ONE SAMPLE PER POINT against a 1-25 ms workload. Warm repeats at 156q
#     measured a 198.7% spread on the Qiskit side (min 22.3, max 96.3 ms).
#     A single sample there is noise, not a measurement. v3 repeats REPS
#     times inside the same process and reports min / median / stdev.
#
# [3] MEMORY WAS NEVER ACTUALLY SAMPLED. The parent polled RSS every 100 ms
#     while the timed work took 1.5-31 ms, i.e. 0.02 to 0.31 samples per
#     measurement. The "Peak_Memory_MB" column in v2 is the child's import
#     footprint, not the compilation peak. v3 samples RSS from a thread
#     inside the child at 0.2 ms during a dedicated (untimed) pass, and also
#     records tracemalloc's exact Python-side peak.
#
# [4] THE TWO COMPILERS WERE NOT PRODUCING COMPARABLE OUTPUT, and the script
#     recorded nothing that would reveal it. At 50 qubits:
#         qiskit opt=0 : 14000 gates, 1500 cx, depth 320
#         qiskit opt=3 :   700 gates,   75 cx, depth  16
#         psf-zero     :   375 gates,   75 2q, depth   9
#     v2 timed PSF-Zero against opt=0 -- the arm that skips the block
#     consolidation PSF-Zero itself performs, and emits a 20x worse circuit.
#     v3 records gate counts and depth for every arm and adds the two honest
#     baselines: opt=3 (same output quality) and entangling_basis="cx"
#     (same output basis). PSF-Zero wins both -- by MORE than the old
#     comparison suggested (4.1-5.6x vs opt=3), which is the point.
#
# [5] "SUCCESS" MEANT "DID NOT RAISE". Nothing checked that PSF-Zero's output
#     was equivalent to its input. v3 runs an Operator-equivalence check at a
#     small qubit count for every arm and records the infidelity, and counts
#     any synthesis fallbacks PSF-Zero reported.
#
# [6] Robustness/portability: Queue.empty() is racy (replaced with a timed
#     get), backend_mock was dead, and the fork/spawn start method changes how
#     warm the child is -- so it is now recorded in the CSV alongside the
#     platform and Qiskit version. Results from a Windows (spawn) run and a
#     Linux (fork) run are NOT directly comparable without it.
# ===========================================================================
from __future__ import annotations

import argparse
import multiprocessing
import platform
import statistics
import sys
import threading
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import psutil
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary

import psf_compile

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
QUBIT_SIZES = [15, 50, 100, 156]
GATES_PER_PAIR = 20
SEEDS = list(range(1, 11))
REPS = 5                     # timed repetitions inside one child process
TIMEOUT_SECONDS = 3600
CHECK_QUBITS = 6             # size used for the Operator-equivalence check
MEM_WINDOW_S = 0.05          # sampling window for the RSS pass
MEM_MAX_CALLS = 50
OUTPUT_CSV = "phase1_v3_benchmark_results.csv"
HW_BASIS = ["rz", "sx", "x", "cx"]
TWO_Q_NAMES = ("cx", "cz", "ecr", "rxx", "ryy", "rzz", "unitary", "swap", "iswap")


# --------------------------------------------------------------------------
# Circuit under test (unchanged from v2)
# --------------------------------------------------------------------------
def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed=2):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    for (a, b) in pairs:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


# --------------------------------------------------------------------------
# Arms. Each is one compiler configuration. "group" says which arms are
# fair to compare on time: they must produce output of comparable quality.
# --------------------------------------------------------------------------
def _qiskit_arm(level):
    def run(qc):
        return transpile(qc, basis_gates=HW_BASIS, optimization_level=level)
    return run


def _psf_arm(**kw):
    def run(qc):
        return psf_compile.compile(qc, **kw)
    return run


ARMS = {
    # kept so v3 numbers stay comparable with the v2 CSV, but note the group:
    # opt=0 does no block consolidation, so its output is ~20x worse and its
    # time is NOT a fair reference for PSF-Zero.
    "qiskit_opt0": dict(fn=_qiskit_arm(0), family="Qiskit",
                        basis="rz/sx/x/cx", group="unconsolidated",
                        note="no consolidation; kept for continuity with v2"),
    "qiskit_opt3": dict(fn=_qiskit_arm(3), family="Qiskit",
                        basis="rz/sx/x/cx", group="consolidated",
                        note="quality-matched baseline"),
    "psf_canonical": dict(fn=_psf_arm(verify=True), family="PSF-Zero",
                          basis="rxx/ryy/rzz + rz/ry", group="consolidated",
                          note="default; verify=True is the cheap core check"),
    "psf_cx": dict(fn=_psf_arm(verify=True, entangling_basis="cx"),
                   family="PSF-Zero", basis="cx + rz/ry", group="consolidated",
                   note="basis-matched baseline vs qiskit_opt3"),
}
DEFAULT_ARMS = ["qiskit_opt0", "qiskit_opt3", "psf_canonical", "psf_cx"]


# --------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------
class RssSampler(threading.Thread):
    """Samples this process's RSS at a high rate. v2 sampled from the PARENT
    every 100 ms, which cannot see a workload that lasts a few milliseconds."""

    def __init__(self, interval=0.0002):
        super().__init__(daemon=True)
        self.interval = interval
        # NOTE: must not be named _stop -- threading.Thread._stop is an
        # internal method, and shadowing it breaks join() and thread teardown.
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
            if rss > self.peak:
                self.peak = rss
            self.samples += 1
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)


def circuit_stats(qc):
    ops = qc.count_ops()
    return dict(
        gates_total=int(sum(ops.values())),
        gates_2q=int(sum(n for g, n in ops.items() if g in TWO_Q_NAMES)),
        depth=int(qc.depth()),
    )


def worker(spec, arm_name, reps, result_queue):
    """Runs entirely in the child. Builds its own circuit so nothing large is
    pickled across, and so the RSS baseline is taken after construction."""
    try:
        num_qubits, gates_per_pair, seed = spec
        arm = ARMS[arm_name]
        fn = arm["fn"]
        qc = build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed)

        # [1] warm-up, OUTSIDE the timer, symmetric across every arm.
        fallback_note = ""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fn(qc)
            if caught:
                fallback_note = " | ".join(str(w.message)[:200] for w in caught)

        # [2] timed repetitions, no sampler thread running.
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn(qc)
            times.append(time.perf_counter() - t0)

        stats = circuit_stats(out)

        # [3] memory in its own untimed pass, sampled from inside the child.
        # tracemalloc gives the exact Python-side peak for ONE call; the RSS
        # sampler needs a window it can actually land samples in, so it runs
        # over repeated calls (the per-call allocation pattern repeats, so the
        # peak over the window is still a per-call peak).
        proc = psutil.Process()
        tracemalloc.start()
        fn(qc)
        _, tm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        baseline = proc.memory_info().rss / (1024 * 1024)
        sampler = RssSampler()
        sampler.start()
        t_end = time.perf_counter() + MEM_WINDOW_S
        mem_calls = 0
        main_peak = baseline
        while time.perf_counter() < t_end and mem_calls < MEM_MAX_CALLS:
            fn(qc)
            mem_calls += 1
            # Arms that stay in pure Python hold the GIL and can starve the
            # sampler thread (psf_cx landed 2 samples where psf_canonical, which
            # releases the GIL in Rust, landed 58), so sample here too.
            rss = proc.memory_info().rss / (1024 * 1024)
            if rss > main_peak:
                main_peak = rss
        sampler.stop()
        peak_rss = max(sampler.peak, main_peak, baseline)

        result_queue.put({
            "status": "success",
            "times": times,
            "baseline_rss_mb": baseline,
            "peak_rss_mb": peak_rss,
            "rss_delta_mb": peak_rss - baseline,
            "rss_samples": sampler.samples + mem_calls,
            "mem_calls": mem_calls,
            "tracemalloc_peak_mb": tm_peak / (1024 * 1024),
            "fallback_note": fallback_note,
            **stats,
        })
    except Exception as e:  # noqa: BLE001 - the point is to report it
        result_queue.put({"status": f"error: {type(e).__name__}: {e}", "times": None})


def equivalence_worker(spec, arm_name, result_queue):
    """[5] Does the arm actually preserve the unitary? Small size only:
    Operator() is 2^n x 2^n."""
    try:
        num_qubits, gates_per_pair, seed = spec
        qc = build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = ARMS[arm_name]["fn"](qc)
        a = Operator(qc).data
        b = Operator(out).data
        d = a.shape[0]
        tr = np.trace(a.conj().T @ b)
        infid = 1.0 - float(abs(tr) ** 2) / (d * d)
        result_queue.put({"status": "success", "infidelity": infid})
    except Exception as e:  # noqa: BLE001
        result_queue.put({"status": f"error: {type(e).__name__}: {e}",
                          "infidelity": None})


def run_isolated(target, args, timeout=TIMEOUT_SECONDS):
    """Spawns the worker, enforces the timeout, and drains the queue BEFORE
    joining. v2 called join() first and then Queue.empty(), which is both racy
    and the classic way to deadlock a child holding a large result."""
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
            return {"status": "timeout", "times": None}

    process.join(timeout=30)
    if result is None:
        return {"status": "crash", "times": None}
    return result


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summarize(df):
    ok = df[df["Status"] == "success"]
    if ok.empty:
        print("\nno successful runs to summarize")
        return

    print("\n" + "=" * 78)
    print("SUMMARY -- median of per-point medians, over seeds")
    print("=" * 78)
    piv = ok.pivot_table(index="Qubits", columns="Arm",
                         values="Time_median_s", aggfunc="median")
    order = [a for a in DEFAULT_ARMS if a in piv.columns]
    piv = piv[order]
    print("\ncompile time (ms)")
    print((piv * 1000).round(2).to_string())

    print("\noutput circuit (median over seeds)")
    q = ok.pivot_table(index="Qubits", columns="Arm",
                       values="Gates_2q", aggfunc="median")[order]
    d = ok.pivot_table(index="Qubits", columns="Arm",
                       values="Depth", aggfunc="median")[order]
    print("  2-qubit gates")
    print(q.astype(int).to_string())
    print("  depth")
    print(d.astype(int).to_string())

    pmin = ok.pivot_table(index="Qubits", columns="Arm",
                          values="Time_min_s", aggfunc="min")[order]
    print("\nspeed-up (higher = PSF-Zero faster). Compare only within a group:")
    print("  qiskit_opt3 is the quality-matched baseline; qiskit_opt0 does no")
    print("  block consolidation and emits ~20x more 2-qubit gates.")
    print("  min-of-all-samples is the noise-resistant statistic; median shows")
    print("  what a single run would typically see.")
    for psf in ("psf_canonical", "psf_cx"):
        if psf not in piv.columns:
            continue
        for base in ("qiskit_opt0", "qiskit_opt3"):
            if base not in piv.columns:
                continue
            tag = "   <- fair" if base == "qiskit_opt3" else ""
            r_med = (piv[base] / piv[psf]).round(2)
            r_min = (pmin[base] / pmin[psf]).round(2)
            print(f"    {base:12s} / {psf:14s}")
            print("        median: "
                  + "  ".join(f"{q}q={v}x" for q, v in r_med.items()) + tag)
            print("        min   : "
                  + "  ".join(f"{q}q={v}x" for q, v in r_min.items()))

    print("\npeak RSS above baseline (MB, measured inside the child)")
    m = ok.pivot_table(index="Qubits", columns="Arm",
                       values="RSS_Delta_MB", aggfunc="median")[order]
    print(m.round(1).to_string())


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="PSF-Zero vs Qiskit, phase 1 (v3)")
    ap.add_argument("--qubits", type=int, nargs="+", default=QUBIT_SIZES)
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="number of seeds, 1..N")
    ap.add_argument("--reps", type=int, default=REPS,
                    help="timed repetitions inside each child process")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                    choices=list(ARMS))
    ap.add_argument("--gates-per-pair", type=int, default=GATES_PER_PAIR)
    ap.add_argument("--check-qubits", type=int, default=CHECK_QUBITS,
                    help="size for the Operator-equivalence check; 0 to skip")
    ap.add_argument("--out", default=OUTPUT_CSV)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 15/50 qubits, 2 seeds, 3 reps")
    args = ap.parse_args()

    if args.quick:
        args.qubits, args.seeds, args.reps = [15, 50], 2, 3

    seeds = list(range(1, args.seeds + 1))
    env = dict(
        Platform=platform.platform(),
        Python=sys.version.split()[0],
        Qiskit=__import__("qiskit").__version__,
        StartMethod=multiprocessing.get_start_method(),
        CPU=platform.processor() or "unknown",
    )
    print("environment:", env)
    print(f"arms={args.arms}  qubits={args.qubits}  seeds={seeds}  reps={args.reps}")

    # ---- [5] correctness first: a fast benchmark of a wrong compiler is worthless
    equivalence = {}
    if args.check_qubits:
        print(f"\n--- equivalence check at {args.check_qubits} qubits ---")
        for arm in args.arms:
            r = run_isolated(equivalence_worker,
                             ((args.check_qubits, args.gates_per_pair, 7), arm),
                             timeout=600)
            equivalence[arm] = r.get("infidelity")
            infid = r.get("infidelity")
            # abs(): 1 - |tr|^2/d^2 can come out very slightly negative at
            # machine precision, which is a pass, not a fail.
            verdict = ("OK" if infid is not None and abs(infid) < 1e-9
                       else "*** FAIL ***")
            shown = "n/a" if infid is None else f"{infid:.3e}"
            print(f"   {arm:14s} 1-fidelity = {shown:12s} {verdict}")
        if any(v is None or abs(v) >= 1e-9 for v in equivalence.values()):
            print("   WARNING: an arm did not reproduce the input unitary. "
                  "Timings below are not meaningful for that arm.")

    results = []
    for q in args.qubits:
        print(f"\n--- Starting evaluation: {q} Qubits ---")
        for seed in seeds:
            for arm in args.arms:
                print(f"[{arm}] Qubits: {q}, Seed: {seed} running...", flush=True)
                r = run_isolated(worker, ((q, args.gates_per_pair, seed),
                                          arm, args.reps))
                times = r.get("times")
                row = {
                    "Arm": arm,
                    "Family": ARMS[arm]["family"],
                    "OutputBasis": ARMS[arm]["basis"],
                    "Group": ARMS[arm]["group"],
                    "Qubits": q,
                    "GatesPerPair": args.gates_per_pair,
                    "Seed": seed,
                    "Reps": args.reps,
                    "Status": r["status"],
                    "Time_min_s": min(times) if times else None,
                    "Time_median_s": statistics.median(times) if times else None,
                    "Time_max_s": max(times) if times else None,
                    "Time_stdev_s": (statistics.stdev(times)
                                     if times and len(times) > 1 else None),
                    "Peak_RSS_MB": r.get("peak_rss_mb"),
                    "Baseline_RSS_MB": r.get("baseline_rss_mb"),
                    "RSS_Delta_MB": r.get("rss_delta_mb"),
                    "RSS_Samples": r.get("rss_samples"),
                    "TraceMalloc_Peak_MB": r.get("tracemalloc_peak_mb"),
                    "Gates_total": r.get("gates_total"),
                    "Gates_2q": r.get("gates_2q"),
                    "Depth": r.get("depth"),
                    "Fallbacks": r.get("fallback_note", ""),
                    "Equiv_Infidelity": equivalence.get(arm),
                    **env,
                }
                results.append(row)
                pd.DataFrame(results).to_csv(args.out, index=False)

                if times:
                    print(f"  -> {r['status']}  min {min(times)*1000:.2f} ms"
                          f"  median {statistics.median(times)*1000:.2f} ms"
                          f"  2q={r.get('gates_2q')} depth={r.get('depth')}"
                          f"  dRSS={r.get('rss_delta_mb'):.1f}MB"
                          f" ({r.get('rss_samples')} samples)")
                    if r.get("fallback_note"):
                        print(f"     fallback: {r['fallback_note']}")
                else:
                    print(f"  -> {r['status']}")

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    summarize(df)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # [6] "spawn" everywhere, for two reasons.
    #  (a) Portability: Windows can only spawn. Under Linux's default "fork"
    #      the child inherits the parent's already-warmed interpreter, so the
    #      two platforms were not measuring the same thing at all.
    #  (b) Correctness: this parent has already touched NumPy/OpenBLAS (importing
    #      psf_compile builds the core eigenbasis), and forking a process with a
    #      live BLAS thread pool deadlocks the child the moment it calls into
    #      BLAS. Qiskit's transpile does, and hung indefinitely under fork.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
