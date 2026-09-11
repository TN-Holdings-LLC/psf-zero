#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_canonical_penalty.py -- Isolating layer by layer why only psf_canonical is slow on this machine

Background
----------
The test1_v3.py output on the AMD machine from 2026-09-10 reproduced the degradation
table (4.35/2.21/1.66/1.44) at 4.23/2.11/1.63/1.39, which the README concluded as a
"single anomalous run, permanently unexplainable". Since the 3 runs on the Intel
machine clump around 8.0/4.1/3.3/3.0, this is a machine difference.

Comparing by arm against Intel run2, the penalty is not uniform:

    qiskit_opt0    1.24 to 1.53x
    qiskit_opt3    1.31 to 1.47x
    psf_cx         1.71 to 2.09x
    psf_canonical  2.50 to 3.30x   <- Stands out here

In other words, this cannot be explained by "this machine is just slow overall".
There is something specific to the canonical path.
This script measures which layer is responsible. **test1_v3.py will not be modified.**

Layers to measure
-----------------
  prep          Collect2qBlocks + ConsolidateBlocks only (Qiskit's preprocessing)
  canonical_F   compile(verify=False, entangling_basis="canonical")  Entire process
  canonical_T   compile(verify=True,  entangling_basis="canonical")  Entire process <- Same condition as test1_v3's psf_canonical
  cx_F          compile(verify=False, entangling_basis="cx")         Entire process
  cx_T          compile(verify=True,  entangling_basis="cx")         Entire process <- Same condition as test1_v3's psf_cx
  synth_1block  Repeatedly synthesizes just a single 2-qubit block (minimizing Qiskit's contribution)
  core_raw      Calls psf_zero_core.geometric_decompose() directly (completely bypassing Qiskit)
  core_raw_ch   Same, but geometric_decompose_checked() (includes numerical checks on the Rust side)
  numpy_ref     eig/matmul loop for 4x4 complex matrices (baseline for the machine's raw numerical performance)

"synthesis = canonical_F − prep" yields the raw synthesis cost.

Updates in v2 (2026-09-11, after seeing the initial Intel run)
--------------------------------------------------------------
The first version had two condition mismatches and couldn't be directly compared with
the test1_v3.py observations.

1. **cx was only measured with verify=False.** test1_v3.py's psf_cx uses verify=True.
   The comparison target was wrong. -> Added cx_verifyTrue and aligned synth_1block
   to verify=True.
2. **The circuit was from phase3_v5** (where UnitaryGates are decomposed into basis
   gates before being appended). test1_v3.py appends UnitaryGates directly, so the
   gate count is about 11x different for the same n. -> Added --fixture and
   **changed the default to test1v3**.

The Intel data collected in the first version is valid as a baseline per layer, but
cannot be used for the cx ÷ canonical comparison. Must be retaken in v2.

Pre-registered Predictions (written before measuring)
---------------------------------------------------
(H1) BLAS Thread Over-allocation Hypothesis
     When BLAS uses multithreading on tiny matrices like 4x4, synchronization overhead
     can exceed the computation itself. Machines with more cores are at a greater disadvantage.
     -> Supported if --threads 1 eliminates or drastically shrinks the canonical penalty.
        Rejected if it doesn't disappear.
(H2) Rust Core Specific Hypothesis
     If only core_raw / synth_1block are 2.5-3.3x slower, while prep and numpy_ref stay
     around 1.3x, the cause lies within the synthesis path.
(H3) Machine is Just Slow Hypothesis (Already doubtful)
     Supported if all layers are uniformly 2.5-3.3x slower. However, this contradicts
     existing data where qiskit_opt3 was only 1.3x slower, so this shouldn't happen.

Usage
-----
    python diag_canonical_penalty.py --out diag_<machine>_<date>.csv

    # Running twice with different thread counts is the main goal (default automatically runs both)
    python diag_canonical_penalty.py --threads-arms default,1 --out diag_...csv

**Run and compare on BOTH machines (AMD and Intel). Running on just one is meaningless.**
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time

import numpy as np

GATES_PER_PAIR = 20
BASIS_GATES = ["rz", "sx", "x", "cx"]


# ----------------------------------------------------------------- fixtures
def build_test1v3_circuit(num_qubits, gates_per_pair, seed=0):
    """Circuit from test1_v3.py. Verbatim copy. Appends UnitaryGates as-is.

    **This is the default.** The phenomenon we want to explain (only psf_canonical
    stalls on AMD) was observed with this circuit, so dividing it up with the same
    circuit is necessary for it to mean anything.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for (a, b) in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed=0):
    """Circuit from phase3_v5_spare_qubits.py. Verbatim copy. UnitaryGates are
    decomposed into basis gates before being appended, resulting in about 11x
    more gates for the same n."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for (a, b) in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


FIXTURES = {"test1v3": build_test1v3_circuit,
            "dense_blocks": build_dense_pair_blocks_circuit}


# ----------------------------------------------------------------- layers
def _time(fn, reps):
    """One warmup call is outside the timer. Returns min and median."""
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def layer_prep(n, seed, reps, fixture="test1v3"):
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    qc = FIXTURES[fixture](n, GATES_PER_PAIR, seed=seed)
    pm = PassManager([Collect2qBlocks(),
                      ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True)])
    return _time(lambda: pm.run(qc), reps)


def layer_compile(n, seed, reps, verify, basis, fixture="test1v3"):
    import psf_compile
    qc = FIXTURES[fixture](n, GATES_PER_PAIR, seed=seed)
    kw = {"verify": verify}
    try:                                  # To work even with versions lacking entangling_basis
        import inspect
        if "entangling_basis" in inspect.signature(psf_compile.compile).parameters:
            kw["entangling_basis"] = basis
        elif basis != "canonical":
            return None, None
    except (TypeError, ValueError):
        pass
    return _time(lambda: psf_compile.compile(qc, **kw), reps)


def layer_synth_1block(n_blocks, seed, reps, verify, basis):
    """Compiles a 1-block 2-qubit circuit n_blocks times. Qiskit's per-call
    fixed cost remains, but since the circuit is minimal, the synthesis
    contribution dominates."""
    import psf_compile
    import inspect
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    circs = []
    for _ in range(n_blocks):
        qc = QuantumCircuit(2)
        blk = QuantumCircuit(2)
        for _ in range(GATES_PER_PAIR):
            blk.append(UnitaryGate(random_unitary(4, seed=int(rng.integers(0, 2**31))).data), [0, 1])
        qc.compose(blk.decompose(), [0, 1], inplace=True)
        circs.append(qc)
    kw = {"verify": verify}
    if "entangling_basis" in inspect.signature(psf_compile.compile).parameters:
        kw["entangling_basis"] = basis
    elif basis != "canonical":
        return None, None

    def run():
        for c in circs:
            psf_compile.compile(c, **kw)
    return _time(run, reps)


def layer_core_raw(n_calls, seed, reps, fname="geometric_decompose"):
    """Calls the Rust core directly. The innermost layer, completely bypassing Qiskit.

    Argument shapes can vary by environment, so we try a few and use the one that works.
    In the sandbox (current psf_zero_core),
    `geometric_decompose(u_r: 2D f64 array, u_i: 2D f64 array)` works.
    If none work, returns None and silently skips.
    """
    try:
        import psf_zero_core
        from qiskit.quantum_info import random_unitary
    except Exception:
        return None, None
    fn = getattr(psf_zero_core, fname, None)
    if fn is None:
        return None, None
    rng = np.random.default_rng(seed)
    mats = [random_unitary(4, seed=int(rng.integers(0, 2**31))).data for _ in range(n_calls)]

    def as_2d(m):
        return (np.ascontiguousarray(m.real), np.ascontiguousarray(m.imag))

    def as_flat(m):
        return (np.ascontiguousarray(m.real.ravel()), np.ascontiguousarray(m.imag.ravel()))

    def as_lists(m):
        return (m.real.tolist(), m.imag.tolist())

    def as_single(m):
        return (m,)

    shaper = None
    for cand in (as_2d, as_flat, as_lists, as_single):
        try:
            fn(*cand(mats[0]))
            shaper = cand
            break
        except Exception:  # noqa: BLE001
            continue
    if shaper is None:
        return None, None
    args = [shaper(m) for m in mats]

    def run():
        for a in args:
            fn(*a)
    return _time(run, reps)


def layer_numpy_ref(n_calls, seed, reps):
    """Machine's raw numerical performance. eig and matmul on 4x4 complex matrices.
    If BLAS thread behavior has an effect, it should appear here too."""
    rng = np.random.default_rng(seed)
    mats = [rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)) for _ in range(n_calls)]

    def run():
        for m in mats:
            np.linalg.eig(m)
            m @ m
    return _time(run, reps)


# ----------------------------------------------------------------- driver
def _child(payload_json):
    """Spawned child process side. Performs 1 measurement and returns JSON."""
    p = json.loads(payload_json)
    kind = p["kind"]
    try:
        if kind == "prep":
            mn, md = layer_prep(p["n"], p["seed"], p["reps"],
                                p.get("fixture", "test1v3"))
        elif kind == "compile":
            mn, md = layer_compile(p["n"], p["seed"], p["reps"], p["verify"],
                                   p["basis"], p.get("fixture", "test1v3"))
        elif kind == "synth_1block":
            mn, md = layer_synth_1block(p["count"], p["seed"], p["reps"], p["verify"], p["basis"])
        elif kind == "core_raw":
            mn, md = layer_core_raw(p["count"], p["seed"], p["reps"],
                                    p.get("fname", "geometric_decompose"))
        elif kind == "numpy_ref":
            mn, md = layer_numpy_ref(p["count"], p["seed"], p["reps"])
        else:
            raise ValueError(kind)
        out = {"status": "success" if mn is not None else "skipped",
               "t_min": mn, "t_med": md}
    except Exception as e:  # noqa: BLE001
        out = {"status": f"error: {type(e).__name__}: {e}", "t_min": None, "t_med": None}
    print("RESULT " + json.dumps(out), flush=True)


def run_child(payload, threads):
    """1 measurement in a separate process with swapped environment variables. Thread settings
    need to take effect before import, so this is done via environment variables + a new process."""
    env = dict(os.environ)
    if threads is not None:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[k] = str(threads)
    cmd = [sys.executable, os.path.abspath(__file__), "--child", json.dumps(payload)]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "t_min": None, "t_med": None}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"status": f"no result (rc={r.returncode}) {r.stderr.strip()[:200]}",
            "t_min": None, "t_med": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--qubits", type=int, nargs="+", default=[15, 50, 156])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads-arms", default="default,1",
                    help="Comma-separated. 'default' keeps the environment as is, a number fixes it to that value")
    ap.add_argument("--block-count", type=int, default=200,
                    help="Number of iterations per measurement for synth_1block / core_raw / numpy_ref")
    ap.add_argument("--fixture", choices=sorted(FIXTURES), default="test1v3",
                    help="How to build the circuit. Default is test1v3 (same circuit as the phenomenon to explain)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.child:
        return _child(args.child)

    import pandas as pd
    env = {"Platform": platform.platform(), "Python": platform.python_version(),
           "StartMethod": "spawn", "CPU": platform.processor(),
           "CPU_count": os.cpu_count()}
    try:
        import qiskit
        env["Qiskit"] = qiskit.__version__
    except Exception:
        env["Qiskit"] = "?"
    env["numpy"] = np.__version__
    env["Fixture"] = args.fixture
    # Allow confirming later whether both machines are using the same Rust binary.
    # (To rule out the possibility that one was built without optimizations)
    try:
        import psf_zero_core
        f = getattr(psf_zero_core, "__file__", None)
        env["core_file"] = os.path.basename(f) if f else "?"
        env["core_bytes"] = os.path.getsize(f) if f and os.path.exists(f) else -1
    except Exception:
        env["core_file"], env["core_bytes"] = "?", -1
    try:
        cfg = np.__config__.show(mode="dicts")
        env["blas"] = str(cfg.get("Build Dependencies", {})
                             .get("blas", {}).get("name", "?"))
    except Exception:
        env["blas"] = "?"
    out_csv = args.out or f"diag_canonical_penalty_{time.strftime('%Y-%m-%d')}.csv"

    print("environment:", env)
    print(f"threads arms = {args.threads_arms}   qubits = {args.qubits}   "
          f"seeds = {args.seeds}   reps = {args.reps}\n")
    print("Pre-registered Predictions:")
    print("  H1 BLAS Thread Over-allocation -> --threads 1 eliminates the canonical penalty")
    print("  H2 Synthesis Path Specific      -> Only synth_1block / core_raw are prominently slower")
    print("  H3 Machine is Uniformly Slow    -> Same ratio across all layers. Contradicts existing data, so unlikely\n")

    rows = []
    for th_s in [t.strip() for t in args.threads_arms.split(",") if t.strip()]:
        threads = None if th_s == "default" else int(th_s)
        print(f"===== threads = {th_s} =====")
        for seed in args.seeds:
            # Layers independent of circuit size are measured only once
            for kind, payload, label in (
                ("core_raw", {"kind": "core_raw", "count": args.block_count,
                              "seed": seed, "reps": args.reps,
                              "fname": "geometric_decompose"}, "core_raw"),
                ("core_raw", {"kind": "core_raw", "count": args.block_count,
                              "seed": seed, "reps": args.reps,
                              "fname": "geometric_decompose_checked"},
                 "core_raw_checked"),
                ("numpy_ref", {"kind": "numpy_ref", "count": args.block_count,
                               "seed": seed, "reps": args.reps}, "numpy_ref"),
                ("synth_1block", {"kind": "synth_1block", "count": 20, "seed": seed,
                                  "reps": args.reps, "verify": True,
                                  "basis": "canonical"}, "synth_1block_canonical"),
                ("synth_1block", {"kind": "synth_1block", "count": 20, "seed": seed,
                                  "reps": args.reps, "verify": True,
                                  "basis": "cx"}, "synth_1block_cx"),
            ):
                r = run_child(payload, threads)
                print(f"  [seed{seed}] {label:<24} {r['status']:<10} "
                      f"min {(r['t_min'] or float('nan'))*1000:9.2f} ms", flush=True)
                rows.append(dict(Threads=th_s, Seed=seed, Qubits=-1, Layer=label,
                                 Status=r["status"], Time_min_s=r["t_min"],
                                 Time_median_s=r["t_med"], **env))
            for n in args.qubits:
                for payload, label in (
                    ({"kind": "prep", "n": n, "seed": seed, "reps": args.reps,
                      "fixture": args.fixture}, "prep"),
                    ({"kind": "compile", "n": n, "seed": seed, "reps": args.reps,
                      "verify": False, "basis": "canonical",
                      "fixture": args.fixture}, "canonical_verifyFalse"),
                    ({"kind": "compile", "n": n, "seed": seed, "reps": args.reps,
                      "verify": True, "basis": "canonical",
                      "fixture": args.fixture}, "canonical_verifyTrue"),
                    ({"kind": "compile", "n": n, "seed": seed, "reps": args.reps,
                      "verify": False, "basis": "cx",
                      "fixture": args.fixture}, "cx_verifyFalse"),
                    ({"kind": "compile", "n": n, "seed": seed, "reps": args.reps,
                      "verify": True, "basis": "cx",
                      "fixture": args.fixture}, "cx_verifyTrue"),
                ):
                    r = run_child(payload, threads)
                    print(f"  [seed{seed}] n={n:<4} {label:<22} {r['status']:<10} "
                          f"min {(r['t_min'] or float('nan'))*1000:9.2f} ms", flush=True)
                    rows.append(dict(Threads=th_s, Seed=seed, Qubits=n, Layer=label,
                                     Status=r["status"], Time_min_s=r["t_min"],
                                     Time_median_s=r["t_med"], **env))
        print()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}\n")

    ok = df[df.Status == "success"]
    if ok.empty:
        print("(No successful measurements)")
        return
    print("=" * 78)
    print("SUMMARY -- min time (ms), median across seeds")
    print("=" * 78)
    piv = ok.pivot_table(index=["Qubits", "Layer"], columns="Threads",
                         values="Time_min_s", aggfunc="median") * 1000
    print(piv.round(3).to_string())

    if {"default", "1"} <= set(ok.Threads.unique()):
        print("\nThread fixing effect (default ÷ 1thread. > 1.0 means fixing made it faster):")
        r = (piv["default"] / piv["1"]).round(2)
        print(r.to_string())
        print("\n  -> If only canonical types significantly exceed 1.0, supports H1 (BLAS threads).")
        print("     If all layers are ~1.0, H1 is rejected.")

    print("\ncx ÷ canonical (verify=True vs verify=True. Same condition as test1_v3's psf_cx ÷ psf_canonical):")
    for th in piv.columns:
        for n in args.qubits:
            try:
                v = piv[th][(n, "cx_verifyTrue")] / piv[th][(n, "canonical_verifyTrue")]
                print(f"  threads={th:<8} n={n:<4} {v:5.2f}")
            except KeyError:
                pass

    print("\nRaw synthesis cost (canonical_verifyFalse − prep, ms):")
    for th in ok.Threads.unique():
        sub = piv[th] if th in piv.columns else None
        if sub is None:
            continue
        for n in args.qubits:
            try:
                v = sub[(n, "canonical_verifyFalse")] - sub[(n, "prep")]
                print(f"  threads={th:<8} n={n:<4} {v:8.3f} ms")
            except KeyError:
                pass


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sys.exit(main())
