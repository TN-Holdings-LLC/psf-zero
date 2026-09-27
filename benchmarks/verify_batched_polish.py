"""verify_batched_polish.py -- Addendum 207: psf_compile.py 2026-09-27.1
(polish batched over all blocks, changelog item 20) against the per-block
polish of 2026-09-26.4, which the same file restores with
USE_BATCHED_POLISH = False.

Every comparison runs both settings on the same input in the same process.

  B1 fresh cliff circuits (FakeNighthawk, 120 logical, 60 pair24 blocks), 30:
     exact per-pair check of the batched output, and per-pair agreement of
     the batched and per-block outputs (both mapped back to logical qubits).
  B2 the number of blocks stepped by the polish, per compile, both settings
     (B1 circuits and B4 circuits).
  B3 cliff compounding loop, 100 laps per setting: per-pair distance to the
     lap-0 circuit at laps 10, 50, 100 (Addendum 206: 6.02e-12 at lap 100).
  B4 fresh training circuits (12 qubits), 30, at block_gate_floor 12 and 8:
     |compiled loss - reference loss|.
  B5 time: polish time per compile and compile median, cliff (B1 circuits,
     settings alternated per circuit, 3 compiles each).
  B6 time: training circuits at floor 8, compile median, both settings.

Usage (repository root; loop_endurance.py in benchmarks/):
    python -u benchmarks/verify_batched_polish.py 2>&1 | tee verify_batched_polish.txt
"""
from __future__ import annotations

import contextlib
import csv
import gc
import io
import math
import os
import platform
import statistics as st
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

import loop_endurance as le
import psf_compile as pc

PC = time.perf_counter
N_FRESH = 30
C_LAPS = 100
C_CHECK = (10, 50, 100)
OUT_CSV = "verify_batched_polish_2026-09-27.csv"
LINE = CouplingMap.from_line(le.E_QUBITS)
SETTINGS = (("per-block", False), ("batched", True))

STATS = {"polish_s": 0.0, "stepped": 0}
ORIG_BATCH = pc._refine_batch
ORIG_SINGLE = pc._refine_decomposition


def _timed_batch(U, p, *a, **k):
    t0 = PC()
    out = ORIG_BATCH(U, p, *a, **k)
    STATS["polish_s"] += PC() - t0
    STATS["stepped"] += int(np.sum(out[1] > pc.REFINE_THRESHOLD))
    return out


def _timed_single(*a, **k):
    t0 = PC()
    out = ORIG_SINGLE(*a, **k)
    STATS["polish_s"] += PC() - t0
    STATS["stepped"] += int(out[1] > pc.REFINE_THRESHOLD)
    return out


def run(fn, batched):
    """Call fn() with the given setting; returns (result, polish seconds, stepped blocks)."""
    pc.USE_BATCHED_POLISH = batched
    STATS["polish_s"], STATS["stepped"] = 0.0, 0
    try:
        out = fn()
    finally:
        pc.USE_BATCHED_POLISH = True
    return out, STATS["polish_s"], STATS["stepped"]


def cliff_circuit(n, theta):
    qc = QuantumCircuit(n)
    for k in range(n // 2):
        le.add_pair24(qc, 2 * k, 2 * k + 1, theta[k])
    return qc


def cliff_compile(qc, backend, native):
    pc._CX_CORE_CACHE.clear()
    t0 = PC()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True, on_unsupported="raise",
                                      seed_transpiler=0)
    return out, PC() - t0


def line_compile(qc, floor):
    pc._CX_CORE_CACHE.clear()
    t0 = PC()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        out = pc.compile_for_hardware(qc, coupling_map=LINE, basis_gates=["cx", "rz", "sx", "x"],
                                      block_gate_floor=floor, entangling_basis="cx",
                                      initial_layout=list(range(le.E_QUBITS)), on_unsupported="keep",
                                      seed_transpiler=0)
    el = PC() - t0
    final = list(out.layout.final_index_layout()) if out.layout is not None else list(range(le.E_QUBITS))
    if final != list(range(le.E_QUBITS)):
        raise RuntimeError(f"compiled circuit permuted qubits: {final}")
    return out, el


def spsa_target():
    rng = np.random.default_rng(7)
    target_theta = rng.uniform(-np.pi, np.pi, le.E_NPARAMS)
    target_theta[8::15] = 0.0
    return Statevector(le.e_circuit(target_theta)), target_theta + rng.normal(0.0, 0.3, le.E_NPARAMS)


def main():
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-27.1":
        print("V0 FAILED: psf_compile.py is not 2026-09-27.1. Stopping.")
        return
    pc._refine_batch = _timed_batch
    pc._refine_decomposition = _timed_single
    t_start = time.time()
    backend, native = le.nighthawk()
    n = backend.coupling_map.size()
    rows = []

    # ---------------- B1, B2, B5: fresh cliff circuits
    print(f"\n=== B1/B2/B5: fresh cliff circuits ({N_FRESH}) ===")
    rng = np.random.default_rng(31)
    circuits = [cliff_circuit(n, rng.uniform(-np.pi, np.pi, (n // 2, 24))) for _ in range(N_FRESH)]
    cliff_compile(circuits[0], backend, native)  # warm-up
    worst_check, worst_agree, step_mismatch = 0.0, 0.0, 0
    t = {name: [] for name, _ in SETTINGS}
    pol = {name: [] for name, _ in SETTINGS}
    for idx, qc in enumerate(circuits):
        outs, steps = {}, {}
        order = SETTINGS if idx % 2 == 0 else SETTINGS[::-1]
        for rep in range(3):
            for name, flag in order:
                (out, el), ps, stp = run(lambda: cliff_compile(qc, backend, native), flag)
                t[name].append(el)
                pol[name].append(ps)
                if rep == 0:
                    outs[name], steps[name] = out, stp
        ok, w = le.pair_check(qc, outs["batched"], n)
        worst_check = max(worst_check, w if ok else math.inf)
        ma = le.pair_matrices(le.back_to_logical(outs["batched"], n), n)
        mb = le.pair_matrices(le.back_to_logical(outs["per-block"], n), n)
        agree = max(le.aligned(mb[k], ma[k]) for k in ma)
        worst_agree = max(worst_agree, agree)
        step_mismatch += int(steps["batched"] != steps["per-block"])
        rows.append(dict(part="B1", index=idx, check=w, agree=agree, stepped_batched=steps["batched"],
                         stepped_per_block=steps["per-block"]))
    for name, _ in SETTINGS:
        rows.append(dict(part="B5", setting=name, compile_median=st.median(t[name]),
                         polish_mean=st.mean(pol[name])))
        print(f"  {name:9s}: compile median {st.median(t[name]) * 1000:.1f} ms, polish {st.mean(pol[name]) * 1000:.2f} "
              f"ms per compile", flush=True)
    print(f"  batched output: worst per-pair check {worst_check:.2e}; worst per-pair distance between batched and "
          f"per-block outputs {worst_agree:.2e}; compiles with a different number of stepped blocks: "
          f"{step_mismatch}/{N_FRESH}", flush=True)

    # ---------------- B3: compounding
    print(f"\n=== B3: cliff compounding loop ({C_LAPS} laps per setting) ===")
    rng13 = np.random.default_rng(13)
    initial = cliff_circuit(n, rng13.uniform(-np.pi, np.pi, (n // 2, 24)))
    ref = le.pair_matrices(initial, n)
    dist = {}
    for name, flag in SETTINGS:
        current, d = initial, {}
        pc.USE_BATCHED_POLISH = flag
        try:
            for lap in range(1, C_LAPS + 1):
                gc.disable()
                out, _ = cliff_compile(current, backend, native)
                current = le.back_to_logical(out, n)
                gc.enable()
                gc.collect()
                if lap in C_CHECK:
                    mats = le.pair_matrices(current, n)
                    d[lap] = max(le.aligned(ref[k], mats[k]) for k in ref)
        finally:
            pc.USE_BATCHED_POLISH = True
        dist[name] = d
        rows.append(dict(part="B3", setting=name, **{f"lap{k}": v for k, v in d.items()}))
        print(f"  {name:9s}: " + ", ".join(f"lap {k} {v:.2e}" for k, v in d.items()), flush=True)

    # ---------------- B4, B6: training circuits
    print(f"\n=== B4/B6: fresh training circuits ({N_FRESH}, 12 qubits) ===")
    target, theta0 = spsa_target()
    trng = np.random.default_rng(32)
    train = [le.e_circuit(theta0 + trng.normal(0.0, 0.3, le.E_NPARAMS)) for _ in range(N_FRESH)]
    ref_loss = [1.0 - abs(target.inner(Statevector(qc))) ** 2 for qc in train]
    b4 = {}
    t8 = {name: [] for name, _ in SETTINGS}
    b4_step_mismatch = 0
    for floor in (12, 8):
        for name, flag in SETTINGS:
            worst = 0.0
            for i, (qc, lr) in enumerate(zip(train, ref_loss)):
                (out, el), _, stp = run(lambda: line_compile(qc, floor), flag)
                worst = max(worst, abs((1.0 - abs(target.inner(Statevector(out))) ** 2) - lr))
                if floor == 8:
                    t8[name].append(el)
                rows.append(dict(part="B4", setting=name, floor=floor, index=i, stepped=stp,
                                 compile_s=el))
            b4[(floor, name)] = worst
            print(f"  floor {floor:2d}, {name:9s}: worst |loss - reference| {worst:.2e}", flush=True)
    by_key = {}
    for r in rows:
        if r.get("part") == "B4":
            by_key.setdefault((r["floor"], r["index"]), {})[r["setting"]] = r["stepped"]
    b4_step_mismatch = sum(1 for v in by_key.values() if v.get("batched") != v.get("per-block"))
    for name, _ in SETTINGS:
        print(f"  floor  8, {name:9s}: compile median {st.median(t8[name]) * 1000:.2f} ms")

    # ---------------- predictions
    print("\n=== predictions ===")

    def v(ok):
        return "CONFIRMED" if ok else "NOT CONFIRMED"

    print(f"B1 batched per-pair check <= 1e-12 and batched vs per-block per-pair distance <= 1e-12 "
          f"({worst_check:.2e}, {worst_agree:.2e}) -> {v(worst_check <= 1e-12 and worst_agree <= 1e-12)}")
    print(f"B2 same number of stepped blocks in every compile (cliff {N_FRESH - step_mismatch}/{N_FRESH}, "
          f"training {len(by_key) - b4_step_mismatch}/{len(by_key)}) -> "
          f"{v(step_mismatch == 0 and b4_step_mismatch == 0)}")
    db, dp = dist["batched"][C_LAPS], dist["per-block"][C_LAPS]
    print(f"B3 drift at lap {C_LAPS}: batched <= 1.5 x per-block and <= 1e-11 ({db:.2e} vs {dp:.2e}) -> "
          f"{v(db <= 1.5 * dp and db <= 1e-11)}")
    w4 = max(b4.values())
    print(f"B4 training, floors 12 and 8, both settings: worst |loss - reference| <= 1e-14 ({w4:.2e}) -> "
          f"{v(w4 <= 1e-14)}")
    pb, pp = st.mean(pol["batched"]), st.mean(pol["per-block"])
    cb, cp = st.median(t["batched"]), st.median(t["per-block"])
    print(f"B5 cliff: batched polish <= 3 ms per compile ({pb * 1000:.2f} vs {pp * 1000:.2f} ms) and compile "
          f"median lower by >= 8 ms ({cb * 1000:.1f} vs {cp * 1000:.1f} ms) -> "
          f"{v(pb <= 0.003 and cp - cb >= 0.008)}")
    tb, tp = st.median(t8["batched"]), st.median(t8["per-block"])
    print(f"B6 training floor 8: batched compile median not above per-block ({tb * 1000:.2f} vs {tp * 1000:.2f} ms) "
          f"-> {v(tb <= tp)}")

    fields = ["part", "setting", "floor", "index", "check", "agree", "stepped_batched", "stepped_per_block",
              "stepped", "compile_s", "compile_median", "polish_mean", "lap10", "lap50", "lap100"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows); total wall time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
