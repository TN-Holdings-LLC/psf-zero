"""verify_exact_fallback.py -- Addendum 217: psf_compile.py 2026-09-27.6
(exact fallback, changelog item 23) against its own USE_EXACT_FALLBACK = False
setting, which is the 2026-09-27.3 behavior.

  V1  The 3,003 training compiles of Addendum 216 (laps 1-3000 of Addendum
      214 part E and the four failing laps), block_gate_floor 8: loss error
      against the uncompiled circuit, both settings.
  V2  Same compiles: two-qubit gate count per compile, both settings; number
      of blocks rebuilt.
  V3  30 fresh cliff circuits (FakeNighthawk, 120 logical, layout_search):
      OpenQASM 2 text identical in both settings.
  V4  Floor-8 training compile time, both settings.
  V5  100 training compiles at the default floor (12): identical output in
      both settings.

Predictions:
  V1  With the fix, every loss error <= 1e-14 (without it, the five
      compiles of Addendum 216 above 1e-10 reappear).
  V2  Two-qubit counts identical in every compile; blocks rebuilt >= 5 and
      <= the 4,298 fallback blocks.
  V3  30 of 30 cliff outputs identical.
  V4  Median compile time with the fix <= 1.10 x without.
  V5  100 of 100 identical.

Usage (repository root; loop_endurance.py in benchmarks/):
    python -u benchmarks/verify_exact_fallback.py 2>&1 | tee verify_exact_fallback.txt
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import os
import platform
import statistics as st
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

import loop_endurance as le
import psf_compile as pc

N_LAPS = 3000
FAIL_LAPS = (1030, 3600, 4580, 6720)
LINE = CouplingMap.from_line(le.E_QUBITS)
OUT_CSV = "verify_exact_fallback_2026-09-27.csv"
PC = time.perf_counter


def quiet(fn):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return fn()


def line_compile(qc, floor):
    pc._CX_CORE_CACHE.clear()
    t0 = PC()
    out = quiet(lambda: pc.compile_for_hardware(qc, coupling_map=LINE, basis_gates=["cx", "rz", "sx", "x"],
                                                block_gate_floor=floor, entangling_basis="cx",
                                                initial_layout=list(range(le.E_QUBITS)), on_unsupported="keep",
                                                seed_transpiler=0))
    return out, PC() - t0


def twoq(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2)


def reset_stats():
    for k in pc.GUARD_STATS:
        pc.GUARD_STATS[k] = 0


def main():
    print(f"platform {platform.platform()} | python {platform.python_version()} | qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-27.6":
        print("V0 FAILED: psf_compile.py is not 2026-09-27.6. Stopping.")
        return
    t_start = time.time()
    rng_e = np.random.default_rng(7)
    target_theta = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
    target_theta[8::15] = 0.0
    target = Statevector(le.e_circuit(target_theta))
    rng_t = np.random.default_rng(202)
    thetas = {lap: target_theta + rng_t.normal(0.0, 0.5, le.E_NPARAMS) for lap in range(1, max(FAIL_LAPS) + 1)}
    laps = sorted(set(range(1, N_LAPS + 1)) | set(FAIL_LAPS))

    def loss(qc):
        return 1 - abs(target.inner(Statevector(qc))) ** 2

    rows, res = [], {}
    for name, flag in (("fixed", True), ("previous", False)):
        pc.USE_EXACT_FALLBACK = flag
        reset_stats()
        errs, counts, times = {}, {}, []
        try:
            for lap in laps:
                qc = le.e_circuit(thetas[lap])
                out, el = line_compile(qc, 8)
                errs[lap] = abs(loss(out) - loss(qc))
                counts[lap] = twoq(out)
                times.append(el)
                rows.append(dict(setting=name, lap=lap, loss_error=errs[lap], twoq=counts[lap], compile_s=el))
        finally:
            pc.USE_EXACT_FALLBACK = True
        res[name] = dict(errs=errs, counts=counts, med=st.median(times), stats=dict(pc.GUARD_STATS))
        bad = sum(1 for e in errs.values() if e > 1e-10)
        print(f"  {name:8s}: worst loss error {max(errs.values()):.2e}; compiles > 1e-10: {bad}; > 1e-14: "
              f"{sum(1 for e in errs.values() if e > 1e-14)}; compile median {res[name]['med'] * 1000:.2f} ms; "
              f"guard {res[name]['stats']}", flush=True)

    backend, native = le.nighthawk()
    n = backend.coupling_map.size()
    rng = np.random.default_rng(31)
    same_cliff = 0
    for i in range(30):
        qc = QuantumCircuit(n)
        th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
        for k in range(n // 2):
            le.add_pair24(qc, 2 * k, 2 * k + 1, th[k])
        hs = []
        for flag in (True, False):
            pc.USE_EXACT_FALLBACK = flag
            pc._CX_CORE_CACHE.clear()
            try:
                out = quiet(lambda: pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                                            entangling_basis="cx", layout_search=True,
                                                            on_unsupported="keep", seed_transpiler=0))
            finally:
                pc.USE_EXACT_FALLBACK = True
            hs.append(hashlib.sha256(qasm2.dumps(out).encode()).hexdigest())
        same_cliff += int(hs[0] == hs[1])
    print(f"  cliff: identical outputs {same_cliff}/30", flush=True)

    same12 = 0
    for lap in range(1, 101):
        qc = le.e_circuit(thetas[lap])
        hs = []
        for flag in (True, False):
            pc.USE_EXACT_FALLBACK = flag
            try:
                out, _ = line_compile(qc, 12)
            finally:
                pc.USE_EXACT_FALLBACK = True
            hs.append(hashlib.sha256(qasm2.dumps(out).encode()).hexdigest())
        same12 += int(hs[0] == hs[1])
    print(f"  floor 12: identical outputs {same12}/100", flush=True)

    print("\n=== predictions ===")

    def v(ok):
        return "CONFIRMED" if ok else "NOT CONFIRMED"

    fx, pv = res["fixed"], res["previous"]
    w1 = max(fx["errs"].values())
    print(f"V1 with the fix every loss error <= 1e-14 (worst {w1:.2e}; without the fix {max(pv['errs'].values()):.2e}, "
          f"{sum(1 for e in pv['errs'].values() if e > 1e-10)} above 1e-10) -> {v(w1 <= 1e-14)}")
    diff = [lap for lap in laps if fx["counts"][lap] != pv["counts"][lap]]
    rebuilt = fx["stats"].get("exact_rebuilt", 0)
    print(f"V2 two-qubit counts identical ({len(laps) - len(diff)}/{len(laps)}); blocks rebuilt {rebuilt} "
          f"(expected 5..4298) -> {v(not diff and 5 <= rebuilt <= 4298)}")
    print(f"V3 cliff outputs identical ({same_cliff}/30) -> {v(same_cliff == 30)}")
    print(f"V4 compile median with the fix {fx['med'] * 1000:.2f} vs {pv['med'] * 1000:.2f} ms "
          f"(ratio {fx['med'] / pv['med']:.3f} <= 1.10) -> {v(fx['med'] <= 1.10 * pv['med'])}")
    print(f"V5 floor 12 outputs identical ({same12}/100) -> {v(same12 == 100)}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "lap", "loss_error", "twoq", "compile_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows); total wall time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
