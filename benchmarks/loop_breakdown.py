"""loop_breakdown.py -- Addendum 203: where the time of one loop lap goes.

Diagnostic (expectations are stated in the preregistration and scored
there as "as expected" / "not as expected", not as pass/fail). All loops
run with the garbage-collection setting recommended by Addendum 202:
gc.collect(); gc.freeze() once after set-up, the collector disabled during
each lap, and gc.collect() at the lap boundary (timed as its own stage).

Part A  Cliff-scale compounding loop (FakeNighthawk, 120 logical qubits,
        60 paired 24-parameter blocks, layout_search=True), as in
        Addenda 200 and 202. Per lap: compile, serialization of the
        compiled circuit with qpy (what is sent to IBM Runtime), mapping
        back to logical qubits, boundary collection.
Part B  Training loop (12 qubits, 165 parameters, SPSA, statevector), as
        Addendum 199 Part E, driven by the loss of the compiled circuit.
        Per step: perturbation, circuit build, compile, simulation, loss,
        parameter update, boundary collection (two evaluations per step).
Part C  Reference: the same training loop the standard Qiskit way -- the
        parameterized circuit is transpiled once (optimization_level=3,
        same line, basis and trivial layout), and each evaluation only binds
        parameters and simulates. Same seeds, so the SPSA path should match
        Part B to rounding.

Parts A and B each run twice from the same starting point: "plain"
(only the lap-level stages above are timed) and "instrumented" (the
functions inside compile_for_hardware are replaced by timed wrappers for
the duration of the phase, and per-pass times of the internal transpile
call are collected through its callback). Both loops are deterministic, so
the two phases compile identical circuits; the script checks that (Part A:
SHA-256 of every compiled circuit's OpenQASM 2 text; Part B: every loss
value) and reports the instrumentation overhead as the ratio of the
compile medians.

Usage (repository root; loop_endurance.py must be in benchmarks/):
    python -u benchmarks/loop_breakdown.py 2>&1 | tee loop_breakdown.txt
"""
from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import io
import os
import platform
import statistics as st
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit, qasm2, qpy, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

import loop_endurance as le
import psf_compile as pc
import psf_smart_layout as psl

A_LAPS = 300
B_STEPS = 300
OUT_CSV = "loop_breakdown_2026-09-27.csv"
PC = time.perf_counter

# Timings of the current lap / step; wrappers add into it.
CUR: dict = defaultdict(float)


# ------------------------------------------------------------ instrumentation

class Instrumentation:
    """Swaps timed wrappers into psf_compile / psf_smart_layout (module or
    class attributes looked up at call time by the code under test) and
    restores the originals on exit."""

    def __init__(self):
        self._saved = []

    def _wrap(self, owner, name, key):
        orig = getattr(owner, name, None)
        if orig is None:
            return

        def wrapper(*a, **k):
            t0 = PC()
            try:
                return orig(*a, **k)
            finally:
                CUR[key] += PC() - t0

        self._saved.append((owner, name, orig))
        setattr(owner, name, wrapper)

    def __enter__(self):
        self._wrap(pc, "compile", "c1 psf.compile")
        orig_pm = pc.PassManager

        class TimedPassManager(orig_pm):
            def run(self_pm, *a, **k):
                t0 = PC()
                try:
                    return super().run(*a, **k)
                finally:
                    CUR["c1a consolidate"] += PC() - t0

        self._saved.append((pc, "PassManager", orig_pm))
        pc.PassManager = TimedPassManager
        self._wrap(pc.SU4GeodesicPSFSynthesizer, "synthesize", "c1b synthesize (all blocks)")
        self._wrap(pc, "_CORE_CHECKED", "c1b1 core decompose (Rust)")
        self._wrap(pc, "geometric_decompose", "c1b1 core decompose (Rust)")
        self._wrap(pc, "_refine_decomposition", "c1b2 polish")
        self._wrap(pc.SU4GeodesicPSFSynthesizer, "_build_circuit", "c1b3 build block circuit")
        self._wrap(pc, "_append_cx_core_closed_form", "c1b3a CX core, closed form")
        self._wrap(pc, "_cx_core_cached", "c1b3b CX core, decomposer + guard")
        self._wrap(psl, "smart_vf2_layout", "c2 layout search")
        self._wrap(pc, "transpile", "c3 qiskit transpile (L1)")
        return self

    def __exit__(self, *exc):
        for owner, name, orig in reversed(self._saved):
            setattr(owner, name, orig)
        self._saved.clear()
        return False


def pass_callback(**kw):
    CUR["p " + kw["pass_"].name()] += kw["time"]


def derive_residuals():
    """Adds 'other' stages so that every parent equals the sum of its parts."""
    if "c1 psf.compile" not in CUR:
        return
    CUR["c1c compile, other (to_matrix, compose, copy)"] = (
        CUR["c1 psf.compile"] - CUR["c1a consolidate"] - CUR["c1b synthesize (all blocks)"])
    CUR["c1b4 synthesize, other"] = (
        CUR["c1b synthesize (all blocks)"] - CUR["c1b1 core decompose (Rust)"] - CUR["c1b2 polish"]
        - CUR["c1b3 build block circuit"])
    CUR["c1b3c build, other (local rotations)"] = (
        CUR["c1b3 build block circuit"] - CUR["c1b3a CX core, closed form"]
        - CUR["c1b3b CX core, decomposer + guard"])
    passes = sum(v for k, v in CUR.items() if k.startswith("p "))
    CUR["c3z transpile, other (conversion, set-up)"] = CUR["c3 qiskit transpile (L1)"] - passes
    CUR["c9 compile_for_hardware, other"] = (
        CUR["lap: compile"] - CUR["c1 psf.compile"] - CUR["c2 layout search"] - CUR["c3 qiskit transpile (L1)"])


# ------------------------------------------------------------ helpers

def lap_gc_begin():
    gc.disable()


def lap_gc_end():
    gc.enable()
    t0 = PC()
    gc.collect()
    CUR["lap: boundary gc"] = PC() - t0


def compile_cliff(qc, backend, native, instrumented):
    pc._CX_CORE_CACHE.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        return pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                       entangling_basis="cx", layout_search=True, on_unsupported="raise",
                                       seed_transpiler=0, callback=pass_callback if instrumented else None)


def compile_line(qc, cmap, instrumented):
    pc._CX_CORE_CACHE.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                                      entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                      on_unsupported="keep", seed_transpiler=0,
                                      callback=pass_callback if instrumented else None)
    return out


def check_identity_layout(out):
    final = list(out.layout.final_index_layout()) if out.layout is not None else list(range(le.E_QUBITS))
    if final != list(range(le.E_QUBITS)):
        raise RuntimeError(f"compiled circuit permuted qubits: {final}")


def twoq(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2)


def record(rows, part, phase, index):
    for k, v in CUR.items():
        rows.append(dict(part=part, phase=phase, index=index, stage=k, seconds=v))


# ------------------------------------------------------------ Part A

def part_a(backend, native, initial, rows, phase):
    n = backend.coupling_map.size()
    instrumented = phase == "instrumented"
    laps, hashes = [], []
    current = initial
    ctx = Instrumentation() if instrumented else contextlib.nullcontext()
    with ctx:
        for lap in range(1, A_LAPS + 1):
            CUR.clear()
            lap_gc_begin()
            t0 = PC()
            out = compile_cliff(current, backend, native, instrumented)
            t1 = PC()
            buf = io.BytesIO()
            qpy.dump(out, buf)
            t2 = PC()
            current = le.back_to_logical(out, n)
            t3 = PC()
            CUR["lap: compile"] = t1 - t0
            CUR["lap: serialize (qpy)"] = t2 - t1
            CUR["lap: back to logical"] = t3 - t2
            lap_gc_end()
            CUR["lap: total"] = PC() - t0
            derive_residuals()
            laps.append(dict(CUR))
            record(rows, "A", phase, lap)
            hashes.append(hashlib.sha256(qasm2.dumps(out).encode()).hexdigest())
    return laps, hashes, len(buf.getvalue())


# ------------------------------------------------------------ Parts B and C

def spsa_setup():
    rng = np.random.default_rng(7)
    target_theta = rng.uniform(-np.pi, np.pi, le.E_NPARAMS)
    target_theta[8::15] = 0.0
    target = Statevector(le.e_circuit(target_theta))
    theta0 = target_theta + rng.normal(0.0, 0.3, le.E_NPARAMS)
    return target, theta0


def spsa_loop(evaluate, target, theta0, rows, part, phase, ctx):
    """evaluate(theta) -> (loss); stages are timed inside `evaluate`."""
    theta = theta0.copy()
    steps, losses = [], []
    with ctx:
        for k in range(B_STEPS):
            CUR.clear()
            lap_gc_begin()
            t0 = PC()
            a_k = 0.2 / (k + 11) ** 0.602
            c_k = 0.1 / (k + 1) ** 0.101
            delta = np.random.default_rng(1000 + k).choice([-1.0, 1.0], le.E_NPARAMS)
            plus, minus = theta + c_k * delta, theta - c_k * delta
            CUR["lap: perturb"] = PC() - t0
            vals = [evaluate(plus), evaluate(minus)]
            t1 = PC()
            theta = theta - a_k * (vals[0] - vals[1]) / (2.0 * c_k) * delta
            CUR["lap: update"] = PC() - t1
            lap_gc_end()
            CUR["lap: total"] = PC() - t0
            derive_residuals()
            steps.append(dict(CUR))
            losses.extend(vals)
            record(rows, part, phase, k + 1)
    return steps, losses, theta


def part_b(target, theta0, rows, phase):
    cmap = CouplingMap.from_line(le.E_QUBITS)
    instrumented = phase == "instrumented"

    def evaluate(theta):
        t0 = PC()
        qc = le.e_circuit(theta)
        t1 = PC()
        out = compile_line(qc, cmap, instrumented)
        t2 = PC()
        check_identity_layout(out)
        t3 = PC()
        sv = Statevector(out)
        t4 = PC()
        loss = 1.0 - abs(target.inner(sv)) ** 2
        t5 = PC()
        CUR["lap: build circuit"] += t1 - t0
        CUR["lap: compile"] += t2 - t1
        CUR["lap: simulate"] += t4 - t3
        CUR["lap: loss"] += t5 - t4
        return loss

    ctx = Instrumentation() if instrumented else contextlib.nullcontext()
    return spsa_loop(evaluate, target, theta0, rows, "B", phase, ctx)


def parametric_circuit():
    pv = ParameterVector("t", le.E_NPARAMS)
    qc = QuantumCircuit(le.E_QUBITS)
    for k, (a, b) in enumerate(le.E_BLOCKS):
        le.add_block(qc, a, b, [pv[i] for i in range(15 * k, 15 * (k + 1))])
    return qc, pv


def part_c(target, theta0, rows, tqc, pv):
    def evaluate(theta):
        t0 = PC()
        bound = tqc.assign_parameters(dict(zip(pv, theta)), strict=False)
        t1 = PC()
        sv = Statevector(bound)
        t2 = PC()
        loss = 1.0 - abs(target.inner(sv)) ** 2
        t3 = PC()
        CUR["lap: bind parameters"] += t1 - t0
        CUR["lap: simulate"] += t2 - t1
        CUR["lap: loss"] += t3 - t2
        return loss

    return spsa_loop(evaluate, target, theta0, rows, "C", "plain", contextlib.nullcontext())


# ------------------------------------------------------------ reporting

def report(title, laps, top_passes=8):
    keys = sorted({k for lap in laps for k in lap})
    total = st.mean(l["lap: total"] for l in laps)
    print(f"  {title}: {len(laps)} laps; lap total mean {total * 1000:.2f} ms, "
          f"median {st.median(l['lap: total'] for l in laps) * 1000:.2f} ms")
    print(f"    {'stage':52s} {'mean ms':>9s} {'median ms':>10s} {'share':>7s}")

    def line(k, label=None):
        vals = [l.get(k, 0.0) for l in laps]
        m = st.mean(vals)
        print(f"    {label or k:52s} {m * 1000:9.3f} {st.median(vals) * 1000:10.3f} {100 * m / total:6.1f}%")

    for k in [k for k in keys if k.startswith("lap: ") and k != "lap: total"]:
        line(k)
    comp = [k for k in keys if k.startswith("c")]
    for k in comp:
        depth = len(k.split(" ")[0]) - 1
        line(k, "      " + "  " * depth + k.split(" ", 1)[1])
    passes = [k for k in keys if k.startswith("p ")]
    if passes:
        means = sorted(((st.mean(l.get(k, 0.0) for l in laps), k) for k in passes), reverse=True)
        for m, k in means[:top_passes]:
            line(k, "            pass: " + k[2:])
        rest = [k for _, k in means[top_passes:]]
        if rest:
            vals = [sum(l.get(k, 0.0) for k in rest) for l in laps]
            m = st.mean(vals)
            print(f"    {'            other passes (' + str(len(rest)) + ')':52s} {m * 1000:9.3f} "
                  f"{st.median(vals) * 1000:10.3f} {100 * m / total:6.1f}%")


def med(laps, key):
    return st.median(l[key] for l in laps)


def mean_share(laps, key):
    return st.mean(l.get(key, 0.0) for l in laps) / st.mean(l["lap: total"] for l in laps)


# ------------------------------------------------------------ main

def main():
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__))
    print("LOADED", psl.__file__, getattr(psl, "LAYOUT_VERSION", "?"), le.normalized_sha256(psl.__file__))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-26.4":
        print("V0 FAILED: psf_compile.py is not 2026-09-26.4. Stopping.")
        return
    t_start = time.time()

    # ---- set-up (everything long-lived is built before gc.freeze()) ----
    backend, native = le.nighthawk()
    n = backend.coupling_map.size()
    rng = np.random.default_rng(13)
    theta = rng.uniform(-np.pi, np.pi, (n // 2, 24))
    initial = QuantumCircuit(n)
    for k in range(n // 2):
        le.add_pair24(initial, 2 * k, 2 * k + 1, theta[k])
    target, theta0 = spsa_setup()
    pqc, pv = parametric_circuit()
    t0 = PC()
    tqc1 = transpile(pqc, coupling_map=CouplingMap.from_line(le.E_QUBITS), basis_gates=["cx", "rz", "sx", "x"],
                     optimization_level=1, initial_layout=list(range(le.E_QUBITS)), seed_transpiler=0)
    t_l1 = PC() - t0
    t0 = PC()
    tqc3 = transpile(pqc, coupling_map=CouplingMap.from_line(le.E_QUBITS), basis_gates=["cx", "rz", "sx", "x"],
                     optimization_level=3, initial_layout=list(range(le.E_QUBITS)), seed_transpiler=0)
    t_l3 = PC() - t0
    check_identity_layout(tqc3)
    compile_cliff(initial, backend, native, False)  # warm-up
    psf_line = compile_line(le.e_circuit(theta0), CouplingMap.from_line(le.E_QUBITS), False)  # warm-up
    Statevector(psf_line)
    gc.collect()
    gc.freeze()
    rows = []

    print(f"\n=== A: cliff-scale compounding loop ({n} logical qubits), {A_LAPS} laps per phase ===")
    a_plain, h_plain, qpy_bytes = part_a(backend, native, initial, rows, "plain")
    a_instr, h_instr, _ = part_a(backend, native, initial, rows, "instrumented")
    a_same = sum(1 for x, y in zip(h_plain, h_instr) if x == y)
    report("plain", a_plain)
    report("instrumented", a_instr)
    a_over = med(a_instr, "lap: compile") / med(a_plain, "lap: compile")
    print(f"  compiled circuits identical in both phases: {a_same}/{A_LAPS}; instrumentation overhead "
          f"(compile median, instrumented / plain) {a_over:.3f}; qpy size {qpy_bytes / 1024:.0f} KiB")

    print(f"\n=== B: training loop through PSF-Zero ({le.E_QUBITS} qubits, {le.E_NPARAMS} parameters), "
          f"{B_STEPS} SPSA steps per phase ===")
    b_plain, l_plain, th_b = part_b(target, theta0, rows, "plain")
    b_instr, l_instr, _ = part_b(target, theta0, rows, "instrumented")
    b_same = sum(1 for x, y in zip(l_plain, l_instr) if x == y)
    report("plain", b_plain)
    report("instrumented", b_instr)
    b_over = med(b_instr, "lap: compile") / med(b_plain, "lap: compile")
    print(f"  loss values identical in both phases: {b_same}/{len(l_plain)}; instrumentation overhead "
          f"{b_over:.3f}; two-qubit gates in the compiled circuit {twoq(psf_line)}")

    print(f"\n=== C: the same training loop, Qiskit parameterized route (transpile once, bind each evaluation) ===")
    print(f"  one-time transpile: L1 {t_l1 * 1000:.0f} ms ({twoq(tqc1)} two-qubit gates), "
          f"L3 {t_l3 * 1000:.0f} ms ({twoq(tqc3)} two-qubit gates); the loop uses L3")
    c_steps, l_c, th_c = part_c(target, theta0, rows, tqc3, pv)
    report("plain", c_steps)
    dl = max(abs(x - y) for x, y in zip(l_plain, l_c))
    dth = float(np.max(np.abs(th_b - th_c)))
    print(f"  against B: largest loss difference {dl:.2e}; largest parameter difference after "
          f"{B_STEPS} steps {dth:.2e}")

    # ---- expectations (preregistered in Addendum 203) ----
    print("\n=== expectations ===")

    def verdict(ok):
        return "as expected" if ok else "NOT as expected"

    s_comp, s_back = mean_share(a_plain, "lap: compile"), mean_share(a_plain, "lap: back to logical")
    print(f"X1 cliff loop: compile >= 60% of the lap and back-to-logical >= 15% "
          f"({100 * s_comp:.1f}%, {100 * s_back:.1f}%) -> {verdict(s_comp >= 0.60 and s_back >= 0.15)}")
    parts = {k: st.mean(l[k] for l in a_instr) for k in ("c1 psf.compile", "c2 layout search",
                                                         "c3 qiskit transpile (L1)")}
    largest = max(parts, key=parts.get)
    listed = ", ".join("%s %.1f ms" % (k.split(" ", 1)[1], v * 1000) for k, v in parts.items())
    print(f"X2 cliff compile: the Qiskit transpile call is the largest of psf.compile / layout / transpile "
          f"({listed}) -> {verdict(largest == 'c3 qiskit transpile (L1)')}")
    s_bc = mean_share(b_plain, "lap: compile")
    print(f"X3 training loop: compile >= 50% of a step ({100 * s_bc:.1f}%) -> {verdict(s_bc >= 0.50)}")
    tb, tc = med(b_plain, "lap: total"), med(c_steps, "lap: total")
    print(f"X4 Qiskit route faster per step ({tc * 1000:.2f} vs {tb * 1000:.2f} ms, ratio {tb / tc:.2f}) and "
          f">= 1.5x the two-qubit gates ({twoq(tqc3)} vs {twoq(psf_line)}) -> "
          f"{verdict(tc < tb and twoq(tqc3) >= 1.5 * twoq(psf_line))}")
    print(f"X5 instrumentation overhead <= 15% (A {a_over:.3f}, B {b_over:.3f}) and identical results "
          f"(A {a_same}/{A_LAPS}, B {b_same}/{len(l_plain)}) -> "
          f"{verdict(a_over <= 1.15 and b_over <= 1.15 and a_same == A_LAPS and b_same == len(l_plain))}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["part", "phase", "index", "stage", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows); total wall time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
