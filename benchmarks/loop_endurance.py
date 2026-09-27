"""loop_endurance.py -- Addendum 199: endurance test of the compile loop.

Purpose: run PSF-Zero (psf_compile.py VERSION 2026-09-26.4) the way a
variational training loop does -- thousands of compiles in one process -- and
record everything that could go wrong: wrong results, slow or stalled laps,
growing memory, exceptions, the correctness guard firing, and drift.

Part E  Training loop, executed (12 qubits, statevector).
        A brick-layer ansatz of 11 two-qubit blocks (15 parameterized gates
        each, so every block is consolidated and resynthesized) is trained
        by SPSA towards a target state whose ZZ angles are zero. Each loss
        evaluation compiles the circuit with compile_for_hardware (line
        coupling map, trivial layout, basis [cx, rz, sx, x],
        entangling_basis="cx") and computes the loss from the compiled
        circuit and, as a reference, from the uncompiled circuit.
        Arms: "ref-driven" (SPSA follows the reference loss; the compiled
        loss is recorded beside it), "ref-driven, guard off" (same, with
        USE_CX_GUARD=False, i.e. the released 2026-09-26.2 behaviour of the
        CX path), and "compiled-driven" (SPSA follows the compiled loss).
Part F  Compile-only loop at cliff scale, fresh each lap (FakeNighthawk,
        120 logical qubits = spare 0). 60 pair blocks of 24 parameterized
        gates; each lap the parameters take a random-walk step and the
        circuit is rebuilt and compiled with layout_search=True. Exact
        per-pair check every CHECK_EVERY laps.
Part C  Compile-only loop at cliff scale, compounding: each lap's routed
        output is mapped back to logical qubits and compiled again (the
        worst case, as in deadline_compound_chain.py, but for many more
        laps). Per-pair distance to the lap-0 circuit every CHECK_EVERY laps.

Every part records per-lap compile time, resident memory (VmRSS) every
CHECK_EVERY laps, exceptions (caught, counted, the loop continues), and
psf_compile.GUARD_STATS.

Usage (repository root):
    python -u benchmarks/loop_endurance.py 2>&1 | tee loop_endurance.txt
    (options: --steps N for Part E, --laps N for Parts F and C)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import os
import platform
import statistics as st
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.fake_provider import FakeNighthawk

import psf_compile as pc
import psf_smart_layout as psl

E_QUBITS = 12
E_STEPS = 600
FC_LAPS = 1000
CHECK_EVERY = 50
WALK_SIGMA = 0.02
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
OUT_CSV = "loop_endurance_2026-09-27.csv"


# ------------------------------------------------------------ helpers

def normalized_sha256(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def reset_guard_stats():
    for k in pc.GUARD_STATS:
        pc.GUARD_STATS[k] = 0


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def time_summary(ts):
    return (f"median {st.median(ts) * 1000:.1f} ms, p99 {pct(ts, 0.99) * 1000:.1f} ms, "
            f"max {max(ts) * 1000:.1f} ms")


def add_block(qc, a, b, p):
    """15 parameterized gates on (a, b): local ZYZ on both, canonical core,
    local ZYZ on both. Consolidated by PSF-Zero (floor 12)."""
    qc.rz(p[0], a); qc.ry(p[1], a); qc.rz(p[2], a)
    qc.rz(p[3], b); qc.ry(p[4], b); qc.rz(p[5], b)
    qc.rxx(p[6], a, b); qc.ryy(p[7], a, b); qc.rzz(p[8], a, b)
    qc.rz(p[9], a); qc.ry(p[10], a); qc.rz(p[11], a)
    qc.rz(p[12], b); qc.ry(p[13], b); qc.rz(p[14], b)


def add_pair24(qc, a, b, p):
    """24 parameterized gates on (a, b): two rounds of (local ZYZ on both,
    canonical core), then a final local layer."""
    k = 0
    for _ in range(2):
        for q in (a, b):
            qc.rz(p[k], q); qc.ry(p[k + 1], q); qc.rz(p[k + 2], q)
            k += 3
        qc.rxx(p[k], a, b); qc.ryy(p[k + 1], a, b); qc.rzz(p[k + 2], a, b)
        k += 3
    for q in (a, b):
        qc.rz(p[k], q); qc.ry(p[k + 1], q); qc.rz(p[k + 2], q)
        k += 3


def pair_check(qc_logical, qc_out, n_logical):
    """Exact per-pair check, verbatim from nighthawk_deadline_cliff.py."""
    lay = qc_out.layout
    if lay is None:
        return False, None
    phys = list(lay.final_index_layout(filter_ancillas=True))
    pairs = [(i, i + 1) for i in range(0, n_logical - 1, 2)]
    owner = {}
    for k, (a, b) in enumerate(pairs):
        owner[phys[a]] = k
        owner[phys[b]] = k
    per_pair = {k: QuantumCircuit(2) for k in range(len(pairs))}
    for inst in qc_out.data:
        qs = [qc_out.find_bit(q).index for q in inst.qubits]
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        ks = {owner.get(q) for q in qs}
        if None in ks:
            if len(qs) == 1:
                continue
            return False, None
        if len(ks) != 1:
            return False, None
        k = ks.pop()
        a, b = pairs[k]
        local = [0 if q == phys[a] else 1 for q in qs]
        per_pair[k].append(inst.operation, local)
    worst = 0.0
    for k, (a, b) in enumerate(pairs):
        ref = QuantumCircuit(2)
        for inst in qc_logical.data:
            qs = [qc_logical.find_bit(q).index for q in inst.qubits]
            if set(qs) <= {a, b}:
                ref.append(inst.operation, [0 if q == a else 1 for q in qs])
        u, v = Operator(per_pair[k]).data, Operator(ref).data
        infid = 1.0 - abs(np.trace(v.conj().T @ u)) / 4.0
        worst = max(worst, float(infid))
    return True, worst


def pair_matrices(qc, n):
    """4x4 operator of each pair (2k, 2k+1) of a logical circuit; raises if a
    two-qubit gate spans two pairs."""
    per = {k: QuantumCircuit(2) for k in range(n // 2)}
    for inst in qc.data:
        if inst.operation.name in ("barrier", "delay", "measure"):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        ks = {q // 2 for q in qs}
        if len(ks) != 1:
            raise RuntimeError(f"{inst.operation.name} on {qs} spans two pairs")
        k = ks.pop()
        per[k].append(inst.operation, [q - 2 * k for q in qs])
    return {k: Operator(c).data for k, c in per.items()}


def aligned(a, b):
    t = np.trace(a.conj().T @ b)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(b - ph * a))


def back_to_logical(routed, n):
    """Routed output -> logical circuit, as in deadline_compound_chain.py."""
    init = list(routed.layout.initial_index_layout(filter_ancillas=True))
    final = list(routed.layout.final_index_layout(filter_ancillas=True))
    if init != final:
        raise RuntimeError("routing permuted qubits")
    to_logical = {p: v for v, p in enumerate(init)}
    out = QuantumCircuit(n)
    for inst in routed.data:
        if inst.operation.name in ("barrier", "delay", "measure"):
            continue
        phys = [routed.find_bit(q).index for q in inst.qubits]
        if any(p not in to_logical for p in phys):
            if len(phys) == 1:
                continue
            raise RuntimeError(f"two-qubit operation outside the layout: {phys}")
        out.append(inst.operation, [to_logical[p] for p in phys])
    return out


# ------------------------------------------------------------ Part E

E_BLOCKS = [(i, i + 1) for i in range(0, E_QUBITS - 1, 2)] + [(i, i + 1) for i in range(1, E_QUBITS - 2, 2)]
E_NPARAMS = 15 * len(E_BLOCKS)


def e_circuit(theta):
    qc = QuantumCircuit(E_QUBITS)
    for k, (a, b) in enumerate(E_BLOCKS):
        add_block(qc, a, b, theta[15 * k:15 * (k + 1)])
    return qc


def part_e(rows, steps):
    print(f"\n=== E: training loop, executed ({E_QUBITS} qubits, {len(E_BLOCKS)} blocks, "
          f"{E_NPARAMS} parameters, {steps} SPSA steps) ===")
    cmap = CouplingMap.from_line(E_QUBITS)
    rng = np.random.default_rng(7)
    target_theta = rng.uniform(-np.pi, np.pi, E_NPARAMS)
    target_theta[8::15] = 0.0  # every block's ZZ angle is zero at the target
    target = Statevector(e_circuit(target_theta))
    theta0 = target_theta + rng.normal(0.0, 0.3, E_NPARAMS)

    def loss_of(qc):
        return 1.0 - abs(target.inner(Statevector(qc))) ** 2

    def compiled_loss(theta, guard):
        qc = e_circuit(theta)
        pc.USE_CX_GUARD = guard
        pc._CX_CORE_CACHE.clear()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                                          entangling_basis="cx", initial_layout=list(range(E_QUBITS)),
                                          on_unsupported="keep", seed_transpiler=0)
        el = time.perf_counter() - t0
        pc.USE_CX_GUARD = True
        final = list(out.layout.final_index_layout()) if out.layout is not None else list(range(E_QUBITS))
        if final != list(range(E_QUBITS)):
            raise RuntimeError(f"compiled circuit permuted qubits: {final}")
        return loss_of(out), loss_of(qc), el

    arms = [("ref-driven", True, "ref"), ("ref-driven, guard off", False, "ref"),
            ("compiled-driven", True, "compiled")]
    finals = {}
    for arm, guard, driver in arms:
        reset_guard_stats()
        theta = theta0.copy()
        times, diffs, errors = [], [], 0
        rss0 = rss_mb()
        rss_trace = [rss0]
        for k in range(steps):
            a_k = 0.2 / (k + 11) ** 0.602
            c_k = 0.1 / (k + 1) ** 0.101
            delta = np.random.default_rng(1000 + k).choice([-1.0, 1.0], E_NPARAMS)
            vals = []
            for sign in (1.0, -1.0):
                try:
                    lc, lr, el = compiled_loss(theta + sign * c_k * delta, guard)
                    times.append(el)
                    diffs.append(abs(lc - lr))
                    vals.append(lc if driver == "compiled" else lr)
                except Exception as exc:  # recorded; the loop continues on the reference
                    errors += 1
                    if errors <= 3:
                        print(f"  [{arm}] step {k}: {type(exc).__name__}: {exc}")
                    vals.append(loss_of(e_circuit(theta + sign * c_k * delta)))
            theta = theta - a_k * (vals[0] - vals[1]) / (2.0 * c_k) * delta
            if (k + 1) % CHECK_EVERY == 0 or k == steps - 1:
                rss_trace.append(rss_mb())
                lc, lr, el = compiled_loss(theta, guard)
                rows.append(dict(part="E", arm=arm, step=k + 1, loss_ref=lr, loss_compiled=lc,
                                 abs_diff=abs(lc - lr), time_s=el, rss_mb=rss_trace[-1]))
        finals[arm] = theta
        n_bad = sum(1 for d in diffs if d > 1e-10)
        print(f"  {arm:22s}: final loss {loss_of(e_circuit(theta)):.3e} (start {loss_of(e_circuit(theta0)):.3e}); "
              f"max |compiled - ref| {max(diffs):.2e} ({n_bad}/{len(diffs)} evals > 1e-10); "
              f"compile {time_summary(times)}; RSS {rss0:.0f} -> {rss_trace[-1]:.0f} MB "
              f"(max {max(rss_trace):.0f}); exceptions {errors}; guard {dict(pc.GUARD_STATS)}", flush=True)
        rows.append(dict(part="E", arm=arm, step="summary", abs_diff=max(diffs), n_over=n_bad,
                         time_median=st.median(times), time_p99=pct(times, 0.99), time_max=max(times),
                         rss_start=rss0, rss_end=rss_trace[-1], exceptions=errors,
                         guard=str(dict(pc.GUARD_STATS))))
    div = float(np.max(np.abs(finals["compiled-driven"] - finals["ref-driven"])))
    print(f"  compiled-driven vs ref-driven: max parameter difference after {steps} steps {div:.2e}")
    rows.append(dict(part="E", arm="divergence", step=steps, abs_diff=div))


# ------------------------------------------------------------ Parts F and C

def nighthawk():
    with contextlib.redirect_stderr(io.StringIO()):
        backend = FakeNighthawk()
    native = [g for g in backend.operation_names if g in NATIVE]
    return backend, native


def cfh(qc, backend, native):
    pc._CX_CORE_CACHE.clear()
    info = {}
    orig = psl.smart_vf2_layout

    def probe(*a, **k):
        out = orig(*a, **k)
        info.update(out[1])
        return out

    psl.smart_vf2_layout = probe
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                          entangling_basis="cx", layout_search=True,
                                          on_unsupported="raise", seed_transpiler=0)
            el = time.perf_counter() - t0
    finally:
        psl.smart_vf2_layout = orig
    return out, el, info.get("phase")


def part_f(rows, laps):
    backend, native = nighthawk()
    n = backend.coupling_map.size()
    npair = n // 2
    print(f"\n=== F: compile-only loop, fresh circuit each lap (FakeNighthawk, {n} logical, {laps} laps) ===")
    rng = np.random.default_rng(11)
    theta = rng.uniform(-np.pi, np.pi, (npair, 24))
    reset_guard_stats()
    times, phases, errors, worst_checks = [], [], 0, []
    rss0 = rss_mb()
    rss_trace = [rss0]
    for lap in range(1, laps + 1):
        theta = theta + rng.normal(0.0, WALK_SIGMA, theta.shape)
        qc = QuantumCircuit(n)
        for k in range(npair):
            add_pair24(qc, 2 * k, 2 * k + 1, theta[k])
        try:
            out, el, phase = cfh(qc, backend, native)
            times.append(el)
            phases.append(phase)
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"  lap {lap}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)
            continue
        if lap % CHECK_EVERY == 0 or lap == laps:
            applicable, worst = pair_check(qc, out, n)
            worst_checks.append(worst if applicable else float("inf"))
            rss_trace.append(rss_mb())
            rows.append(dict(part="F", arm="fresh", step=lap, time_s=el, pair_worst=worst,
                             pair_applicable=applicable, rss_mb=rss_trace[-1], layout_phase=phase))
            if lap % (CHECK_EVERY * 4) == 0:
                print(f"  lap {lap}: last {el * 1000:.1f} ms, pair worst {worst}, RSS {rss_trace[-1]:.0f} MB",
                      flush=True)
    slow = sum(1 for t in times if t > 1.0)
    print(f"  compile {time_summary(times)}; laps over 1 s: {slow}; layout phases {sorted(set(phases), key=str)}; "
          f"pair checks worst {max(worst_checks):.2e} over {len(worst_checks)} checks; "
          f"RSS {rss0:.0f} -> {rss_trace[-1]:.0f} MB (max {max(rss_trace):.0f}); exceptions {errors}; "
          f"guard {dict(pc.GUARD_STATS)}")
    rows.append(dict(part="F", arm="fresh", step="summary", time_median=st.median(times),
                     time_p99=pct(times, 0.99), time_max=max(times), n_over=slow,
                     pair_worst=max(worst_checks), rss_start=rss0, rss_end=rss_trace[-1],
                     exceptions=errors, guard=str(dict(pc.GUARD_STATS))))


def part_c(rows, laps):
    backend, native = nighthawk()
    n = backend.coupling_map.size()
    npair = n // 2
    print(f"\n=== C: compile-only loop, compounding (FakeNighthawk, {n} logical, {laps} laps) ===")
    rng = np.random.default_rng(13)
    theta = rng.uniform(-np.pi, np.pi, (npair, 24))
    current = QuantumCircuit(n)
    for k in range(npair):
        add_pair24(current, 2 * k, 2 * k + 1, theta[k])
    ref = pair_matrices(current, n)
    reset_guard_stats()
    times, errors, dist_trace = [], 0, []
    rss0 = rss_mb()
    rss_trace = [rss0]
    for lap in range(1, laps + 1):
        try:
            out, el, phase = cfh(current, backend, native)
            times.append(el)
            current = back_to_logical(out, n)
        except Exception as exc:
            errors += 1
            print(f"  lap {lap}: {type(exc).__name__}: {exc} -- stopping Part C here")
            break
        if lap % CHECK_EVERY == 0 or lap == laps or lap == 1:
            mats = pair_matrices(current, n)
            d = max(aligned(ref[k], mats[k]) for k in ref)
            dist_trace.append((lap, d))
            rss_trace.append(rss_mb())
            rows.append(dict(part="C", arm="compound", step=lap, time_s=el, pair_dist=d,
                             rss_mb=rss_trace[-1], layout_phase=phase))
            if lap % (CHECK_EVERY * 4) == 0 or lap == 1:
                print(f"  lap {lap}: max per-pair distance {d:.3e}, last {el * 1000:.1f} ms, "
                      f"RSS {rss_trace[-1]:.0f} MB", flush=True)
    if dist_trace:
        last_lap, last_d = dist_trace[-1]
        half = [d for l, d in dist_trace if l <= last_lap // 2]
        ratio = last_d / half[-1] if half and half[-1] > 0 else float("nan")
        print(f"  compile {time_summary(times)}; distance at lap {last_lap}: {last_d:.3e} "
              f"(per lap {last_d / last_lap:.2e}; ratio to half-way {ratio:.2f}); "
              f"RSS {rss0:.0f} -> {rss_trace[-1]:.0f} MB (max {max(rss_trace):.0f}); exceptions {errors}; "
              f"guard {dict(pc.GUARD_STATS)}")
        rows.append(dict(part="C", arm="compound", step="summary", time_median=st.median(times),
                         time_p99=pct(times, 0.99), time_max=max(times), pair_dist=last_d,
                         rss_start=rss0, rss_end=rss_trace[-1], exceptions=errors,
                         guard=str(dict(pc.GUARD_STATS))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=E_STEPS)
    ap.add_argument("--laps", type=int, default=FC_LAPS)
    args = ap.parse_args()
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, normalized_sha256(pc.__file__))
    print("LOADED", psl.__file__, getattr(psl, "LAYOUT_VERSION", "?"), normalized_sha256(psl.__file__))
    print("SCRIPT", os.path.abspath(__file__), normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-26.4":
        print("V0 FAILED: psf_compile.py is not 2026-09-26.4. Stopping.")
        return
    rows = []
    t0 = time.time()
    part_e(rows, args.steps)
    part_f(rows, args.laps)
    part_c(rows, args.laps)
    fields = ["part", "arm", "step", "loss_ref", "loss_compiled", "abs_diff", "n_over", "time_s",
              "time_median", "time_p99", "time_max", "pair_worst", "pair_applicable", "pair_dist",
              "rss_mb", "rss_start", "rss_end", "exceptions", "guard", "layout_phase"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows); total wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
