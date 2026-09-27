"""diag_polish_and_blocks.py -- Addendum 205: two questions raised by Addendum 204.

Part P  The polish step (psf_compile._refine_decomposition, Addendum 186).
        Addendum 204 measured ~0.21 ms per block, which suggests the
        Gauss-Newton step runs on most blocks, not only on the near-face
        inputs it was added for. Every call is recorded: residual before and
        after (Frobenius norm of reconstruction - target), whether a step was
        taken, its time, and |c| (the third Cartan coordinate; the face c = 0
        is where Addendum 185 found the core's error). Sets:
          P1 fresh cliff circuits (FakeNighthawk, 120 logical, 60 pair24
             blocks), 30 circuits
          P2 fresh training circuits (12 qubits, Addendum 199 ansatz), 30
          P3 cliff compounding loop, 100 laps
        each run with the step's trigger threshold at 1e-13 (current), 1e-12,
        1e-11 and "off" (never step). Accuracy per threshold: P1 exact
        per-pair check (worst infidelity), P2 loss against the uncompiled
        reference, P3 per-pair distance to the lap-0 circuit at laps 10, 50
        and 100. Nothing in psf_compile.py is changed; the threshold is
        passed through a wrapper for the duration of each run.
Part G  Two-qubit gate count of the training circuit: 48 measured where 33
        was expected. The consolidated circuit that compile() builds is
        captured, listing which qubit pairs became unitary blocks and which
        two-qubit gates passed through unconsolidated, for block_gate_floor
        in {12 (default), 10, 8, 6, 4, 2}; plus the final two-qubit count,
        compile time and exactness for the training circuit and three other
        families (hardware-efficient ansatz, QAOA on a line, cliff pair24).

Usage (repository root; loop_endurance.py must be in benchmarks/):
    python -u benchmarks/diag_polish_and_blocks.py 2>&1 | tee diag_polish_and_blocks.txt
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
from collections import Counter

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
THRESHOLDS = (("1e-13 (current)", 1e-13), ("1e-12", 1e-12), ("1e-11", 1e-11), ("off", math.inf))
N_FRESH = 30
C_LAPS = 100
C_CHECK = (10, 50, 100)
FLOORS = (12, 10, 8, 6, 4, 2)
OUT_CSV = "diag_polish_and_blocks_2026-09-27.csv"
LINE = CouplingMap.from_line(le.E_QUBITS)
BASIS = ["cx", "rz", "sx", "x"]

ORIG_REFINE = pc._refine_decomposition
REC: list = []


class Polish:
    """Routes every polish call through ORIG_REFINE with the given trigger
    threshold, recording each call into REC."""

    def __init__(self, threshold):
        self.threshold = threshold

    def __enter__(self):
        th = self.threshold

        def wrapper(U, cartan, k1, k2, phase, *a, **k):
            t0 = PC()
            out = ORIG_REFINE(U, cartan, k1, k2, phase, threshold=th)
            dt = PC() - t0
            REC.append((out[1], out[2], dt, out[1] > th, abs(float(cartan[2]))))
            return out

        pc._refine_decomposition = wrapper
        return self

    def __exit__(self, *exc):
        pc._refine_decomposition = ORIG_REFINE
        return False


# ------------------------------------------------------------ circuits and compiles

def cliff_circuit(n, theta):
    qc = QuantumCircuit(n)
    for k in range(n // 2):
        le.add_pair24(qc, 2 * k, 2 * k + 1, theta[k])
    return qc


def cliff_compile(qc, backend, native, floor=pc.DEFAULT_BLOCK_GATE_FLOOR):
    pc._CX_CORE_CACHE.clear()
    t0 = PC()
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                      block_gate_floor=floor, entangling_basis="cx", layout_search=True,
                                      on_unsupported="raise", seed_transpiler=0)
    return out, PC() - t0


def line_compile(qc, floor=pc.DEFAULT_BLOCK_GATE_FLOOR):
    pc._CX_CORE_CACHE.clear()
    t0 = PC()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        out = pc.compile_for_hardware(qc, coupling_map=LINE, basis_gates=BASIS, block_gate_floor=floor,
                                      entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                      on_unsupported="keep", seed_transpiler=0)
    el = PC() - t0
    final = list(out.layout.final_index_layout()) if out.layout is not None else list(range(le.E_QUBITS))
    if final != list(range(le.E_QUBITS)):
        raise RuntimeError(f"compiled circuit permuted qubits: {final}")
    return out, el


def twoq(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2)


def state_infidelity(ref, out):
    return 1.0 - abs(Statevector(ref).inner(Statevector(out))) ** 2


def hea_circuit(rng, layers=4):
    n = le.E_QUBITS
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-np.pi, np.pi), q)
            qc.rz(rng.uniform(-np.pi, np.pi), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def qaoa_line_circuit(rng, p=3):
    n = le.E_QUBITS
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        g, b = rng.uniform(-np.pi, np.pi, 2)
        for q in range(n - 1):
            qc.rzz(g, q, q + 1)
        for q in range(n):
            qc.rx(b, q)
    return qc


# ------------------------------------------------------------ Part P

def summarize(recs, n_compiles):
    if not recs:
        return dict(calls=0)
    before = [r[0] for r in recs]
    stepped = [r for r in recs if r[3]]
    return dict(calls=len(recs), per_compile=len(recs) / n_compiles, stepped=len(stepped),
                stepped_frac=len(stepped) / len(recs),
                before_med=st.median(before), before_p90=le.pct(before, 0.90), before_max=max(before),
                after_max=max(r[1] for r in recs),
                polish_ms_per_compile=1000 * sum(r[2] for r in recs) / n_compiles,
                dt_step_us=1e6 * st.median([r[2] for r in stepped]) if stepped else float("nan"),
                dt_nostep_us=1e6 * st.median([r[2] for r in recs if not r[3]]) if len(stepped) < len(recs)
                else float("nan"))


def print_summary(label, s, extra=""):
    if not s.get("calls"):
        print(f"    {label:16s}: no polish calls")
        return
    print(f"    {label:16s}: {s['calls']} calls ({s['per_compile']:.1f} per compile), step taken on "
          f"{s['stepped']} ({100 * s['stepped_frac']:.1f}%); residual before: median {s['before_med']:.2e}, "
          f"p90 {s['before_p90']:.2e}, max {s['before_max']:.2e}; after: max {s['after_max']:.2e}; "
          f"polish {s['polish_ms_per_compile']:.2f} ms per compile (per call: with step {s['dt_step_us']:.0f} us, "
          f"without {s['dt_nostep_us']:.0f} us){extra}", flush=True)


def part_p(backend, native, rows):
    n = backend.coupling_map.size()
    rng = np.random.default_rng(31)
    fresh_cliff = [cliff_circuit(n, rng.uniform(-np.pi, np.pi, (n // 2, 24))) for _ in range(N_FRESH)]
    target, theta0 = spsa_target()
    fresh_train = [le.e_circuit(theta0 + rng.normal(0.0, 0.3, le.E_NPARAMS)) for _ in range(N_FRESH)]
    ref_losses = [1.0 - abs(target.inner(Statevector(qc))) ** 2 for qc in fresh_train]
    rng13 = np.random.default_rng(13)
    initial = cliff_circuit(n, rng13.uniform(-np.pi, np.pi, (n // 2, 24)))
    ref_pairs = le.pair_matrices(initial, n)
    results = {}
    near_face = None

    print(f"\n=== P1: fresh cliff circuits ({N_FRESH}) ===")
    for label, th in THRESHOLDS:
        REC.clear()
        times, worst = [], 0.0
        with Polish(th):
            for qc in fresh_cliff:
                out, el = cliff_compile(qc, backend, native)
                times.append(el)
                ok, w = le.pair_check(qc, out, n)
                worst = max(worst, w if ok else math.inf)
        s = summarize(REC, N_FRESH)
        s.update(compile_med=st.median(times), worst=worst)
        results[("P1", label)] = s
        if label.startswith("1e-13"):
            near = [r for r in REC if r[4] < 1e-3]
            far = [r for r in REC if r[4] >= 1e-3]
            near_face = (len(near), sum(r[3] for r in near), len(far), sum(r[3] for r in far),
                         st.median([r[0] for r in near]) if near else float("nan"),
                         st.median([r[0] for r in far]) if far else float("nan"))
            for r in REC:
                rows.append(dict(part="P1", threshold=label, before=r[0], after=r[1], seconds=r[2],
                                 stepped=int(r[3]), abs_c=r[4]))
        print_summary(label, s, f"; compile median {s['compile_med'] * 1000:.1f} ms; worst per-pair check "
                                f"{worst:.2e}")
    if near_face:
        nn, ns, fn, fs, nm, fm = near_face
        print(f"    by |c| at the current threshold: |c| < 1e-3: {nn} calls, {ns} stepped, median residual "
              f"{nm:.2e}; |c| >= 1e-3: {fn} calls, {fs} stepped, median residual {fm:.2e}")

    print(f"\n=== P2: fresh training circuits ({N_FRESH}, 12 qubits) ===")
    for label, th in THRESHOLDS:
        REC.clear()
        times, worst = [], 0.0
        with Polish(th):
            for qc, lr in zip(fresh_train, ref_losses):
                out, el = line_compile(qc)
                times.append(el)
                worst = max(worst, abs((1.0 - abs(target.inner(Statevector(out))) ** 2) - lr))
        s = summarize(REC, N_FRESH)
        s.update(compile_med=st.median(times), worst=worst)
        results[("P2", label)] = s
        if label.startswith("1e-13"):
            for r in REC:
                rows.append(dict(part="P2", threshold=label, before=r[0], after=r[1], seconds=r[2],
                                 stepped=int(r[3]), abs_c=r[4]))
        print_summary(label, s, f"; compile median {s['compile_med'] * 1000:.1f} ms; worst |loss - reference| "
                                f"{worst:.2e}")

    print(f"\n=== P3: cliff compounding loop ({C_LAPS} laps per threshold) ===")
    for label, th in THRESHOLDS:
        REC.clear()
        times, dist = [], {}
        current = initial
        with Polish(th):
            for lap in range(1, C_LAPS + 1):
                gc.disable()
                out, el = cliff_compile(current, backend, native)
                current = le.back_to_logical(out, n)
                times.append(el)
                gc.enable()
                gc.collect()
                if lap in C_CHECK:
                    mats = le.pair_matrices(current, n)
                    dist[lap] = max(le.aligned(ref_pairs[k], mats[k]) for k in ref_pairs)
        s = summarize(REC, C_LAPS)
        s.update(compile_med=st.median(times), dist=dist, worst=dist[C_LAPS])
        results[("P3", label)] = s
        if label.startswith("1e-13"):
            for r in REC:
                rows.append(dict(part="P3", threshold=label, before=r[0], after=r[1], seconds=r[2],
                                 stepped=int(r[3]), abs_c=r[4]))
        dtxt = ", ".join(f"lap {k} {v:.2e}" for k, v in dist.items())
        print_summary(label, s, f"; compile median {s['compile_med'] * 1000:.1f} ms; distance {dtxt}")
    for (part, label), s in results.items():
        rows.append(dict(part=part, threshold=label, summary=str({k: v for k, v in s.items()})))
    return results


def spsa_target():
    rng = np.random.default_rng(7)
    target_theta = rng.uniform(-np.pi, np.pi, le.E_NPARAMS)
    target_theta[8::15] = 0.0
    return Statevector(le.e_circuit(target_theta)), target_theta + rng.normal(0.0, 0.3, le.E_NPARAMS)


# ------------------------------------------------------------ Part G

CAPTURED = []


class CaptureConsolidation:
    """Captures the consolidated circuit built inside psf_compile.compile()."""

    def __enter__(self):
        orig = pc.PassManager
        self._orig = orig

        class Capturing(orig):
            def run(self_pm, *a, **k):
                out = super().run(*a, **k)
                CAPTURED.append(out)
                return out

        pc.PassManager = Capturing
        return self

    def __exit__(self, *exc):
        pc.PassManager = self._orig
        return False


def block_listing(qb):
    blocks, raw = [], Counter()
    for inst in qb.data:
        if len(inst.qubits) != 2:
            continue
        pair = tuple(sorted(qb.find_bit(q).index for q in inst.qubits))
        if inst.operation.name == "unitary":
            blocks.append(pair)
        else:
            raw[(inst.operation.name, pair)] += 1
    return blocks, raw


def part_g(backend, native, rows):
    print("\n=== G: training circuit -- which pairs become blocks, by block_gate_floor ===")
    target, theta0 = spsa_target()
    train = le.e_circuit(theta0)
    ref_loss = 1.0 - abs(target.inner(Statevector(train))) ** 2
    res = {}
    for floor in FLOORS:
        CAPTURED.clear()
        with CaptureConsolidation():
            out, _ = line_compile(train, floor)
        blocks, raw = block_listing(CAPTURED[0])
        times = [line_compile(train, floor)[1] for _ in range(5)]
        loss_diff = abs((1.0 - abs(target.inner(Statevector(out))) ** 2) - ref_loss)
        raw_pairs = sorted({p for (_, p) in raw})
        raw_names = Counter(name for (name, _) in raw.elements())
        res[("train", floor)] = dict(twoq=twoq(out), blocks=len(blocks), raw=sum(raw.values()),
                                     time=st.median(times), exact=loss_diff)
        print(f"  floor {floor:2d}: {len(blocks)} blocks on {sorted(set(blocks))}; unconsolidated two-qubit "
              f"gates {sum(raw.values())} {dict(raw_names)} on {raw_pairs}; final two-qubit gates {twoq(out)}; "
              f"compile median {st.median(times) * 1000:.1f} ms; |loss - reference| {loss_diff:.1e}", flush=True)
        rows.append(dict(part="G", family="train", floor=floor, twoq=twoq(out), blocks=len(blocks),
                         raw=sum(raw.values()), seconds=st.median(times), exact=loss_diff,
                         summary=f"blocks {sorted(set(blocks))}; raw {dict(raw_names)} on {raw_pairs}"))

    print("\n=== G: other families, by block_gate_floor ===")
    rng = np.random.default_rng(41)
    hea, qaoa = hea_circuit(rng), qaoa_line_circuit(rng)
    n = backend.coupling_map.size()
    cliff = cliff_circuit(n, rng.uniform(-np.pi, np.pi, (n // 2, 24)))
    for fam, qc in (("hea", hea), ("qaoa", qaoa), ("cliff", cliff)):
        for floor in FLOORS:
            if fam == "cliff":
                outs = [cliff_compile(qc, backend, native, floor) for _ in range(3)]
                out = outs[0][0]
                ok, w = le.pair_check(qc, out, n)
                exact = w if ok else math.inf
            else:
                outs = [line_compile(qc, floor) for _ in range(3)]
                out = outs[0][0]
                exact = state_infidelity(qc, out)
            t = st.median(o[1] for o in outs)
            res[(fam, floor)] = dict(twoq=twoq(out), time=t, exact=exact)
            rows.append(dict(part="G", family=fam, floor=floor, twoq=twoq(out), seconds=t, exact=exact))
        line = "; ".join(f"floor {f}: {res[(fam, f)]['twoq']} 2q, {res[(fam, f)]['time'] * 1000:.1f} ms, "
                         f"exact {res[(fam, f)]['exact']:.1e}" for f in FLOORS)
        print(f"  {fam:5s}: {line}", flush=True)
    return res


# ------------------------------------------------------------ main

def main():
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-26.4":
        print("V0 FAILED: psf_compile.py is not 2026-09-26.4. Stopping.")
        return
    t_start = time.time()
    backend, native = le.nighthawk()
    rows = []
    p = part_p(backend, native, rows)
    g = part_g(backend, native, rows)

    print("\n=== predictions ===")

    def v(ok):
        return "CONFIRMED" if ok else "NOT CONFIRMED"

    cur = "1e-13 (current)"
    fr = {k: p[(k, cur)]["stepped_frac"] for k in ("P1", "P2", "P3")}
    print(f"Q1 at the current threshold the step runs on >= 80% of calls in P1, P2 and P3 "
          f"({', '.join(f'{k} {100 * x:.1f}%' for k, x in fr.items())}) -> {v(all(x >= 0.8 for x in fr.values()))}")
    pooled = [p[(k, cur)]["before_med"] for k in ("P1", "P2", "P3")]
    print(f"Q2 median residual before the polish < 1e-12 in every set "
          f"({', '.join(f'{x:.2e}' for x in pooled)}) -> {v(all(x < 1e-12 for x in pooled))}")
    t_cur, t_12 = p[("P1", cur)]["polish_ms_per_compile"], p[("P1", "1e-12")]["polish_ms_per_compile"]
    w12 = p[("P1", "1e-12")]["worst"]
    d_cur, d_12 = p[("P3", cur)]["dist"][C_LAPS], p[("P3", "1e-12")]["dist"][C_LAPS]
    q3 = t_12 <= 0.3 * t_cur and w12 <= 1e-12 and d_12 <= 2 * d_cur
    print(f"Q3 threshold 1e-12: P1 polish time <= 30% of current ({t_12:.2f} vs {t_cur:.2f} ms), worst per-pair "
          f"check <= 1e-12 ({w12:.2e}), P3 distance at lap {C_LAPS} <= 2 x current ({d_12:.2e} vs {d_cur:.2e}) "
          f"-> {v(q3)}")
    g12 = g[("train", 12)]
    print(f"H1 floor 12: 6 blocks and 15 unconsolidated two-qubit gates in the training circuit "
          f"({g12['blocks']}, {g12['raw']}) -> {v(g12['blocks'] == 6 and g12['raw'] == 15)}")
    g8 = g[("train", 8)]
    print(f"H2 floor 8: 11 blocks, no unconsolidated two-qubit gate, 33 final two-qubit gates, exact "
          f"({g8['blocks']}, {g8['raw']}, {g8['twoq']}, {g8['exact']:.1e}) -> "
          f"{v(g8['blocks'] == 11 and g8['raw'] == 0 and g8['twoq'] == 33 and g8['exact'] <= 1e-10)}")
    h3 = all(g[(f, 8)]["twoq"] <= g[(f, 12)]["twoq"] for f in ("hea", "qaoa", "cliff"))
    h3_exact = all(g[(f, fl)]["exact"] <= 1e-10 for f in ("hea", "qaoa", "cliff") for fl in FLOORS)
    changes = ", ".join("%s %d->%d" % (f, g[(f, 12)]["twoq"], g[(f, 8)]["twoq"]) for f in ("hea", "qaoa", "cliff"))
    print(f"H3 floor 8 does not increase two-qubit gates in hea, qaoa or cliff ({changes}), "
          f"and every compile in G is exact (<= 1e-10) -> {v(h3 and h3_exact)}")

    fields = ["part", "threshold", "family", "floor", "before", "after", "seconds", "stepped", "abs_c",
              "twoq", "blocks", "raw", "exact", "summary"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows); total wall time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
