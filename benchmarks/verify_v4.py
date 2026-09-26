"""verify_v4.py -- Addendum 196.

psf_compile.py VERSION 2026-09-26.4, three changes, each switched by a
module flag for an A/B comparison in one process.

Part G  Guard on Qiskit's CX decomposer (item 17), USE_CX_GUARD off/on.
        Inputs: canonical cores exp(i(a XX + b YY + c ZZ)) in four families
        -- (0.6,0.3,e), (0.7,0.1,e), (0.5,e,e), (pi/4,e,e) -- with e from 1e-9
        to 1e-5 (four per decade), each bare and dressed with random
        single-qubit layers (3 seeds). Bare cores make the Rust core raise and
        exercise the whole-block fallback; dressed ones exercise the cached
        core. Per block: average gate infidelity and phase-included Frobenius
        distance of the synthesized circuit to the input, CX count.
Part N  Closed form with native middle gaps (item 18), USE_NATIVE_GAPS off
        (= 2026-09-26.3 gaps) / on.
  N1    300 random unitaries: phase-included distance, CX count, and `sx`
        after transpile(basis_gates=[rz, sx, x, cx], optimization_level=1).
  N2    compile_for_hardware on FakeNighthawk, dense pair blocks, spare 0
        and 8, seeds 0-4, REPS interleaved calls per arm: time, 2q, sx,
        depth, exact per-pair check; GUARD_STATS over the new arm.
Part W  Single-qubit errors in the weighted layout (item 19). FakeNighthawk
        (120 and 112 logical), FakeTorino and FakeFez (80 logical), seed 0
        (Addendum 195: ESP does not depend on the circuit seed here). Arms:
        edge errors only, edge + qubit errors, Qiskit L3. log10 ESP, mean
        2-qubit edge error, time, exact per-pair check.

Usage (repository root):
    python -u benchmarks/verify_v4.py 2>&1 | tee v4_result.txt
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
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
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, random_unitary
import qiskit_ibm_runtime.fake_provider as fp

import psf_compile as pc
import psf_smart_layout as psl

G_EPS = [10.0 ** (-k / 4) for k in range(20, 37)]  # 1e-5 .. 1e-9
G_FAMILIES = {
    "(0.6,0.3,e)": lambda e: (0.6, 0.3, e),
    "(0.7,0.1,e)": lambda e: (0.7, 0.1, e),
    "(0.5,e,e)": lambda e: (0.5, e, e),
    "(pi/4,e,e)": lambda e: (np.pi / 4, e, e),
}
G_SEEDS = (13, 101, 202)
N1_RANDOM = 300
N2_SPARES = (0, 8)
N2_SEEDS = 5
REPS = 5
HEAVY_HEX_LOGICAL = 80
GATES_PER_PAIR = 20
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
OUT_CSV = "v4_2026-09-26.csv"


def normalized_sha256(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    """Verbatim from bench_cliff_1v1.py."""
    from qiskit.circuit.library import UnitaryGate
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for (a, b) in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


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


def twoq_count(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2 and i.operation.name != "barrier")


def pauli_core(a, b, c):
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)
    h = a * np.kron(X, X) + b * np.kron(Y, Y) + c * np.kron(Z, Z)
    w, v = np.linalg.eigh(h)
    return (v * np.exp(1j * w)) @ v.conj().T


def dressed(core, seed):
    k = [random_unitary(2, seed=seed + i).data for i in range(4)]
    return np.kron(k[0], k[1]) @ core @ np.kron(k[2], k[3])


def dists(u, v):
    """(phase-included Frobenius distance, average gate infidelity)."""
    raw = float(np.linalg.norm(u - v))
    f = abs(np.trace(v.conj().T @ u)) / 4.0
    return raw, float(1.0 - (4.0 * f * f + 1.0) / 5.0)


def synth_cx(u):
    pc._CX_CORE_CACHE.clear()
    synth = pc.SU4GeodesicPSFSynthesizer(
        pc.GeodesicPSFHyper(tol=1e-5, on_unsupported="keep", entangling_basis="cx"), verify=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return synth.synthesize(u)


def esp_log10(qc, target):
    total = 0.0
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "delay") or name not in target.operation_names:
            continue
        qargs = tuple(qc.find_bit(q).index for q in inst.qubits)
        try:
            props = target[name][qargs]
        except KeyError:
            props = None
        if props is not None and props.error:
            total += math.log10(max(1e-300, 1.0 - props.error))
    return total


def mean_edge_error(qc, edge_err):
    used = {tuple(sorted(qc.find_bit(q).index for q in i.qubits))
            for i in qc.data if len(i.qubits) == 2 and i.operation.name != "barrier"}
    vals = [edge_err[e] for e in used if e in edge_err]
    return float(np.mean(vals)) if vals else None


def psf_call(qc, backend, native, **extra):
    pc._CX_CORE_CACHE.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True,
                                      on_unsupported="raise", seed_transpiler=0, **extra)
        el = time.perf_counter() - t0
    return out, el


# ------------------------------------------------------------ Part G

def part_g(rows):
    print("\n=== G: guard on Qiskit's CX decomposer ===")
    res = {False: [], True: []}
    stats_on = None
    for fam, mk in G_FAMILIES.items():
        for e in G_EPS:
            cases = [("bare", pauli_core(*mk(e)))] + [(f"dressed{s}", dressed(pauli_core(*mk(e)), s))
                                                      for s in G_SEEDS]
            for form, u in cases:
                for guard in (False, True):
                    pc.USE_CX_GUARD = guard
                    qc = synth_cx(u)
                    raw, al = dists(u, Operator(qc).data)
                    n2 = qc.count_ops().get("cx", 0)
                    res[guard].append((fam, e, form, raw, al, n2))
                    rows.append(dict(part="G", case=f"{fam}_e{e:.2e}_{form}", arm="on" if guard else "off",
                                     err_raw=raw, infid=al, cx=n2))
    pc.USE_CX_GUARD = True
    for k in pc.GUARD_STATS:
        pc.GUARD_STATS[k] = 0
    for fam, mk in G_FAMILIES.items():  # one more pass with fresh counters, guard on
        for e in G_EPS:
            for form, u in [("bare", pauli_core(*mk(e)))] + [(f"d{s}", dressed(pauli_core(*mk(e)), s))
                                                              for s in G_SEEDS]:
                synth_cx(u)
    stats_on = dict(pc.GUARD_STATS)
    off, on = res[False], res[True]
    n = len(on)
    off_bad = [r for r in off if r[4] > 1e-3]
    on_worst_al = max(r[4] for r in on)
    on_worst_raw = max(r[3] for r in on)
    cx_mismatch = sum(1 for a, b in zip(off, on) if a[4] <= 1e-8 and a[5] != b[5])
    print(f"  {n} blocks per arm")
    print(f"  guard off: worst infidelity {max(r[4] for r in off):.3e}; blocks > 1e-3: {len(off_bad)}")
    for r in off_bad[:4]:
        print(f"    e.g. {r[0]} e={r[1]:.1e} {r[2]}: infidelity {r[4]:.3e}, CX {r[5]}")
    print(f"  guard on:  worst infidelity {on_worst_al:.3e}, worst phase-included Frobenius {on_worst_raw:.3e}; "
          f"max CX {max(r[5] for r in on)}")
    print(f"  CX differs where the guard-off result was already correct: {cx_mismatch}")
    print(f"  GUARD_STATS (guard on, one pass): {stats_on}")
    print(f"G1 (guard on: infidelity <= 1e-8 and phase-included Frobenius <= 1e-3 on every block): "
          f"{'PASS' if on_worst_al <= 1e-8 and on_worst_raw <= 1e-3 else 'FAIL'}")
    print(f"G2 (guard off reproduces the defect, >= 1 block with infidelity > 1e-3): {'PASS' if off_bad else 'FAIL'}")
    print(f"G3 (ZSX rejections >= 1, default-Euler rejections 0): "
          f"{'PASS' if stats_on['zsx_rejected'] >= 1 and stats_on['default_rejected'] == 0 else 'FAIL'}")
    print(f"G4 (max CX <= 3; CX unchanged where guard-off was within 1e-8): "
          f"{'PASS' if max(r[5] for r in on) <= 3 and cx_mismatch == 0 else 'FAIL'}")


# ------------------------------------------------------------ Part N

def part_n1(rows):
    print("\n=== N1: closed form, native gaps vs 2026-09-26.3 gaps, 300 random blocks ===")
    worst = {False: 0.0, True: 0.0}
    sx_diff, cx_diff = 0, 0
    for i in range(N1_RANDOM):
        u = random_unitary(4, seed=5000 + i).data
        r = {}
        for native in (False, True):
            pc.USE_NATIVE_GAPS = native
            qc = synth_cx(u)
            raw, _ = dists(u, Operator(qc).data)
            t = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
            r[native] = (raw, qc.count_ops().get("cx", 0), t.count_ops().get("sx", 0))
            worst[native] = max(worst[native], raw)
            rows.append(dict(part="N1", case=f"random_{i}", arm="native" if native else "v3gaps",
                             err_raw=raw, cx=r[native][1], sx=r[native][2]))
        sx_diff += int(r[True][2] != r[False][2])
        cx_diff += int(r[True][1] != r[False][1])
    pc.USE_NATIVE_GAPS = True
    print(f"  worst phase-included: v3 gaps {worst[False]:.3e}, native {worst[True]:.3e}; "
          f"blocks with different sx: {sx_diff}, different CX: {cx_diff}")
    print(f"N1 (native <= 1e-13, same CX and sx on every block): "
          f"{'PASS' if worst[True] <= 1e-13 and sx_diff == 0 and cx_diff == 0 else 'FAIL'}")


def part_n2(rows, backend, native):
    print("\n=== N2: end to end, FakeNighthawk ===")
    n_phys = backend.coupling_map.size()
    for k in pc.GUARD_STATS:
        pc.GUARD_STATS[k] = 0
    all_ok = True
    for spare in N2_SPARES:
        times = {False: [], True: []}
        for seed in range(N2_SEEDS):
            qc = build_dense_pair_blocks_circuit(n_phys - spare, GATES_PER_PAIR, seed)
            first = {}
            for rep in range(REPS):
                for arm in ((False, True) if rep % 2 == 0 else (True, False)):
                    pc.USE_NATIVE_GAPS = arm
                    out, el = psf_call(qc, backend, native)
                    times[arm].append(el)
                    row = dict(part="N2", case=f"spare{spare}_seed{seed}", arm="native" if arm else "v3gaps",
                               rep=rep, time_s=el, twoq=twoq_count(out))
                    if rep == 0:
                        applicable, worst = pair_check(qc, out, n_phys - spare)
                        ops = out.count_ops()
                        first[arm] = (twoq_count(out), ops.get("sx", 0), out.depth(), applicable, worst)
                        row.update(sx=ops.get("sx", 0), depth=out.depth(), pair_applicable=applicable,
                                   pair_worst=worst)
                    rows.append(row)
            o, n = first[False], first[True]
            ok = o[0] == n[0] and o[1] == n[1] and n[3] and n[4] is not None and n[4] <= 1e-12
            all_ok &= ok
            print(f"spare={spare} seed={seed}: 2q {o[0]}/{n[0]} sx {o[1]}/{n[1]} depth {o[2]}/{n[2]} "
                  f"pair_worst native {n[4]}", flush=True)
        mo, mn = st.median(times[False]), st.median(times[True])
        print(f"spare={spare}: median v3 gaps {mo * 1000:.2f} ms, native {mn * 1000:.2f} ms")
        if spare == 0:
            all_ok &= mn <= 0.055
    pc.USE_NATIVE_GAPS = True
    print(f"  GUARD_STATS over N2 (both arms): {dict(pc.GUARD_STATS)}")
    print(f"N2 (same 2q and sx, per-pair <= 1e-12, spare-0 native median <= 55 ms): {'PASS' if all_ok else 'FAIL'}")


# ------------------------------------------------------------ Part W

def part_w(rows):
    print("\n=== W: single-qubit errors in the weighted layout ===")
    configs = [("FakeNighthawk", 0), ("FakeNighthawk", 8), ("FakeTorino", None), ("FakeFez", None)]
    w1 = w2 = total = 0
    ok_all = True
    for name, spare in configs:
        if not hasattr(fp, name):
            print(f"  {name} not available; skipped")
            continue
        with contextlib.redirect_stderr(io.StringIO()):
            backend = getattr(fp, name)()
        target = backend.target
        native = [g for g in backend.operation_names if g in NATIVE]
        n_phys = backend.coupling_map.size()
        n_log = n_phys - spare if spare is not None else HEAVY_HEX_LOGICAL
        qc = build_dense_pair_blocks_circuit(n_log, GATES_PER_PAIR, 0)
        edge_err = pc.edge_errors_from_target(target)
        qubit_err = pc.qubit_errors_from_target(target)
        arms = {}
        arms["edge only"] = psf_call(qc, backend, native, layout_edge_errors=edge_err)
        arms["edge + qubit"] = psf_call(qc, backend, native, layout_edge_errors=edge_err,
                                        layout_qubit_errors=qubit_err)
        t0 = time.perf_counter()
        out_q = transpile(qc, backend, optimization_level=3, seed_transpiler=0)
        arms["Qiskit L3"] = (out_q, time.perf_counter() - t0)
        res = {}
        for arm, (out, t) in arms.items():
            applicable, worst = pair_check(qc, out, n_log)
            res[arm] = (esp_log10(out, target), mean_edge_error(out, edge_err), t, twoq_count(out),
                        out.count_ops().get("sx", 0), applicable, worst)
            rows.append(dict(part="W", case=f"{name}_n{n_log}", arm=arm, time_s=t, twoq=res[arm][3],
                             sx=res[arm][4], esp_log10=res[arm][0], mean_edge_err=res[arm][1],
                             pair_applicable=applicable, pair_worst=worst))
        e, eq, q = res["edge only"], res["edge + qubit"], res["Qiskit L3"]
        total += 1
        w1 += int(eq[0] >= q[0])
        w2 += int(eq[0] >= e[0])
        ok_all &= bool(eq[5] and eq[6] is not None and eq[6] <= 1e-12)
        print(f"{name} n={n_log}: log10 ESP edge-only {e[0]:.4f} | edge+qubit {eq[0]:.4f} | L3 {q[0]:.4f} ; "
              f"mean edge err {e[1]:.2e}/{eq[1]:.2e}/{q[1]:.2e} ; 2q {e[3]}/{eq[3]}/{q[3]} ; "
              f"sx {e[4]}/{eq[4]}/{q[4]} ; time {e[2]:.3f}/{eq[2]:.3f}/{q[2]:.3f}s", flush=True)
    print(f"W1 (edge+qubit ESP >= Qiskit L3): {w1}/{total} -> {'PASS' if w1 == total else 'FAIL'}")
    print(f"W2 (edge+qubit ESP >= edge-only): {w2}/{total} -> {'PASS' if w2 == total else 'FAIL'}")
    print(f"W3 (edge+qubit per-pair <= 1e-12): {'PASS' if ok_all else 'FAIL'}")


def main():
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, normalized_sha256(pc.__file__))
    print("LOADED", psl.__file__, getattr(psl, "LAYOUT_VERSION", "NO LAYOUT_VERSION"),
          normalized_sha256(psl.__file__))
    print("SCRIPT", os.path.abspath(__file__), normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-26.4" or getattr(psl, "LAYOUT_VERSION", None) != "2026-09-26.m1":
        print("V0 FAILED: not the registered files. Stopping.")
        return
    rows = []
    with contextlib.redirect_stderr(io.StringIO()):
        backend = fp.FakeNighthawk()
    native = [g for g in backend.operation_names if g in NATIVE]
    psf_call(build_dense_pair_blocks_circuit(backend.coupling_map.size(), GATES_PER_PAIR, 99), backend, native)
    part_g(rows)
    part_n1(rows)
    part_n2(rows, backend, native)
    part_w(rows)
    fields = ["part", "case", "arm", "rep", "time_s", "err_raw", "infid", "cx", "twoq", "sx", "depth",
              "pair_applicable", "pair_worst", "esp_log10", "mean_edge_err"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
