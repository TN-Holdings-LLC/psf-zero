"""verify_closed_form_and_weighted.py -- Addendum 194.

Two changes in psf_compile.py VERSION 2026-09-26.3, checked in one process.

Part C  Closed-form CX core (changelog item 15), switched with
        psf_compile.USE_CX_CLOSED_FORM (old arm False, new arm True).
  C1    Block level: 300 random 2-qubit unitaries plus a special set
        (identity, CX, SWAP, iSWAP, local-dressed CX, cores with c = 0 and
        with c = 1e-9, 1e-7, 1e-5, 1e-3). For each unitary and arm: exact
        distance ||Operator(qc) - U||_F (global phase included), CX count,
        whether the closed form was used, and the `sx` count after
        transpile(basis_gates=[rz, sx, x, cx], optimization_level=1).
  C2    End to end: compile_for_hardware on FakeNighthawk, dense pair
        blocks, spare 0 and 8, seeds 0-4, REPS interleaved calls per arm:
        time, 2q count, sx count, depth, exact per-pair check.
Part Q  Error-weighted matching layout (changelog item 16).
        Backends: FakeNighthawk (spare 0 and 8, i.e. 120 and 112 logical
        qubits) and, when available, FakeTorino and FakeFez (80 logical
        qubits). Seeds 0-2. Arms: PSF unweighted (layout_search=True), PSF
        weighted (plus layout_edge_errors=edge_errors_from_target(target)),
        Qiskit transpile(optimization_level=3, seed_transpiler=0).
        Metrics: estimated success probability (ESP) = product over every
        instruction of (1 - error) from the backend's Target (missing
        errors count as 0), reported as log10; mean 2-qubit error over the
        edges used; compile time; exact per-pair check.

Usage (repository root):
    python -u benchmarks/verify_closed_form_and_weighted.py 2>&1 | tee closed_form_weighted_result.txt
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
from qiskit.circuit.library import CXGate, SwapGate, iSwapGate
from qiskit.quantum_info import Operator, random_unitary
import qiskit_ibm_runtime.fake_provider as fp

import psf_compile as pc
import psf_smart_layout as psl

C1_RANDOM = 300
C2_SPARES = (0, 8)
C2_SEEDS = 5
REPS = 5
Q_SEEDS = 3
HEAVY_HEX_LOGICAL = 80
GATES_PER_PAIR = 20
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
OUT_CSV = "closed_form_weighted_2026-09-26.csv"


# ------------------------------------------------------------ helpers

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
    """exp(i(a XX + b YY + c ZZ)) via eigendecomposition (no scipy needed)."""
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)
    h = a * np.kron(X, X) + b * np.kron(Y, Y) + c * np.kron(Z, Z)
    w, v = np.linalg.eigh(h)
    return (v * np.exp(1j * w)) @ v.conj().T


def dressed(core, seed):
    """Random single-qubit layers on both sides of `core` (same local class)."""
    k = [random_unitary(2, seed=seed + i).data for i in range(4)]
    return np.kron(k[0], k[1]) @ core @ np.kron(k[2], k[3])


def esp_log10(qc, target):
    total = 0.0
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(qc.find_bit(q).index for q in inst.qubits)
        err = None
        if name in target.operation_names:
            try:
                props = target[name][qargs]
            except KeyError:
                props = None
            if props is not None:
                err = props.error
        if err:
            total += math.log10(max(1e-300, 1.0 - err))
    return total


def mean_edge_error(qc, edge_err):
    used = set()
    for inst in qc.data:
        if len(inst.qubits) == 2 and inst.operation.name != "barrier":
            used.add(tuple(sorted(qc.find_bit(q).index for q in inst.qubits)))
    vals = [edge_err[e] for e in used if e in edge_err]
    return (float(np.mean(vals)) if vals else None), len(used)


class LayoutProbe:
    def __init__(self):
        self.orig = psl.smart_vf2_layout
        self.info = None

    def __enter__(self):
        def wrapper(*a, **k):
            out = self.orig(*a, **k)
            self.info = out[1]
            return out
        psl.smart_vf2_layout = wrapper
        return self

    def __exit__(self, *exc):
        psl.smart_vf2_layout = self.orig
        return False


def psf_call(qc, backend, native, **extra):
    pc._CX_CORE_CACHE.clear()
    with LayoutProbe() as probe, contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True,
                                      on_unsupported="raise", seed_transpiler=0, **extra)
        el = time.perf_counter() - t0
    return out, el, (probe.info or {})


# ------------------------------------------------------------ Part C

def part_c1(rows):
    print("\n=== C1: block level ===")
    mats = [(f"random_{i}", random_unitary(4, seed=1000 + i).data) for i in range(C1_RANDOM)]
    mats += [
        ("identity", np.eye(4, dtype=complex)),
        ("cx", Operator(CXGate()).data),
        ("swap", Operator(SwapGate()).data),
        ("iswap", Operator(iSwapGate()).data),
        ("cx_dressed", dressed(Operator(CXGate()).data, 7)),
        ("core_c0", dressed(pauli_core(0.6, 0.3, 0.0), 11)),
    ]
    for eps in (1e-9, 1e-7, 1e-5, 1e-3):
        mats.append((f"core_c{eps:g}", dressed(pauli_core(0.6, 0.3, eps), 13)))
    agg = {False: dict(err=[], sx=0, used=0), True: dict(err=[], sx=0, used=0)}
    cx_mismatch = []
    for label, u in mats:
        res = {}
        for arm in (False, True):
            pc.USE_CX_CLOSED_FORM = arm
            pc._CX_CORE_CACHE.clear()
            synth = pc.SU4GeodesicPSFSynthesizer(
                pc.GeodesicPSFHyper(tol=1e-5, on_unsupported="keep", entangling_basis="cx"), verify=True)
            with contextlib.redirect_stdout(io.StringIO()):
                qc = synth.synthesize(u)
            err = float(np.linalg.norm(Operator(qc).data - u))
            ops = qc.count_ops()
            used = "rx" in ops
            t = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
            sx = t.count_ops().get("sx", 0)
            res[arm] = (err, ops.get("cx", 0), sx, used, synth.fallback_count)
            agg[arm]["err"].append(err)
            agg[arm]["sx"] += sx
            agg[arm]["used"] += int(used)
            rows.append(dict(part="C1", case=label, arm="new" if arm else "old", err=err,
                             cx=ops.get("cx", 0), sx=sx, closed_form=used, fallback=synth.fallback_count))
        if res[True][1] != res[False][1]:
            cx_mismatch.append((label, res[False][1], res[True][1]))
        if not label.startswith("random"):
            print(f"  {label:14s} old: err={res[False][0]:.2e} cx={res[False][1]} sx={res[False][2]} | "
                  f"new: err={res[True][0]:.2e} cx={res[True][1]} sx={res[True][2]} "
                  f"closed_form={res[True][3]} fallback={res[True][4]}")
    rnd_used = sum(1 for r in rows if r["part"] == "C1" and r["arm"] == "new"
                   and r["case"].startswith("random") and r["closed_form"])
    print(f"  all {len(mats)}: max err old {max(agg[False]['err']):.2e} new {max(agg[True]['err']):.2e}; "
          f"median old {st.median(agg[False]['err']):.2e} new {st.median(agg[True]['err']):.2e}")
    print(f"  total sx old {agg[False]['sx']} new {agg[True]['sx']}; closed form used on "
          f"{rnd_used}/{C1_RANDOM} random blocks; CX-count mismatches: {cx_mismatch or 'none'}")
    ok = (max(agg[True]["err"]) <= 1e-13 and not cx_mismatch and agg[True]["sx"] <= agg[False]["sx"]
          and rnd_used == C1_RANDOM)
    print(f"C1 {'PASS' if ok else 'FAIL'}")


def part_c2(rows, backend, native):
    print("\n=== C2: end to end, FakeNighthawk ===")
    n_phys = backend.coupling_map.size()
    for spare in C2_SPARES:
        times = {False: [], True: []}
        ok = True
        for seed in range(C2_SEEDS):
            qc = build_dense_pair_blocks_circuit(n_phys - spare, GATES_PER_PAIR, seed)
            first = {}
            for rep in range(REPS):
                order = (False, True) if rep % 2 == 0 else (True, False)
                for arm in order:
                    pc.USE_CX_CLOSED_FORM = arm
                    out, el, _ = psf_call(qc, backend, native)
                    times[arm].append(el)
                    row = dict(part="C2", case=f"spare{spare}_seed{seed}", arm="new" if arm else "old",
                               rep=rep, time_s=el, twoq=twoq_count(out))
                    if rep == 0:
                        applicable, worst = pair_check(qc, out, n_phys - spare)
                        ops = out.count_ops()
                        first[arm] = (twoq_count(out), ops.get("sx", 0), out.depth(), applicable, worst)
                        row.update(sx=ops.get("sx", 0), depth=out.depth(), pair_applicable=applicable,
                                   pair_worst=worst)
                    rows.append(row)
            o, n = first[False], first[True]
            case_ok = (o[0] == n[0] and n[1] <= o[1] and n[3] and n[4] is not None and n[4] <= 1e-12)
            ok &= case_ok
            print(f"spare={spare} seed={seed}: 2q {o[0]}->{n[0]} sx {o[1]}->{n[1]} depth {o[2]}->{n[2]} "
                  f"pair_worst new {n[4]}", flush=True)
        mo, mn = st.median(times[False]), st.median(times[True])
        m_ok = mn <= mo - 0.010
        print(f"spare={spare}: median old {mo * 1000:.2f} ms -> new {mn * 1000:.2f} ms | "
              f"C2 correctness {'PASS' if ok else 'FAIL'} | C2 speed {'PASS' if m_ok else 'FAIL'}")
    pc.USE_CX_CLOSED_FORM = True


# ------------------------------------------------------------ Part Q

def part_q(rows):
    print("\n=== Q: error-weighted matching vs unweighted vs Qiskit L3 ===")
    configs = [("FakeNighthawk", 0), ("FakeNighthawk", 8)]
    for name in ("FakeTorino", "FakeFez"):
        if hasattr(fp, name):
            configs.append((name, None))
        else:
            print(f"  {name} not available in this qiskit_ibm_runtime; skipped")
    q2, q3, q4, total = 0, 0, [], 0
    q1_ok = True
    for name, spare in configs:
        with contextlib.redirect_stderr(io.StringIO()):
            backend = getattr(fp, name)()
        target = backend.target
        native = [g for g in backend.operation_names if g in NATIVE]
        n_phys = backend.coupling_map.size()
        n_log = n_phys - spare if spare is not None else HEAVY_HEX_LOGICAL
        edge_err = pc.edge_errors_from_target(target)
        for seed in range(Q_SEEDS):
            qc = build_dense_pair_blocks_circuit(n_log, GATES_PER_PAIR, seed)
            res = {}
            out_u, t_u, info_u = psf_call(qc, backend, native)
            out_w, t_w, info_w = psf_call(qc, backend, native, layout_edge_errors=edge_err)
            t0 = time.perf_counter()
            out_q = transpile(qc, backend, optimization_level=3, seed_transpiler=0)
            t_q = time.perf_counter() - t0
            for arm, out, t, info in (("PSF unweighted", out_u, t_u, info_u),
                                      ("PSF weighted", out_w, t_w, info_w),
                                      ("Qiskit L3", out_q, t_q, {})):
                applicable, worst = pair_check(qc, out, n_log)
                me, n_edges = mean_edge_error(out, edge_err)
                res[arm] = dict(esp=esp_log10(out, target), mean_edge=me, time=t, twoq=twoq_count(out),
                                sx=out.count_ops().get("sx", 0), phase=info.get("phase"),
                                order=info.get("order_name"), applicable=applicable, worst=worst)
                rows.append(dict(part="Q", case=f"{name}_n{n_log}_seed{seed}", arm=arm, time_s=t,
                                 twoq=res[arm]["twoq"], sx=res[arm]["sx"], esp_log10=res[arm]["esp"],
                                 mean_edge_err=me, layout_phase=info.get("phase"),
                                 layout_order=info.get("order_name"),
                                 pair_applicable=applicable, pair_worst=worst))
            u, w, q = res["PSF unweighted"], res["PSF weighted"], res["Qiskit L3"]
            total += 1
            case_q1 = (w["phase"] == 0 and w["order"] == "matching_weighted" and w["applicable"]
                       and w["worst"] is not None and w["worst"] <= 1e-12 and w["twoq"] == u["twoq"])
            q1_ok &= case_q1
            q2 += int(w["esp"] >= u["esp"])
            q3 += int(w["esp"] >= q["esp"])
            q4.append(w["time"] - u["time"])
            print(f"{name} n={n_log} seed={seed}: log10 ESP  unweighted {u['esp']:.4f} | weighted {w['esp']:.4f} | "
                  f"L3 {q['esp']:.4f} ; mean edge err {u['mean_edge']:.2e} / {w['mean_edge']:.2e} / "
                  f"{q['mean_edge']:.2e} ; 2q {u['twoq']}/{w['twoq']}/{q['twoq']} ; sx {u['sx']}/{w['sx']}/{q['sx']} ; "
                  f"time {u['time']:.3f}/{w['time']:.3f}/{q['time']:.3f}s ; weighted layout {w['order']}",
                  flush=True)
    print(f"\nQ1 (weighted path taken and correct, same 2q as unweighted): {'PASS' if q1_ok else 'FAIL'}")
    print(f"Q2 (weighted ESP >= unweighted ESP): {q2}/{total} -> {'PASS' if q2 == total else 'FAIL'}")
    verdict = "SUPPORTED" if q3 * 3 >= 2 * total else ("NOT SUPPORTED" if q3 * 3 <= total else "INCONCLUSIVE")
    print(f"Q3 (weighted ESP >= Qiskit L3 ESP): {q3}/{total} -> {verdict}")
    print(f"Q4 (weighted - unweighted compile time, median): {st.median(q4) * 1000:.2f} ms -> "
          f"{'PASS' if st.median(q4) <= 0.002 else 'FAIL'}")


def main():
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, normalized_sha256(pc.__file__))
    print("LOADED", psl.__file__, getattr(psl, "LAYOUT_VERSION", "NO LAYOUT_VERSION"),
          normalized_sha256(psl.__file__))
    print("SCRIPT", os.path.abspath(__file__), normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-26.3" or getattr(psl, "LAYOUT_VERSION", None) != "2026-09-26.m1":
        print("C0 FAILED: not the registered files. Stopping.")
        return
    rows = []
    with contextlib.redirect_stderr(io.StringIO()):
        backend = fp.FakeNighthawk()
    native = [g for g in backend.operation_names if g in NATIVE]
    psf_call(build_dense_pair_blocks_circuit(backend.coupling_map.size(), GATES_PER_PAIR, 99), backend, native)
    part_c1(rows)
    part_c2(rows, backend, native)
    part_q(rows)
    fields = ["part", "case", "arm", "rep", "time_s", "err", "cx", "twoq", "sx", "depth", "closed_form",
              "fallback", "pair_applicable", "pair_worst", "esp_log10", "mean_edge_err", "layout_phase",
              "layout_order"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
