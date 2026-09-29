"""pl_heavyhex_chain.py -- pre-registered (workplace, 2026-09-29): the timed PennyLane
compounding loop of Addenda 183-184, moved to a fully occupied heavy-hex device
(FakeKingston, 156 qubits) and run with the candidate stack of 2026-09-29.

Each lap: PennyLane tape -> Qiskit circuit -> compile -> map back to logical qubits ->
PennyLane tape, which is the next lap's input. Every lap's compile time is scored against
a 1 s deadline, and every lap's meaning is checked block by block against PennyLane's own
matrices of the lap-0 tape (qml.matrix; blocks of 2 or 3 wires).

Circuit family T of full_heavyhex_cliff.py: k2 pair blocks and k3 = N - 2M triple
blocks (a 3-wire path a-b-c), filling the device at spare 0. A pair block is 20
Haar-random QubitUnitary on (a, b), as in Addendum 183; a triple block is 10 on (a, b)
followed by 10 on (b, c). Spare s removes s/2 pair blocks.

Arms (one arm per process, chosen with --arm; the core is whatever psf_zero_core is
first on the path, the layout module is taken from --layout-dir when given):
  Q3  transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
  P   compile_for_hardware(..., entangling_basis="cx", layout_search=True,
      on_unsupported="raise", seed_transpiler=0), release core 2026-09-28.1 and
      release psf_smart_layout 2026-09-26.m1
  PN  the same call, candidate core 2026-09-29.1 and candidate psf_smart_layout
      2026-09-29.c1
If a lap's output contains a two-qubit gate between different blocks or permutes
qubits (SWAPs), it cannot be mapped back block by block: that lap is recorded as
"swap" with no meaning check, and the next lap reuses the same input tape.

    PYTHONPATH=<core> python -u benchmarks/pl_heavyhex_chain.py run --arm Q3 [--layout-dir D]
    python -u benchmarks/pl_heavyhex_chain.py score
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
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

SPARES = (0, 16)
LAPS = 20
DEADLINE = 1.0
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
EXPECTED = {"P": ("2026-09-28.1", "2026-09-26.m1"), "PN": ("2026-09-29.1", "2026-09-29.c1")}


class SwapLap(RuntimeError):
    pass


def initial_tape(blocks, n, seed):
    import pennylane as qml
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)

    def ru():
        return random_unitary(4, seed=int(rng.integers(0, 2**31))).data

    ops = []
    for b in blocks:
        if len(b) == 2:
            ops += [qml.QubitUnitary(ru(), wires=[b[0], b[1]]) for _ in range(20)]
        else:
            ops += [qml.QubitUnitary(ru(), wires=[b[0], b[1]]) for _ in range(10)]
            ops += [qml.QubitUnitary(ru(), wires=[b[1], b[2]]) for _ in range(10)]
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


def block_matrices(tape, blocks):
    """PennyLane's own matrix of each block's operations, in tape order."""
    import pennylane as qml
    owner = {q: k for k, b in enumerate(blocks) for q in b}
    per = {k: [] for k in range(len(blocks))}
    for op in tape.operations:
        ks = {owner[int(w)] for w in op.wires}
        if len(ks) != 1:
            raise SwapLap(f"operation {op.name} on wires {list(op.wires)} spans blocks")
        per[ks.pop()].append(op)
    mats = {}
    for k, b in enumerate(blocks):
        ops = per[k]
        mats[k] = (qml.matrix(qml.tape.QuantumTape(ops, measurements=[], shots=None), wire_order=list(b))
                   if ops else np.eye(2 ** len(b), dtype=complex))
    return mats


def aligned(a, b):
    t = np.trace(a.conj().T @ b)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(b - ph * a))


def back_to_logical(routed, n, blocks):
    from qiskit import QuantumCircuit
    init = list(routed.layout.initial_index_layout(filter_ancillas=True))
    final = list(routed.layout.final_index_layout(filter_ancillas=True))
    if init != final:
        raise SwapLap("routing permuted qubits")
    to_logical = {p: v for v, p in enumerate(init)}
    owner = {q: k for k, b in enumerate(blocks) for q in b}
    out = QuantumCircuit(n)
    for inst in routed.data:
        if inst.operation.name in ("barrier", "delay", "measure"):
            continue
        phys = [routed.find_bit(q).index for q in inst.qubits]
        if any(p not in to_logical for p in phys):
            if len(phys) == 1:
                continue
            raise SwapLap(f"two-qubit operation outside the layout: {phys}")
        lq = [to_logical[p] for p in phys]
        if len({owner[q] for q in lq}) != 1:
            raise SwapLap(f"two-qubit operation between blocks: {lq}")
        out.append(inst.operation, lq)
    return out, tuple(init)


def to_tape(logical):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    from psf_pennylane_gpu_prototype import qiskit_to_tape
    wrapped = QuantumCircuit(logical.num_qubits)
    for inst in logical.data:
        qs = [logical.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2:
            wrapped.append(UnitaryGate(Operator(inst.operation).data), qs)
        else:
            wrapped.append(inst.operation, qs)
    return qiskit_to_tape(wrapped, list(range(logical.num_qubits)))


def run(args):
    # The layout module must be found first before anything imports psf_smart_layout
    # (loop_endurance imports it at module level).
    if args.layout_dir:
        sys.path.insert(0, args.layout_dir)
    import warnings
    import pennylane as qml
    import qiskit
    from qiskit import transpile
    from qiskit_ibm_runtime import fake_provider
    import full_heavyhex_cliff as fh
    import loop_endurance as le
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    core_v = layout_v = None
    if args.arm != "Q3":
        import psf_compile as pc
        import psf_smart_layout as sl
        core_v, layout_v = getattr(pc, "CORE_VERSION", None), sl.LAYOUT_VERSION
        print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__), "| CORE_VERSION", core_v)
        print("LAYOUT", sl.__file__, layout_v, le.normalized_sha256(sl.__file__))
    print(f"arm {args.arm} | platform {platform.platform()} | cores {os.cpu_count()} | python "
          f"{platform.python_version()} | qiskit {qiskit.__version__} | pennylane {qml.__version__}")
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = getattr(fake_provider, args.fake)().target
    f = fh.graph_facts(target)
    nq, m = f["nq"], f["matching"]
    print(f"Stage 0: {args.fake}, {nq} qubits, max matching {m}, {{P2,P3}}-factor "
          f"{'found' if f['factor'] else 'NOT FOUND'}")
    if not f["factor"]:
        print("G0 FAILED: nothing is run.")
        return
    cmap = target.build_coupling_map()
    native = [g for g in target.operation_names if g in NATIVE]
    spares = tuple(int(x) for x in args.spares.split(","))
    rows = []
    for spare in spares:
        blocks = fh.layout_blocks("T", spare, nq, m)
        n = sum(len(b) for b in blocks)
        swapfree = fh.expected_twoq(blocks)
        tape = initial_tape(blocks, n, seed=1000 * spare)
        u0 = block_matrices(tape, blocks)
        prev_layout = None
        for lap in range(1, args.laps + 1):
            t_lap = time.perf_counter()
            qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
            t0 = time.perf_counter()
            if args.arm == "Q3":
                routed = transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    routed = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=native,
                                                     entangling_basis="cx", layout_search=True,
                                                     on_unsupported="raise", seed_transpiler=0)
            compile_s = time.perf_counter() - t0
            twoq = sum(1 for i in routed.data if len(i.qubits) == 2 and i.operation.name != "barrier")
            try:
                logical, layout = back_to_logical(routed, n, blocks)
                new_tape = to_tape(logical)
                mats = block_matrices(new_tape, blocks)
                dist = max(aligned(u0[k], mats[k]) for k in u0)
                status, tape = "OK", new_tape
            except SwapLap as e:
                status, dist, layout = f"swap: {e}"[:80], None, None
            lap_s = time.perf_counter() - t_lap
            row = dict(arm=args.arm, spare=spare, logical_qubits=n, lap=lap, status=status, compile_s=compile_s,
                       lap_s=lap_s, within_1s=compile_s <= DEADLINE, twoq=twoq, swapfree_twoq=swapfree,
                       max_block_distance=dist, ops=len(tape.operations),
                       layout_changed=(prev_layout is not None and layout is not None and layout != prev_layout),
                       core_version=core_v, layout_version=layout_v)
            rows.append(row)
            if layout is not None:
                prev_layout = layout
            d_txt = "-" if dist is None else f"{dist:.3e}"
            print(f"{args.arm:2s} spare={spare:2d} lap {lap:3d} compile {compile_s:8.3f}s lap {lap_s:7.2f}s "
                  f"2q={twoq} (swap-free {swapfree}) max_block_dist={d_txt} layout_changed={row['layout_changed']} "
                  f"{status}", flush=True)
    out = f"pl_heavyhex_chain_{args.arm}_{args.tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh_:
        w = csv.DictWriter(fh_, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)\nDONE")


def score(args):
    rows = []
    for arm in ("Q3", "P", "PN"):
        with open(f"pl_heavyhex_chain_{arm}_{args.tag}.csv", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    for r in rows:
        r["compile_s"] = float(r["compile_s"])
        r["within_1s"] = r["within_1s"] == "True"
        r["twoq"], r["swapfree_twoq"], r["spare"] = int(r["twoq"]), int(r["swapfree_twoq"]), int(r["spare"])
        r["max_block_distance"] = float(r["max_block_distance"]) if r["max_block_distance"] not in ("", "None") else None

    def cell(arm, spare):
        return [r for r in rows if r["arm"] == arm and r["spare"] == spare]

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")

    spares = sorted({r["spare"] for r in rows})
    hi = spares[-1]
    laps = max(int(r["lap"]) for r in rows)
    print("=" * 78)
    print("SCORING (thresholds exactly as pre-registered)")
    print("=" * 78)
    vers = {arm: sorted({(r["core_version"], r["layout_version"]) for r in rows if r["arm"] == arm})
            for arm in ("P", "PN")}
    full = all(len(cell(a, s)) == laps for a in ("Q3", "P", "PN") for s in spares)
    ok0 = full and all(vers[a] == [EXPECTED[a]] for a in ("P", "PN"))
    print(f"C0 every arm ran {laps} laps at spares {spares}: {full}; versions {vers} -> {'passed' if ok0 else 'FAILED'}")
    for s in spares:
        for a in ("Q3", "P", "PN"):
            c = cell(a, s)
            ds = [r["max_block_distance"] for r in c if r["max_block_distance"] is not None]
            print(f"   spare {s:2d} {a:2s}: median {st.median(r['compile_s'] for r in c):.3f} s, max "
                  f"{max(r['compile_s'] for r in c):.3f} s, within 1 s {sum(r['within_1s'] for r in c)}/{len(c)}, "
                  f"swap laps {sum(r['status'] != 'OK' for r in c)}, 2q {sorted({r['twoq'] for r in c})} "
                  f"(swap-free {c[0]['swapfree_twoq']}), block distance first/last "
                  f"{(f'{ds[0]:.2e} / {ds[-1]:.2e}' if ds else '-')}, layout changes "
                  f"{sum(r['layout_changed'] == 'True' for r in c)}")
    n = laps
    pn0 = cell("PN", 0)
    w1 = sum(r["within_1s"] for r in pn0)
    print(f"D1 spare 0: PN within 1 s in {w1}/{n} laps (confirmed {n}/{n}, refuted <= {n // 2}) -> "
          f"{v(w1 == n, w1 <= n // 2)}")
    q0 = cell("Q3", 0)
    wq = sum(r["within_1s"] for r in q0)
    print(f"D2 spare 0: Qiskit L3 within 1 s in {wq}/{n} laps (confirmed 0, refuted >= {n // 2}) -> "
          f"{v(wq == 0, wq >= n // 2)}")
    sf = sum(1 for r in pn0 if r["status"] == "OK" and r["twoq"] == r["swapfree_twoq"])
    print(f"D3 spare 0: PN swap-free and mapped back in {sf}/{n} laps (confirmed {n}/{n}, refuted <= {n // 2}) -> "
          f"{v(sf == n, sf <= n // 2)}")
    dpn = [r["max_block_distance"] for r in rows if r["arm"] == "PN" and r["max_block_distance"] is not None]
    dmax = max(dpn) if dpn else None
    print(f"D4 PN meaning kept over {n} compounded laps (both spares): worst block distance to lap 0 {dmax} "
          f"(confirmed <= 1e-12 at every checked lap, refuted > 1e-10 at any) -> "
          f"{v(bool(dpn) and dmax <= 1e-12, bool(dpn) and dmax > 1e-10)}")
    p0 = cell("P", 0)
    sp = sum(1 for r in p0 if r["status"] != "OK")
    print(f"D5 spare 0: the release P's output has SWAPs in {sp}/{n} laps (confirmed {n}/{n}, refuted 0) -> "
          f"{v(sp == n, sp == 0)}")
    qh = sum(r["within_1s"] for r in cell("Q3", hi))
    print(f"D6 spare {hi}: Qiskit L3 within 1 s in {qh}/{n} laps (confirmed >= {n - 2}, refuted <= {n // 2}) -> "
          f"{v(qh >= n - 2, qh <= n // 2)}")
    ph = cell("PN", hi)
    okh = sum(1 for r in ph if r["within_1s"] and r["status"] == "OK" and r["twoq"] == r["swapfree_twoq"])
    print(f"D7 spare {hi}: PN within 1 s, swap-free and mapped back in {okh}/{n} laps (confirmed {n}/{n}, "
          f"refuted <= {n // 2}) -> {v(okh == n, okh <= n // 2)}")
    for a in ("Q3", "P"):
        d = [r["max_block_distance"] for r in rows if r["arm"] == a and r["max_block_distance"] is not None]
        print(f"Reported without prediction: {a} worst block distance over checked laps "
              f"{max(d) if d else None} ({len(d)} checked laps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "score"))
    ap.add_argument("--arm", choices=("Q3", "P", "PN"))
    ap.add_argument("--layout-dir", default=None)
    ap.add_argument("--fake", default="FakeKingston")
    ap.add_argument("--laps", type=int, default=LAPS)
    ap.add_argument("--spares", default=",".join(str(s) for s in SPARES))
    ap.add_argument("--tag", default="2026-09-29")
    args = ap.parse_args()
    if args.mode == "run" and not args.arm:
        ap.error("run needs --arm")
    run(args) if args.mode == "run" else score(args)


if __name__ == "__main__":
    main()
