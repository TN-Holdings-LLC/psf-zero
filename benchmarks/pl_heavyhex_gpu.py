"""pl_heavyhex_gpu.py -- pre-registered (workplace design, run on a RunPod RTX 4090,
2026-09-29): the PennyLane compounding loop on a fully occupied heavy-hex device, with
every lap's compiled circuit checked as a WHOLE circuit on lightning.gpu.

Why a 27-qubit device: a statevector of FakeKingston's 156 qubits cannot be simulated;
FakeAuckland (27 qubits, heavy-hex) fits on the GPU (2 GiB in complex128) and shows the
cliff for the pairs + triples family (workplace dry run of 2026-09-29).

Circuit family T (as in full_heavyhex_cliff.py; helpers copied here so the script is
self-contained): k3 = N - 2M triples (3-wire paths) and k2 = M - k3 pairs; spare s
removes s/2 pairs. Pair block: 20 Haar-random QubitUnitary on (a, b); triple block: 10
on (a, b) then 10 on (b, c).

Each lap: tape -> Qiskit -> compile (timed) -> (1) WHOLE-CIRCUIT check on lightning.gpu:
the routed physical circuit is simulated from |0...0>, and <Z>, <X> of every logical
qubit and <ZZ> of every block edge (read at each logical qubit's FINAL physical
position) are compared with the lap-0 logical tape's values; this applies to every lap,
including laps with SWAPs; (2) if the output is swap-free, it is mapped back to logical
qubits, checked block by block with qml.matrix against lap 0, and becomes the next lap's
tape; otherwise the next lap reuses the same input tape.

Arms (one per process): Q3 (transpile, optimization_level=3, target), P (release core
2026-09-28.1 + psf_smart_layout 2026-09-26.m1), PN (candidate core 2026-09-29.1 +
psf_smart_layout 2026-09-29.c1 via --layout-dir). The core is whatever psf_zero_core is
first on the path.

    PYTHONPATH=<core> python -u pl_heavyhex_gpu.py run --arm PN --layout-dir <dir> --repo <repo>
    python -u pl_heavyhex_gpu.py score
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

import numpy as np

SPARES = (0, 4)
LAPS = 30
DEADLINE = 1.0
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
EXPECTED = {"P": ("2026-09-28.1", "2026-09-26.m1"), "PN": ("2026-09-29.1", "2026-09-29.c1")}


class SwapLap(RuntimeError):
    pass


def norm_sha(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ---------------------------------------------------------------- device and family

def load_target(args):
    import warnings
    if args.generic_heavy_hex:
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.transpiler import CouplingMap
        cm = CouplingMap.from_heavy_hex(args.generic_heavy_hex)
        b = GenericBackendV2(cm.size(), coupling_map=cm, basis_gates=["cx", "rz", "sx", "x", "id"], seed=0)
        return b.target, f"GenericBackendV2 heavy-hex d={args.generic_heavy_hex}"
    from qiskit_ibm_runtime import fake_provider
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(fake_provider, args.fake)().target, args.fake


def graph_facts(target):
    import networkx as nx
    cmap = target.build_coupling_map()
    nq = cmap.size()
    g = nx.Graph()
    g.add_nodes_from(range(nq))
    g.add_edges_from(sorted({tuple(sorted(e)) for e in cmap.get_edges()}))
    m = sorted(tuple(sorted(e)) for e in nx.max_weight_matching(g, maxcardinality=True))
    mate, pid = {}, {}
    for k, (u, v) in enumerate(m):
        mate[u], mate[v] = v, u
        pid[u] = pid[v] = k
    unmatched = [x for x in range(nq) if x not in mate]
    bg = nx.Graph()
    tops = [("u", x) for x in unmatched]
    bg.add_nodes_from(tops)
    for x in unmatched:
        for y in g.neighbors(x):
            bg.add_edge(("u", x), ("p", pid[y]))
    mm = nx.bipartite.maximum_matching(bg, top_nodes=tops) if tops else {}
    return nq, len(m), all(t in mm for t in tops)


def layout_blocks(spare, nq, m):
    k3 = nq - 2 * m
    sizes = [3] * k3 + [2] * (m - k3 - spare // 2)
    blocks, q = [], 0
    for s in sizes:
        blocks.append(tuple(range(q, q + s)))
        q += s
    return blocks


def swapfree_twoq(blocks):
    return sum({2: 3, 3: 6}[len(b)] for b in blocks)


def initial_tape(blocks, seed):
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


# ---------------------------------------------------------------- conversions and checks

def wrap_to_tape(qc):
    """Qiskit circuit -> PennyLane tape; two-qubit gates are wrapped as unitaries first
    (the only multi-qubit form qiskit_to_tape trusts)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    from psf_pennylane_gpu_prototype import qiskit_to_tape
    w = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name in ("barrier", "delay", "measure"):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        w.append(UnitaryGate(Operator(inst.operation).data) if len(qs) == 2 else inst.operation, qs)
    return qiskit_to_tape(w, list(range(qc.num_qubits)))


def observables(blocks, pos):
    """<Z>, <X> of every logical qubit and <ZZ> of every block edge, at positions pos[q]."""
    import pennylane as qml
    obs = []
    for b in blocks:
        for q in b:
            obs += [qml.expval(qml.PauliZ(pos[q])), qml.expval(qml.PauliX(pos[q]))]
        for a, c in zip(b[:-1], b[1:]):
            obs.append(qml.expval(qml.PauliZ(pos[a]) @ qml.PauliZ(pos[c])))
    return obs


def run_expvals(dev, ops, meas):
    import pennylane as qml
    tape = qml.tape.QuantumTape(ops, meas, shots=None)
    return np.asarray(qml.execute([tape], dev)[0], dtype=float)


def block_matrices(tape, blocks):
    import pennylane as qml
    owner = {q: k for k, b in enumerate(blocks) for q in b}
    per = {k: [] for k in range(len(blocks))}
    for op in tape.operations:
        ks = {owner[int(w)] for w in op.wires}
        if len(ks) != 1:
            raise SwapLap("operation spans blocks")
        per[ks.pop()].append(op)
    return {k: qml.matrix(qml.tape.QuantumTape(per[k], [], shots=None), wire_order=list(b))
            for k, b in enumerate(blocks)}


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
            raise SwapLap("two-qubit operation outside the layout")
        lq = [to_logical[p] for p in phys]
        if len({owner[q] for q in lq}) != 1:
            raise SwapLap("two-qubit operation between blocks")
        out.append(inst.operation, lq)
    return out


# ---------------------------------------------------------------- run

def run(args):
    if args.layout_dir:
        sys.path.insert(0, args.layout_dir)
    for p in (args.repo, os.path.join(args.repo, "benchmarks")):
        if p not in sys.path:
            sys.path.insert(1 if args.layout_dir else 0, p)
    import pennylane as qml
    import qiskit
    from qiskit import transpile
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    core_v = layout_v = None
    if args.arm != "Q3":
        import psf_compile as pc
        import psf_smart_layout as sl
        core_v, layout_v = getattr(pc, "CORE_VERSION", None), sl.LAYOUT_VERSION
        print("LOADED psf_compile", pc.VERSION, norm_sha(pc.__file__), "| CORE_VERSION", core_v)
        print("LAYOUT", layout_v, norm_sha(sl.__file__))
    print(f"arm {args.arm} | device {args.device} | platform {platform.platform()} | cores {os.cpu_count()} | "
          f"python {platform.python_version()} | qiskit {qiskit.__version__} | pennylane {qml.__version__}")
    print("SCRIPT", os.path.basename(__file__), norm_sha(os.path.abspath(__file__)))
    target, label = load_target(args)
    nq, m, factor = graph_facts(target)
    print(f"Stage 0: {label}, {nq} qubits, max matching {m}, {{P2,P3}}-factor {'found' if factor else 'NOT FOUND'}")
    if not factor:
        print("G0 FAILED: nothing is run.")
        return
    cmap = target.build_coupling_map()
    native = [g for g in target.operation_names if g in NATIVE]
    spares = tuple(int(x) for x in args.spares.split(","))
    dev = qml.device(args.device, wires=nq)
    print("DEVICE", dev.name, dev.short_name if hasattr(dev, "short_name") else "")
    rows = []
    for spare in spares:
        blocks = layout_blocks(spare, nq, m)
        n = sum(len(b) for b in blocks)
        tape = initial_tape(blocks, seed=1000 * spare)
        u0 = block_matrices(tape, blocks)
        ident = {q: q for q in range(n)}
        t0 = time.perf_counter()
        ref = run_expvals(dev, tape.operations, observables(blocks, ident))
        ref_s = time.perf_counter() - t0
        # C0: a small sub-circuit (the first three blocks) on the GPU device and on the CPU
        sub = [op for op in tape.operations if max(int(w) for w in op.wires) < sum(len(b) for b in blocks[:3])]
        sub_blocks = blocks[:3]
        nsub = sum(len(b) for b in sub_blocks)
        cpu = qml.device("lightning.qubit", wires=nsub)
        small = qml.device(args.device, wires=nsub)
        d_c0 = float(np.max(np.abs(run_expvals(small, sub, observables(sub_blocks, ident))
                                   - run_expvals(cpu, sub, observables(sub_blocks, ident)))))
        # C1: sensitivity control -- an extra RX(1e-6) after the first operation of block 0
        ops_k = list(tape.operations)
        ops_k.insert(1, qml.RX(1e-6, wires=blocks[0][0]))
        d_c1 = float(np.max(np.abs(run_expvals(dev, ops_k, observables(blocks, ident)) - ref)))
        print(f"spare={spare} n={n} reference {len(ref)} values in {ref_s:.2f}s | C0 {nsub}-wire GPU vs CPU {d_c0:.2e} "
              f"| C1 RX(1e-6) control changes the values by {d_c1:.2e}", flush=True)
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
            final = list(routed.layout.final_index_layout(filter_ancillas=True))
            t0 = time.perf_counter()
            vals = run_expvals(dev, wrap_to_tape(routed).operations, observables(blocks, dict(enumerate(final))))
            gpu_s = time.perf_counter() - t0
            whole = float(np.max(np.abs(vals - ref)))
            try:
                logical = back_to_logical(routed, n, blocks)
                new_tape = wrap_to_tape(logical)
                mats = block_matrices(new_tape, blocks)
                bdist = max(aligned(u0[k], mats[k]) for k in u0)
                status, tape = "OK", new_tape
                layout = tuple(final)
            except SwapLap as e:
                status, bdist, layout = f"swap: {e}", None, None
            lap_s = time.perf_counter() - t_lap
            row = dict(arm=args.arm, spare=spare, logical_qubits=n, lap=lap, status=status, compile_s=compile_s,
                       gpu_check_s=gpu_s, lap_s=lap_s, within_1s=compile_s <= DEADLINE, twoq=twoq,
                       swapfree_twoq=swapfree_twoq(blocks), whole_circuit_max_diff=whole, block_distance=bdist,
                       layout_changed=(prev_layout is not None and layout is not None and layout != prev_layout),
                       core_version=core_v, layout_version=layout_v, device=dev.name, c0_gpu_vs_cpu=d_c0,
                       c1_control=d_c1)
            rows.append(row)
            if layout is not None:
                prev_layout = layout
            b_txt = "-" if bdist is None else f"{bdist:.3e}"
            print(f"{args.arm:2s} spare={spare} lap {lap:3d} compile {compile_s:7.3f}s gpu {gpu_s:6.2f}s "
                  f"2q={twoq} (swap-free {row['swapfree_twoq']}) whole={whole:.3e} block={b_txt} {status}", flush=True)
    out = f"pl_heavyhex_gpu_{args.arm}_{args.tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)\nDONE")


# ---------------------------------------------------------------- score

def score(args):
    rows = []
    for arm in ("Q3", "P", "PN"):
        with open(f"pl_heavyhex_gpu_{arm}_{args.tag}.csv", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    for r in rows:
        for k in ("compile_s", "gpu_check_s", "whole_circuit_max_diff", "c0_gpu_vs_cpu", "c1_control"):
            r[k] = float(r[k])
        r["block_distance"] = float(r["block_distance"]) if r["block_distance"] not in ("", "None") else None
        r["within_1s"] = r["within_1s"] == "True"
        r["spare"], r["twoq"], r["swapfree_twoq"] = int(r["spare"]), int(r["twoq"]), int(r["swapfree_twoq"])

    def cell(a, s):
        return [r for r in rows if r["arm"] == a and r["spare"] == s]

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")

    spares = sorted({r["spare"] for r in rows})
    hi, n = spares[-1], max(int(r["lap"]) for r in rows)
    print("=" * 78)
    print("SCORING (thresholds exactly as pre-registered)")
    print("=" * 78)
    vers = {a: sorted({(r["core_version"], r["layout_version"]) for r in rows if r["arm"] == a}) for a in ("P", "PN")}
    devs = sorted({r["device"] for r in rows})
    full = all(len(cell(a, s)) == n for a in ("Q3", "P", "PN") for s in spares)
    c0max = max(r["c0_gpu_vs_cpu"] for r in rows)
    ok0 = full and vers["P"] == [EXPECTED["P"]] and vers["PN"] == [EXPECTED["PN"]] and devs == ["lightning.gpu"] \
        and c0max <= 1e-12
    print(f"C0 laps complete {full}; versions {vers}; device {devs}; small sub-circuit GPU vs CPU max {c0max:.2e} "
          f"(<= 1e-12) -> {'passed' if ok0 else 'FAILED'}")
    c1min = min(r["c1_control"] for r in rows)
    print(f"C1 the RX(1e-6) control is detected: smallest change {c1min:.2e} (>= 1e-9) -> "
          f"{'passed' if c1min >= 1e-9 else 'FAILED'}")
    for s in spares:
        for a in ("Q3", "P", "PN"):
            c = cell(a, s)
            bd = [r["block_distance"] for r in c if r["block_distance"] is not None]
            print(f"   spare {s} {a:2s}: compile median {st.median(r['compile_s'] for r in c):.3f} s "
                  f"(max {max(r['compile_s'] for r in c):.3f}), within 1 s {sum(r['within_1s'] for r in c)}/{len(c)}, "
                  f"swap laps {sum(r['status'] != 'OK' for r in c)}, 2q {sorted({r['twoq'] for r in c})} "
                  f"(swap-free {c[0]['swapfree_twoq']}), whole-circuit max {max(r['whole_circuit_max_diff'] for r in c):.2e} "
                  f"(lap 1 {c[0]['whole_circuit_max_diff']:.2e}, last {c[-1]['whole_circuit_max_diff']:.2e}), "
                  f"block last {(f'{bd[-1]:.2e}' if bd else '-')}, GPU check median {st.median(r['gpu_check_s'] for r in c):.2f} s")
    w = sum(r["within_1s"] for r in cell("PN", 0))
    print(f"E1 spare 0: PN within 1 s in {w}/{n} (confirmed {n}/{n}, refuted <= {n // 2}) -> {v(w == n, w <= n // 2)}")
    w = sum(r["within_1s"] for r in cell("Q3", 0))
    print(f"E2 spare 0: Qiskit L3 within 1 s in {w}/{n} (confirmed 0, refuted >= {n // 2}) -> {v(w == 0, w >= n // 2)}")
    sf = sum(1 for r in cell("PN", 0) if r["status"] == "OK" and r["twoq"] == r["swapfree_twoq"])
    print(f"E3 spare 0: PN swap-free and mapped back in {sf}/{n} (confirmed {n}/{n}, refuted <= {n // 2}) -> "
          f"{v(sf == n, sf <= n // 2)}")
    wpn = max(r["whole_circuit_max_diff"] for r in rows if r["arm"] == "PN")
    print(f"E4 PN whole-circuit check on the GPU, both spares, every lap: max {wpn:.2e} (confirmed <= 1e-12, "
          f"refuted > 1e-10) -> {v(wpn <= 1e-12, wpn > 1e-10)}")
    sw = [r for r in rows if r["arm"] in ("Q3", "P") and r["spare"] == 0]
    wsw = max(r["whole_circuit_max_diff"] for r in sw)
    print(f"E5 spare 0: the outputs of Q3 and P (with SWAPs in the dry run) are correct as whole circuits: max {wsw:.2e} over {len(sw)} laps "
          f"(confirmed <= 1e-10 in every lap, refuted > 1e-6 in any) -> {v(wsw <= 1e-10, wsw > 1e-6)}")
    w = sum(r["within_1s"] for r in cell("Q3", hi))
    print(f"E6 spare {hi}: Qiskit L3 within 1 s in {w}/{n} (confirmed >= {n - 2}, refuted <= {n // 2}) -> "
          f"{v(w >= n - 2, w <= n // 2)}")
    ok = sum(1 for r in cell("PN", hi) if r["within_1s"] and r["status"] == "OK" and r["twoq"] == r["swapfree_twoq"])
    print(f"E7 spare {hi}: PN within 1 s, swap-free and mapped back in {ok}/{n} (confirmed {n}/{n}, refuted <= {n // 2})"
          f" -> {v(ok == n, ok <= n // 2)}")
    for a in ("Q3", "P", "PN"):
        wa = max(r["whole_circuit_max_diff"] for r in rows if r["arm"] == a)
        bd = [r["block_distance"] for r in rows if r["arm"] == a and r["block_distance"] is not None]
        print(f"Reported without prediction: {a} whole-circuit max {wa:.2e}; block distance max "
              f"{(f'{max(bd):.2e}' if bd else '-')} over {len(bd)} mapped laps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "score"))
    ap.add_argument("--arm", choices=("Q3", "P", "PN"))
    ap.add_argument("--layout-dir", default=None)
    ap.add_argument("--repo", default=os.path.expanduser("~/psf-zero"))
    ap.add_argument("--device", default="lightning.gpu")
    ap.add_argument("--fake", default="FakeAuckland")
    ap.add_argument("--generic-heavy-hex", type=int, default=0, help="dry run only: GenericBackendV2 heavy-hex of this distance")
    ap.add_argument("--laps", type=int, default=LAPS)
    ap.add_argument("--spares", default=",".join(str(s) for s in SPARES))
    ap.add_argument("--tag", default="2026-09-29")
    args = ap.parse_args()
    if args.mode == "run" and not args.arm:
        ap.error("run needs --arm")
    run(args) if args.mode == "run" else score(args)


if __name__ == "__main__":
    main()
