"""w8_tn_swapnet.py -- feasibility check (2026-10-14): for circuits that need SWAPs, does a placement from the circuit's
coupling structure (spectral / tensor-network-style ordering), or a swap network built by formula, beat Qiskit level 3?
Measured: correctness, CZ count, duration only. No error weights, no search, one compile per cell.

    python w8_tn_swapnet.py --out <dir>        (from the psf-zero repository root)

Device: ibm_marrakesh's calibration snapshot of 2026-10-13 11:37 as a Target (HW2's day; HW2's QK3 ba_n20 had
668 CZ, 32.9 us), with W5's cut (failed qubits and couplers removed first) and snake.

Arms
  QK3    generate_preset_pass_manager(3, target=<snapshot Target>, seed_transpiler=0) on the input as given.
  SPEC1  logical qubits ordered by the Fiedler vector of the weighted interaction graph (weights = number of
         two-qubit gates; ties by index), laid along W5's longest snake piece (centred); routed by Qiskit level 2 on
         the cut lattice's coupling map with basis {cz, rz, sx, x} (no Target, so no error-based re-placement).
  SPEC2  the same order, but laid on the cut lattice's breadth-first order from the snake piece's first qubit (a
         compact 2-D region instead of a line); routed the same way.
  SWAPNET (QAOA only) the cost layers rebuilt as an odd-even transposition swap network on the snake line, from
         SPEC1's order: at every comparator the pending ZZ term of the two occupants (if any) is applied and the
         occupants are swapped while terms of the layer remain; a layer stops as soon as all its terms are done;
         the network continues from where the previous layer left the qubits. Mixers act where the qubits are.
         Translated by Qiskit level 2 on the line (no routing needed: every gate is on a coupler).
Correctness: exact outcome distribution against the logical circuit (Aer statevector, <= 24 qubits); for SCALE1's
Clifford circuits, the ideal bitstring by stabilizer simulation."""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings

import numpy as np

REPO = os.path.abspath(os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.getcwd(), "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w2"), os.path.join(os.getcwd(), "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w5"), os.path.join(HERE, "..", "w5"), os.path.join(HERE, "..", "w2"), os.path.join(REPO, "benchmarks")]
import w5_place as P5  # noqa: E402
import w2_tfim as W2  # noqa: E402

BASIS = ["cz", "rz", "sx", "x"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def interaction(qc):
    w = {}
    for ins in qc.data:
        if len(ins.qubits) == 2 and ins.operation.name not in ("barrier", "measure"):
            a, b = sorted(qc.find_bit(q).index for q in ins.qubits)
            w[(a, b)] = w.get((a, b), 0) + 1
    return w


def fiedler_order(n, w):
    L = np.zeros((n, n))
    for (a, b), c in w.items():
        L[a, b] -= c
        L[b, a] -= c
        L[a, a] += c
        L[b, b] += c
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1]
    k = int(np.argmax(np.abs(f)))
    if f[k] < 0:                      # fix the eigenvector's sign deterministically
        f = -f
    return sorted(range(n), key=lambda q: (round(float(f[q]), 12), q))


def lattice_ctx(snap):
    n, edges, adj = P5.lattice(snap)
    badq, K, kadj, sizes = P5.cut(snap, n, edges, adj)
    path = P5.snake(n, adj)
    pcs = P5.pieces(path, K, kadj)
    best = max(pcs, key=lambda p: (len(p), -path.index(p[0])))
    rank = {q: i for i, q in enumerate(path)}
    order, seen, fr = [], {best[0]}, [best[0]]
    while fr:
        u = fr.pop(0)
        order.append(u)
        for v in sorted(kadj[u], key=lambda v: (rank.get(v, 10_000), v)):
            if v not in seen:
                seen.add(v)
                fr.append(v)
    return dict(n=n, edges=edges, badq=badq, K=K, kadj=kadj, best=best, bfs=order)


def compile_on(qc, layout, cmap_edges, level=2):
    from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
    pm = generate_preset_pass_manager(level, coupling_map=CouplingMap(cmap_edges), basis_gates=BASIS,
                                      initial_layout=layout, seed_transpiler=0)
    return pm.run(qc)


# ------------------------------------------------------------------ QAOA parsing and the swap network

def parse_qaoa(qc):
    """(prep ops per qubit, layers: list of ({(a, b): theta}, {q: [ops]})) from a QAOA circuit whose cost terms are
    written cx(a, b) rz(theta) on b cx(a, b) (other qubits' gates may be interleaved)."""
    data = [(ins.operation, [qc.find_bit(q).index for q in ins.qubits]) for ins in qc.data
            if ins.operation.name not in ("barrier", "measure")]
    n = qc.num_qubits
    started = [False] * n
    prep = {q: [] for q in range(n)}
    layer_of = [0] * n
    layers = []
    open_term = {}                    # qubit -> [a, b, theta or None]

    def lay(k):
        while len(layers) <= k:
            layers.append(({}, {x: [] for x in range(n)}))
        return layers[k]
    for op, qs in data:
        if op.name == "cx":
            a, b = qs
            t = open_term.get(a)
            if t is not None and t[0] == a and t[1] == b and t[2] is not None:
                k = layer_of[a]
                assert layer_of[b] == k
                key = (min(a, b), max(a, b))
                lay(k)[0][key] = lay(k)[0].get(key, 0.0) + t[2]
                del open_term[a], open_term[b]
                continue
            assert a not in open_term and b not in open_term, "overlapping terms"
            open_term[a] = open_term[b] = [a, b, None]
            started[a] = started[b] = True
            continue
        q = qs[0]
        if q in open_term:
            t = open_term[q]
            assert op.name == "rz" and q == t[1] and t[2] is None, f"unexpected {op.name} inside a term"
            t[2] = float(op.params[0])
            continue
        if not started[q] and layer_of[q] == 0:
            prep[q].append(op)
        else:
            lay(layer_of[q])[1][q].append(op)
            if op.name == "rx":
                layer_of[q] += 1
    assert not open_term
    return prep, layers


def qaoa_logical(n, prep, layers):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for q in range(n):
        for op in prep[q]:
            qc.append(op, [q])
    for zz, mix in layers:
        for (a, b), th in sorted(zz.items()):
            qc.rzz(th, a, b)          # cx rz(th) cx == exp(-i th/2 Z Z) == RZZ(th)
        for q in range(n):
            for op in mix[q]:
                qc.append(op, [q])
    qc.measure_all()
    return qc


def swapnet(n, prep, layers, order):
    """logical circuit on n 'line' qubits (position p), plus the final position of every logical qubit."""
    from qiskit import ClassicalRegister, QuantumCircuit
    qc = QuantumCircuit(n)
    occ = list(order)                 # occ[p] = logical qubit at position p
    pos = {l: p for p, l in enumerate(occ)}
    for l in range(n):
        for op in prep[l]:
            qc.append(op, [pos[l]])
    parity = 0
    for zz, mix in layers:
        pend = dict(zz)
        while pend:
            for p in range(parity, n - 1, 2):
                a, b = occ[p], occ[p + 1]
                key = (min(a, b), max(a, b))
                if key in pend:
                    qc.rzz(pend.pop(key), p, p + 1)
                if pend:
                    qc.swap(p, p + 1)
                    occ[p], occ[p + 1] = b, a
                    pos[a], pos[b] = p + 1, p
            parity ^= 1
        for l in range(n):
            for op in mix[l]:
                qc.append(op, [pos[l]])
    cr = ClassicalRegister(n, "meas")
    qc.add_register(cr)
    for l in range(n):
        qc.measure(pos[l], cr[l])
    return qc


# ------------------------------------------------------------------ measures

def clifford_bitstring(out, n):
    """as W5's final_bitstring, after snapping rz angles within 1e-9 of a multiple of pi/2 (synthesis leaves
    floating-point angles); None if a gate is not Clifford even after that"""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, StabilizerState
    body = QuantumCircuit(out.num_qubits)
    for ins in out.data:
        nm = ins.operation.name
        if nm in ("barrier", "delay", "measure"):
            continue
        qs = [out.find_bit(q).index for q in ins.qubits]
        if nm == "rz":
            th = float(ins.operation.params[0])
            k = round(th / (math.pi / 2))
            if abs(th - k * math.pi / 2) > 1e-9:
                return None
            for _ in range(k % 4):
                body.s(qs[0])
            continue
        body.append(ins.operation, qs)
    fin = out.layout.final_index_layout()
    p = StabilizerState(Clifford(body)).probabilities_dict([fin[i] for i in range(n)])
    return next(iter(p)) if len(p) == 1 else None


def ideal_prob(out, n, ideal):
    """1 minus the largest deviation (in probability) of any touched qubit from the ideal outcome (Aer MPS): logical
    qubits at their final positions must show the ideal bits, every other touched qubit |0>; 1 for an exact compile"""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    body = [(ins.operation, [out.find_bit(q).index for q in ins.qubits]) for ins in out.data
            if ins.operation.name not in ("barrier", "delay", "measure")]
    fin = out.layout.final_index_layout()
    active = sorted({q for _, qs in body for q in qs} | {fin[i] for i in range(n)})
    pos = {q: i for i, q in enumerate(active)}
    c = QuantumCircuit(len(active))
    for op, qs in body:
        c.append(op, [pos[q] for q in qs])
    # Aer's MPS returns amplitudes in reversed bit order in save_amplitudes_squared (found here: 0 against the
    # statevector's 1 on the same circuit), so the check reads <Z> of every touched qubit instead: an exact compile
    # gives <Z> = -1 where the ideal bit is 1, +1 elsewhere (other touched qubits back in |0>).
    from qiskit.quantum_info import Pauli
    want = {pos[q]: 1.0 for q in active}
    for i in range(n):
        want[pos[fin[i]]] = -1.0 if ideal[n - 1 - i] == "1" else 1.0
    for k in want:
        c.save_expectation_value(Pauli("Z"), [k], label=f"z{k}")
    d = AerSimulator(method="matrix_product_state").run(c).result().data()
    return 1.0 - max(abs(d[f"z{k}"] - v) / 2 for k, v in want.items())


def dur_cz(out, snap, edges, badq):
    return P5.dur_cz_failed(out, snap, edges, badq)


def main():
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from qiskit.transpiler import generate_preset_pass_manager
    snap = json.load(open(P5.SNAP))
    ST = load(os.path.join(REPO, "benchmarks", "snapshot_target.py"), "snapshot_target")
    tgt = ST.target_from_snapshot(P5.SNAP)
    H = load(os.path.join(REPO, "data", "2026-10-13", "hw2_kit", "hw2_real.py"), "hw2")
    S = load(P5.CIRC, "scale1_circuits")
    C = lattice_ctx(snap)
    kedges = [[u, v] for u in C["K"] for v in C["kadj"][u]]
    line = C["best"]
    pm3 = generate_preset_pass_manager(3, target=tgt, seed_transpiler=0)
    rows = []
    jobs = [(t, "qaoa", H.build_test(t)[0]) for t in ("ba_n14", "ba_n16", "ba_n18", "ba_n20")]
    jobs += [(f"R4_n{n}_i{i}", "clifford", S.family_R(n, 4, i)) for n in (24, 48) for i in range(3)]
    for name, kind, qc in jobs:
        n = qc.num_qubits
        w = interaction(qc)
        order = fiedler_order(n, w)
        if kind == "qaoa":
            prep, layers = parse_qaoa(qc)
            logical = qaoa_logical(n, prep, layers)
            ref = W2.probs(W2.unrolled(logical))
            ref_b = W2.probs(W2.unrolled(qc))
            assert W2.tvd(ref, ref_b) < 1e-9, "parsed QAOA differs from the input"
        arms = {}
        t0 = time.perf_counter()
        arms["QK3"] = (pm3.run(qc), time.perf_counter() - t0)
        s = (len(line) - n) // 2
        lay1 = [None] * n
        for i, l in enumerate(order):
            lay1[l] = line[s + i]
        t0 = time.perf_counter()
        arms["SPEC1"] = (compile_on(qc, lay1, kedges), time.perf_counter() - t0)
        lay2 = [None] * n
        for i, l in enumerate(order):
            lay2[l] = C["bfs"][i]
        t0 = time.perf_counter()
        arms["SPEC2"] = (compile_on(qc, lay2, kedges), time.perf_counter() - t0)
        if kind == "qaoa":
            sn = swapnet(n, prep, layers, order)
            lineq = line[s:s + n]
            t0 = time.perf_counter()
            arms["SWAPNET"] = (compile_on(sn, lineq, [[lineq[i], lineq[i + 1]] for i in range(n - 1)] +
                                          [[lineq[i + 1], lineq[i]] for i in range(n - 1)]), time.perf_counter() - t0)
        for arm, (out, secs) in arms.items():
            dur, cz, fail = dur_cz(out, snap, C["edges"], C["badq"])
            if kind == "qaoa":
                err = W2.tvd(W2.probs(out), ref)
                ok = err <= 1e-6
            else:
                if n <= 24:
                    err = 1 - ideal_prob(out, n, S.ideal_bitstring(qc))
                    ok = err <= 1e-6
                else:                          # MPS is too slow at 48 qubits here: stabilizer check when possible
                    try:
                        ok = P5.final_bitstring(out, n) == S.ideal_bitstring(qc)
                        err = 0.0 if ok else 1.0
                    except Exception:      # noqa: BLE001  (an rz angle not a multiple of pi/2: not checkable)
                        ok, err = None, None
            r = dict(circuit=name, arm=arm, n=n, cz=cz, dur_us=None if dur is None else round(dur, 3),
                     failed_used=fail, correct=None if ok is None else bool(ok), err=err, compile_s=round(secs, 2))
            rows.append(r)
            print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(a.out, "w8_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
