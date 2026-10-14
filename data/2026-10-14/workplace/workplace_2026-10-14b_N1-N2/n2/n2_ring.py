"""n2_ring.py -- N2 (2026-10-14): a CZ ring of n qubits run on a line of heavy-hex by formula, for DEPTH's data
re-uploading classifier (W3's circuits), compared with Qiskit level 3, c48's constructive pipeline (PSF-C, W3) and
a15/a14 (searched routing as a14's fallback).

The formula (no search; one fixed construction per n):
  * Folded order on a line: positions hold 0, n-1, 1, n-2, 2, n-3, ... . Ring edges (0, n-1) and the middle edge are
    adjacent; every other ring edge spans distance 2.
  * Ring layer = rounds; each round: CZ on every ring edge whose two qubits are adjacent and not yet done (they all
    commute: CZs are diagonal), then one odd-even transposition round of SWAPs on the interior positions 1..n-2
    (comparators (1,2), (3,4), ... in odd rounds, (2,3), (4,5), ... in even rounds), only while edges remain.
  * Mirror: the next layer runs the previous layer's rounds backwards (undoes its SWAPs), so every layer costs the
    same and the qubits return to the folded order every second layer.
  For n = 4: CZ(0,3), CZ(1,2) | SWAP(1,2) | CZ(0,1), CZ(3,2): 4 CZ + 1 SWAP = 7 CZ, depth 1 + 3 + 1 CZ.
  For n = 6: 2 CZ | 2 SWAPs | 2 CZ | 1 SWAP | 2 CZ: 6 CZ + 3 SWAPs = 15 CZ, depth 1 + 3 + 1 + 3 + 1.
The line is placed by item 86's rule (W5's cut, snake, common gate length first)."""
import importlib.util
import json
import os
import sys
import time
import warnings

import numpy as np

REPO = os.path.abspath(os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.getcwd(), "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w2"), os.path.join(HERE, "..", "..", "w17", "w2"), os.path.join(REPO, "benchmarks")]
import w2_tfim as W2  # noqa: E402


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


def ring_rounds(n):
    """[(cz pairs [(logical a, logical b)], swaps [(position p, p+1)])] for one ring layer from the folded order."""
    order = []
    lo, hi = 0, n - 1
    while lo <= hi:
        order.append(lo)
        if hi != lo:
            order.append(hi)
        lo, hi = lo + 1, hi - 1
    edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    occ = list(order)
    rounds, parity = [], 1
    while True:
        czs = []
        for p in range(n - 1):
            e = (min(occ[p], occ[p + 1]), max(occ[p], occ[p + 1]))
            if e in edges:
                czs.append((occ[p], occ[p + 1]))
                edges.discard(e)
        swaps = []
        if edges:
            for p in range(1 if parity else 2, n - 2, 2):
                swaps.append((p, p + 1))
                occ[p], occ[p + 1] = occ[p + 1], occ[p]
            parity ^= 1
        rounds.append((czs, swaps))
        if not edges:
            break
        if len(rounds) > 4 * n:
            raise RuntimeError("no schedule")
    return order, rounds


def build(x, theta, n, L, line, N):
    """the classifier on physical qubits `line` (positions 0..n-1), all N device qubits, measurements at the end"""
    from qiskit import ClassicalRegister, QuantumCircuit
    order, rounds = ring_rounds(n)
    qc = QuantumCircuit(N)
    occ = list(order)
    pos = {l: p for p, l in enumerate(occ)}
    k = 0
    for layer in range(L):
        for q in range(n):
            qc.ry(np.pi / 2 * x[q], line[pos[q]])
        for q in range(n):
            qc.ry(theta[k], line[pos[q]])
            k += 1
            qc.rz(theta[k], line[pos[q]])
            k += 1
        def do_swaps(swaps):
            for p, p1 in swaps:
                qc.swap(line[p], line[p1])
                a, b = occ[p], occ[p1]
                occ[p], occ[p1] = b, a
                pos[a], pos[b] = p1, p

        def do_czs(czs):
            for a, b in czs:
                qc.cz(line[pos[a]], line[pos[b]])
        if layer % 2 == 0:                  # forward: each round's CZs, then its SWAPs
            for czs, swaps in rounds:
                do_czs(czs)
                do_swaps(swaps)
        else:                               # mirrored: rounds backwards, each round's SWAPs undone, then its CZs
            for czs, swaps in reversed(rounds):
                do_swaps(list(reversed(swaps)))
                do_czs(czs)
    for q in range(n):
        qc.ry(theta[k], line[pos[q]])
        k += 1
    cr = ClassicalRegister(n, "meas")
    qc.add_register(cr)
    for q in range(n):
        qc.measure(line[pos[q]], cr[q])
    return qc


def main():
    warnings.simplefilter("ignore")
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
    from qiskit_ibm_runtime import fake_provider
    D = load(os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1", "depth_eval.py"), "depth_eval")
    load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    P = load(os.path.join(REPO, "patches", "psf_compile_c49n1_2026-10-14", "psf_compile.py"), "c49n1")
    w3 = {(r["device"], r["n"], r["L"], r["arm"]): r for r in map(json.loads, open(sys.argv[2]))}
    rng = np.random.default_rng(20261014)            # W3's angles, drawn in W3's order
    angles = {}
    for n in (4, 6):
        for L in range(1, 17):
            angles[(n, L)] = (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L)))
    for n in (4, 6):
        o, r = ring_rounds(n)
        ncz = sum(len(c) for c, _ in r)
        nsw = sum(len(s) for _, s in r)
        print(f"n={n}: folded order {o}; per layer {ncz} CZ + {nsw} SWAP = {ncz + 3 * nsw} CZ; rounds {r}")
    fh = open(os.path.join(out_dir, "n2_rows.jsonl"), "w")
    for dev in ("FakeFez", "FakeTorino"):
        b = getattr(fake_provider, dev)()
        t = b.target
        N = t.num_qubits
        lat = P._chain_lattice(t, 0.5)
        K, kadj, fadj, snake = lat
        for n in (4, 6):
            line, how = P._chain_place(n, K, fadj, snake)
            if line is None:
                line, how = P._chain_place(n, K, kadj, snake)
            cm = CouplingMap()
            for q in range(N):
                cm.add_physical_qubit(q)
            for i in range(n - 1):
                cm.add_edge(line[i], line[i + 1])
                cm.add_edge(line[i + 1], line[i])
            pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                              initial_layout=list(range(N)), routing_method="none", seed_transpiler=0)
            for L in range(1, 17):
                x, th = angles[(n, L)]
                logical = D.circuit(x, th, n, L)
                logical.measure_all()
                t0 = time.perf_counter()
                out = pm.run(build(x, th, n, L, line, N))
                secs = time.perf_counter() - t0
                dur, cz, _ = W2.duration_and_cz(out, t)
                err = W2.tvd(W2.probs(out), W2.probs(logical))
                q3, pc = w3[(dev, n, L, "QK3")], w3[(dev, n, L, "PSF-C")]
                r = dict(device=dev, n=n, L=L, cz=cz, dur_us=round(dur, 4), err=err, correct=bool(err <= 1e-9),
                         isa=W2.isa_ok(out, t), line=line, how=how, secs=round(secs, 2), qk3_cz=q3["cz"],
                         qk3_dur=q3["dur_us"], psfc_cz=pc["cz"], psfc_dur=pc["dur_us"])
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(f"{dev} n={n} L={L:2d} N2 {cz:4d} {dur:6.2f}us | PSF-C {pc['cz']:4d} {pc['dur_us']:6.2f} | QK3 "
                      f"{q3['cz']:4d} {q3['dur_us']:6.2f} | err {err:.1e}", flush=True)


if __name__ == "__main__":
    main()
