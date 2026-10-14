"""w5_place.py -- W5 (2026-10-14): a placement decided by the lattice's structure alone (no weights, no search, no
random choice), prototyped offline on ibm_marrakesh's calibration snapshot of 2026-10-13 11:37 and SCALE1's logical
circuits.

Rule (structure only):
  1. CUT. Remove every qubit with a reported sx or readout error >= 0.5 and every coupler with a two-qubit error
     >= 0.5; then keep the largest connected component K of what is left (ties: the one with the smallest qubit).
  2. SNAKE. Heron r2's heavy-hex is read from the coupling map: rows are maximal runs of consecutive qubits joined by
     couplers (8 rows of 16), bridges are the other qubits (4 between each pair of rows). The snake is the fixed path
     row 0 left to right, down the bridge at the right end, row 1 right to left as far as the bridge at the left end,
     down, and so on (129 qubits on Heron r2; the first qubit of rows 1-6 is not on it).
  3. PLACE. A chain circuit (its two-qubit gates form a path) goes on the longest piece of the snake that survives in K,
     centred in it (start = floor((length - n) / 2)); if no piece is long enough, or for any other circuit, the
     qubits are taken in a fixed order: breadth-first through K from the first snake qubit of the longest piece,
     neighbours visited in snake order (qubits off the snake after those on it, by index). Logical qubit i goes to
     the i-th qubit of that order (for a chain, along the chain).
  4. ROUTE. Qiskit's level-1 pass manager on the coupling map of K only (so no failed element can be used),
     initial layout from step 3, seed_transpiler=0, once.

Measured: whether it compiles (no rejection), failed elements used, ISA on the snapshot Target, the same output on a
second call, the ideal bitstring reproduced (stabilizer simulation of the compiled circuit), CZ count, duration (as
soon as possible on the snapshot's durations). Reported beside SCALE1's PSF and QK3 numbers from its manifest.

    python w5_place.py --out <dir>      (from the psf-zero repository root)"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings

REPO = os.path.abspath(os.getcwd())
SNAP = os.path.join(REPO, "data", "2026-10-13", "hw2", "ns_snapshot_ibm_marrakesh.json")
MAN = os.path.join(REPO, "data", "2026-10-13", "scale1", "scale1_manifest.json")
CIRC = os.path.join(REPO, "data", "2026-10-13", "scale1_kit", "scale1_circuits.py")
FAILED = 0.5
CELLS = (("L", 12, 0), ("L", 24, 0), ("L", 48, 0),   # as scale1_kit/scale1_real.py
         ("R", 24, 2), ("R", 48, 2), ("R", 96, 2), ("R", 144, 2),
         ("R", 24, 4), ("R", 48, 4))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ the rule

def lattice(snap):
    edges = {}
    for a, b, e, _ in snap["couplers"]:
        k = (min(a, b), max(a, b))
        edges[k] = max(edges.get(k, 0.0), e if e is not None else 0.0)
    n = len(snap["qubits"])
    adj = {q: set() for q in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return n, edges, adj


def cut(snap, n, edges, adj):
    badq = {q for q in range(n) for v in [snap["qubits"][str(q)]]
            if (v.get("sx_error") or 0) >= FAILED or (v.get("readout_error") or 0) >= FAILED}
    good = {k for k, e in edges.items() if e < FAILED and k[0] not in badq and k[1] not in badq}
    gadj = {q: set() for q in range(n) if q not in badq}
    for a, b in good:
        gadj[a].add(b)
        gadj[b].add(a)
    seen, comps = set(), []
    for q in sorted(gadj):
        if q in seen:
            continue
        comp, fr = {q}, [q]
        while fr:
            u = fr.pop()
            for v in gadj[u]:
                if v not in comp:
                    comp.add(v)
                    fr.append(v)
        seen |= comp
        comps.append(comp)
    K = max(comps, key=lambda c: (len(c), -min(c)))
    kadj = {q: {v for v in gadj[q] if v in K} for q in K}
    return badq, K, kadj, sorted(len(c) for c in comps)


def snake(n, adj):
    rows, cur = [], [0]
    for q in range(1, n):
        if q in adj[q - 1] and q - 1 in adj[q]:
            cur.append(q)
        else:
            rows.append(cur)
            cur = [q]
    rows.append(cur)
    rows = [r for r in rows if len(r) > 2]
    inrow = {q: (i, j) for i, r in enumerate(rows) for j, q in enumerate(r)}
    path = []
    col = 0                                   # entry column of the current row
    for i, r in enumerate(rows):
        leftward = i % 2 == 1
        if i + 1 < len(rows):                 # the bridge used to go down: at the far end in the direction of travel
            br = [(inrow[a][1], b) for b in range(n) if b not in inrow for a in adj[b]
                  if a in inrow and inrow[a][0] == i and any(c in inrow and inrow[c][0] == i + 1 for c in adj[b])]
            exit_col, bq = (min(br) if leftward else max(br))
        else:
            exit_col, bq = (0 if leftward else len(r) - 1), None
        step = -1 if leftward else 1
        start = col if i else 0
        path += [r[j] for j in range(start, exit_col + step, step)]
        if bq is not None:
            path.append(bq)
            below = next(c for c in adj[bq] if c in inrow and inrow[c][0] == i + 1)
            col = inrow[below][1]
    for a, b in zip(path, path[1:]):
        assert b in adj[a], (a, b)
    return path


def pieces(path, K, kadj):
    """maximal runs of the snake whose qubits are in K and whose consecutive couplers survived the cut"""
    out, cur = [], []
    for q in path:
        if q in K and (not cur or q in kadj[cur[-1]]):
            cur.append(q)
        else:
            if cur:
                out.append(cur)
            cur = [q] if q in K else []
    if cur:
        out.append(cur)
    return out


def is_chain(qc):
    from collections import defaultdict
    deg, pairs = defaultdict(set), set()
    for ins in qc.data:
        if len(ins.qubits) == 2 and ins.operation.name not in ("barrier",):
            a, b = sorted(qc.find_bit(q).index for q in ins.qubits)
            pairs.add((a, b))
    for a, b in pairs:
        deg[a].add(b)
        deg[b].add(a)
    if any(len(v) > 2 for v in deg.values()):
        return None
    ends = [q for q, v in deg.items() if len(v) == 1]
    if len(ends) != 2 or len(deg) != qc.num_qubits:
        return None
    order, prev, cur = [ends[0]], None, ends[0]
    while len(order) < len(deg):
        nxt = [v for v in deg[cur] if v != prev][0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order                               # logical qubits along the chain


def place(qc, path, K, kadj):
    pcs = pieces(path, K, kadj)
    best = max(pcs, key=lambda p: (len(p), -path.index(p[0])))
    chain = is_chain(qc)
    n = qc.num_qubits
    if chain is not None and len(best) >= n:
        s = (len(best) - n) // 2
        phys = best[s:s + n]
        return {chain[i]: phys[i] for i in range(n)}, "snake interval"
    rank = {q: i for i, q in enumerate(path)}
    order, seen, fr = [], {best[0]}, [best[0]]
    while fr and len(order) < n:
        u = fr.pop(0)
        order.append(u)
        for v in sorted(kadj[u], key=lambda v: (rank.get(v, 10_000), v)):
            if v not in seen:
                seen.add(v)
                fr.append(v)
    if len(order) < n:
        return None, f"K has only {len(K)} qubits"
    logical = chain if chain is not None else list(range(n))
    return {logical[i]: order[i] for i in range(n)}, "breadth-first order"


def place_exact(qc, path, K, kadj):
    """Variant B: if the circuit's interaction graph embeds in K with every two-qubit gate on a coupler, take the
    FIRST embedding rustworkx's VF2 returns with id_order (nodes in index order; no score, no error weights); logical
    qubits without two-qubit gates then take the remaining qubits of K in the breadth-first order of `place`.
    Otherwise `place`."""
    import rustworkx as rx
    pairs = sorted({tuple(sorted(qc.find_bit(q).index for q in ins.qubits)) for ins in qc.data
                    if len(ins.qubits) == 2 and ins.operation.name != "barrier"})
    used = sorted({q for p in pairs for q in p})
    if not pairs:
        return place(qc, path, K, kadj)
    G = rx.PyGraph()
    kn = sorted(K)
    G.add_nodes_from(kn)
    ki = {q: i for i, q in enumerate(kn)}
    G.add_edges_from_no_data([(ki[a], ki[b]) for a in kn for b in kadj[a] if a < b])
    H = rx.PyGraph()
    H.add_nodes_from(used)
    ui = {q: i for i, q in enumerate(used)}
    H.add_edges_from_no_data([(ui[a], ui[b]) for a, b in pairs])
    it = rx.vf2_mapping(G, H, subgraph=True, induced=False, id_order=True, call_limit=2_000_000)
    m = next(it, None)
    if m is None:
        lay, how = place(qc, path, K, kadj)
        return lay, how + " (no exact embedding)"
    lay = {used[h]: kn[g] for g, h in m.items()}
    rest_l = [q for q in range(qc.num_qubits) if q not in lay]
    order, _ = place(qc.__class__(len(K)), path, K, kadj)    # breadth-first order of all of K
    rest_p = [order[i] for i in range(len(K)) if order[i] not in set(lay.values())]
    for lq, pq in zip(rest_l, rest_p):
        lay[lq] = pq
    return lay, "exact embedding (first VF2 match)"


def compile_rule(qc, lay, K, kadj, basis):
    from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
    cm = CouplingMap([[a, b] for a in K for b in kadj[a]])
    pm = generate_preset_pass_manager(1, coupling_map=cm, basis_gates=basis, initial_layout=[lay[i] for i in
                                      range(qc.num_qubits)], seed_transpiler=0)
    return pm.run(qc)


# ------------------------------------------------------------------ checks

def sig(out):
    h = hashlib.sha256()
    for ins in out.data:
        h.update(ins.operation.name.encode())
        h.update(repr(tuple(out.find_bit(q).index for q in ins.qubits)).encode())
        h.update(repr(tuple(round(float(p), 10) for p in ins.operation.params)).encode())
    return h.hexdigest()[:16]


def final_bitstring(out, n):
    """the deterministic outcome of the compiled (unmeasured, Clifford) circuit read at the final positions of
    logical qubits 0..n-1, logical qubit 0 rightmost; None if not deterministic"""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, StabilizerState
    body = QuantumCircuit(out.num_qubits)
    for ins in out.data:
        if ins.operation.name not in ("barrier", "delay", "measure"):
            body.append(ins.operation, [out.find_bit(q).index for q in ins.qubits])
    fin = out.layout.final_index_layout()
    p = StabilizerState(Clifford(body)).probabilities_dict([fin[i] for i in range(n)])
    return next(iter(p)) if len(p) == 1 else None


def dur_cz_failed(out, snap, edges, badq):
    q = snap["qubits"]
    two = {(min(a, b), max(a, b)): l for a, b, e, l in snap["couplers"]}
    pos = {x: i for i, x in enumerate(out.qubits)}
    clock, cz, fail = {}, 0, 0
    for ins in out.data:
        nm = ins.operation.name
        if nm in ("barrier", "delay"):
            continue
        qs = tuple(pos[x] for x in ins.qubits)
        if any(x in badq for x in qs):
            fail += 1
        if nm == "measure":
            continue
        if len(qs) == 2:
            cz += 1
            k = (min(qs), max(qs))
            if k not in edges:
                return None, cz, -1
            if edges[k] >= FAILED:
                fail += 1
            d = two[k]
        elif nm == "rz":
            d = 0.0
        else:
            d = q[str(qs[0])]["sx_length_s"]
        s = max(clock.get(x, 0.0) for x in qs)
        for x in qs:
            clock[x] = s + d
    return max(clock.values()) * 1e6, cz, fail


def main():
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    snap = json.load(open(SNAP))
    man = json.load(open(MAN))
    S = load(CIRC, "scale1_circuits")
    n, edges, adj = lattice(snap)
    badq, K, kadj, sizes = cut(snap, n, edges, adj)
    path = snake(n, adj)
    pcs = pieces(path, K, kadj)
    info = dict(snapshot=snap.get("calibration_time"), failed_qubits=sorted(badq),
                failed_couplers=sorted(k for k, e in edges.items() if e >= FAILED), component_sizes=sizes, K=len(K),
                snake_length=len(path), snake_pieces=[len(p) for p in pcs])
    print(json.dumps(info))
    basis = ["cz", "rz", "sx", "x"]
    ref = {(e["cell"], e["inst"], e["arm"]): e for e in man["entries"]}
    rows = []
    for fam, nq, d in CELLS:
        for inst in range(3):
            qc = S.family_L(nq, inst) if fam == "L" else S.family_R(nq, d, inst)
            cell = f"{fam}_n{nq}" if fam == "L" else f"R{d}_n{nq}"
            for variant, placer in (("A", place), ("B", place_exact)):
                t0 = time.perf_counter()
                lay, how = placer(qc, path, K, kadj)
                r = dict(cell=cell, inst=inst, n=nq, variant=variant, how=how)
                if lay is None:
                    r.update(compiled=False)
                else:
                    out = compile_rule(qc, lay, K, kadj, basis)
                    out2 = compile_rule(qc, placer(qc, path, K, kadj)[0], K, kadj, basis)
                    dur, cz, fail = dur_cz_failed(out, snap, edges, badq)
                    ideal = S.ideal_bitstring(qc)
                    got = final_bitstring(out, nq)
                    r.update(compiled=True, cz=cz, dur_us=round(dur, 3) if dur is not None else None, failed_used=fail,
                             deterministic=sig(out) == sig(out2), correct=got == ideal, secs=round(time.perf_counter() - t0, 2),
                             max_phys=max(lay.values()), on_snake=sum(1 for v in lay.values() if v in set(path)))
                for arm in ("PSF", "QK3", "QK3F"):
                    e = ref.get((cell, inst, arm))
                    if e:
                        r[arm] = dict(cz=e["twoq"], dur_us=round(e["dur_us"], 3), failed=e.get("failed_used"),
                                      submitted=e.get("submitted"), part=e.get("part"))
                rows.append(r)
                print(json.dumps(r), flush=True)
    json.dump(dict(info=info, rows=rows), open(os.path.join(a.out, "w5_rows.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
