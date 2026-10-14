"""n4_variantB.py -- N4 (2026-10-14): W5's variant B (cut failed elements; first exact embedding by VF2, no score;
otherwise breadth-first order along the snake; Qiskit level-1 routing on the cut lattice, coupling map and basis only)
against c49's default call, on C49-VAL's 18 tests x 5 devices (c49's numbers read from C49-VAL's rows) and SCALE1's
27 circuits on the marrakesh snapshot. Measured: CZ, duration (C49-VAL's own rule: measurements left out), refusal,
determinism (two calls), correctness (outcome distribution, <= 20 active qubits; SCALE1: the known bitstring).
    python n4_variantB.py <out> <patched c49+n1 psf_compile.py>        (from the repository root)"""
import contextlib, hashlib, io, json, math, os, sys, time, warnings, importlib.util
import numpy as np
warnings.simplefilter("ignore")
REPO = os.getcwd()
sys.path[:0] = ["benchmarks", ".", os.path.join("data", "2026-10-12", "c48_dev"),
                os.path.join(REPO, "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w2"),
                os.path.join(REPO, "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w5")]
import w2_tfim as W2
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
P = load(sys.argv[2], "c49n1_for_n4"); P.WARN_WITHOUT_TARGET = False   # only its lattice helpers are used
C49 = load(os.path.join("patches", "psf_compile_c49_2026-10-14", "psf_compile.py"), "c49_for_n4")
C49.WARN_WITHOUT_TARGET = False
H = load("benchmarks/hw1_real.py", "hw1"); H.KIT = os.path.join("data", "2026-10-12", "hw_kit")
H2 = load(os.path.join("data", "2026-10-13", "hw2_kit", "hw2_real.py"), "hw2"); H2.KIT = H.KIT
import gens
import snapshot_target as ST
S = load(os.path.join("data", "2026-10-13", "scale1_kit", "scale1_circuits.py"), "scale1_circuits")
import w5_place as P5
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
from qiskit_ibm_runtime import fake_provider
SNAP = os.path.join("data", "2026-10-13", "hw2", "ns_snapshot_ibm_marrakesh.json")
TESTS = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12", "tfim_n16", "ghz_n12",
         "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "heis2d_n16_r1", "chem_n12_r1", "qaoa3reg_n20_r3",
         "qaoa3reg_n40_r3", "tfim_n40_r2")
DEVICES = ("FakeFez", "FakeMarrakesh", "FakeKingston", "FakeTorino", "snapshot_marrakesh")


def device(d):
    if d == "snapshot_marrakesh":
        return ST.target_from_snapshot(SNAP)
    return getattr(fake_provider, d)().target


def build(name):
    if name.startswith("tfimP"):
        return H2.build_test(name)[0]
    return H.build_test(name)[0] if name in H.TESTS else gens.build(name)


def metrics(c, t):
    pos = {q: i for i, q in enumerate(c.qubits)}
    clock, n2 = {}, 0
    for ins in c.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = t[nm].get(qs) if nm in t.operation_names else None
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        n2 += len(qs) == 2
    return n2, (max(clock.values()) if clock else 0.0) * 1e6


def sig(c):
    h = hashlib.sha256()
    for ins in c.data:
        h.update(ins.operation.name.encode()); h.update(repr([c.find_bit(q).index for q in ins.qubits]).encode())
        h.update(repr([round(float(p), 10) for p in ins.operation.params]).encode())
    return h.hexdigest()[:16]


def variant_b(qc, t):
    """W5's variant B on a Target. Returns the compiled circuit, or raises."""
    import rustworkx as rx
    K, kadj, fadj, snake = P._chain_lattice(t, 0.5)
    n = qc.num_qubits
    pairs = sorted({tuple(sorted(qc.find_bit(q).index for q in i.qubits)) for i in qc.data
                    if len(i.qubits) == 2 and i.operation.name not in ("barrier", "measure")})
    # breadth-first order from the first qubit of the longest snake piece (W5)
    pcs, cur = [], []
    for q in snake:
        if q in K and (not cur or q in kadj[cur[-1]]):
            cur.append(q)
        else:
            if cur:
                pcs.append(cur)
            cur = [q] if q in K else []
    if cur:
        pcs.append(cur)
    best = max(pcs, key=lambda p: (len(p), -snake.index(p[0])))
    rank = {q: i for i, q in enumerate(snake)}
    order, seen, fr = [], {best[0]}, [best[0]]
    while fr:
        u = fr.pop(0); order.append(u)
        for v in sorted(kadj[u], key=lambda v: (rank.get(v, 10_000), v)):
            if v not in seen:
                seen.add(v); fr.append(v)
    if len(order) < n:
        raise RuntimeError("K too small")
    lay, how = None, "breadth-first"
    used = sorted({q for p in pairs for q in p})
    if pairs:
        kn = sorted(K); ki = {q: i for i, q in enumerate(kn)}
        G = rx.PyGraph(); G.add_nodes_from(kn)
        G.add_edges_from_no_data([(ki[a], ki[b]) for a in kn for b in kadj[a] if a < b])
        Hh = rx.PyGraph(); Hh.add_nodes_from(used); ui = {q: i for i, q in enumerate(used)}
        Hh.add_edges_from_no_data([(ui[a], ui[b]) for a, b in pairs])
        m = next(rx.vf2_mapping(G, Hh, subgraph=True, induced=False, id_order=True, call_limit=2_000_000), None)
        if m is not None:
            lay = {used[h]: kn[g] for g, h in m.items()}
            rest = [p for p in order if p not in set(lay.values())]
            for lq, pq in zip([q for q in range(n) if q not in lay], rest):
                lay[lq] = pq
            how = "exact embedding"
    if lay is None:
        lay = {i: order[i] for i in range(n)}
    N = t.num_qubits
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for a in sorted(K):
        for b in sorted(kadj[a]):
            cm.add_edge(a, b)
    pm = generate_preset_pass_manager(1, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                      initial_layout=[lay[i] for i in range(n)], seed_transpiler=0)
    return pm.run(qc), how


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    c49rows = {(r["device"], r["test"]): r for r in map(json.loads, open(os.path.join("data", "2026-10-14", "c49_val", "run", "c49_val_rows.jsonl")))
               if r["arm"] == "C49"}
    fh = open(os.path.join(out, "n4_rows.jsonl"), "w")
    for d in DEVICES:
        t = device(d)
        for name in TESTS:
            qc = build(name)
            r = dict(set="C49-VAL", device=d, test=name)
            try:
                t0 = time.perf_counter(); c, how = variant_b(qc, t); secs = time.perf_counter() - t0
                c2, _ = variant_b(qc, t)
                n2, dur = metrics(c, t)
                act = len({q for i in c.data for q in i.qubits if i.operation.name not in ("barrier",)})
                err = None
                if qc.num_clbits and act <= 20:
                    err = W2.tvd(W2.probs(c), W2.probs(W2.unrolled(qc)))
                r.update(refused=False, how=how, twoq=n2, dur_us=round(dur, 3), deterministic=sig(c) == sig(c2), err=err,
                         secs=round(secs, 2))
            except Exception as exc:  # noqa: BLE001
                r.update(refused=True, error=f"{type(exc).__name__}: {str(exc)[:120]}")
            ref = c49rows.get((d, name))
            if ref:
                r.update(c49_twoq=ref.get("twoq"), c49_dur=ref.get("dur_us"), c49_error=ref.get("error"))
            fh.write(json.dumps(r) + "\n"); fh.flush(); print(json.dumps(r), flush=True)
    # SCALE1's 27 on the snapshot (W5 numbers for variant B are recomputed here; c49 compiled here, default call)
    t = device("snapshot_marrakesh")
    for fam, n, dd in P5.CELLS:
        for inst in range(3):
            qc = S.family_L(n, inst) if fam == "L" else S.family_R(n, dd, inst)
            cell = f"{fam}_n{n}" if fam == "L" else f"R{dd}_n{n}"
            r = dict(set="SCALE1", device="snapshot_marrakesh", test=f"{cell}_i{inst}")
            c, how = variant_b(qc, t); c2, _ = variant_b(qc, t)
            n2, dur = metrics(c, t)
            r.update(refused=False, how=how, twoq=n2, dur_us=round(dur, 3), deterministic=sig(c) == sig(c2),
                     correct=P5.final_bitstring(c, n) == S.ideal_bitstring(qc))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    c9 = C49.compile_for_hardware(qc, target=t, entangling_basis="cx", layout_search=True, seed_transpiler=0)
                n9, d9 = metrics(c9, t)
                r.update(c49_twoq=n9, c49_dur=round(d9, 3), c49_refused=False)
            except Exception as exc:  # noqa: BLE001
                r.update(c49_refused=True, c49_error=f"{type(exc).__name__}")
            fh.write(json.dumps(r) + "\n"); fh.flush(); print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
