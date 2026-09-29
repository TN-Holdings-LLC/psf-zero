"""full_heavyhex_cliff.py -- pre-registered (workplace, 2026-09-29): does the layout
cliff appear on a heavy-hex device when every one of its qubits is used?

Addendum 246 (home) could not answer this: disjoint pair blocks fill only 2M = 128 of
ibm_kingston's 156 qubits (M = maximum matching), so the device was never saturated.
Here two circuit families occupy all qubits:

  T  (pairs + triples) k2 disjoint pair24 blocks and k3 "triple" blocks, with
     k3 = N - 2M and k2 = M - k3. A triple (a, b, c) is pair24 on (a, b) followed by
     pair24 on (b, c): its interaction graph is a 3-qubit path. At spare 0 the
     interaction graph is a {P2, P3}-factor of the device (Stage 0 checks one exists).
  S  (pairs + single-qubit-only qubits) M pair24 blocks and N - 2M qubits that carry
     only single-qubit gates (two rz-ry-rz layers). All N logical qubits exist, but the
     interaction graph is still a matching.
  B  (bridge) M pair24 blocks, no other qubits: the Addendum 245 family at its spare 0
     (n = 2M), on this script's device. Spare 0 only.

Spare s (T and S) removes s/2 pair blocks, so n = N - s; SPARES = (0, 2, 8, 16).

Device: by default the snapshot FakeKingston (qiskit-ibm-runtime fake provider; no
account, no network). With --target-pkl FILE a pickled live Target is used instead (home
only; a pickle runs code when loaded, so only load one you made yourself).

Arms, each compile in its own spawned child, timed inside the child around the compile
call only, killed at --cap seconds (DNF):
  Q3  qiskit.transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
  P   psf_compile.compile_for_hardware(qc, coupling_map=target's, basis_gates=native,
      entangling_basis="cx", layout_search=True, on_unsupported="raise", seed_transpiler=0)
      with the release benchmarks/psf_smart_layout.py (LAYOUT_VERSION 2026-09-26.m1)
  PX  the same call with the candidate psf_smart_layout.py 2026-09-29.c1 found first on
      sys.path (--px-dir): a corrected feasibility check and a short-path shortcut
Quality: routed two-qubit count; when no two-qubit gate joins different blocks, an exact
per-block check (blocks of 1, 2 or 3 qubits) by phase-aligned Frobenius distance; a
SHA-256 digest of the output (gate names, qubit indices, parameters).
Diagnostic, outside the timed call: for P and PX, what that arm's smart_vf2_layout finds
for the same interaction graph with compile_for_hardware's limits (path or matching
shortcut = phase 0, VF2 phase 1 or 2, or nothing).

    python -u benchmarks/full_heavyhex_cliff.py run --px-dir <candidate_dir> 2>&1 | tee full_heavyhex_cliff.txt
    python -u benchmarks/full_heavyhex_cliff.py score 2>&1 | tee full_heavyhex_score.txt
    (home, optional) ... run --px-dir <candidate_dir> --target-pkl <file> --tag home
    (harness dry run) ... run --px-dir <candidate_dir> --fake FakeAuckland --tag dry --cap 60 --spares 0,2,4
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import multiprocessing as mp
import os
import pickle
import platform
import statistics as st
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SPARES = (0, 2, 8, 16)  # --spares exists only for the harness dry run on a small device
INPUTS = 3
DEADLINES = (0.1, 1.0, 10.0)
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
FAMILY_SEED = {"T": 0, "S": 100_000, "B": 200_000}


# ---------------------------------------------------------------- device and Stage 0

def load_target(args):
    if args.target_pkl:
        with open(args.target_pkl, "rb") as f:
            return pickle.load(f), f"pickled Target ({os.path.basename(args.target_pkl)})"
    import warnings
    from qiskit_ibm_runtime import fake_provider
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = getattr(fake_provider, args.fake)()
    return backend.target, f"{args.fake} (fake provider snapshot)"


def graph_facts(target):
    """Qubit count, undirected edges, degree range, maximum matching and a {P2,P3}-factor
    built by attaching every unmatched qubit to a distinct adjacent matched pair."""
    import networkx as nx
    cmap = target.build_coupling_map()
    nq = cmap.size()
    edges = sorted({tuple(sorted(e)) for e in cmap.get_edges()})
    g = nx.Graph()
    g.add_nodes_from(range(nq))
    g.add_edges_from(edges)
    deg = sorted(d for _, d in g.degree())
    m = sorted(tuple(sorted(e)) for e in nx.max_weight_matching(g, maxcardinality=True))
    mate = {}
    for u, v in m:
        mate[u], mate[v] = v, u
    unmatched = [x for x in range(nq) if x not in mate]
    pid = {}
    for k, (u, v) in enumerate(m):
        pid[u] = pid[v] = k
    bg = nx.Graph()
    tops = [("u", x) for x in unmatched]
    bg.add_nodes_from(tops)
    for x in unmatched:
        for y in g.neighbors(x):
            bg.add_edge(("u", x), ("p", pid[y]))
    mm = nx.bipartite.maximum_matching(bg, top_nodes=tops) if tops else {}
    factor = None
    if all(t in mm for t in tops):
        triples, used = [], set()
        for x in unmatched:
            k = mm[("u", x)][1]
            u, v = m[k]
            centre = u if g.has_edge(x, u) else v
            triples.append((x, centre, mate[centre]))
            used.add(k)
        pairs = [m[k] for k in range(len(m)) if k not in used]
        cover = sorted(q for t in triples for q in t) + sorted(q for p in pairs for q in p)
        if sorted(cover) == list(range(nq)) and all(g.has_edge(a, b) and g.has_edge(b, c) for a, b, c in triples):
            factor = dict(triples=triples, pairs=pairs)
    return dict(nq=nq, edges=len(edges), deg=(deg[0], st.median(deg), deg[-1]), matching=len(m),
                unmatched=len(unmatched), factor=factor)


# ---------------------------------------------------------------- circuits

def layout_blocks(family, spare, nq, m):
    """Logical blocks (tuples of logical qubit indices) for a family and spare."""
    k3 = nq - 2 * m
    if family == "T":
        k2 = m - k3 - spare // 2
        sizes = [3] * k3 + [2] * k2
    elif family == "S":
        k2 = m - spare // 2
        sizes = [2] * k2 + [1] * k3
    else:  # "B"
        sizes = [2] * m
    blocks, q = [], 0
    for s in sizes:
        blocks.append(tuple(range(q, q + s)))
        q += s
    return blocks


def build(family, spare, k, nq, m):
    from qiskit import QuantumCircuit
    import loop_endurance as le
    blocks = layout_blocks(family, spare, nq, m)
    n = sum(len(b) for b in blocks)
    rng = np.random.default_rng(FAMILY_SEED[family] + 1000 * spare + k)
    qc = QuantumCircuit(n)
    for b in blocks:
        if len(b) == 3:
            th = rng.uniform(-np.pi, np.pi, 48)
            le.add_pair24(qc, b[0], b[1], th[:24])
            le.add_pair24(qc, b[1], b[2], th[24:])
        elif len(b) == 2:
            le.add_pair24(qc, b[0], b[1], rng.uniform(-np.pi, np.pi, 24))
        else:
            th = rng.uniform(-np.pi, np.pi, 6)
            for j in (0, 3):
                qc.rz(th[j], b[0]); qc.ry(th[j + 1], b[0]); qc.rz(th[j + 2], b[0])
    return qc, blocks


# ---------------------------------------------------------------- checks

def aligned(a, b):
    t = np.trace(a.conj().T @ b)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(b - ph * a))


def block_check(qc_logical, qc_out, blocks):
    """Per-block phase-aligned operator distance. Returns (applicable, worst)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    lay = qc_out.layout
    if lay is None:
        return False, None
    phys = list(lay.final_index_layout(filter_ancillas=True))
    owner, local = {}, {}
    for k, b in enumerate(blocks):
        for j, q in enumerate(b):
            owner[phys[q]] = k
            local[phys[q]] = j
    per = {k: QuantumCircuit(len(b)) for k, b in enumerate(blocks)}
    for inst in qc_out.data:
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        qs = [qc_out.find_bit(q).index for q in inst.qubits]
        ks = {owner.get(q) for q in qs}
        if None in ks:
            if len(qs) == 1:
                continue
            return False, None
        if len(ks) != 1:
            return False, None
        k = ks.pop()
        per[k].append(inst.operation, [local[q] for q in qs])
    worst = 0.0
    for k, b in enumerate(blocks):
        ref = QuantumCircuit(len(b))
        pos = {q: j for j, q in enumerate(b)}
        for inst in qc_logical.data:
            qs = [qc_logical.find_bit(q).index for q in inst.qubits]
            if set(qs) <= set(b):
                ref.append(inst.operation, [pos[q] for q in qs])
        worst = max(worst, aligned(Operator(ref).data, Operator(per[k]).data))
    return True, worst


def expected_twoq(blocks):
    """Two-qubit gates with no SWAP: 3 per pair block, 6 per triple (two pair24 blocks)."""
    return sum({1: 0, 2: 3, 3: 6}[len(b)] for b in blocks)


# ---------------------------------------------------------------- one compile

def out_digest(out):
    import hashlib
    h = hashlib.sha256()
    for i in out.data:
        h.update(repr((i.operation.name, [out.find_bit(q).index for q in i.qubits],
                       [float(x) for x in i.operation.params])).encode())
    return h.hexdigest()[:16]


def _worker(args_d, arm, family, spare, k, nq, m, q):
    ns = argparse.Namespace(**args_d)
    if arm == "PX":
        sys.path.insert(0, ns.px_dir)
    from qiskit import transpile
    target, _ = load_target(ns)
    qc, blocks = build(family, spare, k, nq, m)
    native = [g for g in target.operation_names if g in NATIVE]
    cmap = target.build_coupling_map()
    if arm != "Q3":
        import psf_compile as pc
        import psf_smart_layout as sl
    t0 = time.perf_counter()
    if arm == "Q3":
        out = transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            out = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=native, entangling_basis="cx",
                                          layout_search=True, on_unsupported="raise", seed_transpiler=0)
    elapsed = time.perf_counter() - t0
    twoq = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier")
    applicable, worst = block_check(qc, out, blocks)
    diag = layout_mod = None
    if arm != "Q3":
        import loop_endurance as le
        layout_mod = f"{sl.LAYOUT_VERSION} {le.normalized_sha256(sl.__file__)[:12]}"
        pairs = sorted({tuple(sorted((qc.find_bit(i.qubits[0]).index, qc.find_bit(i.qubits[1]).index)))
                        for i in qc.data if len(i.qubits) == 2})
        lm, info = sl.smart_vf2_layout(cmap, pairs, qc.num_qubits, per_attempt_call_limit=50_000,
                                       time_budget_s=2.0, fallback_call_limit=2_000_000)
        diag = f"found={lm is not None} phase={info.get('phase')} order={info.get('order_name')} " \
               f"tried={info.get('orderings_tried')} search_s={info.get('elapsed_s'):.3f}"
    q.put(dict(elapsed_s=elapsed, twoq=twoq, expected_twoq=expected_twoq(blocks), block_check_applicable=applicable,
               block_worst=worst, digest=out_digest(out), layout_module=layout_mod, psf_layout_diag=diag))


def run_one(args, arm, family, spare, k, nq, m):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(vars(args), arm, family, spare, k, nq, m, q))
    p.start()
    p.join(args.cap)
    empty = dict(elapsed_s=None, twoq=None, expected_twoq=None, block_check_applicable=None, block_worst=None,
                 digest=None, layout_module=None, psf_layout_diag=None)
    if p.is_alive():
        p.terminate()
        p.join()
        return dict(empty, status="DNF")
    if p.exitcode != 0 or q.empty():
        return dict(empty, status=f"ERROR(exit {p.exitcode})")
    r = q.get()
    r["status"] = "OK"
    return r


# ---------------------------------------------------------------- run and score

ARMS = ("Q3", "P", "PX")


def run(args):
    import qiskit
    import psf_compile as pc
    import loop_endurance as le
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__),
          "| CORE_VERSION", getattr(pc, "CORE_VERSION", None))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    cand = os.path.join(args.px_dir, "psf_smart_layout.py")
    print("PX layout module", cand, le.normalized_sha256(cand))
    target, label = load_target(args)
    f = graph_facts(target)
    native = [x for x in target.operation_names if x in NATIVE]
    fac = f["factor"]
    print(f"Stage 0: device {label}, {f['nq']} qubits, {f['edges']} edges, degree min/median/max "
          f"{f['deg'][0]}/{f['deg'][1]}/{f['deg'][2]}, max matching {f['matching']} pairs "
          f"({2 * f['matching']} qubits, {f['unmatched']} unmatched), native {native}")
    k3 = f["nq"] - 2 * f["matching"]
    print(f"G0 {{P2,P3}}-factor with {k3} triples and {f['matching'] - k3} pairs: "
          f"{'found' if fac else 'NOT FOUND'}" + (f" (triples {fac['triples']})" if fac else ""))
    if not fac:
        print("G0 FAILED: family T cannot fill the device; nothing is run.")
        return
    nq, m = f["nq"], f["matching"]
    spares = tuple(int(x) for x in args.spares.split(","))
    plan = [("B", 0)] + [(fam, s) for fam in ("T", "S") for s in spares]
    rows = []
    for fam, spare in plan:
        for k in range(INPUTS):
            for arm in ARMS:
                blocks = layout_blocks(fam, spare, nq, m)
                n = sum(len(b) for b in blocks)
                r = run_one(args, arm, fam, spare, k, nq, m)
                row = dict(family=fam, spare=spare, logical_qubits=n, physical_qubits=nq, input=k, arm=arm, **r)
                for d in DEADLINES:
                    row[f"within_{d}s"] = r["status"] == "OK" and r["elapsed_s"] <= d
                rows.append(row)
                el = "DNF" if r["elapsed_s"] is None else f"{r['elapsed_s']:.3f}s"
                print(f"{fam} spare={spare:2d} n={n} input={k} {arm:2s} {r['status']:5s} {el:>9s} "
                      f"2q={r['twoq']} (no-swap {r['expected_twoq']}) check={r['block_check_applicable']} "
                      f"worst={r['block_worst']} digest={r['digest']}"
                      + (f" | {r['layout_module']} | {r['psf_layout_diag']}" if r["psf_layout_diag"] else ""),
                      flush=True)
    out_csv = f"full_heavyhex_cliff_{args.tag}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(device=label, qubits=nq, matching=m, factor=fac, native=native, cap_s=args.cap,
                   spares=list(spares), rows=rows),
              open(f"full_heavyhex_cliff_{args.tag}.json", "w"), indent=1, default=str)
    print(f"\nWrote {out_csv} and full_heavyhex_cliff_{args.tag}.json ({len(rows)} rows)\nDONE")


def score(args):
    data = json.load(open(f"full_heavyhex_cliff_{args.tag}.json"))
    rows, cap, spares = data["rows"], data["cap_s"], tuple(data["spares"])
    hi = spares[-1]

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")

    def cell(fam, spare, arm):
        return [r for r in rows if r["family"] == fam and r["spare"] == spare and r["arm"] == arm]

    def med(fam, spare, arm):
        return st.median(r["elapsed_s"] if r["status"] == "OK" else cap for r in cell(fam, spare, arm))

    def fmt_t(r):
        return "DNF" if r["status"] != "OK" else f"{r['elapsed_s']:.3f}"

    def within(fam, spare, arm):
        return sum(r["within_1.0s"] for r in cell(fam, spare, arm))

    print("=" * 78)
    print(f"SCORING (thresholds exactly as pre-registered); device {data['device']}; DNF counted as {cap:.0f} s")
    print("=" * 78)
    err = [r for r in rows if r["status"].startswith("ERROR")]
    psf_ok = all(r["status"] == "OK" for r in rows if r["arm"] in ("P", "PX"))
    mods = {arm: sorted({r["layout_module"].split()[0] for r in rows if r["arm"] == arm and r["layout_module"]})
            for arm in ("P", "PX")}
    ok0 = not err and psf_ok and mods["P"] == ["2026-09-26.m1"] and mods["PX"] == ["2026-09-29.c1"]
    print(f"C0 no errors: {not err}; every PSF-Zero compile finished: {psf_ok}; layout modules {mods} -> "
          f"{'passed' if ok0 else 'FAILED (read the predictions with this caveat)'}")
    for fam in ("B", "T", "S"):
        for s in (spares if fam != "B" else (0,)):
            line = f"   {fam} spare {s:2d}:"
            for arm in ARMS:
                c = cell(fam, s, arm)
                line += (f" | {arm} med {med(fam, s, arm):.3f} [{','.join(fmt_t(r) for r in c)}] 2q "
                         f"{[r['twoq'] for r in c]} <=1s {within(fam, s, arm)}/{len(c)}")
            print(line + f" | no-swap {cell(fam, s, 'P')[0]['expected_twoq']}")
    n = INPUTS
    h1 = med("T", 0, "Q3") / med("T", hi, "Q3")
    print(f"H1 T: Qiskit L3 median spare 0 / spare {hi} = {h1:.1f} (confirmed >= 10, refuted < 3) -> "
          f"{v(h1 >= 10, h1 < 3)}")
    rt = cell("T", 0, "P")
    sw = sum(1 for r in rt if r["status"] == "OK" and r["twoq"] > r["expected_twoq"])
    print(f"H2 T spare 0: the release PSF-Zero (P) inserts SWAPs (2q above the no-swap count) in {sw}/{n} "
          f"(confirmed {n}/{n}, refuted 0) -> {v(sw == n, sw == 0)}")
    rx_ = cell("T", 0, "PX")
    nsw = sum(1 for r in rx_ if r["status"] == "OK" and r["twoq"] == r["expected_twoq"])
    wx = within("T", 0, "PX")
    print(f"H3 T spare 0: the candidate (PX) is swap-free in {nsw}/{n} and within 1 s in {wx}/{n} "
          f"(confirmed both {n}/{n}, refuted either <= 1) -> {v(nsw == n and wx == n, nsw <= 1 or wx <= 1)}")
    h4 = med("T", 0, "Q3") / med("T", 0, "PX")
    print(f"H4 T spare 0: Qiskit L3 median / PX median = {h4:.1f} (confirmed >= 10, refuted < 3) -> "
          f"{v(h4 >= 10, h4 < 3)}")
    h5 = med("S", 0, "Q3") / med("S", hi, "Q3")
    print(f"H5 S: Qiskit L3 median spare 0 / spare {hi} = {h5:.1f} (confirmed < 3, refuted >= 10) -> "
          f"{v(h5 < 3, h5 >= 10)}")
    ws = min(within("S", 0, "P"), within("S", 0, "PX"))
    print(f"H6 S spare 0: P and PX within 1 s, fewer of the two: {ws}/{n} (confirmed {n}/{n}, refuted <= 1) -> "
          f"{v(ws == n, ws <= 1)}")
    key = lambda r: (r["family"], r["spare"], r["input"])  # noqa: E731
    by = {(r["arm"],) + key(r): r for r in rows}
    bs = [r for r in rows if r["arm"] == "P" and r["family"] in ("B", "S")]
    same_d = sum(1 for r in bs if r["status"] == "OK" and by[("PX",) + key(r)]["digest"] == r["digest"])
    print(f"H7 B and S (matching-shaped): PX output identical to P (digest) in {same_d}/{len(bs)} "
          f"(confirmed all, refuted any differs) -> {v(same_d == len(bs), same_d < len(bs))}")
    pairs_x = [(by[("Q3",) + key(r)], r) for r in rows if r["arm"] == "PX"]
    pairs_x = [(a, b) for a, b in pairs_x if a["status"] == "OK" and b["status"] == "OK"]
    worse = sum(b["twoq"] > a["twoq"] for a, b in pairs_x)
    print(f"H8 PX two-qubit count never above Qiskit L3: above in {worse} of {len(pairs_x)} "
          f"(confirmed 0, refuted any) -> {v(worse == 0, worse > 0)}")
    pw = [r["block_worst"] for r in rows if r["arm"] in ("P", "PX") and r["block_check_applicable"]]
    napp = sum(1 for r in rows if r["arm"] in ("P", "PX") and r["status"] == "OK")
    wmax = max(pw) if pw else None
    print(f"H9 P and PX exact per block (<= 1e-12) wherever the check applies: {len(pw)}/{napp} applicable, "
          f"worst {wmax} -> {v(bool(pw) and wmax <= 1e-12, bool(pw) and wmax > 1e-12)}")
    qw = [r["block_worst"] for r in rows if r["arm"] == "Q3" and r["block_check_applicable"]]
    nq3 = sum(1 for r in rows if r["arm"] == "Q3" and r["status"] == "OK")
    print(f"Reported without prediction: Qiskit L3 per-block worst {max(qw) if qw else None} over {len(qw)}/{nq3} "
          f"applicable, above 1e-12: {sum(1 for x in qw if x > 1e-12)}; B ratio Q3/P at spare 0 "
          f"{med('B', 0, 'Q3') / med('B', 0, 'P'):.1f}; P above Qiskit L3 in 2q: "
          f"{sum(1 for r in rows if r['arm'] == 'P' and r['status'] == 'OK' and by[('Q3',) + key(r)]['status'] == 'OK' and r['twoq'] > by[('Q3',) + key(r)]['twoq'])}")
    diags = sorted({(r["arm"], r["family"], r["spare"], r["psf_layout_diag"].split(" search_s")[0])
                    for r in rows if r["psf_layout_diag"]})
    for d in diags:
        print("   PSF layout search (diagnostic):", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "score"))
    ap.add_argument("--fake", default="FakeKingston")
    ap.add_argument("--target-pkl", default=None)
    ap.add_argument("--px-dir", default=None, help="folder holding the candidate psf_smart_layout.py")
    ap.add_argument("--tag", default="2026-09-29")
    ap.add_argument("--cap", type=float, default=180.0)
    ap.add_argument("--spares", default="0,2,8,16")
    args = ap.parse_args()
    if args.mode == "run" and not args.px_dir:
        ap.error("run needs --px-dir")
    run(args) if args.mode == "run" else score(args)


if __name__ == "__main__":
    main()
