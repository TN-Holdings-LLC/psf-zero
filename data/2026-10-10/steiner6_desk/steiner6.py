"""steiner6.py -- STEINER6 (2026-10-10; exploratory): STEINER5 (steiner5.py's method, unchanged otherwise)
with one more family of placement candidates, for chains: the logical qubits in a linear order (spectral: the Fiedler
vector of the terms' interaction graph; also the natural and the folded natural order) laid along a long path of the
device's working graph (a reachability-aware greedy walk from 12 low-degree starts); logical qubits beyond the path's
length are placed greedily next to their partners. Up to 4 such placements (best by sum(weight x distance)) join the
candidates, all scored by the construction's own cost as in STEINER5. STEINER5 found that VF2 does not find the
embedding of an 80-120-qubit chain within its call limit, and that those tests kept the greedy placement.

STEINER5's description follows.

STEINER5 (2026-10-10; exploratory): STEINER4's arm O (STEINER3's construction, then Qiskit level-3
optimisation on the fixed layout with no routing) with the placement computed instead of guessed, and, in arm PE,
the device's errors in both the placement and the trees. STEINER4 (Addendum 435) found the remaining gap in the
placement: a periodic chain that embeds in the heavy-hex lattice was placed with bridges on every term, and the
working elements' errors were ignored.

Placement candidates, scored by the construction's own cost over all terms, the cheapest kept:
  - STEINER3's greedy placement (four starts), as before;
  - embeddings: the terms' interaction graph (logical qubits, an edge per pair that share a term, weighted by how
    many terms they share) is pruned greedily, heaviest edges first, to the largest edge set that is still a
    subgraph (monomorphism, rustworkx VF2) of the device's working graph; up to 200 of its embeddings are scored by
    sum(weight x distance) and the 6 best are kept as candidates.
Arms (each test, device and arm in its own process):
  P   hop distances: placement cost = STEINER3's tree cost (CNOTs) summed over the terms; trees on hop distances
  PE  error distances: a coupler costs -ln(1 - its two-qubit error); placement cost = sum over the trees' couplers of
      that cost times the CNOTs placed on it; trees on these distances
Both: STEINER3's gadgets, then Qiskit level 3 with initial_layout = every qubit in order and routing_method="none".
Exactness on the final circuit (tests with at most 12 logical and 20 touched qubits), as STEINER4.

    cd <psf-zero repository>
    python <this file> run --bp <benchpress clone> --out DIR [--par 4] [--smoke]
    python <this file> report --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), HERE, REPO]
import steiner3 as S3  # noqa: E402  (benchmarks/steiner3.py)

DEVICES = S3.DEVICES
ARMS = ("P", "PE")
TIMEOUT, BUDGET = 1200, 7200
VF2_CHECK_LIMIT, VF2_MAPS, KEEP = 20_000, 200, 6
PATH_STARTS, PATH_KEEP = 12, 4


# ------------------------------------------------------------------ distances

def error_distances(adj, target):
    """All-pairs Dijkstra on the working couplers, a coupler costing -ln(1 - its two-qubit error) (the larger of the
    two directions); (dist, par) with STEINER3's interface (par[s][x] = x's parent in the tree from s)."""
    g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    err = {}
    for qa, p in target[g2].items():
        if qa is None or p is None or p.error is None:
            continue
        k = (min(qa), max(qa))
        err[k] = max(err.get(k, 0.0), p.error)
    w = {k: -math.log(1.0 - min(e, 0.999)) + 1e-9 for k, e in err.items()}
    dist, par = {}, {}
    for s in adj:
        d, p, heap = {s: 0.0}, {s: None}, [(0.0, s)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > d[u]:
                continue
            for v in sorted(adj[u]):
                nd = du + w.get((min(u, v), max(u, v)), 1.0)
                if nd < d.get(v, math.inf):
                    d[v], p[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        dist[s], par[s] = d, p
    return dist, par, w


# ------------------------------------------------------------------ placement

def interaction(n, terms):
    w = {}
    for lb, _ in terms:
        sup = [j for j, ch in enumerate(reversed(lb)) if ch != "I"]
        for i in range(len(sup)):
            for k in range(i + 1, len(sup)):
                e = (sup[i], sup[k])
                w[e] = w.get(e, 0) + 1
    return w


def embeddings(n, terms, adj, dist):
    """Up to KEEP placements from VF2 embeddings of the heaviest embeddable part of the interaction graph."""
    import rustworkx as rx
    nodes = sorted(adj)
    idx = {q: i for i, q in enumerate(nodes)}
    dev = rx.PyGraph()
    dev.add_nodes_from(nodes)
    dev.add_edges_from_no_data(sorted({(idx[a], idx[b]) for a in adj for b in adj[a] if a < b}))
    w = interaction(n, terms)
    keep = []
    for e in sorted(w, key=lambda e: (-w[e], e)):
        pat = rx.PyGraph()
        pat.add_nodes_from(range(n))
        pat.add_edges_from_no_data(keep + [e])
        if max(pat.degree(i) for i in range(n)) > max(dev.degree(i) for i in range(len(nodes))):
            continue
        if rx.is_subgraph_isomorphic(dev, pat, id_order=False, induced=False, call_limit=VF2_CHECK_LIMIT):
            keep.append(e)
    pat = rx.PyGraph()
    pat.add_nodes_from(range(n))
    pat.add_edges_from_no_data(keep)
    scored = []
    try:
        it = rx.vf2_mapping(dev, pat, id_order=False, subgraph=True, induced=False, call_limit=VF2_CHECK_LIMIT * 10)
        for k, m in enumerate(it):
            if k >= VF2_MAPS:
                break
            phys = [None] * n
            for dnode, lnode in m.items():
                phys[lnode] = nodes[dnode]
            if any(p is None for p in phys):
                continue
            score = sum(c * dist[phys[a]].get(phys[b], 1e6) for (a, b), c in w.items())
            scored.append((score, phys))
    except Exception:  # noqa: BLE001 - no embedding found within the call limit: the greedy placement remains
        pass
    scored.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, phys in scored:
        if tuple(phys) not in seen:
            seen.add(tuple(phys))
            out.append(phys)
        if len(out) >= KEEP:
            break
    return out, len(keep), len(w)


def long_paths(adj):
    """Long simple paths of the working graph: from each of PATH_STARTS low-degree qubits, walk to the unvisited
    neighbour from which the most unvisited qubits stay reachable (fewer onward neighbours, then the index, break
    ties). Distinct paths, longest first."""
    from collections import deque

    def reach(v, vis):
        seen, dq = {v}, deque([v])
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if w not in vis and w not in seen:
                    seen.add(w)
                    dq.append(w)
        return len(seen)
    out, seen_paths = [], set()
    for s in sorted(adj, key=lambda q: (len(adj[q]), q))[:PATH_STARTS]:
        p, vis = [s], {s}
        while True:
            nb = [v for v in adj[p[-1]] if v not in vis]
            if not nb:
                break
            v = max(nb, key=lambda v: (reach(v, vis), -sum(1 for w in adj[v] if w not in vis), -v))
            p.append(v)
            vis.add(v)
        if tuple(p) not in seen_paths:
            seen_paths.add(tuple(p))
            out.append(p)
    return sorted(out, key=len, reverse=True)


def linear_orders(n, w):
    """Linear orders of the logical qubits: spectral (Fiedler vector of the weighted interaction graph, made
    connected by 1e-6 on every pair), natural, folded natural (0, n-1, 1, n-2, ...)."""
    import numpy as np
    lap = np.full((n, n), -1e-6)
    for (a, b), c in w.items():
        lap[a, b] -= c
        lap[b, a] -= c
    np.fill_diagonal(lap, 0.0)
    np.fill_diagonal(lap, -lap.sum(axis=1))
    orders = []
    if n >= 3:
        vals, vecs = np.linalg.eigh(lap)
        orders.append([int(i) for i in np.argsort(vecs[:, 1], kind="stable")])
    orders.append(list(range(n)))
    folded = []
    lo, hi = 0, n - 1
    while lo <= hi:
        folded.append(lo)
        if hi != lo:
            folded.append(hi)
        lo, hi = lo + 1, hi - 1
    orders.append(folded)
    uniq = []
    for o in orders:
        if o not in uniq:
            uniq.append(o)
    return uniq


def path_placements(n, terms, adj, dist):
    """Up to PATH_KEEP placements along long paths (see the module docstring)."""
    w = interaction(n, terms)
    partners = {i: [] for i in range(n)}
    for (a, b), c in w.items():
        partners[a].append((b, c))
        partners[b].append((a, c))
    paths = long_paths(adj)[:3]
    cands = []
    for order in linear_orders(n, w):
        for path in paths:
            for p in (path, path[::-1]):
                length = min(len(p), n)
                offsets = sorted({0, max(0, len(p) - n), max(0, (len(p) - n) // 2)}) if len(p) > n else [0]
                for off in offsets:
                    phys = [None] * n
                    used = set()
                    for k in range(length):
                        phys[order[k]] = p[off + k]
                        used.add(p[off + k])
                    for lq in order[length:]:  # beyond the path: next to the placed partners
                        free = [v for v in adj if v not in used]
                        if not free:
                            break
                        placed = [(phys[o], c) for o, c in partners[lq] if phys[o] is not None]
                        best = min(free, key=lambda v: (sum(c * dist[q].get(v, 1e6) for q, c in placed),
                                                        -len(adj[v]), v))
                        phys[lq] = best
                        used.add(best)
                    if any(x is None for x in phys):
                        continue
                    score = sum(c * dist[phys[a]].get(phys[b], 1e6) for (a, b), c in w.items())
                    cands.append((score, phys))
    cands.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, phys in cands:
        if tuple(phys) not in seen:
            seen.add(tuple(phys))
            out.append(phys)
        if len(out) >= PATH_KEEP:
            break
    return out


def placement_cost(phys, terms, dist, par, edge_w):
    """Hop arm (edge_w None): STEINER3's tree cost summed. Error arm: the trees' couplers' costs x CNOTs on them."""
    cost = 0.0
    for lb, _ in terms:
        sup = [phys[j] for j, ch in enumerate(reversed(lb)) if ch != "I"]
        if len(sup) < 2:
            continue
        if any(dist[sup[0]].get(v) is None for v in sup):
            return math.inf
        r, p = S3.steiner_tree(sup, dist, par)
        if edge_w is None:
            cost += S3.tree_cost(r, p, set(sup))
        else:
            term = set(sup)
            cost += sum(edge_w.get((min(c, q), max(c, q)), 1.0) * (2 if c in term else 4) for c, q in p.items())
    return cost


def build(qc, target, arm, reading):
    """(physical circuit, placement, evo, info) or a reason (str)."""
    s = S3.synthesise(qc, target, reading=reading)
    if isinstance(s, str):
        return s
    _, phys0, meas, _, evo = s
    from qiskit import QuantumCircuit
    adj, _ = S3.working_graph(target)
    if arm == "P":
        dist, par = S3.bfs_all(adj)
        edge_w = None
    else:
        dist, par, edge_w = error_distances(adj, target)
    triples = S3._variants(evo.operation)[reading]
    terms = [(lb, c) for lb, c, _ in triples]
    n = len(phys0)
    emb, kept, total = embeddings(n, terms, adj, dist)
    paths = path_placements(n, terms, adj, dist)
    cands = ([("greedy", phys0)] + [(f"vf2-{i}", p) for i, p in enumerate(emb)]
             + [(f"path-{i}", p) for i, p in enumerate(paths)])
    costs = [(placement_cost(p, terms, dist, par, edge_w), name, p) for name, p in cands]
    best = min(costs, key=lambda x: (x[0], x[1]))
    phys = best[2]
    out = QuantumCircuit(target.num_qubits, qc.num_clbits)
    gphase = 0.0
    for lb, c, t in triples:
        for name, q, prm in S3.gadget_ops(lb, c, t, phys, dist, par):
            if name == "gphase":
                gphase += prm[0]
            elif name == "cx":
                out.cx(*q)
            elif name == "rz":
                out.rz(prm[0], q[0])
            else:
                getattr(out, name)(q[0])
    out.global_phase = gphase
    for q, cb in meas:
        out.measure(phys[q], cb)
    greedy_cost = costs[0][0]
    info = dict(chosen=best[1], candidates=len(cands), edges_embedded=kept, edges_total=total,
                cost=round(best[0], 3), greedy_cost=round(greedy_cost, 3))
    return out, phys, evo, info


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    from qiskit import transpile
    qc, bp_backend = B.build(a.bp, kind, arg)
    target = S3.device(a.device, bp_backend).target
    rec = dict(stratum=stratum, test=tid, device=a.device, arm=a.arm, input_qubits=qc.num_qubits)
    reading = S3.calibrate()
    if reading is None:
        print(json.dumps(dict(rec, error="no reading of PauliEvolutionGate matches Qiskit's Operator")), flush=True)
        return
    t0 = time.perf_counter()
    s = build(qc, target, a.arm, reading)
    if isinstance(s, str):
        print(json.dumps(dict(rec, eligible=False, why=s)), flush=True)
        return
    phys_circ, phys, evo, info = s
    rec["t_construct"] = round(time.perf_counter() - t0, 3)
    out = transpile(phys_circ, target=target, initial_layout=list(range(target.num_qubits)), routing_method="none",
                    optimization_level=3, seed_transpiler=0)
    rec["t"] = round(time.perf_counter() - t0, 3)
    lay = out.layout
    moved = lay is not None and list(lay.final_index_layout()) != list(range(target.num_qubits))
    rec.update(eligible=True, n=len(phys), terms=len(evo.operation.operator), moved=moved, **info)
    rec["q2"] = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "delay"))
    rec["esp"], rec["on_failed"] = S3.esp_of(out, target)
    try:
        rec["exact"] = "layout moved" if moved else S3.check(qc, evo, phys, out)
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["exact"] = f"check failed: {type(exc).__name__}: {exc}"[:200]
    print(json.dumps(rec), flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    ts = S3.tests(a.bp)
    if a.smoke:
        small = [t for t in ts if "enc_unary_dvalues_4-4" in t[1] or "Lx-26" in t[1]]
        ts = (small or ts)[:2]
    jobs = sorted((hashlib.sha256(("STEINER6|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "steiner6.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par))) + "\n")
    print(f"STEINER6: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
    t_start = time.perf_counter()

    def go(job):
        _, t, d, arm = job
        base = dict(stratum=t[0], test=t[1], device=d, arm=arm)
        if time.perf_counter() - t_start > BUDGET:
            return dict(base, error="not started (budget)")
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--device", d,
                                "--arm", arm, "--job", json.dumps(list(t))], capture_output=True, text=True,
                               timeout=TIMEOUT, cwd=REPO, env=dict(os.environ, PSF_ZERO_REPO=REPO))
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            return json.loads(lines[-1]) if lines else dict(base, error=(p.stderr or "")[-400:], exit=p.returncode)
        except subprocess.TimeoutExpired:
            return dict(base, error=f"timeout {TIMEOUT} s")

    n = 0
    with ThreadPoolExecutor(a.par) as ex, open(path, "a", encoding="utf-8", newline="\n") as fh:
        for f in as_completed([ex.submit(go, j) for j in jobs]):
            r = f.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
            msg = (r["error"][:70] if "error" in r else
                   (f"{r['t']} s, q2 {r['q2']}, {r['chosen']}, exact {r['exact']}, esp {r['esp']}"
                    if r.get("eligible") else f"not eligible: {r['why']}"))
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:5.0f} s] {r['device'][4:]:8s} {r['arm']:2s} "
                  f"{r['test'][-40:]}: {msg}", flush=True)
    report(a)


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "steiner6.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    base = os.path.dirname(os.path.abspath(a.out))
    other = {}
    for f, arms in (("esp_ft/esp_ft.jsonl", ("QK2", "QK3", "PSFR")), ("steiner4/steiner4.jsonl", ("O",)),
                    ("steiner5/steiner5.jsonl", ("P", "PE"))):
        p = os.path.join(base, f)
        if os.path.exists(p):
            for x in [json.loads(y) for y in open(p, encoding="utf-8")][1:]:
                if x.get("arm") in arms and "error" not in x and x.get("eligible", True):
                    key = {"O": "S4O", "P": "S5P", "PE": "S5PE"}.get(x["arm"], x["arm"]) if "steiner" in f else x["arm"]
                    other[(x["device"], x["test"], key)] = x
    mine = {(r["device"], r["test"], r["arm"]): r for r in recs if r.get("eligible")}
    allr = {**other, **mine}
    lg = lambda r: -math.inf if r.get("esp") is None else r["esp"]  # noqa: E731
    gm = lambda xs: math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else float("nan")  # noqa: E731
    L = ["# STEINER6 (exploratory)", "",
         f"git head {meta['git_head']}; {meta['tests']} HamLib FakeTorino tests on {', '.join(DEVICES)}; "
         f"{meta['jobs']} jobs (arms {', '.join(ARMS)}). Others: ESP-FT (QK2, QK3, PSFR), STEINER4's O (S4O), "
         "STEINER5's P and PE (S5P, S5PE).", ""]
    for d in DEVICES:
        L += [f"## {d}", ""]
        for arm in ARMS:
            rs = [r for r in recs if r["device"] == d and r["arm"] == arm and r.get("eligible")]
            bad = [r for r in rs if r["exact"] not in (True, None)]
            ch = {}
            for r in rs:
                k = r["chosen"].split("-")[0]
                ch[k] = ch.get(k, 0) + 1
            L.append(f"- {arm}: {len(rs)} eligible; exact {sum(1 for r in rs if r['exact'] is True)}, not checked "
                     f"{sum(1 for r in rs if r['exact'] is None)}, NOT exact {len(bad)}; on failed elements "
                     f"{sum(1 for r in rs if r['on_failed'])}; placement chosen {ch}; errors "
                     f"{sum(1 for r in recs if r['device'] == d and r['arm'] == arm and 'error' in r)}; time "
                     f"{sum(r['t'] for r in rs):.1f} s summed")
        L += ["", "| arm / other | q2 ratio (gmean, +1) | fewer / equal / more | ESP ratio (gmean, both > 0; best >= "
              "0.01) | ESP 10% better / worse |", "|---|---|---|---|---|"]
        for arm in ARMS:
            for o in ("S5P", "S5PE", "S4O", "QK2", "QK3", "PSFR"):
                pr = [(mine[k], allr[(d, k[1], o)]) for k in mine if k[0] == d and k[2] == arm and (d, k[1], o) in allr]
                if not pr:
                    continue
                qr = gm([(r["q2"] + 1) / (x["q2"] + 1) for r, x in pr])
                fe = sum(1 for r, x in pr if r["q2"] < x["q2"])
                eq = sum(1 for r, x in pr if r["q2"] == x["q2"])
                run_ = [(r, x) for r, x in pr if max(lg(r), lg(x)) >= -2]
                dv = [lg(r) - lg(x) for r, x in run_]
                fin = [v for v in dv if math.isfinite(v)]
                L.append(f"| {arm} / {o} | {qr:.3f} | {fe} / {eq} / {len(pr) - fe - eq} | "
                         f"{10 ** (sum(fin) / len(fin)) if fin else float('nan'):.3f} ({len(run_)}) | "
                         f"{sum(1 for v in dv if v >= math.log10(1.1))} / {sum(1 for v in dv if v <= -math.log10(1.1))} |")
        L += ["", "| test | n | terms | S5P | P | S5PE | PE | QK2 | QK3 | PSFR | PE log10 ESP | QK3 log10 ESP | P placement | PE placement |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        g = lambda t, o, k: allr.get((d, t, o), {}).get(k, "-")  # noqa: E731
        for t in sorted({k[1] for k in mine if k[0] == d}, key=lambda t: (g(t, "P", "n") if g(t, "P", "n") != "-" else 0, t)):
            L.append(f"| {t.split('[')[-1].rstrip(']')[:40]} | {g(t, 'P', 'n')} | {g(t, 'P', 'terms')} | "
                     f"{g(t, 'S5P', 'q2')} | {g(t, 'P', 'q2')} | {g(t, 'S5PE', 'q2')} | {g(t, 'PE', 'q2')} | {g(t, 'QK2', 'q2')} | "
                     f"{g(t, 'QK3', 'q2')} | {g(t, 'PSFR', 'q2')} | {g(t, 'PE', 'esp')} | {g(t, 'QK3', 'esp')} | "
                     f"{g(t, 'P', 'chosen')} | {g(t, 'PE', 'chosen')} |")
        L += [f"- error ({r['arm']}, {r['test'][-40:]}): {r['error'][:160]}" for r in recs
              if r["device"] == d and "error" in r]
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "steiner6.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "report"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--device")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "one": one, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
