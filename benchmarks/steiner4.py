"""steiner4.py -- STEINER4 (2026-10-10; exploratory): STEINER3's construction (benchmarks/steiner3.py) with the two
things it lacked against Qiskit's synthesis, which shares and cancels CNOTs between consecutive terms:

  A  aligned trees: each term's parity is collected onto the previous term's root when that qubit is in the term
     (otherwise onto the term's qubit nearest to it), and its Steiner tree is grown from that root by a multi-source
     shortest-path search in which a coupler of the previous term's tree costs 1 and any other coupler 2. Consecutive
     terms then share CNOTs in the same direction, so the uncompute of one and the compute of the next can cancel.
  O  Qiskit's level-3 optimisation (block consolidation and re-synthesis, commutative cancellation) on the
     constructed circuit, with the layout fixed to the construction's qubits and no routing ("none": the run fails
     if any gate is not on a coupler).

Arms, each test, device and arm in its own process: A (aligned, level-1 translation as STEINER3), O (STEINER3's
construction + O), AO (aligned + O). Exactness: A is checked on the construction before translation, as STEINER3;
O and AO on the final, translated circuit (so an inexact optimisation would show). Placement is STEINER3's.

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
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), HERE, REPO]
import steiner3 as S3  # noqa: E402  (benchmarks/steiner3.py, committed in 07be729)

DEVICES = S3.DEVICES
ARMS = ("A", "O", "AO")
TIMEOUT, BUDGET = 1200, 7200


# ------------------------------------------------------------------ aligned trees

def aligned_tree(terms_q, adj, dist, state):
    """(root, {child: parent}) spanning terms_q, grown from a root chosen by the previous tree, preferring its
    couplers. state = (previous root, set of the previous tree's undirected couplers)."""
    prev_root, prev_edges = state
    ts = list(dict.fromkeys(terms_q))
    if prev_root in ts:
        root = prev_root
    elif prev_root is None:
        root = ts[0]
    else:
        root = min(ts, key=lambda q: (dist[prev_root].get(q, 10 ** 6), q))
    if len(ts) == 1:
        return root, {}
    tree, edges, remaining = {root}, set(), set(ts) - {root}
    while remaining:
        best = {v: (0, 0) for v in tree}
        back, heap, found = {}, [(0, 0, v) for v in sorted(tree)], None
        while heap:
            d, h, u = heapq.heappop(heap)
            if (d, h) > best[u]:
                continue
            if u in remaining:
                found = u
                break
            for w in sorted(adj[u]):
                nd, nh = d + (1 if (min(u, w), max(u, w)) in prev_edges else 2), h + 1
                if (nd, nh) < best.get(w, (math.inf, math.inf)):
                    best[w], back[w] = (nd, nh), u
                    heapq.heappush(heap, (nd, nh, w))
        if found is None:
            return None
        path, x = [found], found
        while x not in tree:
            x = back[x]
            path.append(x)
        for a, b in zip(path, path[1:]):
            edges.add((min(a, b), max(a, b)))
        tree.update(path)
        remaining -= tree
    nbr = {n: set() for n in tree}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    parent, dq = {root: None}, deque([root])
    while dq:
        u = dq.popleft()
        for w in sorted(nbr[u]):
            if w not in parent:
                parent[w] = u
                dq.append(w)
    term, changed = set(ts), True
    while changed:  # prune non-terminal leaves (none are expected: every path ends at a terminal)
        changed = False
        kids = {}
        for c, p in parent.items():
            if p is not None:
                kids.setdefault(p, []).append(c)
        for c in list(parent):
            if c not in term and not kids.get(c) and parent[c] is not None:
                del parent[c]
                changed = True
    return root, {c: p for c, p in parent.items() if p is not None}


def gadget_ops_aligned(label, coeff, t, phys, adj, dist, state):
    """STEINER3's gadget with an aligned tree; returns (ops, new state)."""
    sup = [(j, ch) for j, ch in enumerate(reversed(label)) if ch != "I"]
    if not sup:
        return [("gphase", (), (-t * coeff,))], state
    pre, post = [], []
    for j, ch in sup:
        q = phys[j]
        if ch == "X":
            pre.append(("h", (q,), ()))
            post.append(("h", (q,), ()))
        elif ch == "Y":
            pre += [("sdg", (q,), ()), ("h", (q,), ())]
            post += [("h", (q,), ()), ("s", (q,), ())]
    terms_q = [phys[j] for j, _ in sup]
    r = aligned_tree(terms_q, adj, dist, state)
    if r is None:
        raise RuntimeError("a term's qubits are not connected on the working couplers")
    root, parent = r
    kids = {}
    for c, p in parent.items():
        kids.setdefault(p, []).append(c)
    term = set(terms_q)
    collect = []

    def visit(v):
        p = parent.get(v)
        if p is not None and v not in term:
            collect.append(("cx", (v, p), ()))
        for c in sorted(kids.get(v, [])):
            visit(c)
        if p is not None:
            collect.append(("cx", (v, p), ()))
    for c in sorted(kids.get(root, [])):
        visit(c)
    new_state = (root, {(min(c, p), max(c, p)) for c, p in parent.items()}) if parent else state
    return pre + collect + [("rz", (root,), (2 * t * coeff,))] + collect[::-1] + post, new_state


def synthesise(qc, target, aligned, reading):
    """STEINER3's synthesise(), with the aligned gadgets when `aligned`."""
    s = S3.synthesise(qc, target, reading=reading)
    if isinstance(s, str) or not aligned:
        return s
    _, phys, meas, cost, evo = s
    from qiskit import QuantumCircuit
    adj, _ = S3.working_graph(target)
    dist, _ = S3.bfs_all(adj)
    out = QuantumCircuit(target.num_qubits, qc.num_clbits)
    gphase, state = 0.0, (None, set())
    for lb, c, t in S3._variants(evo.operation)[reading]:
        ops, state = gadget_ops_aligned(lb, c, t, phys, adj, dist, state)
        for name, q, prm in ops:
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
    return out, phys, meas, cost, evo


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    from qiskit import transpile
    qc, bp_backend = B.build(a.bp, kind, arg)
    backend = S3.device(a.device, bp_backend)
    target = backend.target
    rec = dict(stratum=stratum, test=tid, device=a.device, arm=a.arm, input_qubits=qc.num_qubits)
    reading = S3.calibrate()
    if reading is None:
        print(json.dumps(dict(rec, error="no reading of PauliEvolutionGate matches Qiskit's Operator")), flush=True)
        return
    t0 = time.perf_counter()
    s = synthesise(qc, target, aligned=a.arm in ("A", "AO"), reading=reading)
    if isinstance(s, str):
        print(json.dumps(dict(rec, eligible=False, why=s)), flush=True)
        return
    phys_circ, phys, meas, cost, evo = s
    rec["t_construct"] = round(time.perf_counter() - t0, 3)
    if a.arm == "A":
        out = transpile(phys_circ, target=target, layout_method="trivial", routing_method="none",
                        optimization_level=1, seed_transpiler=0)
    else:
        out = transpile(phys_circ, target=target, initial_layout=list(range(target.num_qubits)),
                        routing_method="none", optimization_level=3, seed_transpiler=0)
    rec["t"] = round(time.perf_counter() - t0, 3)
    lay = out.layout
    moved = lay is not None and (list(lay.final_index_layout()) != list(range(target.num_qubits)))
    rec.update(eligible=True, n=len(phys), terms=len(evo.operation.operator), moved=moved)
    rec["q2"] = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "delay"))
    rec["esp"], rec["on_failed"] = S3.esp_of(out, target)
    try:
        rec["exact"] = ("layout moved" if moved else
                        S3.check(qc, evo, phys, phys_circ if a.arm == "A" else out))
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
        small = [t for t in ts if "enc_unary_dvalues_4-4" in t[1] or "ham_JW-8" in t[1] or "JW8" in t[1]]
        ts = (small or ts)[:2]
    jobs = sorted((hashlib.sha256(("STEINER4|" + t[1] + d + arm).encode()).hexdigest(), t, d, arm)
                  for t in ts for d in DEVICES for arm in ARMS)
    os.makedirs(a.out)
    path = os.path.join(a.out, "steiner4.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par))) + "\n")
    print(f"STEINER4: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
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
                   (f"{r['t']} s, q2 {r['q2']}, exact {r['exact']}, esp {r['esp']}" if r.get("eligible")
                    else f"not eligible: {r['why']}"))
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:5.0f} s] {r['device'][4:]:8s} {r['arm']:2s} "
                  f"{r['test'][-40:]}: {msg}", flush=True)
    report(a)


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "steiner4.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    base = os.path.dirname(os.path.abspath(a.out))
    other = {}
    for f, arms in (("esp_ft/esp_ft.jsonl", ("QK2", "QK3", "PSFR")), ("esp_c32/esp_c32.jsonl", ("C32D",)),
                    ("steiner3/steiner3.jsonl", ("ST",))):
        p = os.path.join(base, f)
        if os.path.exists(p):
            for x in [json.loads(y) for y in open(p, encoding="utf-8")][1:]:
                if x.get("arm") in arms and "error" not in x and x.get("eligible", True):
                    other[(x["device"], x["test"], "S3" if x["arm"] == "ST" else x["arm"])] = x
    mine = {(r["device"], r["test"], r["arm"]): r for r in recs if r.get("eligible")}
    allr = {**other, **mine}
    lg = lambda r: -math.inf if r.get("esp") is None else r["esp"]  # noqa: E731
    gm = lambda xs: math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else float("nan")  # noqa: E731
    L = ["# STEINER4 (exploratory)", "",
         f"git head {meta['git_head']}; {meta['tests']} HamLib FakeTorino tests on {', '.join(DEVICES)}; "
         f"{meta['jobs']} jobs (arms {', '.join(ARMS)}). Others: ESP-FT (QK2, QK3, PSFR), ESP-C32 (C32D), "
         "STEINER3 (S3).", ""]
    for d in DEVICES:
        L += [f"## {d}", ""]
        for arm in ARMS:
            rs = [r for r in recs if r["device"] == d and r["arm"] == arm and r.get("eligible")]
            bad = [r for r in rs if r["exact"] not in (True, None)]
            L.append(f"- {arm}: {len(rs)} eligible; exact {sum(1 for r in rs if r['exact'] is True)}, not checked "
                     f"{sum(1 for r in rs if r['exact'] is None)}, NOT exact {len(bad)}"
                     + (f" ({'; '.join(str(r['exact'])[:40] for r in bad[:3])})" if bad else "")
                     + f"; on failed elements {sum(1 for r in rs if r['on_failed'])}; errors "
                     f"{sum(1 for r in recs if r['device'] == d and r['arm'] == arm and 'error' in r)}; "
                     f"time {sum(r['t'] for r in rs):.1f} s summed")
        L += ["", "| arm / other | q2 ratio (gmean, +1) | fewer / equal / more two-qubit gates | ESP ratio (gmean, both > 0; "
              "best >= 0.01) | ESP 10% better / worse |", "|---|---|---|---|---|"]
        for arm in ARMS:
            for o in ("S3", "QK2", "QK3", "PSFR", "C32D"):
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
        L += ["", "| test | n | terms | S3 | A | O | AO | QK2 | QK3 | PSFR | AO log10 ESP | QK3 log10 ESP | AO s | QK3 s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        g = lambda t, o, k: allr.get((d, t, o), {}).get(k, "-")  # noqa: E731
        tests_d = sorted({k[1] for k in mine if k[0] == d}, key=lambda t: (g(t, "AO", "n") if g(t, "AO", "n") != "-"
                                                                          else 0, t))
        for t in tests_d:
            L.append(f"| {t.split('[')[-1].rstrip(']')[:40]} | {g(t, 'AO', 'n')} | {g(t, 'AO', 'terms')} | "
                     f"{g(t, 'S3', 'q2')} | {g(t, 'A', 'q2')} | {g(t, 'O', 'q2')} | {g(t, 'AO', 'q2')} | "
                     f"{g(t, 'QK2', 'q2')} | {g(t, 'QK3', 'q2')} | {g(t, 'PSFR', 'q2')} | {g(t, 'AO', 'esp')} | "
                     f"{g(t, 'QK3', 'esp')} | {g(t, 'AO', 't')} | {g(t, 'QK3', 't')} |")
        L += [f"- error ({r['arm']}, {r['test'][-40:]}): {r['error'][:160]}" for r in recs
              if r["device"] == d and "error" in r]
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "steiner4.md"), "w", encoding="utf-8", newline="\n").write(txt)
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
