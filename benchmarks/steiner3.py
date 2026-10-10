"""steiner3.py -- STEINER3 (2026-10-10; steiner2.py with the reference taken as the circuit Qiskit synthesises for
the gate (its definition, the first-order product formula that the transpiler compiles), not the gate's exact
exponential, which Operator and Statevector use and which no product formula equals; steiner.py with the terms taken in the order and scale Qiskit's own
synthesis of the gate uses, found by `calibrate()` against Qiskit's Operator of a small gate; exploratory, nothing predicted): a first prototype of compiling by construction
instead of by search. A HamLib test (one PauliEvolutionGate, first-order product formula) is synthesised directly
on the device, term by term in the given order, with no routing step:
  - placement: computed once per circuit (greedy, from the terms' co-occurrence and the device's hop distances on its
    working couplers; four starts, the one whose trees are smallest is kept);
  - each term exp(-i t c P): a basis change on its qubits (H for X, S-dagger then H for Y), a CNOT tree along a
    Steiner tree of the device graph that collects the parity of the term's qubits onto one of them (a qubit of the
    tree outside the term is added to its parent once more, so that it cancels), Rz(2 t c) there, then the CNOTs
    and the basis change undone;
  - Qiskit level 1 on the Target with the trivial layout and no routing, for the translation and the cancellation of
    adjacent CNOT pairs.
Every gate acts on a working coupler or qubit by construction. Exactness is checked before the translation on the
tests with at most 12 logical qubits and at most 20 qubits touched: a random product state (and |0> on the other
touched qubits) evolved by the output equals the input circuit's state, up to a global phase.

On BP-FINAL's HamLib FakeTorino tests, on FakeTorino and FakeKingston, each test in its own process (one call, as
ESP-FT timed its arms). The report compares the two-qubit count, ESP and time with ESP-FT's QK2, QK3 and PSFR and
ESP-C32's C32D.

    cd <psf-zero repository>
    python <this file> run --bp <benchpress clone> --out DIR [--par 4] [--smoke]
    python <this file> report --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
HERE = os.path.join(REPO, "benchmarks")
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), HERE, REPO]
DEVICES = ("FakeTorino", "FakeKingston")
FAILED = 0.5
TIMEOUT, BUDGET = 900, 5400
CHECK_N, CHECK_M = 12, 20


def tests(bp):
    import bp_final as F
    return [t for t in F.population(bp) if t[0] == "HamLib, FakeTorino"]


def device(name, bp_backend):
    if name == "FakeTorino":
        return bp_backend
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


def esp_of(out, target):
    lg, n_failed, dead = 0.0, 0, False
    for ins in out.data:
        if ins.operation.name in ("barrier", "delay"):
            continue
        qargs = tuple(out.find_bit(q).index for q in ins.qubits)
        try:
            props = target[ins.operation.name][qargs]
        except KeyError:
            continue
        e = props.error if props is not None and props.error is not None else 0.0
        n_failed += e >= FAILED
        if e >= 1.0:
            dead = True
        else:
            lg += math.log10(1.0 - e)
    return (None if dead else round(lg, 6)), int(n_failed)


# ------------------------------------------------------------------ the device

def working_graph(target):
    """(adjacency of the working couplers, failed qubits): couplers and qubits with error >= FAILED, and qubits whose
    measurement fails, are left out."""
    g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    bad_q = set()
    for name in ("sx", "measure"):
        if name in target.operation_names:
            for qa, p in target[name].items():
                if qa is not None and p is not None and p.error is not None and p.error >= FAILED:
                    bad_q.add(qa[0])
    adj = {q: set() for q in range(target.num_qubits) if q not in bad_q}
    for qa, p in target[g2].items():
        if qa is None or (p is not None and p.error is not None and p.error >= FAILED):
            continue
        a, b = qa
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return adj, bad_q


def bfs_all(adj):
    """Hop distances and BFS parents from every working qubit."""
    dist, par = {}, {}
    for s in adj:
        d, p, dq = {s: 0}, {s: None}, deque([s])
        while dq:
            u = dq.popleft()
            for w in sorted(adj[u]):
                if w not in d:
                    d[w], p[w] = d[u] + 1, u
                    dq.append(w)
        dist[s], par[s] = d, p
    return dist, par


def steiner_tree(terminals, dist, par):
    """An approximate Steiner tree (edges as a parent map rooted at terminals[0]) spanning the terminals: Prim's tree on
    their hop distances, each edge expanded into a shortest path, then a BFS tree of the union with non-terminal
    leaves pruned."""
    ts = list(dict.fromkeys(terminals))
    if len(ts) == 1:
        return ts[0], {}
    inside, nodes, edges = {ts[0]}, {ts[0]}, set()
    while len(inside) < len(ts):
        best = min(((dist[u][v], u, v) for u in inside for v in ts if v not in inside), key=lambda x: x[:3])
        _, u, v = best
        x = v
        while x != u:  # walk v -> u along u's BFS tree
            y = par[u][x]
            edges.add((min(x, y), max(x, y)))
            nodes.update((x, y))
            x = y
        inside.add(v)
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    root, parent, dq = ts[0], {ts[0]: None}, deque([ts[0]])
    while dq:
        u = dq.popleft()
        for w in sorted(adj[u]):
            if w not in parent:
                parent[w] = u
                dq.append(w)
    term = set(ts)
    changed = True
    while changed:  # prune non-terminal leaves
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


def tree_cost(root, parent, terminals):
    steiner = sum(1 for c in parent if c not in terminals)
    return 2 * (len(parent) + steiner)


# ------------------------------------------------------------------ the circuit

def pauli_terms(evo):
    """[(label, coefficient)] in the order of the operator; label character j from the right on the gate's qubit j."""
    op = evo.operation.operator
    return [(lb, float(complex(c).real)) for lb, c in op.to_list()]


def placement(n, terms, adj, dist, start):
    """Greedy: logical qubits in order of their co-occurrence with those already placed, each on the free working
    qubit closest (by co-occurrence-weighted hop distance) to its placed partners."""
    w = {}
    for lb, _ in terms:
        sup = [j for j, ch in enumerate(reversed(lb)) if ch != "I"]
        for i in range(len(sup)):
            for k in range(i + 1, len(sup)):
                a, b = sup[i], sup[k]
                w[(a, b)] = w.get((a, b), 0) + 1
                w[(b, a)] = w.get((b, a), 0) + 1
    deg = [sum(w.get((i, j), 0) for j in range(n)) for i in range(n)]
    order = [max(range(n), key=lambda i: (deg[i], -i))]
    phi, free = {order[0]: start}, set(adj) - {start}
    while len(phi) < n:
        cand = [i for i in range(n) if i not in phi]
        nxt = max(cand, key=lambda i: (sum(w.get((i, j), 0) for j in phi), deg[i], -i))
        best = min(free, key=lambda v: (sum(w.get((nxt, j), 0) * dist[phi[j]].get(v, 10 ** 6) for j in phi),
                                        -len(adj[v]), v))
        phi[nxt] = best
        free.discard(best)
    return [phi[i] for i in range(n)]


def gadget_ops(label, coeff, t, phys, dist, par):
    """The gates of exp(-i t c P) on the device: [(name, qubits, params)]."""
    sup = [(j, ch) for j, ch in enumerate(reversed(label)) if ch != "I"]
    if not sup:
        return [("gphase", (), (-t * coeff,))]
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
    root, parent = steiner_tree(terms_q, dist, par)
    kids = {}
    for c, p in parent.items():
        kids.setdefault(p, []).append(c)
    term = set(terms_q)
    collect = []

    def visit(v):
        p = parent.get(v)
        if p is not None and v not in term:
            collect.append(("cx", (v, p), ()))  # a Steiner qubit's own bit is added twice to its parent: it cancels
        for c in sorted(kids.get(v, [])):
            visit(c)
        if p is not None:
            collect.append(("cx", (v, p), ()))
    for c in sorted(kids.get(root, [])):
        visit(c)
    return pre + collect + [("rz", (root,), (2 * t * coeff,))] + collect[::-1] + post


def _variants(gate):
    """Candidate readings of a PauliEvolutionGate as [(full label, coefficient, time)] in the order applied."""
    op, t = gate.operator, float(gate.time)
    n = gate.num_qubits
    given = [(lb, float(complex(c).real), t) for lb, c in op.to_list()]
    out = {"given": given, "reversed": given[::-1]}
    try:
        from qiskit.synthesis import LieTrotter
        syn = gate.synthesis if gate.synthesis is not None else LieTrotter()
        seq = syn.expand(gate)
        for mapping in ("fwd", "rev"):
            for scale in (1.0, 0.5):
                terms = []
                for item in seq:
                    pl, idx, tau = item[0], list(item[1]), float(item[2])
                    chars = list(pl) if mapping == "fwd" else list(pl)[::-1]
                    full = ["I"] * n
                    for ch, q in zip(chars, idx):
                        full[n - 1 - q] = ch
                    terms.append(("".join(full), tau * scale, 1.0))
                out[f"expand_{mapping}_{scale}"] = terms
    except Exception:  # noqa: BLE001 - this Qiskit has no expand(): the other readings remain
        pass
    return out


def calibrate():
    """The reading of a PauliEvolutionGate that Qiskit's own Operator of it confirms, on a small gate with
    non-commuting terms."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Operator, SparsePauliOp
    op = SparsePauliOp.from_list([("XY", 0.31), ("ZI", -0.52), ("YZ", 0.77), ("IX", 0.13), ("ZZ", 0.41)])
    g = PauliEvolutionGate(op, time=0.9)
    u = Operator(g.definition)  # the product formula Qiskit's transpiler compiles, not the exact exponential
    adj = {0: {1}, 1: {0}}
    dist, par = bfs_all(adj)
    for name, terms in _variants(g).items():
        c = QuantumCircuit(2)
        gp = 0.0
        for lb, coeff, t in terms:
            for gname, q, prm in gadget_ops(lb, coeff, t, [0, 1], dist, par):
                if gname == "gphase":
                    gp += prm[0]
                elif gname == "cx":
                    c.cx(*q)
                elif gname == "rz":
                    c.rz(prm[0], q[0])
                else:
                    getattr(c, gname)(q[0])
        if Operator(c).equiv(u):
            return name
    return None


def synthesise(qc, target, n_starts=4, reading="given"):
    """(physical circuit before translation, placement, measurements) or a reason (str)."""
    from qiskit import QuantumCircuit
    evo, meas = None, []
    for ins in qc.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        if name == "PauliEvolution" and evo is None and not meas:
            evo = ins
        elif name == "measure":
            meas.append((qc.find_bit(ins.qubits[0]).index, qc.find_bit(ins.clbits[0]).index))
        else:
            return f"instruction {name}"
    if evo is None:
        return "no PauliEvolutionGate"
    if isinstance(evo.operation.operator, list):
        return "a list of operators"
    syn = getattr(evo.operation, "synthesis", None)
    if syn is not None and (type(syn).__name__ != "LieTrotter" or getattr(syn, "reps", 1) != 1):
        return f"synthesis {type(syn).__name__}"
    eq = [qc.find_bit(q).index for q in evo.qubits]
    n = len(eq)
    if eq != list(range(n)):
        return "the evolution is not on qubits 0..n-1 in order"
    adj, _ = working_graph(target)
    if len(adj) < n:
        return "not enough working qubits"
    dist, par = bfs_all(adj)
    variants = _variants(evo.operation)
    if reading not in variants:
        return f"reading {reading} not available"
    triples = variants[reading]
    terms = [(lb, c) for lb, c, _ in triples]
    starts = sorted(adj, key=lambda v: (-len(adj[v]), v))[:n_starts]
    best = None
    for s in starts:
        phys = placement(n, terms, adj, dist, s)
        if any(phys[i] not in adj for i in range(n)):
            continue
        cost = 0
        for lb, _ in terms:
            sup = [phys[j] for j, ch in enumerate(reversed(lb)) if ch != "I"]
            if len(sup) > 1:
                if any(dist[sup[0]].get(v) is None for v in sup):
                    cost = None
                    break
                r, p = steiner_tree(sup, dist, par)
                cost += tree_cost(r, p, set(sup))
        if cost is not None and (best is None or cost < best[0]):
            best = (cost, phys)
    if best is None:
        return "no connected placement"
    phys = best[1]
    out = QuantumCircuit(target.num_qubits, qc.num_clbits)
    gphase = 0.0
    for lb, c, t in triples:
        for name, q, prm in gadget_ops(lb, c, t, phys, dist, par):
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
    return out, phys, meas, best[0], evo


def check(qc, evo, phys, out):
    """True if out (before translation) maps a random product state of the logical qubits (|0> elsewhere) to the input
    circuit's state, up to a global phase; a message otherwise; None if too large to check."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    n = len(phys)
    touched = sorted({out.find_bit(b).index for ins in out.data if ins.operation.name != "measure" for b in ins.qubits}
                     | set(phys))
    if n > CHECK_N or len(touched) > CHECK_M:
        return None
    rng = random.Random(70)
    angles = [(rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)) for _ in range(n)]
    ref = QuantumCircuit(n)
    for i, (a, b) in enumerate(angles):
        ref.ry(a, i)
        ref.rz(b, i)
    ref.compose(evo.operation.definition, list(range(n)), inplace=True)  # the product formula, as calibrate()
    sv_ref = Statevector(ref).data
    m = len(touched)
    k = {p: j for j, p in enumerate(touched)}
    small = QuantumCircuit(m)
    for i, (a, b) in enumerate(angles):
        small.ry(a, k[phys[i]])
        small.rz(b, k[phys[i]])
    for ins in out.data:
        if ins.operation.name == "measure":
            continue
        small.append(ins.operation, [k[out.find_bit(b).index] for b in ins.qubits])
    sv = Statevector(small).data.reshape([2] * m)  # axis m-1-j is touched qubit j (little-endian)
    idx = [0] * m
    for j in range(m):
        idx[m - 1 - j] = 0 if touched[j] not in phys else slice(None)
    red = sv[tuple(idx)]  # the remaining axes: touched logical-hosting qubits in descending touched order
    hosts = [touched[j] for j in range(m - 1, -1, -1) if touched[j] in phys]  # axis order of red
    logical_of = {p: i for i, p in enumerate(phys)}
    perm = [n - 1 - logical_of[h] for h in hosts]  # red axis a holds logical qubit logical_of[h]; ref axis n-1-i
    red = np.transpose(red, np.argsort(perm)).reshape(-1)
    ov = abs(np.vdot(sv_ref, red))
    return True if abs(ov - 1.0) < 1e-9 else f"overlap {ov:.12f}"


def one(a):
    warnings.simplefilter("ignore")
    stratum, tid, kind, arg = json.loads(a.job)
    import bp_mock as B
    from qiskit import transpile
    qc, bp_backend = B.build(a.bp, kind, arg)
    backend = device(a.device, bp_backend)
    target = backend.target
    rec = dict(stratum=stratum, test=tid, device=a.device, arm="ST", input_qubits=qc.num_qubits)
    reading = calibrate()
    if reading is None:
        print(json.dumps(dict(rec, error="no reading of PauliEvolutionGate matches Qiskit's Operator")), flush=True)
        return
    rec["reading"] = reading
    t0 = time.perf_counter()
    s = synthesise(qc, target, reading=reading)
    if isinstance(s, str):
        print(json.dumps(dict(rec, eligible=False, why=s)), flush=True)
        return
    phys_circ, phys, meas, cost, evo = s
    out = transpile(phys_circ, target=target, layout_method="trivial", routing_method="none", optimization_level=1,
                    seed_transpiler=0)
    rec["t"] = round(time.perf_counter() - t0, 3)
    rec.update(eligible=True, n=len(phys), terms=len(evo.operation.operator), tree_cnots=cost)
    rec["q2"] = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "delay"))
    rec["esp"], rec["on_failed"] = esp_of(out, target)
    try:
        rec["exact"] = check(qc, evo, phys, phys_circ)
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["exact"] = f"check failed: {type(exc).__name__}: {exc}"[:200]
    print(json.dumps(rec), flush=True)


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    ts = tests(a.bp)
    if a.smoke:
        small = [t for t in ts if "enc_unary_dvalues_4-4" in t[1] or "ham_JW-8" in t[1] or "JW8" in t[1]]
        ts = (small or ts)[:2]
    jobs = sorted((hashlib.sha256(("STEINER3|" + t[1] + d).encode()).hexdigest(), t, d) for t in ts for d in DEVICES)
    os.makedirs(a.out)
    path = os.path.join(a.out, "steiner3.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(meta=dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           git_head=git("rev-parse", "--short", "HEAD"),
                                           dirty_tracked=git("status", "--porcelain", "--untracked-files=no"),
                                           tests=len(ts), jobs=len(jobs), par=a.par))) + "\n")
    print(f"STEINER3: {len(ts)} tests, {len(jobs)} jobs, par {a.par}", flush=True)
    t_start = time.perf_counter()

    def go(job):
        _, t, d = job
        base = dict(stratum=t[0], test=t[1], device=d, arm="ST")
        if time.perf_counter() - t_start > BUDGET:
            return dict(base, error="not started (budget)")
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--device", d,
                                "--job", json.dumps(list(t))], capture_output=True, text=True, timeout=TIMEOUT,
                               cwd=REPO, env=dict(os.environ, PSF_ZERO_REPO=REPO))
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
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:5.0f} s] {r['device'][4:]:8s} "
                  f"{r['test'][-42:]}: {msg}", flush=True)
    report(a)


def report(a):
    lines = [json.loads(x) for x in open(os.path.join(a.out, "steiner3.jsonl"), encoding="utf-8")]
    meta, recs = lines[0]["meta"], lines[1:]
    base = os.path.dirname(os.path.abspath(a.out))
    other = {}
    for f, arms in (("esp_ft/esp_ft.jsonl", ("QK2", "QK3", "PSFR")), ("esp_c32/esp_c32.jsonl", ("C32D",))):
        p = os.path.join(base, f)
        if os.path.exists(p):
            for x in [json.loads(y) for y in open(p, encoding="utf-8")][1:]:
                if x.get("arm") in arms and "error" not in x:
                    other[(x["device"], x["test"], x["arm"])] = x
    lg = lambda r: -math.inf if r.get("esp") is None else r["esp"]  # noqa: E731
    gm = lambda xs: math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else float("nan")  # noqa: E731
    L = ["# STEINER3 (exploratory, nothing predicted)", "",
         f"git head {meta['git_head']}; {meta['tests']} HamLib FakeTorino tests on {', '.join(DEVICES)}; "
         f"{meta['jobs']} jobs. Others from ESP-FT (QK2, QK3, PSFR) and ESP-C32 (C32D).", ""]
    why = {}
    for r in recs:
        if not r.get("eligible") and "why" in r and r["device"] == DEVICES[0]:
            why[r["why"]] = why.get(r["why"], 0) + 1
    L += ["Not eligible (FakeTorino): " + ("; ".join(f"{k}: {v}" for k, v in sorted(why.items())) or "none"), ""]
    for d in DEVICES:
        rs = [r for r in recs if r["device"] == d and r.get("eligible")]
        run_ = [r for r in rs if max([lg(r)] + [lg(other[(d, r["test"], o)]) for o in ("QK2", "QK3", "PSFR", "C32D")
                                                 if (d, r["test"], o) in other]) >= -2]
        L += [f"## {d}", "", f"{len(rs)} eligible; exact {sum(1 for r in rs if r['exact'] is True)}, not checked (too "
              f"large) {sum(1 for r in rs if r['exact'] is None)}, failed check "
              f"{sum(1 for r in rs if r['exact'] not in (True, None))}; on failed elements "
              f"{sum(1 for r in rs if r['on_failed'])}; {len(run_)} where the best ESP >= 0.01.", "",
              "| ST against | q2 ratio (gmean, +1) | fewer / more two-qubit gates (tests) | ESP ratio (gmean, both > 0; "
              "where the best >= 0.01) | ESP 10% better / worse | time: ST / other (summed) |", "|---|---|---|---|---|---|"]
        for o in ("QK2", "QK3", "PSFR", "C32D"):
            pr = [(r, other[(d, r["test"], o)]) for r in rs if (d, r["test"], o) in other]
            if not pr:
                continue
            qr = gm([(r["q2"] + 1) / (x["q2"] + 1) for r, x in pr])
            fe, mo = sum(1 for r, x in pr if r["q2"] < x["q2"]), sum(1 for r, x in pr if r["q2"] > x["q2"])
            pe = [(r, other[(d, r["test"], o)]) for r in run_ if (d, r["test"], o) in other]
            dv = [lg(r) - lg(x) for r, x in pe]
            fin = [v for v in dv if math.isfinite(v)]
            L.append(f"| {o} | {qr:.3f} | {fe} / {mo} | {10 ** (sum(fin) / len(fin)) if fin else float('nan'):.3f} | "
                     f"{sum(1 for v in dv if v >= math.log10(1.1))} / {sum(1 for v in dv if v <= -math.log10(1.1))} | "
                     f"{sum(r['t'] for r, _ in pr):.1f} / {sum(x['t'] for _, x in pr):.1f} |")
        L += ["", "| test | n | terms | ST q2 | QK2 | QK3 | C32D | PSFR | ST log10 ESP | QK3 log10 ESP | exact | ST s | QK3 s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        g = lambda r, o, k: other.get((d, r["test"], o), {}).get(k, "-")  # noqa: E731
        for r in sorted(rs, key=lambda r: (r["n"], r["test"])):
            L.append(f"| {r['test'].split('[')[-1].rstrip(']')[:40]} | {r['n']} | {r['terms']} | {r['q2']} | "
                     f"{g(r, 'QK2', 'q2')} | {g(r, 'QK3', 'q2')} | {g(r, 'C32D', 'q2')} | {g(r, 'PSFR', 'q2')} | "
                     f"{r['esp']} | {g(r, 'QK3', 'esp')} | {r['exact']} | {r['t']} | {g(r, 'QK3', 't')} |")
        L += [f"- error ({r['test'][-40:]}): {r['error'][:160]}" for r in recs if r["device"] == d and "error" in r]
        L.append("")
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "steiner3.md"), "w", encoding="utf-8", newline="\n").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one", "report"))
    ap.add_argument("--bp")
    ap.add_argument("--out")
    ap.add_argument("--device")
    ap.add_argument("--job")
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"run": run, "one": one, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
