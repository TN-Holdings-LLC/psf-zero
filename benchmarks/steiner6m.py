"""steiner6m.py -- STEINER6 (2026-10-11; exploratory, local only): compiling a HamLib Pauli evolution by construction
with the qubits MOVED during the construction. STEINER3-5 fixed the placement once; a term whose qubits are far apart
paid "bridge" CNOTs every time. Here the construction keeps a current placement and, term by term, either runs a term
on it or moves qubits with a SWAP first, deciding from the terms still to come (they are all known in advance).

Exactness: terms are run in an order that only exchanges terms that commute (a term never passes an earlier
anticommuting term), so the circuit equals Qiskit's product formula of the gate exactly, up to the final placement
of the logical qubits, which the output records.

The construction, with the current placement pi (logical -> physical):
  - eligible terms: within a window of the next W unrun terms, those that commute with every earlier unrun term;
  - a term's cost under pi: STEINER3's Steiner-tree CNOT count on the device's working couplers; its excess: cost
    minus 2 (k - 1), the cost if its k qubits were connected;
  - every eligible term with excess 0 (and every single-qubit term) is run at once;
  - otherwise a SWAP on a working coupler next to a qubit of the first F eligible terms is chosen to lower
    H = sum of their excess + 0.5 x the mean excess of the next E eligible ones; a SWAP on the two qubits of an
    eligible two-qubit term is merged with it (the term runs, then the SWAP; Qiskit's level 3 fuses the two into one
    block of at most 3 CNOTs), so it costs 1 CNOT more than the term instead of 3;
  - if no SWAP lowers H by more than its cost, the eligible term of the least excess is run with its bridges.
Then Qiskit level 3 on the fixed layout with no routing (STEINER4's arm O), for the translation and the fusing.

    cd <psf-zero repository>
    python benchmarks/steiner6m.py one --bp <benchpress> --device FakeTorino --test SUBSTR [--qk3]
    python benchmarks/steiner6m.py run --bp <benchpress> --out DIR --qk3 [--devices ...] [--par 2]
Settings (environment, as in run v6): S6_ALGO=v4, S6_ORDER=chain3, S6_LAMBDA=0.1, S6_MU=0.03, S6_F=12, S6_PASSES=1,
S6_EMBED=2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
BP = os.environ.get("BENCHPRESS", "")  # a Benchpress clone: --bp, or the environment variable
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
import steiner3 as S3  # noqa: E402

W, F, E = 64, int(os.environ.get("S6_F", "12")), 24
ALGO = os.environ.get("S6_ALGO", "v4")
ORDER = os.environ.get("S6_ORDER", "chain3")
PASSES = int(os.environ.get("S6_PASSES", "1"))
EMBED = int(os.environ.get("S6_EMBED", "2"))
BIG_TERMS = 2000  # VF2 embeddings (STEINER5) added per mode as initial placements
CHECK_N, CHECK_M = 12, 22


# ------------------------------------------------------------------ terms

def term_masks(label):
    """(x mask, z mask, support) of a label; character j from the right acts on logical qubit j."""
    x = z = 0
    sup = []
    for j, ch in enumerate(reversed(label)):
        if ch == "I":
            continue
        sup.append(j)
        if ch in "XY":
            x |= 1 << j
        if ch in "ZY":
            z |= 1 << j
    return x, z, tuple(sup)


def commute(a, b):
    return (bin((a[0] & b[1]) ^ (a[1] & b[0])).count("1") & 1) == 0


# ------------------------------------------------------------------ the construction

class Mover:
    def __init__(self, n, triples, adj, dist, par, pi0, w=W, f=F, e=E, err=None):
        """err: None (hop counts only) or (error distances, their parents, coupler weight -ln(1 - e) by sorted pair):
        bridged terms then take their trees on the error distances, and a SWAP's cost is scaled by its coupler's
        weight over the median coupler's."""
        self.n, self.adj, self.dist, self.par = n, adj, dist, par
        self.err = err
        self._memo = {}
        self._pmemo = {}
        if err is not None:
            ws = sorted(err[2].values())
            self.w_med = ws[len(ws) // 2] if ws else 1.0
        self.triples = triples
        self.masks = [term_masks(lb) for lb, _, _ in triples]
        self.pi = list(pi0)
        self.inv = {p: i for i, p in enumerate(self.pi)}
        self.w, self.f, self.e = w, f, e
        self.ops = []  # (name, physical qubits, params)
        self.gphase = 0.0
        self.stats = dict(swaps=0, merged_swaps=0, bridged_terms=0, bridge_excess=0, terms=0)
        self.order = []
        self.last2q = {}  # physical qubit -> the pair of the last two-qubit gate on it (single-qubit gates join blocks)

    # costs ---------------------------------------------------------
    def cost(self, j, pi=None):
        pi = self.pi if pi is None else pi
        sup = self.masks[j][2]
        if len(sup) <= 1:
            return 0
        key = tuple(sorted(pi[q] for q in sup))
        c = self._memo.get(key)
        if c is None:
            r, p = S3.steiner_tree(list(key), self.dist, self.par)
            c = self._memo[key] = S3.tree_cost(r, p, set(key))
        return c

    def excess(self, j, pi=None):
        k = len(self.masks[j][2])
        return 0 if k <= 1 else self.cost(j, pi) - 2 * (k - 1)

    def proxy_excess(self, j, pi):
        """A cheap stand-in for the excess when scoring SWAPs: for two qubits, 2 (d - 1) x 2 (a path of d hops costs
        2 d + 2 (d - 1) CNOTs); for more, the same on the minimum spanning tree of their hop distances."""
        sup = self.masks[j][2]
        k = len(sup)
        if k <= 1:
            return 0
        phys = [pi[q] for q in sup]
        if k == 2:
            d = self.dist[phys[0]][phys[1]]
            return 4 * (d - 1)
        key = tuple(sorted(phys))
        c = self._pmemo.get(key)
        if c is not None:
            return c
        c = self._pmemo[key] = self._mst_excess(list(key))
        return c

    def _mst_excess(self, phys):
        inside = {phys[0]}
        rest = set(phys[1:])
        tot = 0
        best = {v: self.dist[phys[0]][v] for v in rest}
        while rest:
            v = min(rest, key=lambda u: (best[u], u))
            tot += best[v] - 1
            rest.discard(v)
            inside.add(v)
            for u in rest:
                du = self.dist[v][u]
                if du < best[u]:
                    best[u] = du
        return 4 * tot

    # emission ------------------------------------------------------
    def run_term(self, j):
        lb, c, t = self.triples[j]
        ex = self.excess(j)
        dist, par = (self.dist, self.par) if (self.err is None or ex == 0) else (self.err[0], self.err[1])
        for name, q, prm in S3.gadget_ops(lb, c, t, self.pi, dist, par):
            if name == "gphase":
                self.gphase += prm[0]
            else:
                self.ops.append((name, q, prm))
                if len(q) == 2:
                    pair = frozenset(q)
                    self.last2q[q[0]] = self.last2q[q[1]] = pair
        if ex > 0:
            self.stats["bridged_terms"] += 1
            self.stats["bridge_excess"] += ex
        self.stats["terms"] += 1
        self.order.append(j)

    def mergeable(self, p, q):
        """A SWAP on (p, q) joins the block of the last two-qubit gates there: Qiskit's level 3 fuses them into one
        two-qubit unitary of at most 3 CNOTs, so the SWAP adds at most 1 CNOT to a block that had 2 (a ZZ term)."""
        pair = frozenset((p, q))
        return self.last2q.get(p) == pair and self.last2q.get(q) == pair

    def swap(self, p, q):
        if self.mergeable(p, q):
            self.stats["merged_swaps"] += 1
        self.ops.append(("swap", (p, q), ()))
        self.last2q[p] = self.last2q[q] = frozenset((p, q))
        a, b = self.inv.get(p), self.inv.get(q)
        if a is not None:
            self.pi[a] = q
        if b is not None:
            self.pi[b] = p
        self.inv = {}
        for i, v in enumerate(self.pi):
            self.inv[v] = i
        self.stats["swaps"] += 1

    # scheduling ----------------------------------------------------
    def build(self):
        T = len(self.triples)
        unrun = list(range(T))  # in order
        pos = 0
        guard = 0
        stuck_swaps = 0
        while unrun:
            guard += 1
            if guard > 50 * T + 10000:
                raise RuntimeError("no progress")
            window = unrun[: self.w]
            elig = []
            for idx, j in enumerate(window):
                mj = self.masks[j]
                if all(commute(self.masks[i], mj) for i in window[:idx]):
                    elig.append(j)
            # run every eligible term that needs no bridge
            free = [j for j in elig if self.excess(j) == 0]
            if free:
                for j in free:
                    self.run_term(j)
                done = set(free)
                unrun = [j for j in unrun if j not in done]
                stuck_swaps = 0
                continue
            front = elig[: self.f]
            ext = elig[self.f: self.f + self.e]
            # beyond the window: the next terms in order, as extended set when the window is small
            h0 = self.score(front, ext, self.pi)
            best = None
            cand = set()
            for j in front:
                for q in self.masks[j][2]:
                    p = self.pi[q]
                    for v in self.adj[p]:
                        cand.add((min(p, v), max(p, v)))
            for (p, v) in sorted(cand):
                pi2 = list(self.pi)
                a, b = self.inv.get(p), self.inv.get(v)
                if a is not None:
                    pi2[a] = v
                if b is not None:
                    pi2[b] = p
                merged = self.mergeable(p, v)
                c_swap = 1 if merged else 3
                if self.err is not None:
                    c_swap *= self.err[2].get((p, v), self.w_med) / self.w_med
                gain = h0 - self.score(front, ext, pi2)
                key = (gain - c_swap, merged, -p, -v)
                if best is None or key > best[0]:
                    best = (key, p, v, merged, gain, c_swap)
            min_ex = min((self.excess(j), j) for j in front)
            if best is not None and best[4] > best[5] and stuck_swaps < 4 * self.n:
                _, p, v, merged, _, _ = best
                self.swap(p, v)
                stuck_swaps += 1
                continue
            # no SWAP pays: run the cheapest eligible term with its bridges
            j = min_ex[1]
            self.run_term(j)
            unrun = [x for x in unrun if x != j]
            stuck_swaps = 0
        return self

    def score(self, front, ext, pi):
        h = sum(self.proxy_excess(j, pi) for j in front)
        if ext:
            h += 0.5 * sum(self.proxy_excess(j, pi) for j in ext) / len(ext) * len(front) / max(1, len(front))
        return h


class Mover2(Mover):
    """STEINER6 v4: every unrun term that commutes with all earlier unrun terms is eligible (not only those in a
    window); the anticommutation between terms is computed once. A SWAP is scored on every eligible term that
    contains one of the two logical qubits it moves (weight 1) and on the blocked ones (weight LAMBDA), by the change
    of their proxy excess; it is made if that gain exceeds its cost (3 CNOTs, 1 when it joins the last block on the
    pair, scaled by the coupler's error weight in the error mode). Otherwise the eligible term of the least excess
    (earliest on a tie) runs with its bridges."""
    LAMBDA = float(os.environ.get("S6_LAMBDA", "0.1"))
    LOOK = 200
    MU = float(os.environ.get("S6_MU", "0.03"))

    def build(self):
        import numpy as np
        T = len(self.triples)
        n = self.n
        X = np.zeros((T, n), dtype=np.uint8)
        Z = np.zeros((T, n), dtype=np.uint8)
        for j, (x, z, sup) in enumerate(self.masks):
            for q in sup:
                X[j, q] = (x >> q) & 1
                Z[j, q] = (z >> q) & 1
        succ = [[] for _ in range(T)]
        blocked = [0] * T
        B = 512
        for i0 in range(0, T, B):
            a = ((X[i0:i0 + B].astype(np.int32) @ Z.T.astype(np.int32)) + (Z[i0:i0 + B].astype(np.int32)
                 @ X.T.astype(np.int32))) & 1
            for ii in range(a.shape[0]):
                i = i0 + ii
                later = np.nonzero(a[ii, i + 1:])[0] + i + 1
                succ[i] = later.tolist()
                for j in succ[i]:
                    blocked[j] += 1
        terms_of = [[] for _ in range(n)]
        for j, (_, _, sup) in enumerate(self.masks):
            for q in sup:
                terms_of[q].append(j)
        unrun = [True] * T
        elig = set(j for j in range(T) if blocked[j] == 0)
        ex = {j: self.excess(j) for j in elig}
        left = T
        stuck = 0
        guard = 0
        ptr = 0
        while left:
            guard += 1
            if guard > 20 * T + 100000:
                raise RuntimeError("no progress")
            free = sorted(j for j in elig if ex[j] == 0)
            if free:
                if ORDER == "chain3":
                    small = [k for k in free if len(self.masks[k][2]) <= 2]
                    if small:  # terms on one or two qubits: in the given order, all at once, as "index"
                        for k in small:
                            self._run(k, unrun, elig, ex, succ, blocked)
                            left -= 1
                    else:  # terms on 3 or more qubits commute pairwise: the most similar to the last one next
                        j = max(free, key=lambda k: (self.similar(self.last_term, k), -k))
                        self._run(j, unrun, elig, ex, succ, blocked)
                        left -= 1
                elif ORDER == "chain":
                    j = max(free, key=lambda k: (self.similar(self.last_term, k), -k))
                    self._run(j, unrun, elig, ex, succ, blocked)
                    left -= 1
                else:
                    for j in free:
                        self._run(j, unrun, elig, ex, succ, blocked)
                        left -= 1
                stuck = 0
                continue
            # the blocked terms looked at: the next LOOK unrun ones in the given order
            while ptr < T and not unrun[ptr]:
                ptr += 1
            look = set()
            k = ptr
            while k < T and len(look) < self.LOOK:
                if unrun[k] and k not in elig:
                    look.add(k)
                k += 1
            # SWAP candidates: couplers next to the qubits of the eligible terms of least excess
            order = sorted(elig, key=lambda j: (ex[j], j))
            focus = order[: self.f]
            cand = set()
            for j in focus:
                for q in self.masks[j][2]:
                    p = self.pi[q]
                    for v in self.adj[p]:
                        cand.add((min(p, v), max(p, v)))
            best = None
            for (p, v) in sorted(cand):
                a, b = self.inv.get(p), self.inv.get(v)
                pi2 = list(self.pi)
                if a is not None:
                    pi2[a] = v
                if b is not None:
                    pi2[b] = p
                touched = set()
                for lq in (a, b):
                    if lq is not None:
                        touched.update(terms_of[lq])
                gain = 0.0
                fset = set(focus)
                for j in touched:
                    if not unrun[j] or not (j in elig or j in look):
                        continue
                    wgt = 1.0 if j in fset else (self.MU if j in elig else self.LAMBDA)
                    gain += wgt * (self.proxy_excess(j, self.pi) - self.proxy_excess(j, pi2))
                merged = self.mergeable(p, v)
                c_swap = 1 if merged else 3
                if self.err is not None:
                    c_swap *= self.err[2].get((p, v), self.w_med) / self.w_med
                key = (gain - c_swap, merged, -p, -v)
                if best is None or key > best[0]:
                    best = (key, p, v, a, b)
            if best is not None and best[0][0] > 0 and stuck < 4 * n:
                _, p, v, a, b = best
                self.swap(p, v)
                for lq in (a, b):
                    if lq is not None:
                        for j in terms_of[lq]:
                            if j in elig:
                                ex[j] = self.excess(j)
                stuck += 1
                continue
            j = order[0]
            self._run(j, unrun, elig, ex, succ, blocked)
            left -= 1
            stuck = 0
        return self

    last_term = None

    def similar(self, i, k):
        """Qubits on which terms i and k carry the same Pauli, minus half those where they differ (0 for no i)."""
        if i is None:
            return 0
        xi, zi, si = self.masks[i]
        xk, zk, sk = self.masks[k]
        both = set(si) & set(sk)
        same = sum(1 for q in both if ((xi >> q) & 1) == ((xk >> q) & 1) and ((zi >> q) & 1) == ((zk >> q) & 1))
        return same - 0.5 * (len(both) - same)

    def _run(self, j, unrun, elig, ex, succ, blocked):
        self.last_term = j
        self.run_term(j)
        unrun[j] = False
        elig.discard(j)
        ex.pop(j, None)
        for k in succ[j]:
            blocked[k] -= 1
            if blocked[k] == 0 and unrun[k]:
                elig.add(k)
                ex[k] = self.excess(k)


# ------------------------------------------------------------------ test harness

def device(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def synthesise(qc, target, modes=("H", "E")):
    """(physical circuit, pi0, pi_final, measurements, stats, evo) or a reason (str)."""
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
    eq = [qc.find_bit(q).index for q in evo.qubits]
    n = len(eq)
    if eq != list(range(n)):
        return "the evolution is not on qubits 0..n-1 in order"
    adj, _ = S3.working_graph(target)
    if len(adj) < n:
        return "not enough working qubits"
    dist, par = S3.bfs_all(adj)
    triples = S3._variants(evo.operation)["given"]
    terms = [(lb, c) for lb, c, _ in triples]
    import steiner5 as S5
    derr, perr, wedge = S5.error_distances(adj, target)
    readout = {}
    if "measure" in target.operation_names:
        for qa, pr in target["measure"].items():
            if qa is not None and pr is not None and pr.error is not None:
                readout[qa[0]] = -math.log(1.0 - min(pr.error, 0.999))
    w_med = sorted(wedge.values())[len(wedge) // 2]
    starts_h = sorted(adj, key=lambda v: (-len(adj[v]), v))[:2]
    nb = {v: sum(wedge.get((min(v, u), max(v, u)), w_med) for u in adj[v]) / max(1, len(adj[v])) for v in adj}
    starts_e = sorted((v for v in adj if len(adj[v]) >= 2), key=lambda v: (nb[v], v))[:2]
    cands = []
    inits = {"H": [("greedy", S3.placement(n, terms, adj, dist, s0)) for s0 in starts_h],
             "E": [("greedy", S3.placement(n, terms, adj, derr, s0)) for s0 in starts_e]}
    if EMBED:
        S5.VF2_CHECK_LIMIT, S5.VF2_MAPS, S5.KEEP = 20_000, 200, 6
        for mode, (dd, pp, ew) in (("H", (dist, par, None)), ("E", (derr, perr, wedge))):
            try:
                emb, _, _ = S5.embeddings(n, terms, adj, dd)
            except Exception:  # noqa: BLE001 - no embedding: the greedy placements remain
                emb = []
            scored = sorted(((S5.placement_cost(e, terms, dd, pp, ew), k) for k, e in enumerate(emb)))[:EMBED]
            inits[mode] += [("vf2", emb[k]) for _, k in scored]
    if len(triples) > BIG_TERMS:  # large: the error mode only, its first greedy start and best embedding
        modes = ("E",)
        inits["E"] = inits["E"][:1] + [x for x in inits["E"] if x[0] == "vf2"][:1]
    for mode in modes:
        for s0, pi0 in inits[mode]:
            cls = Mover2 if ALGO == "v4" else Mover
            errw = None if mode == "H" else (derr, perr, wedge)
            for _ in range(PASSES - 1):  # Sabre's idea: the placement the construction ends in, run backwards
                fwd = cls(n, triples, adj, dist, par, pi0, err=errw).build()
                bwd = cls(n, triples[::-1], adj, dist, par, fwd.pi, err=errw).build()
                pi0 = list(bwd.pi)
            m = cls(n, triples, adj, dist, par, pi0, err=errw).build()
            # estimated -ln ESP of the two-qubit part and the readout: CNOTs and SWAPs (3, or 1 merged) by coupler
            est = 0.0
            for name, q, _ in m.ops:
                if len(q) == 2:
                    est += wedge.get((min(q), max(q)), w_med) * (1 if name == "cx" else 3)
            est -= 2 * w_med * m.stats["merged_swaps"]
            est += sum(readout.get(m.pi[q], 0.0) for q, _ in meas)
            cands.append((est, mode, s0, m, pi0))
    best = min(cands, key=lambda c: (c[0], c[1], c[2]))
    _, mode, s0, m, pi0 = best
    m.stats["mode"] = mode
    m.stats["init"] = s0
    best = (None, m, pi0)
    _, m, pi0 = best
    out = QuantumCircuit(target.num_qubits, qc.num_clbits)
    for name, q, prm in m.ops:
        if name == "cx":
            out.cx(*q)
        elif name == "swap":
            out.swap(*q)
        elif name == "rz":
            out.rz(prm[0], q[0])
        else:
            getattr(out, name)(q[0])
    out.global_phase = m.gphase
    for q, cb in meas:
        out.measure(m.pi[q], cb)
    return out, list(pi0), list(m.pi), meas, m.stats, evo


def check(evo, pi0, pif, out):
    """True if out maps a random product state placed at pi0 (|0> elsewhere) to the product formula's state, read at
    pif; a message otherwise; None if too large."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    n = len(pi0)
    touched = sorted({out.find_bit(b).index for ins in out.data if ins.operation.name != "measure"
                      for b in ins.qubits} | set(pi0) | set(pif))
    if n > CHECK_N or len(touched) > CHECK_M:
        return None
    rng = random.Random(71)
    ang = [(rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)) for _ in range(n)]
    ref = QuantumCircuit(n)
    for i, (a, b) in enumerate(ang):
        ref.ry(a, i)
        ref.rz(b, i)
    ref.compose(evo.operation.definition, list(range(n)), inplace=True)
    sv_ref = Statevector(ref).data
    m = len(touched)
    k = {p: j for j, p in enumerate(touched)}
    small = QuantumCircuit(m)
    for i, (a, b) in enumerate(ang):
        small.ry(a, k[pi0[i]])
        small.rz(b, k[pi0[i]])
    for ins in out.data:
        if ins.operation.name == "measure":
            continue
        small.append(ins.operation, [k[out.find_bit(b).index] for b in ins.qubits])
    small.global_phase = out.global_phase
    sv = Statevector(small).data.reshape([2] * m)
    idx = [0] * m
    hosts_set = set(pif)
    for j in range(m):
        idx[m - 1 - j] = slice(None) if touched[j] in hosts_set else 0
    red = sv[tuple(idx)]
    hosts = [touched[j] for j in range(m - 1, -1, -1) if touched[j] in hosts_set]
    logical_of = {p: i for i, p in enumerate(pif)}
    perm = [n - 1 - logical_of[h] for h in hosts]
    red = np.transpose(red, np.argsort(perm)).reshape(-1)
    ov = abs(np.vdot(sv_ref, red))
    return True if abs(ov - 1.0) < 1e-8 else f"overlap {ov:.12f}"


def relabel(out, target):
    """Qiskit's exact VF2PostLayout on the output, applied by renaming qubits (only ESP is read from the result; the
    exactness check runs on the circuit before translation). (circuit, applied?)."""
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler.passes import VF2PostLayout
    from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason
    dag = circuit_to_dag(out)
    p = VF2PostLayout(target=target, seed=-1, call_limit=3_000_000, max_trials=25_000, strict_direction=True)
    p.run(dag)
    if p.property_set["VF2PostLayout_stop_reason"] != VF2PostLayoutStopReason.SOLUTION_FOUND:
        return out, False
    lay = p.property_set["post_layout"]
    m = {i: lay[dag.qubits[i]] for i in range(len(dag.qubits))}
    new = QuantumCircuit(out.num_qubits, out.num_clbits, global_phase=out.global_phase)
    for ins in out.data:
        new.append(ins.operation, [m[out.find_bit(q).index] for q in ins.qubits],
                   [out.find_bit(c).index for c in ins.clbits])
    return new, True


def q2(out):
    return sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "delay"))


def one(a):
    warnings.simplefilter("ignore")
    import bp_mock as B
    from qiskit import transpile
    from qiskit.transpiler import generate_preset_pass_manager
    stratum, tid, kind, arg = json.loads(a.job)
    qc, _ = B.build(BP, kind, arg)
    backend = device(a.device)
    target = backend.target
    rec = dict(test=tid, device=a.device)
    t0 = time.perf_counter()
    s = synthesise(qc, target)
    if isinstance(s, str):
        print(json.dumps(dict(rec, eligible=False, why=s)), flush=True)
        return
    phys_circ, pi0, pif, meas, stats, evo = s
    t_build = time.perf_counter() - t0
    out = transpile(phys_circ, target=target, initial_layout=list(range(target.num_qubits)), routing_method="none",
                    optimization_level=3, seed_transpiler=0)
    rec.update(eligible=True, n=len(pi0), terms=len(evo.operation.operator), t_build=round(t_build, 2),
               t=round(time.perf_counter() - t0, 2), stats=stats, pre_q2=q2(phys_circ))
    rec["q2"] = q2(out)
    rec["esp_before_relabel"], _ = S3.esp_of(out, target)
    out, rec["relabelled"] = relabel(out, target)
    rec["esp"], rec["on_failed"] = S3.esp_of(out, target)
    if a.qk3:
        t1 = time.perf_counter()
        o3 = generate_preset_pass_manager(3, backend, seed_transpiler=0).run(qc)
        rec.update(qk3_q2=q2(o3), qk3_t=round(time.perf_counter() - t1, 2))
        rec["qk3_esp"], rec["qk3_failed"] = S3.esp_of(o3, target)
    try:
        rec["exact"] = check(evo, pi0, pif, phys_circ)
    except Exception as exc:  # noqa: BLE001 - recorded
        rec["exact"] = f"check failed: {type(exc).__name__}: {exc}"[:200]
    print(json.dumps(rec), flush=True)


def jobs_for(a):
    ts = S3.tests(BP)
    if a.test:
        ts = [t for t in ts if a.test in t[1]]
    return [(t, d) for t in ts for d in a.devices.split(",")]


def run(a):
    if os.path.exists(a.out):
        sys.exit(f"STOP: {a.out} exists")
    os.makedirs(a.out)
    import shutil
    shutil.copyfile(os.path.abspath(__file__), os.path.join(a.out, "steiner6m_as_run.py"))
    jobs = jobs_for(a)
    path = os.path.join(a.out, "s6.jsonl")
    t_start = time.perf_counter()

    def go(job):
        t, d = job
        try:
            p = subprocess.run([sys.executable, os.path.join(a.out, "steiner6m_as_run.py"), "one", "--device", d,
                                "--job", json.dumps(list(t))] + (["--qk3"] if a.qk3 else []),
                               capture_output=True, text=True, timeout=a.timeout)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            return json.loads(lines[-1]) if lines else dict(test=t[1], device=d, error=(p.stderr or "")[-400:])
        except subprocess.TimeoutExpired:
            return dict(test=t[1], device=d, error=f"timeout {a.timeout} s")

    with ThreadPoolExecutor(a.par) as ex, open(path, "w", encoding="utf-8") as fh:
        n = 0
        for fu in as_completed([ex.submit(go, j) for j in jobs]):
            r = fu.result()
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
            print(f"[{n}/{len(jobs)} {time.perf_counter() - t_start:6.0f} s] {r.get('device')} {r.get('test', '')[-40:]}"
                  f" q2 {r.get('q2')} qk3 {r.get('qk3_q2')} exact {r.get('exact')} {r.get('error', '')[:80]}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("one", "run"))
    ap.add_argument("--device", default="FakeTorino")
    ap.add_argument("--devices", default="FakeTorino,FakeKingston,FakeMarrakesh,FakeFez")
    ap.add_argument("--job")
    ap.add_argument("--test")
    ap.add_argument("--out")
    ap.add_argument("--par", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--qk3", action="store_true")
    ap.add_argument("--bp", help="a Benchpress clone (or the environment variable BENCHPRESS)")
    a = ap.parse_args()
    global BP
    if a.bp:
        BP = os.path.abspath(a.bp)
        os.environ["BENCHPRESS"] = BP  # the run's subprocesses read it
    if not BP:
        sys.exit("STOP: give the Benchpress clone with --bp or BENCHPRESS")
    if a.mode == "one" and a.job is None:
        t = [t for t in S3.tests(BP) if a.test in t[1]][0]
        a.job = json.dumps(list(t))
    {"one": one, "run": run}[a.mode](a)


if __name__ == "__main__":
    main()
