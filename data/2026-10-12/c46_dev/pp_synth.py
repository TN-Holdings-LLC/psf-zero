"""Prototype: SWAP-free compilation of circuits by phase-polynomial regions, synthesised on the device's coupling graph.

The circuit (after translation to cx, swap, diagonal gates and single-qubit gates) is cut into regions of
{cx, swap, x, diagonal gates}; each region is a phase polynomial (terms: parity of the region's input wires, angle) and
a final affine map. A region is synthesised in place on a connected set of physical qubits: each term's parity is
built on one wire along an error-weighted Steiner tree of the wires that sum to it and is not undone (later terms use
what is there), then the region's final linear map is completed by RowCol elimination up to a permutation of the wires
(the logical qubits are relabelled, no gate). Regions are separated by the non-diagonal single-qubit gates, applied in
place. Measurements at the end are placed after a final region in which only the measured parities matter.
"""
import math

# the functions _lin_* and _st_* are taken from the psf_compile module given to compile_pp


def _terms_of(name, params, qs):
    """Diagonal gates as [(qubits, rz angle)] (exp(-i angle/2 Z...)); None if not diagonal or not handled."""
    if name in ("rz",):
        return [((qs[0],), float(params[0]))]
    if name in ("p", "u1"):
        return [((qs[0],), float(params[0]))]
    if name == "t":
        return [((qs[0],), math.pi / 4)]
    if name == "tdg":
        return [((qs[0],), -math.pi / 4)]
    if name == "s":
        return [((qs[0],), math.pi / 2)]
    if name == "sdg":
        return [((qs[0],), -math.pi / 2)]
    if name == "z":
        return [((qs[0],), math.pi)]
    if name == "cz":
        return [((qs[0],), math.pi / 2), ((qs[1],), math.pi / 2), ((qs[0], qs[1]), -math.pi / 2)]
    if name in ("cp", "cu1"):
        t = float(params[0])
        return [((qs[0],), t / 2), ((qs[1],), t / 2), ((qs[0], qs[1]), -t / 2)]
    if name == "crz":
        t = float(params[0])
        return [((qs[1],), t / 2), ((qs[0], qs[1]), -t / 2)]
    if name == "rzz":
        return [((qs[0], qs[1]), float(params[0]))]
    if name in ("id", "delay"):
        return []
    return None


LINEAR = ("cx", "swap", "x")
BASIS = ["cx", "swap", "x", "rz", "p", "t", "tdg", "s", "sdg", "z", "cz", "cp", "crz", "rzz", "h", "sx", "sxdg", "rx",
         "ry", "u"]


class PPSynth:
    def __init__(self, M, n, phys, adj, wedge, pool=32):
        self.M = M  # the psf_compile module
        self.n = n
        self.phys = phys  # wire -> physical qubit
        self.adj = adj
        self.wedge = wedge
        self.pool = pool
        alive = set(phys)
        self.sub = {p: {q for q in adj.get(p, ()) if q in alive} for p in alive}
        self.dist, self.par = M._lin_dijkstra(alive, self.sub, wedge)
        self.ops = []  # physical ("cx", c, t) | ("rz", angle, p) | ("g", op, p) | ("measure", p, clbit)
        self.where = list(range(n))  # logical -> wire

    # ---------------------------------------------------------------- one region
    def region(self, P, sign, terms, partial=None):
        """P[l]: parity (mask over wires) logical l holds; sign[l]; terms {mask: angle}. Emits the region's gates and
        relabels self.where. partial: the logicals whose state matters afterwards (None: all)."""
        n = self.n
        Mrow = [1 << w for w in range(n)]  # wire contents, in the region's input basis
        Minv = [1 << w for w in range(n)]  # rows of the inverse

        def cx(c, t):  # wires
            Mrow[t] ^= Mrow[c]
            for j in range(n):  # inverse: column c ^= column t
                if Minv[j] >> t & 1:
                    Minv[j] ^= 1 << c
            self.ops.append(("cx", self.phys[c], self.phys[t]))

        def coeffs(S):  # wires whose rows sum to S
            v = 0
            j = 0
            s = S
            while s:
                if s & 1:
                    v ^= Minv[j]
                s >>= 1
                j += 1
            return v

        def plan(S):
            T = coeffs(S)
            ws = [w for w in range(n) if T >> w & 1]
            if len(ws) == 1:
                return 0, ws, None
            best = None
            for r in ws[:4]:  # a few roots
                root, parent = self.M._st_steiner_tree([self.phys[r]] + [self.phys[w] for w in ws if w != r],
                                                       self.dist, self.par)
                term = {self.phys[w] for w in ws}
                cost = sum(1 + (c not in term) for c in parent)
                if best is None or cost < best[0]:
                    best = (cost, ws, (root, parent))
            return best

        inv_phys = {p: w for w, p in enumerate(self.phys)}
        pending = dict(terms)
        have = {}
        for S in list(pending):
            if abs(math.remainder(pending[S], 4 * math.pi)) < 1e-12:
                del pending[S]
        while pending:
            hit = [S for S in pending if S in set(Mrow)]
            if hit:
                pos = {m: w for w, m in enumerate(Mrow)}
                for S in sorted(hit):
                    self.ops.append(("rz", pending.pop(S), self.phys[pos[S]]))
                continue
            cands = sorted(pending, key=lambda S: (bin(coeffs(S)).count("1"), S))[: self.pool]
            best = None
            for S in cands:
                p = plan(S)
                if best is None or p[0] < best[0][0]:
                    best = (p, S)
            (cost, ws, tree), S = best
            root, parent = tree
            order = self.M._lin_post_order(root, parent)
            kids = {}
            for c, p in parent.items():
                kids.setdefault(p, []).append(c)
            term = {self.phys[w] for w in ws}
            for c, p in order:  # Steiner vertices: their own rows cancelled
                if c not in term:
                    cx(inv_phys[c], inv_phys[min(kids[c])])
            for c, p in order:
                cx(inv_phys[c], inv_phys[p])
            assert Mrow[inv_phys[root]] == S
            self.ops.append(("rz", pending.pop(S), root))
        # the final linear map, up to a permutation: logical l must end on a wire holding P[l]
        keep = [l for l in range(n) if partial is None or l in partial]
        spec = [(coeffs(P[l]), 0) for l in keep]
        got = self.M._lin_targets(n, spec)
        if got is None:
            raise RuntimeError("no completion")
        rows, place = got
        if any(rows[w] != Mrow and False for w in range(n)):
            pass
        unit = all(rows[w] == 1 << w for w in range(n))
        if not unit:
            cnots = self.M._lin_synth(rows, list(self.phys), self.adj, self.wedge)
            if cnots is None:
                raise RuntimeError("unconnected")
            for c, t in cnots:
                cx(inv_phys[c], inv_phys[t])
        new_where = list(self.where)
        for k, l in enumerate(keep):
            new_where[l] = place[k]
            if sign[l]:
                self.ops.append(("g", "x", self.phys[place[k]]))
        if partial is not None:  # the others: any free wires
            used = {new_where[l] for l in keep}
            free = iter(w for w in range(n) if w not in used)
            for l in range(n):
                if l not in partial:
                    new_where[l] = next(free)
        self.where = new_where


def compile_pp(M, qc, target, region_phys=None, pool=32):
    """The physical circuit (on target.num_qubits qubits) or None when the circuit is not handled."""
    from qiskit import QuantumCircuit, transpile
    n = qc.num_qubits
    try:
        flat = transpile(qc, basis_gates=BASIS, optimization_level=0)
    except Exception:  # noqa: BLE001
        return None
    gates = []
    for ins in flat.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        qs = [flat.find_bit(q).index for q in ins.qubits]
        if name == "measure":
            gates.append(("measure", qs, flat.find_bit(ins.clbits[0]).index, None))
            continue
        if ins.clbits or getattr(ins.operation, "condition", None) is not None:
            return None
        gates.append((name, qs, None, ins.operation))
    adj, _ = M._st_working_graph(target)
    g2 = next(g for g in ("cz", "ecr", "cx") if g in target.operation_names)
    wedge = {}
    for qa, pr in target[g2].items():
        if qa is not None and pr is not None and pr.error is not None:
            k = (min(qa), max(qa))
            wedge[k] = max(wedge.get(k, 0.0), -math.log(1.0 - min(pr.error, 0.999)) + 1e-9)
    if region_phys is None or len(region_phys) != n:
        return None
    syn = PPSynth(M, n, region_phys, adj, wedge, pool)
    # per-qubit queues of gate indices
    queues = [[] for _ in range(n)]
    for i, (name, qs, _, _) in enumerate(gates):
        for q in qs:
            queues[q].append(i)
    head = [0] * n
    done = [False] * len(gates)

    def front(i):
        return all(queues[q][head[q]] == i for q in gates[i][1] if head[q] < len(queues[q]))

    def take(i):
        done[i] = True
        for q in gates[i][1]:
            head[q] += 1

    measured = {}
    while True:
        # a region: block-compatible gates at the front
        P = [1 << syn.where[l] for l in range(n)]
        sign = [0] * n
        terms = {}
        progress = True
        any_gate = False
        while progress:
            progress = False
            for q in range(n):
                while head[q] < len(queues[q]):
                    i = queues[q][head[q]]
                    name, qs, _, op = gates[i]
                    if not front(i):
                        break
                    if name in LINEAR:
                        if name == "cx":
                            P[qs[1]] ^= P[qs[0]]
                            sign[qs[1]] ^= sign[qs[0]]
                        elif name == "swap":
                            P[qs[0]], P[qs[1]] = P[qs[1]], P[qs[0]]
                            sign[qs[0]], sign[qs[1]] = sign[qs[1]], sign[qs[0]]
                        else:
                            sign[qs[0]] ^= 1
                        take(i)
                        progress = any_gate = True
                        continue
                    tt = _terms_of(name, op.params if op is not None else [], qs)
                    if tt is None:
                        break
                    for lq, ang in tt:
                        S, sg = 0, 0
                        for l in lq:
                            S ^= P[l]
                            sg ^= sign[l]
                        if S:
                            terms[S] = terms.get(S, 0.0) + (-ang if sg else ang)
                    take(i)
                    progress = any_gate = True
        rest = [i for i in range(len(gates)) if not done[i]]
        final = all(gates[i][0] == "measure" for i in rest)
        if final:
            partial = {gates[i][1][0] for i in rest} if rest else None
            syn.region(P, sign, terms, partial if partial else None)
            for i in rest:
                syn.ops.append(("measure", syn.phys[syn.where[gates[i][1][0]]], gates[i][2]))
            break
        syn.region(P, sign, terms)
        # separators at the front
        moved = False
        for q in range(n):
            if head[q] < len(queues[q]):
                i = queues[q][head[q]]
                name, qs, cb, op = gates[i]
                if len(qs) == 1 and front(i) and name != "measure":
                    syn.ops.append(("g", op, syn.phys[syn.where[qs[0]]]))
                    take(i)
                    moved = True
        if not moved:
            return None  # a mid-circuit measurement or an unhandled gate
    N = target.num_qubits
    out = QuantumCircuit(N, qc.num_clbits)
    for o in syn.ops:
        if o[0] == "cx":
            out.cx(o[1], o[2])
        elif o[0] == "rz":
            out.rz(o[1], o[2])
        elif o[0] == "measure":
            out.measure(o[1], o[2])
        else:
            op = o[1]
            if isinstance(op, str):
                getattr(out, op)(o[2])
            else:
                out.append(op, [o[2]])
    return out, syn
