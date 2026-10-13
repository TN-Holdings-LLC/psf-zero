"""Prototype (c48 development): STEINER6's mover with a time-aware SWAP cost.

_StMover2 scores a SWAP by the CNOTs it saves against the CNOTs it costs. A SWAP also takes time on its two qubits,
and when one of the logical qubits it moves has much work left (a hub of the interaction graph), that time lands on
the circuit's critical path. This mover keeps the as-soon-as-possible clock of every physical qubit (in CX units, in
emission order) and a lower bound on each logical qubit's remaining work (2 CX units per unrun multi-qubit term on
it). The bound on the makespan is LB = max over logical qubits of (clock of its place + its remaining work). A SWAP
that would raise LB pays KAPPA x the rise, in CNOT units, on top of its CNOT cost."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).



def make(M, kappa=1.0):
    base = M._StMover2

    class _StMover3(base):
        KAPPA = kappa

        def _tick(self):
            clock = self.__dict__.setdefault("_clock", {})
            k = self.__dict__.get("_seen", 0)
            for name, q, _ in self.ops[k:]:
                s = max(clock.get(x, 0.0) for x in q)
                d = 3.0 if name == "swap" else (1.0 if name == "cx" else (0.0 if name == "rz" else 0.3))
                for x in q:
                    clock[x] = s + d
            self._seen = len(self.ops)

        def _run(self, j, unrun, elig, ex, succ, blocked):
            super()._run(j, unrun, elig, ex, succ, blocked)
            sup = self.masks[j][2]
            if len(sup) >= 2:
                for q in sup:
                    self._rem[q] -= 2.0
            self._tick()

        def swap(self, p, q):
            super().swap(p, q)
            if "_rem" in self.__dict__:
                self._tick()

        def build(self):
            self._rem = [0.0] * self.n
            for _, _, sup in self.masks:
                if len(sup) >= 2:
                    for q in sup:
                        self._rem[q] += 2.0
            self._clock, self._seen = {}, 0
            return self._build2()

        def _lb(self, pi):
            c = self._clock
            return max(c.get(pi[q], 0.0) + self._rem[q] for q in range(self.n))

        def swap_penalty(self, p, v, a, b):
            """KAPPA x the rise of the makespan bound if the SWAP on (p, v) is emitted now."""
            if self.KAPPA == 0:
                return 0.0
            c = self._clock
            lb = self._lb(self.pi)
            end = max(c.get(p, 0.0), c.get(v, 0.0)) + 3.0
            new = 0.0
            for lq in (a, b):
                if lq is not None:
                    new = max(new, end + self._rem[lq])
            return self.KAPPA * max(0.0, new - lb)

    # The SWAP choice lives inside _StMover2.build; it is re-implemented here with the penalty added to the key.
    def build2(self):
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
                small = [k for k in free if len(self.masks[k][2]) <= 2]
                if small:
                    for k in small:
                        self._run(k, unrun, elig, ex, succ, blocked)
                        left -= 1
                else:
                    j = max(free, key=lambda k: (self.similar(self.last_term, k), -k))
                    self._run(j, unrun, elig, ex, succ, blocked)
                    left -= 1
                stuck = 0
                continue
            while ptr < T and not unrun[ptr]:
                ptr += 1
            look = set()
            k = ptr
            while k < T and len(look) < self.LOOK:
                if unrun[k] and k not in elig:
                    look.add(k)
                k += 1
            order = sorted(elig, key=lambda j: (ex[j], j))
            focus = order[: self.f]
            cand = set()
            for j in focus:
                for q in self.masks[j][2]:
                    p = self.pi[q]
                    for v in self.adj[p]:
                        cand.add((min(p, v), max(p, v)))
            best = None
            fset = set(focus)
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
                for j in touched:
                    if not unrun[j] or not (j in elig or j in look):
                        continue
                    wgt = 1.0 if j in fset else (self.MU if j in elig else self.LAMBDA)
                    gain += wgt * (self.proxy_excess(j, self.pi) - self.proxy_excess(j, pi2))
                merged = self.mergeable(p, v)
                c_swap = 1 if merged else 3
                if self.err is not None:
                    c_swap *= self.err[2].get((p, v), self.w_med) / self.w_med
                pen = self.swap_penalty(p, v, a, b)
                key = (gain - c_swap - pen, merged, -p, -v)
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

    _StMover3._build2 = build2
    return _StMover3
