"""Q4: a ring schedule with constant depth for any n: folded order, then two swap layers.
Positions hold 0, n-1, 1, n-2, ... (chain A on even positions, chain B on odd positions): every ring edge is at
distance <= 2. Round 1: CZ on adjacent ring edges. SA = swaps (p, p+1), p = 1, 5, 9, ... Round 2: CZ on newly adjacent
edges. SB = undo SA together with swaps (p, p+1), p = 3, 7, 11, ... (disjoint, one layer). Round 3: CZ on the rest."""
def schedule(n, variant):
    order, lo, hi = [], 0, n - 1
    while lo <= hi:
        order.append(lo)
        if hi != lo: order.append(hi)
        lo, hi = lo + 1, hi - 1
    edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    occ = list(order); rounds = []
    def czs():
        d = []
        for p in range(n - 1):
            e = (min(occ[p], occ[p + 1]), max(occ[p], occ[p + 1]))
            if e in edges: d.append(e); edges.discard(e)
        return d
    def sw(pairs):
        for p, q in pairs: occ[p], occ[q] = occ[q], occ[p]
    SA = [(p, p + 1) for p in range(1, n - 1, 4) if p + 1 < n]
    SB = [(p, p + 1) for p in range(3, n - 1, 4) if p + 1 < n]
    plan = [SA, SA + SB, SB, SA + SB] if variant else []
    d = czs(); k = 0
    while edges and k < len(plan):
        rounds.append((d, plan[k])); sw(plan[k]); d = czs(); k += 1
    rounds.append((d, []))
    return order, rounds, edges
for n in range(4, 17):
    order, rounds, left = schedule(n, True)
    ncz = sum(len(d) for d, _ in rounds); nsw = sum(len(s) for _, s in rounds)
    depth = sum(1 for d, _ in rounds if d) + 3 * sum(1 for _, s in rounds if s)
    print(n, "left", sorted(left), "rounds", len(rounds), "CZ", ncz, "SWAP", nsw, "total", ncz + 3 * nsw, "depth", depth)
