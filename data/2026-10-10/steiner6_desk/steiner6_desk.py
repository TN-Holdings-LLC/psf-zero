"""steiner6_desk.py -- STEINER6's desk check (2026-10-10; Addendum 439): before STEINER6 was delivered, its chain
placement (steiner6.py's path_placements: the logical qubits in a linear order laid along a long path of the working
graph) was compared with STEINER3's greedy placement on ibm_kingston's live working graph (the Target pickled on
2026-09-28, as data/2026-10-10/kingston_target/kingston_target.json lists it, without its failed couplers and qubits),
for periodic chains of 26, 80 and 120 qubits with shuffled labels. Score: sum over the chain's couplings of the hop
distance between the placed qubits (the chain's length if every coupling is adjacent). No Qiskit needed.

    cd <psf-zero repository>; python data/2026-10-10/steiner6_desk/steiner6_desk.py
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("PSF_ZERO_REPO", REPO)
sys.path[:0] = [HERE, os.path.join(REPO, "benchmarks")]
import steiner3 as S3  # noqa: E402
import steiner6 as S6  # noqa: E402

d = json.load(open(os.path.join(REPO, "data", "2026-10-10", "kingston_target", "kingston_target.json")))
bad_q = {113, 121, 146}
bad_e = {(112, 113), (130, 131), (145, 146), (146, 147)}
adj = {q: set() for q in range(d["qubits"]) if q not in bad_q}
for a, b in d["live_edges"]:
    if (a, b) in bad_e or a in bad_q or b in bad_q:
        continue
    adj[a].add(b)
    adj[b].add(a)
dist, par = S3.bfs_all(adj)
print(f"working qubits {len(adj)}; longest path found {max(len(p) for p in S6.long_paths(adj))}")
for n in (26, 80, 120):
    perm = list(range(n))
    random.Random(n).shuffle(perm)
    terms = []
    for i in range(n):
        a, b = perm[i], perm[(i + 1) % n]
        lab = ["I"] * n
        lab[n - 1 - a] = lab[n - 1 - b] = "Z"
        terms.append(("".join(lab), 1.0))
    w = S6.interaction(n, terms)
    score = lambda ph: sum(c * dist[ph[x]][ph[y]] for (x, y), c in w.items())  # noqa: E731
    best_path = S6.path_placements(n, terms, adj, dist)[0]
    greedy = S3.placement(n, terms, adj, dist, sorted(adj, key=lambda v: (-len(adj[v]), v))[0])
    print(f"periodic chain of {n}: chain length {n}; path placement {score(best_path)}; greedy placement {score(greedy)}")
