"""component_decomposition.py -- for every circuit family this project
has used, report the interaction graph's component decomposition
(component count, size of each component, and a size-frequency
histogram), independent of any Qiskit run.

## Why this exists

Addenda 63-73 established that timing and stop-reason data alone do not
explain why some compositions cliff and others do not: matched edge
counts, matched component counts, and matched bare-edge counts have all
individually failed to separate fast from slow outcomes on their own
(Addendum 63's `balanced_3q4q` vs. `mixed_uneven`; Addendum 72's
`n_bare_edges(17)` vs. `mixed_uneven` despite similar coverage). What has
never been produced is the actual component-size DISTRIBUTION for every
tested configuration side by side. This script produces exactly that,
from the same edge-construction functions `circuit_family_sweep.py`
uses (copied here verbatim, not reimplemented), so no new Qiskit
measurement is needed -- every number in the output CSV is a pure graph-
theoretic property of a circuit already characterized by timing in a
prior addendum.

## What it does NOT do

It does not re-derive or predict timing/stop-reason outcomes -- those
are looked up from the addenda that already measured them (hardcoded in
`KNOWN_OUTCOMES` below, with the source addendum cited per entry) and
merged in for side-by-side display only. If an entry's outcome is not
yet confirmed (e.g. depends on a single run rather than a reproduced
one), that is noted in the `outcome_confidence` column rather than
presented as equally solid.

Usage:
    python component_decomposition.py
    (writes component_decomposition_2026-09-18.csv and prints a summary
    table to stdout; no arguments needed -- every configuration is
    fixed and pre-registered-in-spirit as the twelve points the user
    specified)
"""
from __future__ import annotations

import networkx as nx
import pandas as pd
from collections import Counter


# ==================================================================
# Edge-construction functions, copied verbatim from
# circuit_family_sweep.py (Addenda 51-72). `rng` arguments are kept
# for signature compatibility but are irrelevant here -- component
# structure does not depend on the random gate content, only on which
# qubits are connected, which is fully determined by the other
# parameters (verified in Addenda 60/67: seed/rng never affects graph
# topology in this project's families, only gate content).
# ==================================================================

def _edges_dense_pairs(n, rng):
    return [(i, i + 1) for i in range(0, n - 1, 2)]


def _edges_linear_chain(n, rng):
    return [(i, i + 1) for i in range(n - 1)]


def _edges_merged_pairs(n, rng, m=0):
    base_edges = _edges_dense_pairs(n, rng)
    n_mergeable_pairs = len(base_edges) // 2
    if m > n_mergeable_pairs:
        raise ValueError(f"m={m} exceeds mergeable pair count "
                         f"({n_mergeable_pairs}) for n={n}")
    edges = []
    i = 0
    merged_count = 0
    while i < len(base_edges):
        if merged_count < m and i + 1 < len(base_edges):
            (a, b) = base_edges[i]
            (c, d) = base_edges[i + 1]
            edges.append((a, b))
            edges.append((b, c))
            edges.append((c, d))
            merged_count += 1
            i += 2
        else:
            edges.append(base_edges[i])
            i += 1
    return edges


def _edges_balanced_3q4q(n, rng, n_4q=10, n_3q=8):
    n_needed = 4 * n_4q + 3 * n_3q
    if n_needed != n:
        raise ValueError(f"n_4q={n_4q}, n_3q={n_3q} need exactly "
                         f"{n_needed} qubits, got n={n}")
    edges = []
    q = 0
    for _ in range(n_4q):
        edges += [(q, q + 1), (q + 1, q + 2), (q + 2, q + 3)]
        q += 4
    for _ in range(n_3q):
        edges += [(q, q + 1), (q + 1, q + 2)]
        q += 3
    return edges


def _edges_n_bare_edges(n, rng, n_bare=1):
    remaining = n - n_bare * 2
    if remaining < 0:
        raise ValueError(f"n_bare={n_bare} needs {n_bare*2} qubits, "
                         f"exceeds n={n}")
    n_3q, leftover = divmod(remaining, 3)
    if 0 < remaining < 3:
        raise ValueError(
            f"n_bare={n_bare} on n={n} leaves remaining={remaining} "
            f"qubits (0 < remaining < 3) -- no zero-idle construction.")
    edges = []
    q = 0
    for _ in range(n_bare):
        edges.append((q, q + 1))
        q += 2
    n_plain_3q = n_3q - 1 if leftover > 0 else n_3q
    for _ in range(n_plain_3q):
        edges.append((q, q + 1))
        edges.append((q + 1, q + 2))
        q += 3
    if leftover > 0:
        extra_size = 3 + leftover
        for i in range(extra_size - 1):
            edges.append((q + i, q + i + 1))
        q += extra_size
    return edges


def _edges_single_bare_edge(n, rng, n_3q=19, extra_chain_size=5):
    n_needed = 2 + n_3q * 3 + extra_chain_size
    if n_needed != n:
        raise ValueError(f"n_3q={n_3q}, extra_chain_size={extra_chain_size} "
                         f"need exactly {n_needed} qubits, got n={n}")
    edges = []
    q = 0
    edges.append((q, q + 1))
    q += 2
    for _ in range(n_3q):
        edges.append((q, q + 1))
        edges.append((q + 1, q + 2))
        q += 3
    for i in range(extra_chain_size - 1):
        edges.append((q + i, q + i + 1))
    return edges


def _edges_large_dominant_no_bare_edges(n, rng, n_small=8, small_size=3,
                                        big_size=40):
    n_needed = n_small * small_size + big_size
    if n_needed != n:
        raise ValueError(f"n_small={n_small}, small_size={small_size}, "
                         f"big_size={big_size} need exactly {n_needed} "
                         f"qubits, got n={n}")
    edges = []
    q = 0
    for _ in range(n_small):
        for i in range(small_size - 1):
            edges.append((q + i, q + i + 1))
        q += small_size
    for i in range(big_size - 1):
        edges.append((q + i, q + i + 1))
    return edges


def _edges_mixed_uneven(n, rng, n_small=17, small_size=2, big_size=30):
    n_needed = n_small * small_size + big_size
    if n_needed != n:
        raise ValueError(f"n_small={n_small}, small_size={small_size}, "
                         f"big_size={big_size} need exactly {n_needed} "
                         f"qubits, got n={n}")
    edges = []
    q = 0
    for _ in range(n_small):
        for i in range(small_size - 1):
            edges.append((q + i, q + i + 1))
        q += small_size
    for i in range(big_size - 1):
        edges.append((q + i, q + i + 1))
    return edges


# ==================================================================
# The twelve configurations the user specified, with each one's
# already-measured outcome cited from its source addendum. This is
# lookup data, not a new measurement -- see each entry's `source`.
# ==================================================================

CONFIGURATIONS = [
    # -- fast group --
    dict(label="linear_chain (n=42)", group="fast", n=42,
        fn=_edges_linear_chain, kwargs={},
        outcome="solution found", time_ms=None,
        source="Addendum 51", confidence="single run"),
    dict(label="mixed_uneven (n=64)", group="fast", n=64,
        fn=_edges_mixed_uneven, kwargs={},
        outcome="solution found", time_ms=30.16,
        source="Addendum 63", confidence="single run -- NOT independently reproduced (Addendum 72 Section 6 flags this as a priority)"),
    dict(label="merged_pairs m=3 (n=64)", group="fast", n=64,
        fn=_edges_merged_pairs, kwargs=dict(m=3),
        outcome="solution found", time_ms=32.62,
        source="Addendum 65 (control point)", confidence="single run"),
    dict(label="merged_pairs m=4 (n=64)", group="fast", n=64,
        fn=_edges_merged_pairs, kwargs=dict(m=4),
        outcome="solution found", time_ms=32.31,
        source="Addendum 65 (control point)", confidence="single run"),
    dict(label="merged_pairs m=6 (n=64)", group="fast", n=64,
        fn=_edges_merged_pairs, kwargs=dict(m=6),
        outcome="solution found", time_ms=29.31,
        source="Addendum 59", confidence="single run"),
    dict(label="n_bare_edges n_bare=16 (n=42)", group="fast", n=42,
        fn=_edges_n_bare_edges, kwargs=dict(n_bare=16),
        outcome="solution found", time_ms=26.20,
        source="Addenda 70, 73", confidence="30 runs, 5 seeds -- reproduced"),

    # -- slow (cliffing) group --
    dict(label="dense_pairs (n=42)", group="slow", n=42,
        fn=_edges_dense_pairs, kwargs={},
        outcome="nonexistent solution", time_ms=6616.47,
        source="Addenda 34/51/53/55", confidence="reproduced many times"),
    dict(label="balanced_3q4q (n=64)", group="slow", n=64,
        fn=_edges_balanced_3q4q, kwargs={},
        outcome="nonexistent solution", time_ms=10233.00,
        source="Addendum 63", confidence="single run"),
    dict(label="shrinking_dominant / n_bare_edges small_size=3,big_size=13 (n=64)", group="slow", n=64,
        fn=_edges_mixed_uneven, kwargs=dict(n_small=17, small_size=3, big_size=13),
        outcome="nonexistent solution", time_ms=12402.02,
        source="Addendum 64", confidence="single run"),
    dict(label="large_dominant_no_bare_edges (n=64)", group="slow", n=64,
        fn=_edges_large_dominant_no_bare_edges, kwargs={},
        outcome="nonexistent solution", time_ms=14351.92,
        source="Addendum 65", confidence="single run"),
    dict(label="single_bare_edge (n=64)", group="slow", n=64,
        fn=_edges_single_bare_edge, kwargs={},
        outcome="nonexistent solution", time_ms=10841.73,
        source="Addendum 66", confidence="single run"),
    dict(label="n_bare_edges n_bare=19 (n=42)", group="slow", n=42,
        fn=_edges_n_bare_edges, kwargs=dict(n_bare=19),
        outcome="nonexistent solution", time_ms=6693.30,
        source="Addendum 70", confidence="6 runs, 3 seeds"),
    dict(label="n_bare_edges n_bare=17 (n=64) [see note]", group="slow", n=64,
        fn=_edges_n_bare_edges, kwargs=dict(n_bare=17),
        outcome="nonexistent solution", time_ms=9204.35,
        source="Addendum 72", confidence="23 runs, 5 seeds, independently reproduced -- NOT the same graph as mixed_uneven despite matching coverage (see this script's own output for the component-count proof)"),
]


def analyze(edges: list[tuple[int, int]], n: int) -> dict:
    """Computes the component decomposition of an interaction graph."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    components = list(nx.connected_components(g))
    sizes = sorted((len(c) for c in components), reverse=True)
    size_histogram = dict(sorted(Counter(sizes).items()))
    idle = sum(1 for s in sizes if s == 1)
    return dict(
        n_qubits=n,
        n_edges=len(edges),
        n_components=len(components),
        n_idle_qubits=idle,
        component_sizes=sizes,
        size_histogram=size_histogram,
        max_component_size=max(sizes) if sizes else 0,
        n_bare_2q_components=size_histogram.get(2, 0),
    )


def main():
    rows = []
    print(f"{'label':<55} {'group':>5} {'n_comp':>7} {'idle':>5} "
          f"{'max_sz':>7} {'#bare2q':>8} {'histogram':<30} {'outcome':>22}")
    print("-" * 165)
    for cfg in CONFIGURATIONS:
        edges = cfg["fn"](cfg["n"], None, **cfg["kwargs"])
        info = analyze(edges, cfg["n"])
        hist_str = ",".join(f"{sz}x{cnt}" for sz, cnt in info["size_histogram"].items())
        print(f"{cfg['label']:<55} {cfg['group']:>5} "
              f"{info['n_components']:>7} {info['n_idle_qubits']:>5} "
              f"{info['max_component_size']:>7} {info['n_bare_2q_components']:>8} "
              f"{hist_str:<30} {cfg['outcome']:>22}")
        rows.append(dict(
            label=cfg["label"], group=cfg["group"], n_qubits=cfg["n"],
            n_edges=info["n_edges"], n_components=info["n_components"],
            n_idle_qubits=info["n_idle_qubits"],
            max_component_size=info["max_component_size"],
            n_bare_2q_components=info["n_bare_2q_components"],
            component_sizes=str(info["component_sizes"]),
            size_histogram=str(info["size_histogram"]),
            outcome=cfg["outcome"], time_ms=cfg["time_ms"],
            source_addendum=cfg["source"], outcome_confidence=cfg["confidence"],
        ))

    df = pd.DataFrame(rows)
    out_path = "component_decomposition_2026-09-18.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
