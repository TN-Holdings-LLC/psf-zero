"""verify_topology_generalization.py -- Addendum 104.

Every verification of the four fixes so far (Addenda 88-95, 101-103)
used square grids only. `psf_smart_layout.py`'s own docstring records
that its BFS orderings found nothing on `brick`; heavy-hex's own
bipartite imbalance (Addendum 39-40, this repository's own established
finding) makes qubit-count "spare=0" impossible on that topology at
all. This tests whether the natural-ordering-first fix (Addendum 88)
helps, is neutral, or costs something measurable on these two
topologies -- it cannot reduce coverage on any topology (a new first
attempt; every prior ordering still follows if it fails), so this is
not a coverage regression test.

`brick_edges` is copied verbatim from this repository's own
`verify_vf2_sparse_topology.py`, not re-derived. Heavy-hex's own
"spare=0 relative to its own maximum matching" convention is copied
from this repository's own `occupancy_sweep_heavy_hex.py`, not
re-derived.

Usage:
    python verify_topology_generalization.py
"""
from __future__ import annotations

import csv
import time

import networkx as nx
import rustworkx as rx
from qiskit.transpiler import CouplingMap

from psf_smart_layout import (
    smart_vf2_layout,
    _has_feasible_matching,
    _candidate_orderings,
    _fallback_orderings,
    _relabel,
    _try_mapping,
)


# ---------------------------------------------------------------------
# brick_edges -- copied verbatim from verify_vf2_sparse_topology.py
# ---------------------------------------------------------------------
def brick_edges(rows, cols):
    """Each row is fully connected as a chain; vertical edges are kept on
    only half the checkerboard pattern."""
    idx = lambda r, c: r * cols + c
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows and (r + c) % 2 == 0:
                e.append((idx(r, c), idx(r + 1, c)))
    return e


def dense_pairs(n):
    return [(i, i + 1) for i in range(0, n - 1, 2)]


def cm_from_edges(n_nodes, edges):
    """A minimal CouplingMap-like object with just the two methods
    `smart_vf2_layout`/`_has_feasible_matching` actually call -- avoids
    depending on whether `CouplingMap` accepts arbitrary non-grid edge
    lists the same way on every Qiskit version."""
    class _CM:
        def __init__(self, n, e):
            self._n, self._e = n, e
        def size(self):
            return self._n
        def get_edges(self):
            return self._e
    return _CM(n_nodes, edges)


def run_search_no_bypass(cm, pairs, n):
    """The real, unbypassed entry point."""
    t0 = time.perf_counter()
    layout, info = smart_vf2_layout(cm, pairs, n)
    return layout, info, time.perf_counter() - t0


def run_search_skip_natural(cm, pairs, n, per_attempt_call_limit=50_000,
                            time_budget_s=2.0, fallback_call_limit=2_000_000):
    """Reproduces the pre-Addendum-88 sequence for comparison: builds the
    same graphs, but skips the first ('natural') entry from
    `_candidate_orderings` before iterating -- isolating whether the new
    attempt itself costs something on a topology it may not help on."""
    phys = rx.PyGraph()
    for i in range(cm.size()):
        phys.add_node(i)
    for a, b in cm.get_edges():
        if not phys.has_edge(a, b):
            phys.add_edge(a, b, None)
    im = rx.PyGraph()
    idx_of_logical = {}
    for a, b in pairs:
        for q in (a, b):
            if q not in idx_of_logical:
                idx_of_logical[q] = im.add_node(q)
        im.add_edge(idx_of_logical[a], idx_of_logical[b], None)

    t0 = time.perf_counter()
    rate = None
    tried = 0
    orderings = list(_candidate_orderings(phys))[1:]  # skip "natural"
    for name, order in orderings:
        remaining = time_budget_s - (time.perf_counter() - t0)
        if remaining <= 0.005:
            break
        cl = per_attempt_call_limit if rate is None else max(
            1, min(per_attempt_call_limit, int(rate * remaining)))
        relabeled = _relabel(phys, order)
        m, el = _try_mapping(relabeled, im, order, idx_of_logical,
                             id_order=True, call_limit=cl)
        tried += 1
        if m is None and el > 0:
            rate = cl / el
        if m is not None:
            return m, tried, time.perf_counter() - t0
    rate = None
    for name, order in _fallback_orderings(phys):
        remaining = time_budget_s - (time.perf_counter() - t0)
        if remaining <= 0.005:
            break
        cl = fallback_call_limit if rate is None else max(
            1, min(fallback_call_limit, int(rate * remaining)))
        relabeled = _relabel(phys, order)
        m, el = _try_mapping(relabeled, im, order, idx_of_logical,
                             id_order=False, call_limit=cl)
        tried += 1
        if m is None and el > 0:
            rate = cl / el
        if m is not None:
            return m, tried, time.perf_counter() - t0
    return None, tried, time.perf_counter() - t0


def main():
    rows = []

    # ---------------- brick ----------------
    print("=" * 100)
    print("Addendum 104 / P1 -- brick topology, dense_pairs at spare=0")
    print("=" * 100)
    for r, c in [(6, 7), (8, 8)]:
        n = r * c
        edges = brick_edges(r, c)
        cm = cm_from_edges(n, edges)
        pairs = dense_pairs(n)

        feasible = _has_feasible_matching(cm, pairs)
        layout_full, info_full, t_full = run_search_no_bypass(cm, pairs, n)
        layout_skip, tried_skip, t_skip = run_search_skip_natural(cm, pairs, n)

        found_full = layout_full is not None
        found_skip = layout_skip is not None
        print(f"  {r}x{c} brick (n={n}, {len(edges)} edges): "
              f"feasible={feasible}, "
              f"with-natural: found={found_full} tried={info_full.get('orderings_tried')} "
              f"time={t_full*1000:.2f}ms, "
              f"without-natural: found={found_skip} tried={tried_skip} "
              f"time={t_skip*1000:.2f}ms")
        rows.append(dict(topology=f"brick_{r}x{c}", n_qubits=n,
                         feasible=feasible, found_with_natural=found_full,
                         time_with_natural_ms=t_full*1000,
                         found_without_natural=found_skip,
                         time_without_natural_ms=t_skip*1000))

    # ---------------- heavy-hex ----------------
    print()
    print("=" * 100)
    print("Addendum 104 / P2, P3 -- heavy-hex, spare_pairs=0 relative to its "
          "OWN maximum matching")
    print("=" * 100)
    for d in [3, 5]:
        cm = CouplingMap.from_heavy_hex(d)
        n = cm.size()
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from([tuple(e) for e in cm.get_edges()])
        max_m = len(nx.max_weight_matching(g, maxcardinality=True))
        pairs_exact = dense_pairs(2 * max_m)
        pairs_over = dense_pairs(2 * (max_m + 1))

        feasible_exact = _has_feasible_matching(cm, pairs_exact)
        feasible_over = _has_feasible_matching(cm, pairs_over)
        layout, info, t = run_search_no_bypass(cm, pairs_exact, 2 * max_m)

        print(f"  heavy_hex d={d} (n={n}, max_matching={max_m}): "
              f"P2 exact-capacity accepted={feasible_exact} (True expected), "
              f"P2 one-over rejected={not feasible_over} (True expected), "
              f"P3 search found={layout is not None} tried={info.get('orderings_tried')} "
              f"order={info.get('order_name')} time={t*1000:.2f}ms")
        rows.append(dict(topology=f"heavy_hex_d{d}", n_qubits=n,
                         max_matching=max_m,
                         feasible_at_capacity=feasible_exact,
                         feasible_one_over=feasible_over,
                         search_found=layout is not None,
                         orderings_tried=info.get("orderings_tried"),
                         order_used=info.get("order_name"),
                         time_ms=t*1000))

    print("=" * 100)
    out_path = "topology_generalization_2026-09-20.csv"
    # Two different row shapes (brick vs heavy_hex) -- write as one CSV
    # via the union of all fieldnames, blanks where not applicable,
    # rather than two separate files, for a single artifact.
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
