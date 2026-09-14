#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A prototype layout search for PSF-Zero (2026-09-14, building on spare-qubit-cliff addenda 8-12)

## Motivation

Established in addenda 8-10:
  - Success is cheap (tens of microseconds to a few milliseconds). Failure is
    expensive -- when Qiskit's `VF2Layout` pass fails, it burns through its
    entire assigned `call_limit`, then switches to `SabreLayout`, and the
    subsequent swap-insertion/optimization passes also end up heavier than
    usual (addendum-10; at L3 there was a case reaching nearly 3x the failure
    cost).
  - `VF2Layout` is fixed at `seed=-1` and is not controlled by
    `seed_transpiler` (addendum-9) -- it relies on "a single roll-of-the-dice
    shuffle."

Established in addenda 9 and 12 (though **these results come from the public
`rustworkx.vf2_mapping()`**, and whether Qiskit's internal implementation
`qiskit._accelerate.vf2_layout` behaves the same way is unconfirmed -- see
"Important caveat" below):
  - On a grid physical graph, depth-first search (DFS) is fragile and
    strongly dependent on the starting point and the grid's row/column
    parity, while breadth-first search (BFS) is far more robust.
  - On the same graph, whether a solution is found or not depends on the
    node ordering supplied.

## Additional findings discovered while building this prototype (2026-09-14,
   found through this prototype's own smoke tests and diagnostics; not yet
   confirmed on real hardware -- sandbox only)

The "BFS is robust" finding above was confirmed on **pure grid physical
graphs such as 8x8/9x9**. Applying this prototype not just to an 8x8 grid but
also to addendum-8's brick and diluted_p topologies (spare=0, tight) found
that **this finding does not generalize as-is**:

  - grid, line: with a BFS-family ordering (especially
    bfs_from_min_degree) + id_order=True, a solution is found instantly,
    using only a tiny fraction of `call_limit=50,000` (under a microsecond,
    effectively one step).
  - brick: none of 6 BFS-family orderings tried, with id_order=True, found a
    solution even at `call_limit` raised to 10,000,000. Switching to
    id_order=False (VF2's built-in heuristic ordering) and raising
    `call_limit` to 30,000,000 (roughly Qiskit's L3 budget) still found
    nothing. This **is consistent** with addendum-8's already-confirmed
    result that "brick shows the cliff when tight" (i.e. Qiskit's own
    `VF2Layout` pass genuinely fails on it when tight too) -- brick really is
    hard.
  - diluted_p0.75: even though Qiskit's default pipeline (the internal
    implementation, `seed=-1`, `call_limit=5,000,000` @ L2) succeeds 100%
    (confirmed in addendum-10/11), trying the same condition against the
    public rustworkx finds that **a fixed BFS ordering with id_order=True
    finds nothing even at call_limit=10,000,000**. However, **switching to
    id_order=False (VF2's heuristic ordering) found a solution at
    `call_limit=1,000,000`, in 0.05 seconds** (though this only held for
    some of the starting orderings tried -- degree_desc succeeded, while
    other BFS-family starting orderings still failed even with the same
    id_order=False. So even under id_order=False, the initial node numbering
    is not irrelevant).

**Interpretation**: when the logical interaction pattern is "a set of
disjoint edges (a matching-shaped pattern)," backtracking under a fixed
node-visit order (id_order=True) is likely prone to exponential backtracking,
because a greedy, component-by-component assignment can conflict with the
assignment of a later component. VF2's built-in heuristic (id_order=False,
which dynamically picks the "most constrained node" based on degree and
similar criteria) can work structurally in favour of this kind of pattern --
though for a candidate that is inherently hard, like brick, it still cannot
solve it.

**How this shaped this prototype's design**: rather than committing to a
single strategy (BFS + id_order=True), it uses two stages.

  1. A cheap stage: several BFS-family orderings x id_order=True x a small
     call_limit. Catches, at near-zero cost, cases like grid and line where
     "it's instant once the order is right."
  2. An expensive stage: if stage 1 finds nothing and time budget remains,
     tries id_order=False (the VF2 heuristic) across several starting
     orderings (degree_desc, natural order, etc.) x a larger call_limit.
     Aims to catch cases like diluted_p0.75, where "the ordering heuristic
     works, but a fixed ordering cannot see it."

**Important caveat (unverified)**: including stage 2, every finding here was
confirmed against the **public `rustworkx.vf2_mapping()`**, and it has not
been verified whether the actual `VF2Layout` pass -- which calls a
**separate compiled Rust module**, `qiskit._accelerate.vf2_layout.
vf2_layout_pass_average` (a discovery from addendum-9) -- behaves the same
way. This prototype module is therefore deliberately **implemented directly
on top of the public rustworkx** (it does not rewrite or imitate Qiskit's
internal implementation). Also, the additional brick/diluted_p0.75
diagnostics above were run only in the sandbox (a 2-core Linux VM);
reproduction on real hardware has not yet been confirmed. Before replacing
Qiskit's `VF2Layout` pass itself, it would be worth checking whether the same
ordering effects and stage-2 benefit reproduce in the internal implementation
too -- that is this prototype's next task, and is out of scope here.

## Usage

    from psf_smart_layout import smart_vf2_layout
    layout_map, info = smart_vf2_layout(coupling_map, interaction_pairs, num_qubits)
    # layout_map: {logical_qubit: physical_qubit}, or None if nothing was found
    # info: a dict of diagnostics (number of orderings tried, time spent,
    #       feasibility-check result, etc.)
"""
from __future__ import annotations

import time

# Do not start a new attempt if less than this much time remains (seconds).
MIN_ATTEMPT_S = 0.005


def _has_feasible_matching(cmap, num_logical_pairs):
    """A cheap feasibility check before calling VF2. Returns False rather than
    None when no matching exists (same logic as
    `vf2_probe_common.has_perfect_matching`; reimplemented independently here
    to reduce dependencies)."""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(cmap.size()))
    g.add_edges_from([tuple(e) for e in cmap.get_edges()])
    m = nx.max_weight_matching(g, maxcardinality=True)
    return len(m) >= num_logical_pairs


def _bfs_order_from(graph, start):
    import rustworkx as rx
    layers = rx.bfs_layers(graph, [start])
    order = [n for layer in layers for n in layer]
    # Append any unreachable nodes at the end (a safeguard for disconnected graphs).
    seen = set(order)
    for n in graph.node_indices():
        if n not in seen:
            order.append(n)
    return order


def _candidate_orderings(graph, extra_seeds=(0, 1)):
    """Returns a diverse set of cheap candidate node orderings. BFS-first
    (known from addenda 9/12 to be more robust than DFS -- though that
    finding is limited to grid physical graphs; see the docstring above)."""
    import random
    nodes = list(graph.node_indices())
    degrees = {n: len(graph.neighbors(n)) for n in nodes}
    max_deg_node = max(nodes, key=lambda n: degrees[n])
    min_deg_node = min(nodes, key=lambda n: degrees[n])

    orderings = []
    orderings.append(("bfs_from_max_degree", _bfs_order_from(graph, max_deg_node)))
    orderings.append(("bfs_from_min_degree", _bfs_order_from(graph, min_deg_node)))
    orderings.append(("bfs_from_node0", _bfs_order_from(graph, nodes[0])))
    orderings.append(("degree_desc", sorted(nodes, key=lambda n: -degrees[n])))
    for seed in extra_seeds:
        rng = random.Random(seed)
        start = rng.choice(nodes)
        orderings.append((f"bfs_from_random_seed{seed}", _bfs_order_from(graph, start)))
    return orderings


def _fallback_orderings(graph):
    """Candidate starting orderings for stage 2 (id_order=False). Diagnostics
    found degree_desc solved diluted_p0.75 while other starting orderings did
    not, even under the same id_order=False, so several starting orderings
    are kept on hand."""
    nodes = list(graph.node_indices())
    degrees = {n: len(graph.neighbors(n)) for n in nodes}
    orderings = []
    orderings.append(("natural", list(nodes)))
    orderings.append(("degree_desc", sorted(nodes, key=lambda n: -degrees[n])))
    orderings.append(("degree_asc", sorted(nodes, key=lambda n: degrees[n])))
    return orderings


def _relabel(phys, order):
    import rustworkx as rx
    relabeled = rx.PyGraph()
    old_to_new = {}
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = relabeled.add_node(old_idx)
    for a, b in phys.edge_list():
        relabeled.add_edge(old_to_new[a], old_to_new[b], None)
    return relabeled


def _try_mapping(relabeled, im, order, idx_of_logical, id_order, call_limit):
    """One vf2_mapping attempt. Returns (layout_map, elapsed) if found, or
    (None, elapsed) if not."""
    import rustworkx as rx
    t0 = time.perf_counter()
    it = rx.vf2_mapping(relabeled, im, subgraph=True, id_order=id_order,
                        induced=False, call_limit=call_limit)
    m = next(iter(it), None)
    el = time.perf_counter() - t0
    if m is None:
        return None, el

    # rx.vf2_mapping(first, second, ...) returns a mapping "from first's node
    # indices to second's node indices" (per the official docstring, verified
    # against a path_graph example). Here first=relabeled (physical),
    # second=im (the logical interaction graph), so m is
    # {relabeled_phys_idx: im_idx}.
    #
    # An earlier version of this code misread this as the reverse
    # ({im_idx: relabeled_phys_idx}), which produced a hard-to-notice bug:
    # when num_physical == num_logical (spare=0), this did not raise a
    # KeyError but instead **returned the wrong physical qubits**.
    # (Discovered and fixed by directly validating the layout's correctness
    # in the smoke test.)
    im_idx_to_relabeled = {im_idx: relabeled_idx for relabeled_idx, im_idx in m.items()}
    layout_map = {}
    for logical_q, im_idx in idx_of_logical.items():
        relabeled_phys_idx = im_idx_to_relabeled[im_idx]
        original_phys = order[relabeled_phys_idx]
        layout_map[logical_q] = original_phys
    return layout_map, el


def smart_vf2_layout(coupling_map, interaction_pairs, num_qubits,
                     per_attempt_call_limit=50_000, time_budget_s=2.0,
                     extra_seeds=(0, 1),
                     fallback_call_limit=2_000_000, use_fallback=True):
    """Tries several node orderings and strategies in order of increasing
    budget, stopping as soon as one succeeds.

    Stage 1: several BFS-family orderings x id_order=True x
             per_attempt_call_limit (for cases like grid and line, where
             it's instant once the ordering is right).
    Stage 2: if use_fallback=True and time budget remains, tries
             id_order=False (the VF2 heuristic) across several starting
             orderings x fallback_call_limit (aimed at cases like
             diluted_p0.75, where "the heuristic works but a fixed ordering
             cannot see it." For a candidate that is inherently hard, like
             brick, this can still find nothing -- which is itself
             consistent with addendum-8's confirmation of the "cliff").

    Returns:
        (layout_map, info) -- layout_map is a dict of {logical: physical},
        or None if nothing was found. info holds diagnostics.
    """
    import rustworkx as rx

    t0 = time.perf_counter()
    info = dict(feasible=None, attempts=[], found=False, orderings_tried=0,
               elapsed_s=None, order_name=None, phase=None)

    if not _has_feasible_matching(coupling_map, len(interaction_pairs)):
        info["feasible"] = False
        info["elapsed_s"] = time.perf_counter() - t0
        return None, info
    info["feasible"] = True

    # Build the physical graph as a rustworkx.PyGraph.
    phys = rx.PyGraph()
    for i in range(coupling_map.size()):
        phys.add_node(i)
    for a, b in coupling_map.get_edges():
        if not phys.has_edge(a, b):
            phys.add_edge(a, b, None)

    im = rx.PyGraph()
    idx_of_logical = {}
    for a, b in interaction_pairs:
        for q in (a, b):
            if q not in idx_of_logical:
                idx_of_logical[q] = im.add_node(q)
        im.add_edge(idx_of_logical[a], idx_of_logical[b], None)

    # The observed "calls consumed per second." Used to keep within the time
    # budget (see below).
    rate = None

    def _remaining():
        return time_budget_s - (time.perf_counter() - t0)

    def _budgeted_call_limit(nominal):
        """Shrinks call_limit to fit within the remaining time.

        In a 2026-09-14 real-hardware run, the search took 2.66 seconds
        despite `time_budget_s=2.0` (diluted_p0.5). The cause was that the
        budget was **only checked between attempts** -- once an attempt
        started, it did not stop until `call_limit` was exhausted, so the
        last attempt overran the budget in full. Here, call_limit is shrunk
        by estimating, from the call-consumption rate observed so far, how
        many calls can be consumed in the remaining time."""
        if rate is None:
            return nominal
        rem = _remaining()
        if rem <= 0:
            return 0
        return max(1, min(nominal, int(rate * rem)))

    # --- Stage 1: BFS-family orderings + id_order=True (cheap) ---
    for order_name, order in _candidate_orderings(phys, extra_seeds=extra_seeds):
        if _remaining() <= MIN_ATTEMPT_S:
            break
        cl = _budgeted_call_limit(per_attempt_call_limit)
        if cl <= 0:
            break
        relabeled = _relabel(phys, order)
        layout_map, el = _try_mapping(relabeled, im, order, idx_of_logical,
                                      id_order=True, call_limit=cl)
        info["attempts"].append(dict(phase=1, order=order_name, id_order=True,
                                     found=layout_map is not None, time_s=el,
                                     call_limit=cl))
        info["orderings_tried"] += 1
        if layout_map is None and el > 0:
            # A failure means call_limit was exhausted, so the consumption
            # rate can be measured from it.
            rate = cl / el
        if layout_map is not None:
            info["found"] = True
            info["order_name"] = order_name
            info["phase"] = 1
            info["elapsed_s"] = time.perf_counter() - t0
            return layout_map, info

    # --- Stage 2: id_order=False (the VF2 heuristic, expensive) ---
    if use_fallback:
        rate = None  # re-measure, since stage 1 (id_order=True) and stage 2
                     # have different consumption rates
        for order_name, order in _fallback_orderings(phys):
            if _remaining() <= MIN_ATTEMPT_S:
                break
            cl = _budgeted_call_limit(fallback_call_limit)
            if cl <= 0:
                break
            relabeled = _relabel(phys, order)
            layout_map, el = _try_mapping(relabeled, im, order, idx_of_logical,
                                          id_order=False, call_limit=cl)
            info["attempts"].append(dict(phase=2, order=f"heuristic_{order_name}",
                                         id_order=False, found=layout_map is not None,
                                         time_s=el, call_limit=cl))
            info["orderings_tried"] += 1
            if layout_map is None and el > 0:
                rate = cl / el
            if layout_map is not None:
                info["found"] = True
                info["order_name"] = f"heuristic_{order_name}"
                info["phase"] = 2
                info["elapsed_s"] = time.perf_counter() - t0
                return layout_map, info

    info["elapsed_s"] = time.perf_counter() - t0
    return None, info
