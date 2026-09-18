"""circuit_family_sweep.py -- is a single bare 2-qubit edge sufficient
to prevent the cliff, or does count matter?

Pre-registered in `spare-qubit-cliff-addendum-51-preregistration-2026-09-18.md`
through `spare-qubit-cliff-addendum-66-preregistration-2026-09-18.md`.
See each addendum for the family it introduced.

## This revision: single_bare_edge (Addendum 66)

Addendum 65 established that "contains at least one bare 2-qubit edge"
perfectly separates fast from cliffing outcomes across five
compositions -- but every "fast" case tested had MANY bare edges (17, a
majority of the graph). This revision isolates presence from count:
`single_bare_edge` has exactly ONE bare 2-qubit edge, plus 19 disjoint
3-qubit chains, plus one 5-qubit chain (64-2=62 is not divisible by 3
alone, so the 5-qubit chain makes the qubit count exact), 21 components,
zero idle.

If fast: a single bare edge is sufficient -- presence alone matters.
If it cliffs: some larger count is needed -- Addendum 65's finding is a
quantitative threshold, not a simple presence/absence rule.

No CLI parameters beyond `--families` -- a fixed construction.

Usage:
    python circuit_family_sweep.py --rows 8 --cols 8 --spares 0 \\
        --levels 3 --seeds 3 --repeats 2 \\
        --families single_bare_edge
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from datetime import date

import networkx as nx
import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap

BASIS_GATES = ["rz", "sx", "x", "cx"]
GATES_PER_PAIR = 20
SEED_TRANSPILER = 42

# Passes whose combined time is the "layout stage" for prediction P3.
LAYOUT_PASSES = ("VF2Layout", "SabreLayout", "VF2PostLayout", "TrivialLayout",
                 "DenseLayout", "SetLayout", "ApplyLayout", "FullAncillaAllocation",
                 "EnlargeWithAncilla")
ROUTING_PASSES = ("SabreSwap", "StochasticSwap", "BasicSwap", "LookaheadSwap")


def build_dense_pair_blocks_circuit(num_qubits: int, gates_per_pair: int = GATES_PER_PAIR,
                                    seed: int = 0) -> QuantumCircuit:
    """Verbatim from this project's cliff-sniper family -- do not 'improve'.

    Note that for odd `num_qubits` the final qubit is left unpaired; the
    number of pairs actually built is recorded as `n_pairs` in the output
    so this is visible in the data rather than hidden.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for a, b in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


# ---------------------------------------------------------------- families
# All four families place the SAME number of random-unitary blocks per
# interacting pair, so total gate count scales with edge count rather than
# being held artificially equal -- edge count is recorded per row
# (`n_interaction_edges`) so the difference is visible in the data rather
# than silently confounding a timing comparison.

def _edges_dense_pairs(n, rng):
    """The project's standing family: N/2 disjoint edges. Maximally
    symmetric and disconnected -- see Addendum 51's pre-registration for
    why that is the thing being controlled for here."""
    return [(i, i + 1) for i in range(0, n - 1, 2)]


def _edges_linear_chain(n, rng):
    """A single connected path: (0,1), (1,2), (2,3), ... Connected and far
    less symmetric than disjoint pairs, but still embeds into a grid in
    many ways, so it is not harder or easier by construction."""
    return [(i, i + 1) for i in range(n - 1)]


def _edges_random_regular(n, rng, degree=3):
    """A random `degree`-regular graph, retried until connected. Minimal
    repeated structure. Requires n*degree to be even; falls back to n-1
    nodes if not, which is recorded via `n_interaction_edges` rather than
    silently adjusted."""
    import networkx as nx
    if (n * degree) % 2 != 0:
        n = n - 1
    for attempt in range(50):
        g = nx.random_regular_graph(degree, n, seed=int(rng.integers(0, 2**31)))
        if nx.is_connected(g):
            return sorted(tuple(sorted(e)) for e in g.edges())
    raise RuntimeError(f"no connected {degree}-regular graph on {n} nodes in 50 tries")


def _edges_ghz_star(n, rng):
    """A star: (0,1), (0,2), (0,3), ... **Deliberately unembeddable** in a
    degree-4 grid for n > 5, since the hub needs degree n-1. This is the
    feasibility control of Addendum 51 P3: if `VF2Layout_stop_reason`
    tracks genuine embeddability rather than merely saturation, this
    family should report "nonexistent solution" at EVERY occupancy, not
    only at spare=0."""
    return [(0, i) for i in range(1, n)]


def _edges_k_chains(n, rng, k=1):
    """Addendum 52: `k` disjoint connected chains of equal length,
    interpolating between `dense_pairs` (k = n/2, every chain is a single
    edge) and `linear_chain` (k = 1, one chain spans everything). `n` is
    truncated down to a multiple of `k` if it does not divide evenly, so
    every chain has exactly `n // k` qubits; the truncation is recorded
    via `n_interaction_edges` in the output rather than hidden.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    group_size = n // k
    if group_size < 1:
        raise ValueError(f"k={k} too large for n={n}")
    edges = []
    for g in range(k):
        start = g * group_size
        for i in range(group_size - 1):
            edges.append((start + i, start + i + 1))
    return edges


def _edges_merged_pairs(n, rng, m=0):
    """Addendum 54, P1: starts from `dense_pairs`' 21 disjoint edges on
    n=42 and merges `m` adjacent PAIRS of edges end-to-end into single
    4-qubit chains, reducing component count while using the exact same
    42 qubits -- zero qubits are ever excluded from the interaction
    graph, unlike `k_chains`, where component count and idle-qubit count
    change together. `m=0` reproduces `dense_pairs` exactly (the
    pre-registered sanity check, P3).

    Concretely: edges (0,1),(2,3),(4,5),(6,7),... are grouped in
    consecutive pairs; for the first `m` such pairs, edge `2i`=(a,b) and
    edge `2i+1`=(c,d) are merged into the 4-qubit chain a-b-c-d (edges
    (a,b),(b,c),(c,d)) instead of remaining as two separate 2-qubit
    edges (a,b) and (c,d). Requires m <= n//4 (21 edges = 10 mergeable
    pairs with 1 edge left over on n=42; the leftover single edge is
    left as-is regardless of m).
    """
    base_edges = _edges_dense_pairs(n, rng)  # 21 disjoint pairs on n=42
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
            # Reconnect b-c so the two original edges become one 4-qubit
            # chain a-b-c-d; a,b,c,d are unchanged qubit identities, only
            # which pairs interact changes.
            edges.append((a, b))
            edges.append((b, c))
            edges.append((c, d))
            merged_count += 1
            i += 2
        else:
            edges.append(base_edges[i])
            i += 1
    return edges


def _edges_merged_pairs_variant(n, rng, m=0, order="sequential"):
    """Addendum 60: like `merged_pairs`, but which of the base
    `dense_pairs` edge-PAIRS get merged is controlled by `order`, holding
    component count (and therefore `m`) fixed while changing WHICH
    specific qubits end up in the m four-qubit chains versus the
    remaining 2-qubit edges. Isolates whether Addendum 59's m=5/m=8
    oscillation at 8x8 depends on component count alone or on the
    specific qubits merged.

    - "sequential": identical to `_edges_merged_pairs` -- merges
      edge-pairs (0,1), (2,3), (4,5), ... in index order (the control,
      reproducing Addendum 59 exactly).
    - "reverse": merges edge-pairs starting from the LAST available
      indices instead of the first.
    - "random": selects m of the available edge-pairs uniformly at
      random (using `rng`, so seeded and reproducible) to merge, rather
      than a contiguous block.

    In all three cases, exactly `m` edge-pairs become 4-qubit chains and
    the rest remain untouched 2-qubit edges -- component count is
    identical across all three orderings for the same `m`.
    """
    base_edges = _edges_dense_pairs(n, rng)
    n_mergeable_pairs = len(base_edges) // 2
    if m > n_mergeable_pairs:
        raise ValueError(f"m={m} exceeds mergeable pair count "
                         f"({n_mergeable_pairs}) for n={n}")

    if order == "sequential":
        merge_pair_indices = list(range(m))
    elif order == "reverse":
        merge_pair_indices = list(range(n_mergeable_pairs - m, n_mergeable_pairs))
    elif order == "random":
        merge_pair_indices = sorted(
            rng.choice(n_mergeable_pairs, size=m, replace=False).tolist())
    else:
        raise ValueError(f"unknown order {order!r}; "
                         f"known: sequential, reverse, random")
    merge_set = set(merge_pair_indices)

    edges = []
    for pair_idx in range(n_mergeable_pairs):
        i = pair_idx * 2
        (a, b) = base_edges[i]
        (c, d) = base_edges[i + 1]
        if pair_idx in merge_set:
            edges.append((a, b))
            edges.append((b, c))
            edges.append((c, d))
        else:
            edges.append((a, b))
            edges.append((c, d))
    # If n_mergeable_pairs * 2 < len(base_edges), one edge is left over
    # (odd number of base edges) -- kept as-is, matching `merged_pairs`'
    # own handling.
    if n_mergeable_pairs * 2 < len(base_edges):
        edges.append(base_edges[-1])
    return edges


def _edges_uniform_2q(n, rng, n_components=18):
    """Addendum 63, P0 (sanity check, not a fair mod-3 test): builds
    exactly `n_components` disjoint 2-qubit edges, using
    `2 * n_components` of the `n` qubits and leaving the rest idle by
    construction. This deliberately reintroduces idle qubits (Addenda
    53-55 established these alone kill the cliff) specifically to
    confirm that effect still dominates in this new construction,
    alongside -- not instead of -- the real composition test
    (`_edges_balanced_3q4q`, `_edges_mixed_uneven`).
    """
    n_used = 2 * n_components
    if n_used > n:
        raise ValueError(f"n_components={n_components} needs {n_used} "
                         f"qubits, exceeds n={n}")
    return [(i, i + 1) for i in range(0, n_used, 2)]


def _edges_balanced_3q4q(n, rng, n_4q=10, n_3q=8):
    """Addendum 63, P1 (primary): builds `n_4q` disjoint 4-qubit chains
    and `n_3q` disjoint 3-qubit chains, using every one of
    `4*n_4q + 3*n_3q` qubits (zero idle when this equals `n` exactly --
    the caller is responsible for choosing n_4q/n_3q so this holds, as
    the default 10/8 does for n=64: 10*4 + 8*3 = 64). Component count is
    `n_4q + n_3q` (18 by default, divisible by 3, matching one of the
    values already associated with "easy" outcomes in Addenda 59/61/62)
    -- but with a genuinely different composition (two chain lengths
    mixed) from anything `merged_pairs` can produce at this component
    count.
    """
    n_needed = 4 * n_4q + 3 * n_3q
    if n_needed != n:
        raise ValueError(f"n_4q={n_4q}, n_3q={n_3q} need exactly "
                         f"{n_needed} qubits, got n={n} (must match "
                         f"exactly to keep zero idle qubits)")
    edges = []
    q = 0
    for _ in range(n_4q):
        edges += [(q, q + 1), (q + 1, q + 2), (q + 2, q + 3)]
        q += 4
    for _ in range(n_3q):
        edges += [(q, q + 1), (q + 1, q + 2)]
        q += 3
    return edges


def _edges_single_bare_edge(n, rng, n_3q=19, extra_chain_size=5):
    """Addendum 66: exactly ONE bare 2-qubit edge, plus `n_3q` disjoint
    3-qubit chains, plus one `extra_chain_size`-qubit chain to use up
    the remainder exactly (64 is not reachable by a bare edge plus pure
    3-qubit chains alone: 64-2=62 is not divisible by 3). Default
    (n_3q=19, extra_chain_size=5): 19*3 + 5 + 2 = 64 exactly, zero idle,
    21 components, of which exactly one is a bare 2-qubit edge and none
    of the rest is below 3 qubits. Tests whether a SINGLE bare edge is
    sufficient to prevent the cliff (Addendum 65 found many bare edges
    sufficient; this isolates whether count or mere presence matters).
    """
    n_needed = 2 + n_3q * 3 + extra_chain_size
    if n_needed != n:
        raise ValueError(f"n_3q={n_3q}, extra_chain_size={extra_chain_size} "
                         f"need exactly {n_needed} qubits, got n={n} "
                         f"(must match exactly to keep zero idle qubits)")
    edges = []
    q = 0
    # the single bare edge
    edges.append((q, q + 1))
    q += 2
    # n_3q disjoint 3-qubit chains
    for _ in range(n_3q):
        edges.append((q, q + 1))
        edges.append((q + 1, q + 2))
        q += 3
    # one extra_chain_size-qubit chain to use the remainder exactly
    for i in range(extra_chain_size - 1):
        edges.append((q + i, q + i + 1))
    return edges


def _edges_large_dominant_no_bare_edges(n, rng, n_small=8, small_size=3,
                                        big_size=40):
    """Addendum 65: one connected path of `big_size` qubits, plus
    `n_small` disjoint `small_size`-qubit chains -- **no bare 2-qubit
    edges anywhere**, unlike `mixed_uneven` (Addendum 63), which always
    paired its many small components as bare edges. Tests directly
    whether a LARGE dominant component (default 40 qubits, 62.5% of a
    64-qubit device -- larger than `mixed_uneven`'s own 30-qubit/47%
    fast case) still produces a fast result when the small components
    are chains rather than bare edges, isolating Addendum 64's
    unresolved confound (dominant-component size vs. bare-edge
    presence). Uses the same chain-building logic as
    `_edges_mixed_uneven` (touches every qubit in each small component,
    not just the endpoints -- see Addendum 64's own bug-fix note for why
    this matters for small_size > 2).
    """
    n_needed = n_small * small_size + big_size
    if n_needed != n:
        raise ValueError(f"n_small={n_small}, small_size={small_size}, "
                         f"big_size={big_size} need exactly {n_needed} "
                         f"qubits, got n={n} (must match exactly to keep "
                         f"zero idle qubits)")
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
    """Addendum 63, P1 (primary): `n_small` disjoint `small_size`-qubit
    edges plus ONE large connected path of `big_size` qubits, using
    every one of `n_small*small_size + big_size` qubits (zero idle when
    this equals `n` exactly -- default 17*2+30=64). Component count is
    `n_small + 1` (18 by default, divisible by 3) -- a deliberately
    extreme, non-uniform size distribution, unlike anything
    `merged_pairs`'s own mild mixtures can produce.
    """
    n_needed = n_small * small_size + big_size
    if n_needed != n:
        raise ValueError(f"n_small={n_small}, small_size={small_size}, "
                         f"big_size={big_size} need exactly {n_needed} "
                         f"qubits, got n={n} (must match exactly to keep "
                         f"zero idle qubits)")
    edges = []
    q = 0
    for _ in range(n_small):
        # Addendum 64 bug fix: for small_size > 2, a single edge from
        # q to q+small_size-1 skips the intermediate qubits entirely,
        # leaving them untouched (idle) -- caught before running by
        # checking small_size=3's actual component/idle count against
        # its intended zero-idle design (found 17 idle qubits instead of
        # 0). Fixed to build a genuine (small_size-1)-edge chain
        # touching every qubit in the component, matching how every
        # other multi-qubit component in this project (merged_pairs'
        # 4-qubit chains, etc.) is built.
        for i in range(small_size - 1):
            edges.append((q + i, q + i + 1))
        q += small_size
    for i in range(big_size - 1):
        edges.append((q + i, q + i + 1))
    return edges


def _edges_dense_pairs_with_idle(n, rng, j=0):
    """Addendum 54, P2: `dense_pairs` structure (maximal disjointness --
    every component is a single edge) built on only `n - j` of the `n`
    qubits, leaving `j` qubits with zero interactions (idle wires in the
    circuit) while every OTHER qubit is still in exactly one disjoint
    pair -- the opposite manipulation from `merged_pairs`: component
    count and pair structure are held maximal/fixed, idle-qubit count is
    varied instead. `j=0` reproduces `dense_pairs` exactly (sanity check,
    P3). `j` should be even so pairs stay intact; the idle qubits are the
    LAST `j` qubit indices, left with no edges at all.
    """
    if j < 0 or j % 2 != 0:
        raise ValueError(f"j={j} must be a non-negative even number")
    n_active = n - j
    return _edges_dense_pairs(n_active, rng)  # idle qubits are simply
    # never referenced by any edge; the circuit still has n qubits total
    # (built by the caller), so they exist as idle wires, not as a
    # smaller circuit.


CIRCUIT_FAMILIES = {
    "dense_pairs": _edges_dense_pairs,
    "linear_chain": _edges_linear_chain,
    "random_regular": _edges_random_regular,
    "ghz_star": _edges_ghz_star,
    "k_chains": _edges_k_chains,
    "merged_pairs": _edges_merged_pairs,
    "dense_pairs_with_idle": _edges_dense_pairs_with_idle,
    "merged_pairs_variant": _edges_merged_pairs_variant,
    "uniform_2q": _edges_uniform_2q,
    "balanced_3q4q": _edges_balanced_3q4q,
    "mixed_uneven": _edges_mixed_uneven,
    "shrinking_dominant": _edges_mixed_uneven,
    "large_dominant_no_bare_edges": _edges_large_dominant_no_bare_edges,
    "single_bare_edge": _edges_single_bare_edge,
}


def build_circuit_from_family(num_qubits: int, family: str,
                              gates_per_pair: int = GATES_PER_PAIR,
                              seed: int = 0, k: int = 1, m: int = 0,
                              j: int = 0, order: str = "sequential",
                              small_size: int = 2, big_size: int = 30):
    """Builds a circuit whose interaction graph is given by `family`, using
    the same random-unitary blocks per edge as the original generator, so
    the only thing that changes between families is WHICH pairs interact.
    `k` is only used by `k_chains` (Addendum 52); `m` only by
    `merged_pairs`/`merged_pairs_variant` (Addendum 54, P1 / Addendum
    60); `j` only by `dense_pairs_with_idle` (Addendum 54, P2); `order`
    only by `merged_pairs_variant` (Addendum 60); ignored otherwise.
    Returns (circuit, edge_list)."""
    if family not in CIRCUIT_FAMILIES:
        raise ValueError(f"unknown family {family!r}; "
                         f"known: {sorted(CIRCUIT_FAMILIES)}")
    rng = np.random.default_rng(seed)
    if family == "k_chains":
        edges = _edges_k_chains(num_qubits, rng, k=k)
    elif family == "merged_pairs":
        edges = _edges_merged_pairs(num_qubits, rng, m=m)
    elif family == "merged_pairs_variant":
        edges = _edges_merged_pairs_variant(num_qubits, rng, m=m, order=order)
    elif family == "dense_pairs_with_idle":
        edges = _edges_dense_pairs_with_idle(num_qubits, rng, j=j)
    elif family == "shrinking_dominant":
        # Addendum 64: reuses _edges_mixed_uneven's own logic, with
        # small_size/big_size overridable via CLI so the two-point
        # family (small_size=2/big=30, the sanity check reproducing
        # mixed_uneven exactly; small_size=3/big=13, the new point) can
        # be run without duplicating the construction function.
        edges = _edges_mixed_uneven(num_qubits, rng, n_small=17,
                                    small_size=small_size, big_size=big_size)
    else:
        edges = CIRCUIT_FAMILIES[family](num_qubits, rng)
    qc = QuantumCircuit(num_qubits)
    for a, b in edges:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc, edges


def check_routing_validity(qc: QuantumCircuit, cm: CouplingMap) -> tuple[bool, int]:
    edges = set(map(tuple, cm.get_edges()))
    violations = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            i = qc.find_bit(inst.qubits[0]).index
            j = qc.find_bit(inst.qubits[1]).index
            if (i, j) not in edges and (j, i) not in edges:
                violations += 1
    return violations == 0, violations


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


def matching_feasibility(cm: CouplingMap, n_pairs: int) -> dict:
    """Independent (non-Qiskit) check of whether the circuit's interaction
    graph -- `n_pairs` disjoint edges -- can be embedded in the coupling
    graph at all.

    This is prediction P4's evidence: if a perfect matching exists and
    VF2Layout still reports NO_SOLUTION_FOUND, the failure is a search-
    budget failure and not an infeasibility, which is the whole basis of
    explanation (A).
    """
    g = nx.Graph()
    g.add_nodes_from(range(cm.size()))
    g.add_edges_from([tuple(e) for e in cm.get_edges()])
    m = nx.max_weight_matching(g, maxcardinality=True)
    max_matching = len(m)
    return dict(
        max_matching_size=max_matching,
        n_pairs_required=n_pairs,
        embedding_feasible=bool(max_matching >= n_pairs),
        perfect_matching_exists=bool(max_matching * 2 == cm.size()),
        matching_slack=max_matching - n_pairs,
    )


def timed_transpile(qc: QuantumCircuit, cm: CouplingMap, level: int) -> dict:
    """One transpile, instrumented per pass.

    `transpile(callback=...)` fires once per pass with that pass's own
    execution time, so summing by pass name attributes the total. The
    live `property_set` is also handed to the callback, which is how
    VF2Layout's stop reason is captured -- `transpile()` itself does not
    return the property set.
    """
    per_pass: dict[str, float] = {}
    state = {"stop_reason": None, "post_stop_reason": None}
    # Which passes appeared in the callback at all -- separates "the pass
    # was skipped" from "the pass ran and returned in ~0 ms", which an
    # exact 0.0 timing alone cannot distinguish (Addenda 43-44).
    passes_seen: set[str] = set()

    def cb(pass_, dag, time, property_set, count, **_):
        name = type(pass_).__name__
        passes_seen.add(name)
        per_pass[name] = per_pass.get(name, 0.0) + float(time)
        reason = property_set.get("VF2Layout_stop_reason")
        if reason is not None:
            state["stop_reason"] = getattr(reason, "value", str(reason))
        post_reason = property_set.get("VF2PostLayout_stop_reason")
        if post_reason is not None:
            state["post_stop_reason"] = getattr(post_reason, "value", str(post_reason))

    t0 = time.perf_counter()
    out = transpile(qc, coupling_map=cm, basis_gates=BASIS_GATES,
                    optimization_level=level, seed_transpiler=SEED_TRANSPILER,
                    callback=cb)
    total_ms = (time.perf_counter() - t0) * 1000.0

    layout_ms = sum(v for k, v in per_pass.items() if k in LAYOUT_PASSES) * 1000.0
    routing_ms = sum(v for k, v in per_pass.items() if k in ROUTING_PASSES) * 1000.0
    accounted_ms = sum(per_pass.values()) * 1000.0
    slowest = max(per_pass.items(), key=lambda kv: kv[1]) if per_pass else ("", 0.0)

    return dict(
        out=out,
        total_ms=total_ms,
        layout_ms=layout_ms,
        routing_ms=routing_ms,
        accounted_ms=accounted_ms,
        layout_frac=(layout_ms / total_ms) if total_ms > 0 else None,
        vf2layout_ms=per_pass.get("VF2Layout", 0.0) * 1000.0,
        sabrelayout_ms=per_pass.get("SabreLayout", 0.0) * 1000.0,
        vf2postlayout_ms=per_pass.get("VF2PostLayout", 0.0) * 1000.0,
        sabreswap_ms=per_pass.get("SabreSwap", 0.0) * 1000.0,
        slowest_pass=slowest[0],
        slowest_pass_ms=slowest[1] * 1000.0,
        n_passes=len(per_pass),
        vf2_stop_reason=state["stop_reason"] or "",
        vf2post_stop_reason=state["post_stop_reason"] or "",
        vf2postlayout_ran=("VF2PostLayout" in passes_seen),
        vf2layout_ran=("VF2Layout" in passes_seen),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--spares", type=str, default="0,1,2,3,4,5,6,8,10,12,16,20,24",
                    help="deliberately fine-grained near zero -- locating the "
                         "edge is the entire point of this script")
    ap.add_argument("--levels", type=str, default="1,3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--families", type=str,
                    default="dense_pairs,linear_chain,random_regular,ghz_star",
                    help="comma-separated interaction-graph families to "
                         "sweep. dense_pairs is this project's standing "
                         "family and is included by default so every run "
                         "carries its own baseline rather than relying on "
                         "a comparison against a different session's data")
    ap.add_argument("--ks", type=str, default="1",
                    help="comma-separated k values for the k_chains family "
                         "(Addendum 52) -- ignored for other families. "
                         "k=n/2 reproduces dense_pairs; k=1 reproduces "
                         "linear_chain; both are natural sanity-check "
                         "values to include alongside any intermediate k")
    ap.add_argument("--ms", type=str, default="0",
                    help="comma-separated m values for merged_pairs "
                         "(Addendum 54, P1) -- reduces component count "
                         "while keeping zero idle qubits. m=0 reproduces "
                         "dense_pairs (sanity check).")
    ap.add_argument("--js", type=str, default="0",
                    help="comma-separated j values for "
                         "dense_pairs_with_idle (Addendum 54, P2) -- "
                         "introduces idle qubits while keeping maximal "
                         "disjointness. j=0 reproduces dense_pairs "
                         "(sanity check). Must be even.")
    ap.add_argument("--order", type=str, default="sequential",
                    choices=["sequential", "reverse", "random"],
                    help="merge order for merged_pairs_variant "
                         "(Addendum 60) -- ONE value per run (run the "
                         "script multiple times to compare orders, "
                         "matching Addendum 60's own design of one "
                         "invocation per order). --ms is shared with "
                         "merged_pairs for the m values to sweep.")
    ap.add_argument("--small-size", type=int, default=2,
                    help="qubits per small component for "
                         "shrinking_dominant (Addendum 64) -- default 2 "
                         "reproduces mixed_uneven exactly (sanity check); "
                         "3 is the only other valid value at n=64, "
                         "component count 18 (see Addendum 64's own "
                         "pre-registration for why only two points exist "
                         "in this design). Ignored by other families.")
    ap.add_argument("--big-size", type=int, default=30,
                    help="qubits in the single dominant component for "
                         "shrinking_dominant (Addendum 64) -- must equal "
                         "64 - 17*small_size exactly to keep zero idle "
                         "qubits and 18 components (30 for small_size=2, "
                         "13 for small_size=3). Ignored by other "
                         "families.")
    ap.add_argument("--gates-per-pair", type=int, default=GATES_PER_PAIR)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    grid_size = args.rows * args.cols
    spares = [int(s) for s in args.spares.split(",")]
    levels = [int(l) for l in args.levels.split(",")]
    seeds = list(range(args.seeds))
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for f in families:
        if f not in CIRCUIT_FAMILIES:
            raise SystemExit(f"unknown family {f!r}; known: "
                             f"{sorted(CIRCUIT_FAMILIES)}")

    print("=" * 100)
    print(f"OCCUPANCY SWEEP: {args.rows}x{args.cols} grid ({grid_size} physical qubits)")
    print(f"spares={spares}  levels={levels}  seeds={seeds}  repeats={args.repeats}")
    print("Measuring Qiskit only -- PSF-Zero is not involved in this experiment.")
    print("=" * 100)

    rows_out = []
    t_start = time.perf_counter()

    for spare in spares:
        n = grid_size - spare
        if n < 2:
            print(f"  spare={spare} skipped (fewer than 2 circuit qubits)")
            continue
        cm = CouplingMap.from_grid(args.rows, args.cols)
        n_pairs = len(range(0, n - 1, 2))
        feas = matching_feasibility(cm, n_pairs)
        occupancy = n / grid_size

        ks_list = [int(k) for k in args.ks.split(",")] if "k_chains" in families else [None]
        ms_list = ([int(x) for x in args.ms.split(",")]
                  if ("merged_pairs" in families or "merged_pairs_variant" in families)
                  else [None])
        js_list = [int(x) for x in args.js.split(",")] if "dense_pairs_with_idle" in families else [None]
        for family in families:
         param_list = (ks_list if family == "k_chains"
                       else ms_list if family in ("merged_pairs", "merged_pairs_variant")
                       else js_list if family == "dense_pairs_with_idle"
                       else [None])
         param_name = ("k" if family == "k_chains"
                       else "m" if family in ("merged_pairs", "merged_pairs_variant")
                       else "j" if family == "dense_pairs_with_idle"
                       else None)
         for param_val in param_list:
          for seed in seeds:
            kw = {}
            if param_name == "k":
                kw["k"] = param_val if param_val is not None else 1
            elif param_name == "m":
                kw["m"] = param_val if param_val is not None else 0
                if family == "merged_pairs_variant":
                    kw["order"] = args.order
            elif param_name == "j":
                kw["j"] = param_val if param_val is not None else 0
            if family == "shrinking_dominant":
                kw["small_size"] = args.small_size
                kw["big_size"] = args.big_size
            try:
                qc, edges = build_circuit_from_family(
                    n, family, gates_per_pair=args.gates_per_pair, seed=seed,
                    **kw)
            except Exception as exc:
                print(f"  spare={spare:2d} family={family} "
                      f"{param_name}={param_val} seed={seed} "
                      f"CIRCUIT BUILD FAILED -- {type(exc).__name__}: {exc}")
                continue
            n_edges = len(edges)
            max_interaction_degree = max(
                (sum(1 for e in edges if q in e) for q in range(n)), default=0)
            for level in levels:
                try:
                    timed_transpile(qc, cm, level)  # warm-up, discarded
                except Exception as exc:
                    print(f"  spare={spare:2d} family={family} "
                          f"{param_name}={param_val} seed={seed} L{level} "
                          f"WARM-UP FAILED -- {type(exc).__name__}: {exc}")

                for rep in range(args.repeats):
                    base = dict(
                        family=family,
                        k=(param_val if family == "k_chains" else None),
                        m=(param_val if family in ("merged_pairs", "merged_pairs_variant") else None),
                        order=(args.order if family == "merged_pairs_variant" else None),
                        j=(param_val if family == "dense_pairs_with_idle" else None),
                        n_interaction_edges=n_edges,
                        max_interaction_degree=max_interaction_degree,
                        spare=spare, n=n, occupancy=occupancy, n_pairs=n_pairs,
                        seed=seed, optimization_level=level, repeat=rep,
                        elapsed_since_start_s=time.perf_counter() - t_start,
                        **feas,
                    )
                    try:
                        r = timed_transpile(qc, cm, level)
                        ok, violations = check_routing_validity(r["out"], cm)
                        if not ok:
                            raise RuntimeError(
                                f"output has {violations} coupling violation(s)")
                        rows_out.append(dict(
                            base,
                            time_ms=r["total_ms"],
                            layout_ms=r["layout_ms"],
                            routing_ms=r["routing_ms"],
                            accounted_ms=r["accounted_ms"],
                            layout_frac=r["layout_frac"],
                            vf2layout_ms=r["vf2layout_ms"],
                            sabrelayout_ms=r["sabrelayout_ms"],
                            vf2postlayout_ms=r["vf2postlayout_ms"],
                            sabreswap_ms=r["sabreswap_ms"],
                            slowest_pass=r["slowest_pass"],
                            slowest_pass_ms=r["slowest_pass_ms"],
                            n_passes=r["n_passes"],
                            vf2_stop_reason=r["vf2_stop_reason"],
                            vf2post_stop_reason=r["vf2post_stop_reason"],
                            vf2postlayout_ran=r["vf2postlayout_ran"],
                            vf2layout_ran=r["vf2layout_ran"],
                            two_qubit_gates=two_qubit_gate_count(r["out"]),
                            depth=r["out"].depth(),
                            error="",
                        ))
                    except Exception as exc:
                        rows_out.append(dict(
                            base, time_ms=None, layout_ms=None, routing_ms=None,
                            accounted_ms=None, layout_frac=None, vf2layout_ms=None,
                            sabrelayout_ms=None, vf2postlayout_ms=None,
                            sabreswap_ms=None, slowest_pass="", slowest_pass_ms=None,
                            n_passes=None, vf2_stop_reason="",
                            vf2post_stop_reason="", vf2postlayout_ran=None,
                            vf2layout_ran=None,
                            two_qubit_gates=None, depth=None,
                            error=f"{type(exc).__name__}: {exc}",
                        ))

          print(f"  spare={spare:2d} family={family} "
                f"{param_name}={param_val} "
                f"(occupancy {occupancy:5.1%}, n={n:3d}) seed={seed} "
                f"done  (elapsed {time.perf_counter() - t_start:7.1f}s)")

    df = pd.DataFrame(rows_out)
    df["cpu"] = platform.processor()
    df["platform"] = platform.platform()
    df["python_version"] = platform.python_version()
    df["qiskit_version"] = qiskit.__version__
    df["grid_rows"] = args.rows
    df["grid_cols"] = args.cols
    df["grid_size"] = grid_size
    df["gates_per_pair"] = args.gates_per_pair
    df["seed_transpiler"] = SEED_TRANSPILER

    good = df[df.error == ""]
    if not good.empty:
        print("=" * 100)
        print("Median total compile time (ms) by (spare, optimization_level):")
        print(good.pivot_table(index=["family", "spare"], columns="optimization_level",
                               values="time_ms", aggfunc="median").round(2))
        print()
        print("Median layout-stage fraction of total time:")
        print(good.pivot_table(index="spare", columns="optimization_level",
                               values="layout_frac", aggfunc="median").round(3))
        print()
        print("VF2Layout stop reason by (spare, optimization_level) -- the mechanism:")
        print(good.groupby(["family", "spare", "optimization_level"])["vf2_stop_reason"]
              .agg(lambda s: "/".join(sorted(set(x for x in s if x)) or ["(not run)"])))
        print()
        print("VF2PostLayout stop reason by (spare, optimization_level):")
        print(good.groupby(["family", "spare", "optimization_level"])["vf2post_stop_reason"]
              .agg(lambda s: "/".join(sorted(set(x for x in s if x)) or ["(no reason set)"])))
        print()
        print("Did VF2PostLayout run at all (appeared in the pass callback)?")
        print(good.groupby(["family", "spare", "optimization_level"])["vf2postlayout_ran"]
              .agg(lambda s: "/".join(sorted(set(str(x) for x in s)))))
        print()
        print("Step-to-step ratio of median time (this is where the cliff shows up):")
        for level in sorted(good.optimization_level.unique()):
            sub = (good[good.optimization_level == level]
                   .groupby("spare")["time_ms"].median().sort_index())
            print(f"  optimization_level={level}")
            prev_spare, prev_val = None, None
            for sp, val in sub.items():
                if prev_val is not None and val > 0:
                    print(f"    spare {prev_spare:2d} -> {sp:2d}: "
                          f"{prev_val:9.2f} -> {val:9.2f} ms   "
                          f"ratio {prev_val / val:7.2f}x")
                prev_spare, prev_val = sp, val
        print()
        print("Feasibility (independent of Qiskit -- prediction P4):")
        print(good.groupby("spare")[["n_pairs_required", "max_matching_size",
                                     "embedding_feasible", "matching_slack"]].first())
        print("=" * 100)

    cpu_tag = (platform.processor() or "unknown_cpu").replace(" ", "_").replace(",", "")
    base_name = (f"circuit_family_sweep_{args.rows}x{args.cols}_{cpu_tag}_"
                 f"{date.today().isoformat()}")
    out_path = os.path.join(args.out_dir, base_name + ".csv")
    if os.path.exists(out_path):
        i = 2
        while os.path.exists(os.path.join(args.out_dir, f"{base_name}_run{i}.csv")):
            i += 1
        out_path = os.path.join(args.out_dir, f"{base_name}_run{i}.csv")

    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df)} rows)")
    print(
        "Before pasting this file's contents anywhere outside this machine: "
        "check it for a local file path or any other machine-identifying "
        "string beyond the CPU signature recorded above, per this project's "
        "standing record-keeping rules."
    )


if __name__ == "__main__":
    main()
