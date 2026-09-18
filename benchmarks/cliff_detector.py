"""cliff_detector.py -- a candidate, cheap, pre-compile detector for the
two structural conditions Addenda 51-55 found necessary for the
occupancy cliff: (a) the interaction graph has multiple disjoint
components, and (b) no qubit is idle (every qubit participates in at
least one interaction).

**This is a detector for the necessary conditions found so far, not a
guarantee.** Addendum 56-57 showed these two conditions are not
sufficient at every grid size (no cliff at 4x4 despite satisfying both).
This tool answers "does this circuit have the structural shape that
COULD cliff," not "will this circuit cliff on this specific device."
That gap is stated explicitly in every function's docstring below, not
just here.
"""
from __future__ import annotations

import time
import networkx as nx


def interaction_graph_from_edges(n_qubits: int, edges: list[tuple[int, int]]) -> nx.Graph:
    """Builds the interaction graph a real detector would extract from a
    QuantumCircuit by walking its two-qubit gates -- here taking the edge
    list directly, since this module is tested against this project's own
    circuit-family generators (which already expose their edges) rather
    than against live QuantumCircuit objects. Isolated qubits (present in
    n_qubits but touched by no edge) are added explicitly, since a
    zero-degree node would otherwise never appear in the graph at all --
    and it is exactly the node whose absence-or-presence this detector
    needs to see.
    """
    g = nx.Graph()
    g.add_nodes_from(range(n_qubits))
    g.add_edges_from(edges)
    return g


def detect_cliff_risk_shape(n_qubits: int, edges: list[tuple[int, int]]) -> dict:
    """Checks the two structural conditions Addenda 51-55 found necessary
    (not sufficient -- see module docstring) for the occupancy cliff, plus
    timing for the check itself.

    Returns a dict with:
      - n_components: number of connected components in the interaction
        graph (isolated qubits count as their own singleton component)
      - n_idle_qubits: qubits with degree 0 (touched by no two-qubit gate)
      - multiple_components: n_components > 1
      - zero_idle_qubits: n_idle_qubits == 0
      - cliff_risk_shape: multiple_components AND zero_idle_qubits --
        satisfies BOTH necessary conditions found in Addenda 51-55.
        **Does not itself confirm a cliff will occur** -- see Addendum
        56-57 (4x4 satisfies this and shows no cliff; grid size is a
        third, not-yet-understood factor).
      - detection_time_s: wall-clock time for this function's own graph
        analysis, excluding circuit construction -- the number that
        matters for "is this cheap enough to run before every compile."
    """
    t0 = time.perf_counter()
    g = interaction_graph_from_edges(n_qubits, edges)
    n_components = nx.number_connected_components(g)
    idle_qubits = [q for q in g.nodes if g.degree(q) == 0]
    n_idle = len(idle_qubits)
    detection_time_s = time.perf_counter() - t0

    multiple_components = n_components > 1
    zero_idle = n_idle == 0

    return dict(
        n_qubits=n_qubits,
        n_edges=len(edges),
        n_components=n_components,
        n_idle_qubits=n_idle,
        multiple_components=multiple_components,
        zero_idle_qubits=zero_idle,
        cliff_risk_shape=(multiple_components and zero_idle),
        detection_time_s=detection_time_s,
    )
