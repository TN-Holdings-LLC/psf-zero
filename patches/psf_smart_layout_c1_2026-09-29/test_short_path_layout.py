"""Tests for the candidate psf_smart_layout 2026-09-29.c1: the short-path shortcut
(Stage 0b) and the corrected feasibility check."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import networkx as nx
import pytest
from qiskit.transpiler import CouplingMap

import psf_smart_layout as psl


def _undirected(cm):
    return {tuple(sorted(e)) for e in cm.get_edges()}


def _check_valid(cm, pairs, layout_map):
    edges = _undirected(cm)
    assert len(set(layout_map.values())) == len(layout_map)
    for a, b in pairs:
        assert tuple(sorted((layout_map[a], layout_map[b]))) in edges


def _paths_interaction(n_triples, n_pairs):
    """Logical triples (3k, 3k+1, 3k+2) as paths, then pairs."""
    inter, q = [], 0
    for _ in range(n_triples):
        inter += [(q, q + 1), (q + 1, q + 2)]
        q += 3
    for _ in range(n_pairs):
        inter.append((q, q + 1))
        q += 2
    return inter, q


def test_version_is_candidate():
    assert psl.LAYOUT_VERSION == "2026-09-29.c1"


def test_heavy_hex_fully_occupied_takes_path_shortcut():
    cm = CouplingMap.from_heavy_hex(5)
    g = nx.Graph(list(_undirected(cm)))
    m = len(nx.max_weight_matching(g, maxcardinality=True))
    k3 = cm.size() - 2 * m
    assert k3 > 0
    inter, n = _paths_interaction(k3, m - k3)
    assert n == cm.size()
    lm, info = psl.smart_vf2_layout(cm, inter, n)
    assert info["phase"] == 0 and info["order_name"] == "path_direct"
    assert len(lm) == n
    _check_valid(cm, inter, lm)


def test_line_of_three():
    cm = CouplingMap.from_line(3)
    lm, info = psl.smart_vf2_layout(cm, [(0, 1), (1, 2)], 3)
    assert info["order_name"] == "path_direct"
    _check_valid(cm, [(0, 1), (1, 2)], lm)
    assert lm[1] == 1


def test_longer_path_is_not_a_short_path():
    assert psl._short_path_components([(0, 1), (1, 2), (2, 3)]) is None
    assert psl._short_path_components([(0, 1), (2, 3)]) is None  # no triple: matching shortcut's case
    assert psl._short_path_components([(0, 1), (1, 2), (0, 2)]) is None  # triangle
    assert psl._short_path_components([(0, 1), (1, 2), (3, 4)]) == ([(3, 4)], [(0, 1, 2)])


def test_feasibility_uses_interaction_matching(monkeypatch):
    # Two 3-qubit paths on a line of 6: 4 interaction edges, but only 2 disjoint
    # edges are needed. The old check (4 > matching 3) called this infeasible.
    monkeypatch.setattr(psl, "USE_PATH_SHORTCUT", False)
    cm = CouplingMap.from_line(6)
    inter = [(0, 1), (1, 2), (3, 4), (4, 5)]
    assert psl._interaction_matching_size(inter) == 2
    lm, info = psl.smart_vf2_layout(cm, inter, 6)
    assert info["feasible"] is True and info["phase"] in (1, 2)
    _check_valid(cm, inter, lm)


def test_edge_weights_skip_path_shortcut():
    cm = CouplingMap.from_line(3)
    w = {(0, 1): 1, (1, 2): 1}
    lm, info = psl.smart_vf2_layout(cm, [(0, 1), (1, 2)], 3, edge_weights=w)
    assert info["order_name"] != "path_direct"
    _check_valid(cm, [(0, 1), (1, 2)], lm)


def test_construction_failure_falls_back_to_vf2():
    # Star with centre 0 and leaves 1..3: a maximum matching has one edge, so a
    # triple can attach, but two triples cannot fit at all.
    cm = CouplingMap([(0, 1), (0, 2), (0, 3)])
    assert psl.short_path_layout(cm, [], [(0, 1, 2), (3, 4, 5)]) is None
    lm, info = psl.smart_vf2_layout(cm, [(0, 1), (1, 2)], 3)
    assert info["order_name"] == "path_direct"
    _check_valid(cm, [(0, 1), (1, 2)], lm)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_heavy_hex_partial_occupation(d):
    cm = CouplingMap.from_heavy_hex(d)
    g = nx.Graph(list(_undirected(cm)))
    m = len(nx.max_weight_matching(g, maxcardinality=True))
    k3 = cm.size() - 2 * m
    inter, n = _paths_interaction(k3, max(0, m - k3 - 2))
    lm, info = psl.smart_vf2_layout(cm, inter, n)
    assert info["found"]
    _check_valid(cm, inter, lm)
