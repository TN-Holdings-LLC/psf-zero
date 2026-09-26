"""Tests for the matching shortcut in psf_smart_layout (Addendum 192)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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


def test_grid_perfect_matching_takes_shortcut():
    cm = CouplingMap.from_grid(4, 4)
    pairs = [(i, i + 1) for i in range(0, 16, 2)]
    lm, info = psl.smart_vf2_layout(cm, pairs, 16, use_matching_shortcut=True)
    assert info["phase"] == 0 and info["found"] and info["feasible"]
    _check_valid(cm, pairs, lm)


def test_no_matching_is_infeasible():
    cm = CouplingMap.from_line(5)
    lm, info = psl.smart_vf2_layout(cm, [(0, 1), (2, 3), (4, 5)], 6, use_matching_shortcut=True)
    assert lm is None and info["feasible"] is False


def test_non_matching_interaction_uses_vf2():
    cm = CouplingMap.from_grid(3, 3)
    pairs = [(0, 1), (1, 2), (2, 3)]
    lm, info = psl.smart_vf2_layout(cm, pairs, 4, use_matching_shortcut=True)
    assert info["phase"] in (1, 2)
    _check_valid(cm, pairs, lm)


def test_shortcut_off_uses_vf2():
    cm = CouplingMap.from_grid(4, 4)
    pairs = [(i, i + 1) for i in range(0, 16, 2)]
    lm, info = psl.smart_vf2_layout(cm, pairs, 16, use_matching_shortcut=False)
    assert info["phase"] in (1, 2)
    _check_valid(cm, pairs, lm)


def test_weighted_prefers_heavy_edge():
    cm = CouplingMap.from_line(4)
    w = {(0, 1): 5, (1, 2): 100, (2, 3): 5}
    assert psl.matching_layout(cm, [(0, 1)], edge_weights=w) == {0: 1, 1: 2}
    lm = psl.matching_layout(cm, [(0, 1), (2, 3)], edge_weights=w)
    _check_valid(cm, [(0, 1), (2, 3)], lm)


@pytest.mark.parametrize("pairs,expected", [
    ([(0, 1), (2, 3)], True),
    ([(0, 1), (1, 2)], False),
    ([(0, 0)], False),
])
def test_interaction_is_matching(pairs, expected):
    assert psl._interaction_is_matching(pairs) is expected
