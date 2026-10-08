"""Tests for candidate psf_compile 2026-10-09.c25 (changelog item 52: maximum matching sizes in the layout search
come from rustworkx, the coupling map's kept) against release 2026-10-07.1, on which it is based.

The candidate changes only how two sizes are computed, so the tests check that every size equals networkx's (the
release's). That outputs are unchanged end to end is checked by benchmarks/c25_identity.py.

Run from the repository root:  python -m pytest patches/psf_compile_c25_2026-10-09/test_c25.py -q
"""
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    rel = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout_rel_c25_test")
    c25 = H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout_c25_test")
    return dict(rel=rel, c25=c25)


def nx_size(num_nodes, edges):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edges)
    return len(nx.max_weight_matching(g, maxcardinality=True))


def test_versions(mods):
    assert mods["c25"].LAYOUT_VERSION.startswith("2026-10-09.c25")
    assert mods["rel"].LAYOUT_VERSION == "2026-10-01.1"


def test_matching_size_equals_networkx_on_random_graphs(mods):
    rng = random.Random(20261009)
    for _ in range(400):
        n = rng.randint(1, 40)
        p = rng.choice((0.02, 0.05, 0.1, 0.3, 0.8))
        edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < p]
        edges += [(b, a) for a, b in edges if rng.random() < 0.3]  # both directions, as a coupling map lists them
        assert mods["c25"]._max_matching_size(n, edges) == nx_size(n, edges)


def test_interaction_matching_size_equals_release(mods):
    rng = random.Random(52)
    for _ in range(300):
        labels = rng.sample(range(200), rng.randint(2, 30))  # sparse qubit indices, as a large device's circuit has
        pairs = [(a, b) for a in labels for b in labels if a < b and rng.random() < 0.15]
        if not pairs:
            continue
        assert mods["c25"]._interaction_matching_size(pairs) == mods["rel"]._interaction_matching_size(pairs)
    assert mods["c25"]._interaction_matching_size([]) == 0


def test_feasibility_equals_release_on_coupling_maps(mods):
    from qiskit.transpiler import CouplingMap
    from qiskit_ibm_runtime import fake_provider
    maps = [CouplingMap.from_line(20), CouplingMap.from_grid(5, 6), CouplingMap.from_heavy_hex(5),
            CouplingMap.from_full(7), fake_provider.FakeTorino().coupling_map, CouplingMap.from_line(1)]
    for _ in range(2):  # the second pass reads the kept sizes
        for cm in maps:
            ref = nx_size(cm.size(), [tuple(e) for e in cm.get_edges()])
            for k in (0, ref - 1, ref, ref + 1):
                assert mods["c25"]._has_feasible_matching(cm, k) == mods["rel"]._has_feasible_matching(cm, k) == (
                    ref >= k)


def test_kept_sizes_are_bounded(mods):
    from qiskit.transpiler import CouplingMap
    for n in range(2, 2 + 3 * mods["c25"]._PHYSICAL_MATCHING_MAX):
        mods["c25"]._has_feasible_matching(CouplingMap.from_line(n), 1)
    assert len(mods["c25"]._PHYSICAL_MATCHING) == mods["c25"]._PHYSICAL_MATCHING_MAX
