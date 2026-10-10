"""Tests for candidate psf_compile 2026-10-10.c36 (changelog item 64): release 2026-10-10.3 with
routing_optimization_level="auto", which is 3 when the device's Target is given and 1 otherwise.

Run from the repository root:  python -m pytest patches/psf_compile_c36_2026-10-10/test_c36.py -q
"""
import contextlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_for_c36_test")
    c36 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c36_test")
    rel.WARN_WITHOUT_TARGET = c36.WARN_WITHOUT_TARGET = False
    return dict(rel=rel, c36=c36)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits(n=6, seed=64_100):
    from qiskit.circuit.random import random_circuit
    return [random_circuit(4 + k, 8 + 2 * k, max_operands=2, measure=k % 2 == 0, seed=seed + k) for k in range(n)]


def _call(mod, qc, **kw):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return mod.compile_for_hardware(qc, entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)


def test_version(mods):
    assert mods["c36"].VERSION == "2026-10-10.c36"
    assert mods["rel"].VERSION == "2026-10-10.3"


def test_file_is_the_release_plus_item_64():
    with open(os.path.join(REPO, "psf_compile.py"), encoding="utf-8") as f:
        a = f.read().splitlines()
    with open(os.path.join(HERE, "psf_compile.py"), encoding="utf-8") as f:
        b = f.read().splitlines()
    assert len(b) - len(a) == 8


def test_with_the_device_level_3(mods, torino):
    """The default with backend= is the release's call with routing_optimization_level=3, by value."""
    for qc in _circuits():
        assert _sig(_call(mods["c36"], qc, backend=torino)) == \
            _sig(_call(mods["rel"], qc, backend=torino, routing_optimization_level=3))


def test_without_the_device_unchanged(mods, torino):
    """Without a Target, the release's default call, by value."""
    cm = torino.target.build_coupling_map()
    basis = [g for g in torino.target.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    for qc in _circuits(seed=64_200):
        assert _sig(_call(mods["c36"], qc, coupling_map=cm, basis_gates=basis)) == \
            _sig(_call(mods["rel"], qc, coupling_map=cm, basis_gates=basis))


def test_explicit_level_keeps_its_meaning(mods, torino):
    for qc in _circuits(n=3, seed=64_300):
        assert _sig(_call(mods["c36"], qc, backend=torino, routing_optimization_level=1)) == \
            _sig(_call(mods["rel"], qc, backend=torino))


def test_no_failed_element_with_the_device(mods, torino):
    tgt = torino.target
    edges, qubits = mods["c36"]._failed_elements(tgt, 0.5)
    for qc in _circuits(seed=64_400):
        assert not mods["c36"]._uses_failed(_call(mods["c36"], qc, backend=torino), edges, qubits)
