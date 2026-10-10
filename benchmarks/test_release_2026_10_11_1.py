"""Tests for release psf_compile 2026-10-11.1: candidate 2026-10-10.c36 (changelog item 64, routing at Qiskit's level 3
whenever the device's Target is given), accepted by the pre-registered C36-VAL (Addenda 440-441). The candidate's
own tests are patches/psf_compile_c36_2026-10-10/test_c36.py. The previous release, 2026-10-10.3, is kept unchanged
in patches/psf_compile_release_2026-10-10.3/.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_11_1.py -q
"""
import contextlib
import hashlib
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

REL = os.path.join(REPO, "psf_compile.py")
C36 = os.path.join(REPO, "patches", "psf_compile_c36_2026-10-10", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-10.3", "psf_compile.py")


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip() for ln in f.read().replace("\r\n", "\n").split("\n")]


def nsha(path):
    ls = lines(path)
    while ls and not ls[-1]:
        ls.pop()
    return hashlib.sha256("\n".join(ls).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(REL, "psf_compile_rel101101_test")
    prev = H.load_module(PREV, "psf_compile_prev101101_test")
    return dict(rel=rel, prev=prev)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits(n=6, seed=81_000):
    from qiskit.circuit.random import random_circuit
    return [random_circuit(4 + k, 8 + 2 * k, max_operands=2, measure=k % 2 == 0, seed=seed + k) for k in range(n)]


def _call(mod, qc, **kw):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return mod.compile_for_hardware(qc, entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-11.1"
    assert mods["prev"].VERSION == "2026-10-10.3"


def test_file_is_c36_except_the_version_lines():
    a, b = lines(REL), lines(C36)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-11.1 -- release") and diff[0][1].startswith("VERSION: 2026-10-10.c36")
    assert diff[1][0].startswith('VERSION = "2026-10-11.1"') and diff[1][1].startswith('VERSION = "2026-10-10.c36"')


def test_candidate_and_previous_release_kept_unchanged():
    assert nsha(C36) == "3fa5f226d2faddc882e4d7b37a7be0e700a7f708819cf9ea1b92c6925d9c4f8f"
    assert nsha(PREV) == "adb025944775c92b7cbb3fe8f6f8a8f496bf3f81f4436057a84b1ad1e9a22956"


def test_with_the_device_routing_level_3(mods, torino):
    """The plain call with backend= is 2026-10-10.3's call with routing_optimization_level=3, by value."""
    for qc in _circuits():
        assert _sig(_call(mods["rel"], qc, backend=torino)) == \
            _sig(_call(mods["prev"], qc, backend=torino, routing_optimization_level=3))


def test_without_the_device_unchanged(mods, torino):
    """Without a Target, 2026-10-10.3's default call, by value."""
    cm = torino.target.build_coupling_map()
    basis = [g for g in torino.target.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    mods["rel"].WARN_WITHOUT_TARGET = mods["prev"].WARN_WITHOUT_TARGET = False
    try:
        for qc in _circuits(seed=81_100):
            assert _sig(_call(mods["rel"], qc, coupling_map=cm, basis_gates=basis)) == \
                _sig(_call(mods["prev"], qc, coupling_map=cm, basis_gates=basis))
    finally:
        mods["rel"].WARN_WITHOUT_TARGET = mods["prev"].WARN_WITHOUT_TARGET = True


def test_no_failed_element_with_the_device(mods, torino):
    tgt = torino.target
    edges, qubits = mods["rel"]._failed_elements(tgt, 0.5)
    for qc in _circuits(seed=81_200):
        assert not mods["rel"]._uses_failed(_call(mods["rel"], qc, backend=torino), edges, qubits)


def test_warns_once_without_the_device(mods, torino):
    rel = mods["rel"]
    rel._WARNED["no_target"] = False
    cm = torino.target.build_coupling_map()
    qc = _circuits(n=1, seed=81_300)[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            for _ in range(2):
                rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                         entangling_basis="cx", seed_transpiler=0)
    assert len([x for x in w if "without the device's Target" in str(x.message)]) == 1
