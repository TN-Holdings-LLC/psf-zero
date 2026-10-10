"""Tests for candidate psf_compile 2026-10-10.c35 (changelog item 63): its default is c34 with ABSORB_SYNTH="qiskit",
and ABSORB_SYNTH="psf" gives c34's outputs; outputs are exact.

Run from the repository root:  python -m pytest patches/psf_compile_c35_2026-10-10/test_c35.py -q
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    c34 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c34_2026-10-10", "psf_compile.py"),
                        "psf_compile_c34_for_c35_test")
    c35 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c35_test")
    c34.WARN_WITHOUT_TARGET = c35.WARN_WITHOUT_TARGET = False
    return dict(c34=c34, c35=c35)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits(n=8, seed=63_100):
    from qiskit.circuit.random import random_circuit
    return [random_circuit(3 + k % 7, 6 + 2 * k, max_operands=2, measure=k % 2 == 0, seed=seed + k) for k in range(n)]


def _call(mod, qc, be):
    return mod.compile_for_hardware(qc, backend=be, entangling_basis="cx", layout_search=True, seed_transpiler=0)


def test_version(mods):
    assert mods["c35"].VERSION == "2026-10-10.c35" and mods["c35"].ABSORB_SYNTH == "qiskit"


def test_default_is_c34_with_qiskit_synthesis(mods, torino):
    c34 = mods["c34"]
    c34.ABSORB_SYNTH = "qiskit"
    try:
        for qc in _circuits():
            assert _sig(_call(c34, qc, torino)) == _sig(_call(mods["c35"], qc, torino))
    finally:
        c34.ABSORB_SYNTH = "psf"


def test_psf_switch_is_c34(mods, torino):
    c35 = mods["c35"]
    c35.ABSORB_SYNTH = "psf"
    try:
        for qc in _circuits(5, 63_200):
            assert _sig(_call(mods["c34"], qc, torino)) == _sig(_call(c35, qc, torino))
    finally:
        c35.ABSORB_SYNTH = "qiskit"


def test_exact(mods, torino):
    for qc in _circuits(6, 63_300):
        bare = qc.remove_final_measurements(inplace=False)
        assert mods["c35"]._implements(bare, _call(mods["c35"], bare, torino)) is True
