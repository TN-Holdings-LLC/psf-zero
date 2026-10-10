"""Tests for candidate psf_compile 2026-10-10.c34 (changelog item 62) against candidate 2026-10-10.c33: with its
switches at their defaults, every output equals c33's; the layout search's matching sizes equal the release's; the
switches give exact outputs.

Run from the repository root:  python -m pytest patches/psf_compile_c34_2026-10-10/test_c34.py -q
"""
import contextlib
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
    rel_lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout_rel_c34")
    c34_lay = H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout_c34")
    c33 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c33_2026-10-10", "psf_compile.py"),
                        "psf_compile_c33_for_c34_test")
    c34 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c34_test")
    c33.WARN_WITHOUT_TARGET = c34.WARN_WITHOUT_TARGET = False
    return dict(c33=c33, c34=c34, rel_lay=rel_lay, c34_lay=c34_lay)


@contextlib.contextmanager
def layout(mod):
    old = sys.modules.get("psf_smart_layout")
    sys.modules["psf_smart_layout"] = mod
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop("psf_smart_layout", None)
        else:
            sys.modules["psf_smart_layout"] = old


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits(n=8, seed=62_100):
    from qiskit.circuit.random import random_circuit
    return [random_circuit(3 + k % 7, 6 + 2 * k, max_operands=2, measure=k % 2 == 0, seed=seed + k) for k in range(n)]


def test_version(mods):
    assert mods["c34"].VERSION == "2026-10-10.c34"


def test_matching_sizes_equal_the_release(mods):
    from qiskit.transpiler import CouplingMap
    rng = random.Random(62)
    for k in range(60):
        n = rng.randint(2, 30)
        pairs = sorted({tuple(sorted(rng.sample(range(n), 2))) for _ in range(rng.randint(1, 3 * n))})
        assert mods["c34_lay"]._interaction_matching_size(pairs) == mods["rel_lay"]._interaction_matching_size(pairs)
        cm = CouplingMap([list(p) for p in pairs] + [list(p[::-1]) for p in pairs])
        for need in (1, n // 3, n // 2):
            assert (mods["c34_lay"]._has_feasible_matching(cm, need) ==
                    mods["rel_lay"]._has_feasible_matching(cm, need))


def test_defaults_give_c33s_outputs(mods, torino):
    edges, qubits = mods["c34"]._failed_elements(torino.target, 0.5)
    for qc in _circuits():
        with layout(mods["rel_lay"]):
            a = mods["c33"].compile_for_hardware(qc, backend=torino, entangling_basis="cx", layout_search=True,
                                                 seed_transpiler=0)
        with layout(mods["c34_lay"]):
            b = mods["c34"].compile_for_hardware(qc, backend=torino, entangling_basis="cx", layout_search=True,
                                                 seed_transpiler=0)
        assert _sig(a) == _sig(b)
        for out in (a, b):
            assert mods["c34"]._uses_failed(out, edges, qubits) == mods["c33"]._uses_failed(out, edges, qubits)


@pytest.mark.parametrize("switch", [("ABSORB_SYNTH", "qiskit"), ("COMPRESS", False)])
def test_switches_give_exact_outputs(mods, torino, switch):
    c34 = mods["c34"]
    name, value = switch
    old = getattr(c34, name)
    setattr(c34, name, value)
    try:
        with layout(mods["c34_lay"]):
            for qc in _circuits(6, 62_300):
                bare = qc.remove_final_measurements(inplace=False)
                out = c34.compile_for_hardware(bare, backend=torino, entangling_basis="cx", layout_search=True,
                                               seed_transpiler=0)
                assert c34._implements(bare, out) is True
    finally:
        setattr(c34, name, old)
