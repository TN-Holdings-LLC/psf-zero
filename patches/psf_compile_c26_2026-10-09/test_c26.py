"""Tests for candidate psf_compile 2026-10-09.c26 (changelog item 53: gate matrices kept; one-qubit gates embedded
without np.kron) against release 2026-10-07.1, on which it is based.

Run from the repository root:  python -m pytest patches/psf_compile_c26_2026-10-09/test_c26.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c26_test")
    c26 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c26_test")
    return dict(rel=rel, c26=c26)


def test_versions(mods):
    assert mods["c26"].VERSION == "2026-10-09.c26" and mods["rel"].VERSION == "2026-10-07.1"


def test_embed_equals_release(mods):
    rng = np.random.default_rng(53)
    for _ in range(500):
        k = int(rng.integers(1, 4))
        qubits = tuple(int(x) for x in rng.choice(20, size=k, replace=False))
        mats = {q: rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for q in qubits if rng.random() < 0.7}
        a, b = mods["rel"]._embed_1q(mats, qubits), mods["c26"]._embed_1q(mats, qubits)
        assert a.shape == b.shape and np.array_equal(a, b)


def test_gate_matrix_equals_to_matrix(mods):
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import CZGate, ECRGate, RZGate, SXGate, UnitaryGate, XGate
    g = mods["c26"]._gate_matrix
    for op in (XGate(), SXGate(), CZGate(), ECRGate(), RZGate(0.3), RZGate(np.float64(-1.2)), RZGate(2)):
        for _ in range(2):  # the second time from the kept matrices
            m = g(op)
            assert np.array_equal(m, np.asarray(op.to_matrix(), dtype=complex)) and m.dtype == complex
            assert not m.flags.writeable
    u = UnitaryGate(np.eye(2))
    assert g(u).flags.writeable  # not a standard gate: built as before
    with pytest.raises(Exception):
        g(RZGate(Parameter("t")))  # unbound: raises as before


def test_kept_matrices_are_bounded(mods):
    from qiskit.circuit.library import RZGate
    c = mods["c26"]
    for k in range(c._GATE_MATRICES_MAX + 100):
        c._gate_matrix(RZGate(k * 1e-3))
    assert len(c._GATE_MATRICES) == c._GATE_MATRICES_MAX
