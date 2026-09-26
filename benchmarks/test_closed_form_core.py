"""Tests for the closed-form CX core and the edge-weight helpers in
psf_compile.py VERSION 2026-09-26.3 (Addendum 194), updated for
2026-09-26.4 (Addendum 196): the closed form is no longer identified by `rx`."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import psf_compile as pc


def _core(a, b, c):
    qc = QuantumCircuit(2)
    qc.rxx(-2 * a, 0, 1)
    qc.ryy(-2 * b, 0, 1)
    qc.rzz(-2 * c, 0, 1)
    return Operator(qc).data


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_is_exact_including_phase(seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        a, b, c = rng.uniform(-np.pi, np.pi, 3)
        qc = QuantumCircuit(2)
        if not pc._append_cx_core_closed_form(qc, a, b, c):
            continue
        assert qc.count_ops().get("cx", 0) == 3
        assert np.linalg.norm(Operator(qc).data - _core(a, b, c)) < 1e-13


@pytest.mark.parametrize("abc", [(0.3, 0.2, 0.0), (0.3, 0.2, 1e-8), (np.pi / 2, 0.2, 0.1),
                                 (np.pi / 4, 0.0, 0.0)])
def test_degenerate_triples_are_left_to_the_decomposer(abc):
    qc = QuantumCircuit(2)
    assert pc._append_cx_core_closed_form(qc, *abc) is False
    assert len(qc.data) == 0


def test_flag_off_restores_previous_path():
    from qiskit.quantum_info import random_unitary
    u = random_unitary(4, seed=5).data
    saved = pc.USE_CX_CLOSED_FORM
    try:
        out = {}
        for flag in (False, True):
            pc.USE_CX_CLOSED_FORM = flag
            synth = pc.SU4GeodesicPSFSynthesizer(
                pc.GeodesicPSFHyper(tol=1e-5, on_unsupported="raise", entangling_basis="cx"), verify=True)
            qc = synth.synthesize(u)
            out[flag] = qc
            assert np.linalg.norm(Operator(qc).data - u) < 1e-12
        # The closed form's gate sequence differs from the decomposer's
        # (2026-09-26.4: native gaps emit no rx, so presence of rx no longer
        # identifies it).
        assert [i.operation.name for i in out[False].data] != [i.operation.name for i in out[True].data]
        assert out[False].count_ops()["cx"] == out[True].count_ops()["cx"] == 3
    finally:
        pc.USE_CX_CLOSED_FORM = saved


def test_edge_weights_order_and_positivity():
    w = pc._edge_weights_from_errors({(0, 1): 0.001, (1, 2): 0.01, (2, 3): 0.9})
    assert w[(0, 1)] > w[(1, 2)] > w[(2, 3)] > 0
