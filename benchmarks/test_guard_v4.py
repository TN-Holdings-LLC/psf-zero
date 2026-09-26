"""Tests for psf_compile.py VERSION 2026-09-26.4 (Addendum 196): the guard on
Qiskit's CX decomposer, the native-gap closed form, and single-qubit errors
in the layout weights."""
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


def _canonical(a, b, c):
    qc = QuantumCircuit(2)
    qc.rxx(-2 * a, 0, 1)
    qc.ryy(-2 * b, 0, 1)
    qc.rzz(-2 * c, 0, 1)
    return Operator(qc).data


@pytest.mark.parametrize("c", [3e-8, 1e-7, 3e-7])
def test_guard_repairs_the_zsx_band(c):
    u = _canonical(0.6, 0.3, c)
    circ, ok = pc._guarded_cx_synthesis(u)
    assert ok
    assert np.linalg.norm(Operator(circ).data - u) < 1e-6  # phase included


@pytest.mark.parametrize("c", [3e-8, 1e-7, 3e-7])
def test_cx_synthesizer_is_correct_in_the_band(c):
    u = _canonical(0.6, 0.3, c)
    pc._CX_CORE_CACHE.clear()
    synth = pc.SU4GeodesicPSFSynthesizer(
        pc.GeodesicPSFHyper(tol=1e-5, on_unsupported="keep", entangling_basis="cx"), verify=True)
    qc = synth.synthesize(u)
    assert np.linalg.norm(Operator(qc).data - u) < 1e-6


def test_native_gaps_emit_only_rz_sx_cx():
    qc = QuantumCircuit(2)
    assert pc._append_cx_core_closed_form(qc, 0.6, 0.3, 0.2)
    ops = qc.count_ops()
    assert set(ops) <= {"rz", "sx", "cx"}
    assert ops["sx"] == 2 and ops["cx"] == 3


def test_qubit_errors_lower_the_weight_of_a_dead_qubit():
    w = pc._edge_weights_from_errors({(0, 1): 0.003, (1, 2): 0.003}, {0: 1.0, 1: 1e-4, 2: 1e-4},
                                     n2=3.0, n1=7.0)
    assert 0 < w[(0, 1)] < w[(1, 2)]
