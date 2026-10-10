"""Tests for psf_compile.py 2026-09-28.1 (changelog items 26-27, Addendum 243):
REFINE_THRESHOLD 1e-14 everywhere it is used, CORE_VERSION exported, and a
two-qubit block still synthesized exactly."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import psf_compile as pc  # noqa: E402
import psf_zero_core  # noqa: E402


def test_version():
    assert pc.VERSION == "2026-10-10.3"  # current release (this file was written for 2026-09-28.1)
    assert pc.__version__ == pc.VERSION


def test_refine_threshold_is_1e14_everywhere():
    assert pc.REFINE_THRESHOLD == 1e-14
    assert pc.REFINE_THRESHOLD == pc._REFINE_TARGET
    # the defaults are bound when the functions are defined; they must follow the constant
    assert pc._refine_batch.__defaults__[0] == 1e-14
    assert pc._refine_decomposition.__defaults__[0] == 1e-14


def test_core_version_exported():
    assert "CORE_VERSION" in pc.__all__
    assert pc.CORE_VERSION == getattr(psf_zero_core, "CORE_VERSION", None)


@pytest.mark.parametrize("seed", range(5))
def test_random_block_exact(seed):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, random_unitary
    u = random_unitary(4, seed=seed)
    qc = QuantumCircuit(2)
    qc.unitary(u, [0, 1])
    out = pc.compile(qc, block_gate_floor=0)
    a, b = Operator(out).data, u.data
    z = np.vdot(a.ravel(), b.ravel())
    z = z / abs(z)
    assert np.linalg.norm(b - z * a) <= 1e-13
