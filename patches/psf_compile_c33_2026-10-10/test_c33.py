"""Tests for candidate psf_compile 2026-10-10.c33 (changelog item 61: with a Target, one compile on the map without
the failed elements) against candidate 2026-10-10.c32.

Run from the repository root:  python -m pytest patches/psf_compile_c33_2026-10-10/test_c33.py -q
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
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c32 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c32_2026-10-10", "psf_compile.py"),
                        "psf_compile_c32_for_c33_test")
    c33 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c33_test")
    c32.WARN_WITHOUT_TARGET = c33.WARN_WITHOUT_TARGET = False
    return dict(c32=c32, c33=c33)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits():
    from qiskit.circuit.random import random_circuit
    return [random_circuit(3 + k % 6, 6 + k, max_operands=2, measure=k % 2 == 0, seed=61_100 + k) for k in range(8)]


def test_version(mods):
    assert mods["c33"].VERSION == "2026-10-10.c33"


def test_prune_first_off_is_c32(mods, torino):
    c33 = mods["c33"]
    c33.PRUNE_FIRST = False
    try:
        for qc in _circuits():
            a = mods["c32"].compile_for_hardware(qc, backend=torino, entangling_basis="cx", layout_search=True,
                                                 seed_transpiler=0)
            b = c33.compile_for_hardware(qc, backend=torino, entangling_basis="cx", layout_search=True,
                                         seed_transpiler=0)
            assert _sig(a) == _sig(b)
    finally:
        c33.PRUNE_FIRST = True


def test_one_compile_and_no_failed_element(mods, torino):
    c33 = mods["c33"]
    edges, qubits = c33._failed_elements(torino.target, 0.5)
    assert edges  # FakeTorino reports failed couplers
    for qc in _circuits():
        before = dict(c33.PRUNE_STATS)
        out = c33.compile_for_hardware(qc, backend=torino, entangling_basis="cx", layout_search=True, seed_transpiler=0)
        assert not c33._uses_failed(out, edges, qubits)
        assert c33.PRUNE_STATS["pruned_first"] == before["pruned_first"] + 1
        assert c33.PRUNE_STATS["recompiled"] == before["recompiled"]


def test_raise_or_keep_when_no_placement_avoids_them(mods):
    from qiskit import QuantumCircuit
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import InstructionProperties
    c33 = mods["c33"]
    be = GenericBackendV2(num_qubits=5, basis_gates=["cz", "rz", "sx", "x", "id"],
                          coupling_map=[[i, i + 1] for i in range(4)] + [[i + 1, i] for i in range(4)], seed=61)
    t = be.target
    for qargs in ((1, 2), (2, 1)):
        t.update_instruction_properties("cz", qargs, InstructionProperties(error=1.0, duration=t["cz"][qargs].duration))
    qc = QuantumCircuit(4)
    qc.h(0)
    for k in range(3):
        qc.cx(k, k + 1)
    with pytest.raises(c33.FailedElementsError):
        c33.compile_for_hardware(qc, backend=be, entangling_basis="cx")
    with pytest.warns(RuntimeWarning):
        out = c33.compile_for_hardware(qc, backend=be, entangling_basis="cx", on_failed_elements="keep")
    assert out.num_qubits == 5
