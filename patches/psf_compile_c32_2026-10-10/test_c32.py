"""Tests for candidate psf_compile 2026-10-10.c32 (changelog item 60) against candidate 2026-10-10.c31: with a
Target, the default call is c31's call with placement_refine=True; everything else is c31's.

Run from the repository root:  python -m pytest patches/psf_compile_c32_2026-10-10/test_c32.py -q
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c31 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c31_2026-10-10", "psf_compile.py"),
                        "psf_compile_c31_for_c32_test")
    c32 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c32_test")
    c31.WARN_WITHOUT_TARGET = c32.WARN_WITHOUT_TARGET = False
    return dict(c31=c31, c32=c32)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits():
    from qiskit.circuit.random import random_circuit
    return [random_circuit(3 + k % 6, 6 + k, max_operands=2, measure=k % 2 == 0, seed=60_100 + k) for k in range(8)]


def test_version(mods):
    assert mods["c32"].VERSION == "2026-10-10.c32"
    assert mods["c31"].VERSION == "2026-10-10.c31"


def test_default_with_target_is_c31_with_placement_refine(mods, torino):
    for qc in _circuits():
        a = mods["c31"].compile_for_hardware(qc, backend=torino, placement_refine=True, entangling_basis="cx",
                                             layout_search=True, seed_transpiler=0)
        b = mods["c32"].compile_for_hardware(qc, backend=torino, entangling_basis="cx", layout_search=True,
                                             seed_transpiler=0)
        assert _sig(a) == _sig(b)


def test_everything_else_is_c31(mods, torino):
    t = torino.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    base = dict(coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx", layout_search=True,
                seed_transpiler=0)
    for qc in _circuits()[:5]:
        for kw in (base, dict(base, target=t, placement_refine=False), dict(base, target=t, **RECOMMENDED)):
            a = mods["c31"].compile_for_hardware(qc, **kw)
            b = mods["c32"].compile_for_hardware(qc, **kw)
            assert _sig(a) == _sig(b), sorted(kw)


def test_auto_without_target_is_off(mods):
    from qiskit import QuantumCircuit
    from qiskit.transpiler import CouplingMap
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = mods["c32"].compile_for_hardware(qc, coupling_map=CouplingMap.from_line(3), basis_gates=["cz", "rz", "sx", "x"])
    assert out.num_qubits == 3
    with pytest.raises(ValueError):
        mods["c32"].compile_for_hardware(qc, coupling_map=CouplingMap.from_line(3), placement_refine=True)
