"""Tests of item 93 (and item 95's reach) on prototype psf_compile 2026-10-14.c55 (item 93: every output checked exact at a tight tolerance, with
fallbacks; item 91, the workplace's, is tested in test_91.py; c54's suite runs as *_on_c55.py).

Run from the repository root:  python -m pytest patches/psf_compile_c56_2026-10-14/test_93_on_c55.py -q
"""
import contextlib
import io
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "benchmarks"), REPO]


@pytest.fixture(scope="module")
def m():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    mod = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c55_93_test")
    mod.WARN_WITHOUT_TARGET = False
    mod.SCOPE_GATE = False  # c56: item 96 (scope gate) off; these are the older suites' semantics (tested in test_c56.py)
    return mod


@pytest.fixture(scope="module")
def fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez()


def _snap_circuit():
    """The workplace's Q3 block (a two-qubit block Qiskit's Weyl decomposition snaps), between Hadamards, measured."""
    from qiskit import QuantumCircuit, qpy
    bb = qpy.load(open(os.path.join(HERE, "fixtures", "bad_block.qpy"), "rb"))[0]
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    qc.compose(bb, inplace=True)
    qc.measure_all()
    return qc


def _call(mod, qc, fez, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=fez.target, entangling_basis="cx", layout_search=True,
                                        seed_transpiler=0, **kw)


def test_version(m):
    assert m.VERSION == "2026-10-14.c56"


def test_check_sees_the_snap(m, fez):
    """Qiskit level 3 snaps the block (1.4e-5 in distribution): item 93's check says not exact; item 39's tolerance
    would not see it."""
    from qiskit.transpiler import generate_preset_pass_manager
    qc = _snap_circuit()
    l3 = generate_preset_pass_manager(3, target=fez.target, seed_transpiler=0).run(qc)
    assert m._exact93(qc, l3) is False


def test_output_is_exact_after_fallback(m, fez):
    qc = _snap_circuit()
    before = dict(m.EXACT93_STATS)
    out = _call(m, qc, fez)
    assert m._exact93(qc, out) is True
    assert m.EXACT93_STATS["failed"] >= before["failed"]


def test_switch_off_gives_the_inner_output(m, fez):
    qc = _snap_circuit()
    m.EXACT_CHECK = False
    try:
        off = _call(m, qc, fez)
    finally:
        m.EXACT_CHECK = True
    with contextlib.redirect_stdout(io.StringIO()):
        inner = m._compile_for_hardware_c53g(qc, target=fez.target, entangling_basis="cx", layout_search=True,
                                             seed_transpiler=0)
    from c29_identity import value_sig
    assert value_sig(off) == value_sig(inner)


def test_unmeasured_checked_and_exact(m, fez):
    from qiskit import QuantumCircuit
    import random
    rng = random.Random(9301)
    qc = QuantumCircuit(5)
    for _ in range(12):
        a, b = rng.sample(range(5), 2)
        qc.rzz(rng.uniform(-2, 2), a, b)
        qc.rx(rng.uniform(-2, 2), rng.randrange(5))
    before = m.EXACT93_STATS["checked"]
    out = _call(m, qc, fez)
    assert m.EXACT93_STATS["checked"] == before + 1
    assert m._implements(qc, out, tol=1e-12)


def test_twenty_qubits_checked(m, fez):
    """Item 95: a measured circuit with 20 active qubits (above RESYNTH_MAX_QUBITS) is now checked, and exact."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(20)
    qc.h(0)
    for q in range(19):
        qc.cx(q, q + 1)
    qc.measure_all()
    before = dict(m.EXACT93_STATS)
    _call(m, qc, fez)
    assert m.EXACT93_STATS["checked"] == before["checked"] + 1
    assert m.EXACT93_STATS["passed"] == before["passed"] + 1


def test_large_circuit_not_checked_and_unchanged(m, fez):
    """Above EXACT_CHECK_MAX_QUBITS active qubits nothing is checked and the inner output is returned."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(21)
    qc.h(0)
    for q in range(20):
        qc.cx(q, q + 1)
    qc.measure_all()
    before = m.EXACT93_STATS["not_checkable"]
    out = _call(m, qc, fez)
    assert m.EXACT93_STATS["not_checkable"] == before + 1
    with contextlib.redirect_stdout(io.StringIO()):
        inner = m._compile_for_hardware_c53g(qc, target=fez.target, entangling_basis="cx", layout_search=True,
                                             seed_transpiler=0)
    from c29_identity import value_sig
    assert value_sig(out) == value_sig(inner)
