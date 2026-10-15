"""a21's suite run on a22 (copied from patches/psf_ai_compile_a21_2026-10-15/test_a21.py; only the module, the version
string and this paragraph changed). With psf_compile c59, which has no item 99, a22 must behave as a21.

Tests for prototype psf_ai_compile 2026-10-15.a21 (items 37-41, the workplace's review of a18) on psf_compile
c59. a20's own suite runs on a21 as test_a20_on_a21.py.

Run from the repository root:  python -m pytest patches/psf_ai_compile_a22_2026-10-15/test_a21_on_a22.py -q
"""
import contextlib
import copy
import io
import math
import os
import random
import sys
import time
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
warnings.simplefilter("ignore")
BASIS = ["cz", "rz", "sx", "x"]


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(os.path.join(REPO, "patches", "psf_compile_c59_2026-10-15", "psf_compile.py"), "psf_compile")
    pc.WARN_WITHOUT_TARGET = False
    a = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a22_as_a21_test")
    a20 = H.load_module(os.path.join(REPO, "patches", "psf_ai_compile_a20_2026-10-15", "psf_ai_compile.py"),
                        "psf_ai_compile_a20_for_a21_test")
    a20.EXACT_CHECK = False  # a20's hand-off check is what item 37 replaces; not needed to compare outputs
    return dict(pc=pc, a=a, a20=a20)


@pytest.fixture(scope="module")
def fez():
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.transpiler import InstructionProperties
    from qiskit_ibm_runtime.fake_provider import FakeFez
    t = FakeFez().target
    f = copy.deepcopy(t)
    f.add_instruction(RZZGate(Parameter("t")), {q: InstructionProperties(duration=p.duration, error=p.error)
                                                 for q, p in t["cz"].items()})
    f.add_instruction(RXGate(Parameter("t")), {q: InstructionProperties(duration=p.duration, error=p.error)
                                               for q, p in t["sx"].items()})
    return t, f


def classifier(n, layers=3, seed=1):
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-math.pi, math.pi), q)
            qc.rz(rng.uniform(-math.pi, math.pi), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(rng.uniform(-math.pi, math.pi), q)
    qc.measure_all()
    return qc


def diag_ring(n, gate="rzz", p=2, seed=2):
    """A QAOA-like ring: H, then p times (a ring of `gate` with signed angles, an rx layer)."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        for q in range(n):
            a, b = (q, (q + 1) % n) if rng.random() < 0.5 else ((q + 1) % n, q)
            getattr(qc, gate)(rng.uniform(-1.5, 1.5), a, b)
        beta = rng.uniform(-1, 1)
        for q in range(n):
            qc.rx(beta, q)
    qc.measure_all()
    return qc


def pe_chain(n, steps=2):
    """A transverse-field Ising chain as one PauliEvolutionGate per step (the input W1-A's check mishandled)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    terms = [("ZZ", [q, q + 1], 1.0) for q in range(n - 1)] + [("X", [q], 0.7) for q in range(n)]
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    qc = QuantumCircuit(n)
    for _ in range(steps):
        qc.append(PauliEvolutionGate(H, time=0.3), range(n))
    qc.measure_all()
    return qc


def _call(a, qc, target, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return a.compile_for_model_circuit(qc, target.build_coupling_map(), BASIS, target=target, return_info=True,
                                           **kw)


def test_versions_and_defaults(mods):
    a = mods["a"]
    assert a.AI_COMPILE_VERSION == "2026-10-15.a22"
    assert a.ONE_CHECK and not a.HANDOFF_CHECK and a.RING_MAX_QUBITS == 20 and a.DIAG_RINGS
    assert a.GATE_CLOSED and a.FORMULA_ONLY and not a.FORMULA_CHECK


def test_item37_pauli_evolution_judged_by_its_product_formula(mods, fez):
    """item 30 alone calls a correct compile of a PauliEvolutionGate inexact; item 37 calls it exact"""
    a, pc = mods["a"], mods["pc"]
    t, _ = fez
    qc = pe_chain(8)
    with contextlib.redirect_stdout(io.StringIO()):
        out = pc.compile_for_hardware(qc, target=t, seed_transpiler=0)
    s = time.perf_counter()
    assert a._exact(qc, out) is True
    assert time.perf_counter() - s < 30
    a.ONE_CHECK = False
    try:
        assert a._exact(qc, out) is False  # the exact exponential against the product formula: the old false verdict
    finally:
        a.ONE_CHECK = True


def test_item38_hand_off_not_simulated(mods, fez, monkeypatch):
    a, pc = mods["a"], mods["pc"]
    t, _ = fez
    calls = []
    real = pc._exact93
    monkeypatch.setattr(pc, "_exact93", lambda *x, **k: calls.append(1) or real(*x, **k))
    out, info = _call(a, pe_chain(12), t)
    assert info["path"] == "hand-off" and info["exact"] is None and not calls
    a.HANDOFF_CHECK = True
    try:
        out, info = _call(a, pe_chain(8), t)
        assert info["exact"] is True and calls
    finally:
        a.HANDOFF_CHECK = False


def test_item39_seed_keyword(mods, fez):
    from c29_identity import value_sig
    a = mods["a"]
    t, _ = fez
    qc = pe_chain(8)
    out, _ = _call(a, qc, t, seed_transpiler=0)  # raised TypeError before
    ref, _ = _call(a, qc, t)
    assert value_sig(out) == value_sig(ref)


@pytest.mark.parametrize("n", [9, 10, 12])
def test_item40_rings_above_eight_by_formula(mods, fez, n):
    a, pc = mods["a"], mods["pc"]
    for tg in fez:
        qc = classifier(n, 2, n)
        out, info = _call(a, qc, tg)
        assert info["path"] == "formula" and info["chosen"] in ("ring", "ring2")
        assert pc._isa_ok(out, tg) and pc._exact93(qc, out) is True
    a.RING_MAX_QUBITS = 8
    try:
        with pytest.raises(pc.TranspilerError):  # handed to c59, which reads no CZ ring and refuses (item 98)
            _call(a, classifier(n, 2, n), fez[0])
    finally:
        a.RING_MAX_QUBITS = 20


@pytest.mark.parametrize("gate,n", [("rzz", 5), ("rzz", 8), ("cp", 6), ("crz", 6)])
def test_item41_diagonal_rings(mods, fez, gate, n):
    a, pc = mods["a"], mods["pc"]
    qc = diag_ring(n, gate)
    for tg in fez:
        out, info = _call(a, qc, tg)
        assert info["path"] == "formula"
        assert pc._isa_ok(out, tg) and pc._rzz_in_range(out) and pc._exact93(qc, out) is True
        if "rzz" in tg.operation_names and gate == "rzz":
            assert any(i.operation.name == "rzz" for i in out.data)  # the device's own rzz is used


def test_item41_cx_ring_not_read(mods, fez):
    """a ring of cx does not commute: not read as a ring; handed to c59, which refuses it (it does not embed)"""
    from qiskit import QuantumCircuit
    a, pc = mods["a"], mods["pc"]
    qc = QuantumCircuit(6)
    qc.h(range(6))
    for q in range(6):
        qc.cx(q, (q + 1) % 6)
    qc.measure_all()
    assert a._ring_read2(qc) is None
    with pytest.raises(pc.TranspilerError):
        _call(a, qc, fez[0])


@pytest.mark.parametrize("n,L", [(4, 2), (6, 3), (8, 4)])
def test_cz_rings_as_in_a20(mods, fez, n, L):
    from c29_identity import value_sig
    for tg in fez:
        qc = classifier(n, L, 10 * n + L)
        out, _ = _call(mods["a"], qc, tg)
        with contextlib.redirect_stdout(io.StringIO()):
            ref = mods["a20"].compile_for_model_circuit(qc, tg.build_coupling_map(), BASIS, target=tg)
        assert value_sig(out) == value_sig(ref)


def test_template_stays_cz_only(mods, fez):
    a = mods["a"]
    t, _ = fez
    assert a.ring_template(diag_ring(6, "rzz"), t) is None
    tpl = a.ring_template(classifier(6, 2, 3), t)
    assert tpl is not None
