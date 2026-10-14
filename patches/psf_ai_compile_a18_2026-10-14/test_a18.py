"""Tests for psf_ai_compile 2026-10-14.a18: item 32 (a14's seed search is no longer a candidate) and item 33 (an
input not built by the ring formula is handed on unchanged: to psf_compile's default call with a Target, to Qiskit's
level 3 without one). a17's own tests run on a18 with the two switched off in test_a17_on_a18.py.

Run from the repository root (psf_compile c55, item 94 off; or PSF_COMPILE pointing to another):
    python -m pytest patches/psf_ai_compile_a18_2026-10-14/test_a18.py -q
"""
import contextlib
import copy
import io
import math
import os
import random
import sys
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("PSF_REPO", os.getcwd())
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)
PSF = os.environ.get("PSF_COMPILE", os.path.join(REPO, "patches", "psf_compile_c55_2026-10-14", "psf_compile.py"))  # home: c55, item 94 off


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(PSF, "psf_compile")
    pc.WARN_WITHOUT_TARGET = False
    pc.ZZ_RELAYER = False  # item 94 dropped (Addendum 488)
    a16 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a18")
    return dict(pc=pc, a16=a16, a15=a16.a15)


def fractional(target):
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.transpiler import InstructionProperties
    t = copy.deepcopy(target)
    t.add_instruction(RZZGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                                 for qa, p in target["cz"].items()})
    t.add_instruction(RXGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                               for qa, p in target["sx"].items()})
    return t


@pytest.fixture(scope="module")
def fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    t = FakeFez().target
    return t, fractional(t)


def _basis(t):
    return [g for g in t.operation_names if g in ("cz", "rz", "sx", "x")]


def classifier(n, layers, seed, measure=True, shuffle=False):
    """PennyLane-style: angle encoding, then per layer ry/rz on every qubit and a CZ ring (W3's shape)."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-1, 1) * math.pi / 2, q)
        for q in range(n):
            qc.ry(rng.uniform(-math.pi, math.pi), q)
            qc.rz(rng.uniform(-math.pi, math.pi), q)
        edges = [(q, (q + 1) % n) for q in range(n)]
        if shuffle:
            rng.shuffle(edges)
        for a, b in edges:
            qc.cz(a, b)
    for q in range(n):
        qc.ry(rng.uniform(-math.pi, math.pi), q)
    if measure:
        qc.measure_all()
    return qc


def _probs(circ):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    meas, body = {}, []
    for ins in circ.data:
        nm = ins.operation.name
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = circ.find_bit(ins.qubits[0]).index
        elif nm not in ("barrier", "delay"):
            body.append((ins.operation, [circ.find_bit(q).index for q in ins.qubits]))
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(active)}
    c = QuantumCircuit(len(active))
    for op, qs in body:
        c.append(op, [pos[q] for q in qs])
    c.save_probabilities([pos[meas[k]] for k in range(circ.num_clbits)])
    return np.asarray(AerSimulator(method="statevector").run(c).result().data()["probabilities"])


def same_distribution(out, qc):
    return 0.5 * float(np.abs(_probs(out) - _probs(qc)).sum()) < 1e-9


def _two_q(c):
    return sum(1 for i in c.data if len(i.qubits) == 2 and i.operation.name != "barrier")


def _call(a16, qc, t, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return a16.compile_for_model_circuit(qc, t.build_coupling_map(), _basis(t), target=t, return_info=True, **kw)


def test_version(mods):
    a = mods["a16"]
    assert a.AI_COMPILE_VERSION == "2026-10-14.a18"
    assert a.SEARCH_FALLBACK is False and a.HAND_OFF is True


def test_no_search_on_ring_classifiers(mods, fez, monkeypatch):
    """item 32: a14's search is never called; the ring formula is chosen and exact"""
    a = mods["a16"]
    def boom(*args, **kw):
        raise AssertionError("a14's search was called")
    monkeypatch.setattr(a.a14, "compile_for_model_circuit", boom)
    t, ft = fez
    for tg in (t, ft):
        for n, L in ((4, 1), (4, 6), (6, 3)):
            qc = classifier(n, L, 70 + n + L)
            out, info = _call(a, qc, tg)
            assert info["path"] == "small-constructed" and info["chosen"] != "search"
            assert same_distribution(out, qc)
            assert mods["pc"]._isa_ok(out, tg)


def test_non_ring_input_handed_to_the_compiler(mods, fez):
    """item 33: a chain, not a ring, goes to psf_compile's default call unchanged"""
    a, pc = mods["a16"], mods["pc"]
    from c29_identity import value_sig
    from qiskit import QuantumCircuit
    t, _ = fez
    qc = QuantumCircuit(5)
    for q in range(5):
        qc.ry(0.3 + 0.1 * q, q)
    for q in range(4):
        qc.cz(q, q + 1)
    qc.measure_all()
    out, info = _call(a, qc, t)
    assert info["path"] == "hand-off" and info["chosen"] == "compiler" and info["exact"] is True
    with contextlib.redirect_stdout(io.StringIO()):
        ref = pc.compile_for_hardware(qc, target=t, basis_gates=_basis(t), entangling_basis="cx", layout_search=True,
                                      seed_transpiler=0)
    assert value_sig(out) == value_sig(ref)


def test_no_target_handed_to_qiskit_level3(mods, fez):
    """item 33: without a Target, Qiskit's level 3 with the coupling map and basis, unchanged"""
    a = mods["a16"]
    from c29_identity import value_sig
    from qiskit import transpile
    t, _ = fez
    qc = classifier(4, 2, 80)
    cm = t.build_coupling_map()
    with contextlib.redirect_stdout(io.StringIO()):
        out, info = a.compile_for_model_circuit(qc, cm, _basis(t), target=None, return_info=True)
    assert info["path"] == "hand-off" and info["chosen"] == "QK3"
    ref = transpile(qc, coupling_map=cm, basis_gates=_basis(t), optimization_level=3, seed_transpiler=0)
    assert value_sig(out) == value_sig(ref)
    assert same_distribution(out, qc)


def test_large_input_handed_to_the_compiler(mods, fez):
    a = mods["a16"]
    t, _ = fez
    qc = classifier(a.SMALL_MAX_QUBITS + 1, 1, 81)
    out, info = _call(a, qc, t)
    assert info["path"] == "hand-off" and info["chosen"] == "compiler"
    assert mods["pc"]._isa_ok(out, t)


def test_switches_off_give_a17(mods, fez):
    """SEARCH_FALLBACK = True and HAND_OFF = False restore a17 (the workplace's file) on a ring and on a chain"""
    import core_fix_c2_eval as H
    from c29_identity import value_sig
    from qiskit import QuantumCircuit
    a = mods["a16"]
    a17 = H.load_module(os.path.join(REPO, "patches", "psf_ai_compile_a17_2026-10-14", "psf_ai_compile.py"),
                        "psf_ai_compile_a17_for_a18_test")
    t, _ = fez
    chain = QuantumCircuit(4)
    for q in range(4):
        chain.ry(0.2 * (q + 1), q)
    for q in range(3):
        chain.cz(q, q + 1)
    chain.measure_all()
    a.SEARCH_FALLBACK, a.HAND_OFF = True, False
    try:
        for qc in (classifier(4, 3, 90), chain):
            got = _call(a, qc, t)[0]
            with contextlib.redirect_stdout(io.StringIO()):
                ref = a17.compile_for_model_circuit(qc, t.build_coupling_map(), _basis(t), target=t)
            assert value_sig(got) == value_sig(ref)
    finally:
        a.SEARCH_FALLBACK, a.HAND_OFF = False, True


def test_deterministic(mods, fez):
    from c29_identity import value_sig
    a = mods["a16"]
    _, ft = fez
    qc = classifier(6, 4, 91)
    assert value_sig(_call(a, qc, ft)[0]) == value_sig(_call(a, qc, ft)[0])
