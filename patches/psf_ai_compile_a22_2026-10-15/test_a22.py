"""Tests for prototype psf_ai_compile 2026-10-15.a22 (items 42-43) on psf_compile c61 (items 99-100). a21's suite runs
on a22 with c59 as test_a21_on_a22.py, where a22 must behave as a21 (c59 has no item 99).

Run from the repository root:  python -m pytest patches/psf_ai_compile_a22_2026-10-15/test_a22.py -q
"""
import contextlib
import copy
import io
import math
import os
import random
import sys
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
    pc = H.load_module(os.path.join(REPO, "patches", "psf_compile_c61_2026-10-15", "psf_compile.py"), "psf_compile")
    pc.WARN_WITHOUT_TARGET = False
    a = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a22_test")
    a21 = H.load_module(os.path.join(REPO, "patches", "psf_ai_compile_a21_2026-10-15", "psf_ai_compile.py"),
                        "psf_ai_compile_a21_for_a22_test")
    return dict(pc=pc, a=a, a21=a21)


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


def chain_cz(n, layers=2, seed=3):
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q)
        for q in range(n - 1):
            qc.cz(q, q + 1)
    qc.measure_all()
    return qc


def _call(a, qc, target):
    with contextlib.redirect_stdout(io.StringIO()):
        return a.compile_for_model_circuit(qc, target.build_coupling_map(), BASIS, target=target, return_info=True)


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def test_versions_and_defaults(mods):
    a = mods["a"]
    assert a.AI_COMPILE_VERSION == "2026-10-15.a22" and a.EMBED_RINGS is True
    assert mods["pc"].VERSION == "2026-10-15.c61"


def test_item42_embeddable_ring_goes_to_item99(mods, fez):
    a, a21, pc = mods["a"], mods["a21"], mods["pc"]
    for tg in fez:
        qc = classifier(12, 3, 4)
        out, info = _call(a, qc, tg)
        ref, _ = _call(a21, qc, tg)
        assert info["path"] == "hand-off" and info["chosen"] == "compiler"
        assert pc._isa_ok(out, tg) and pc._exact93(qc, out) is True
        assert pc._duration_s(out, tg) < 0.6 * pc._duration_s(ref, tg)
        assert sum(1 for i in out.data if len(i.qubits) == 2) == 3 * 12  # one CZ per edge and layer, no SWAP


@pytest.mark.parametrize("n", [4, 6, 9, 16])
def test_item42_rings_that_do_not_embed_are_as_in_a21(mods, fez, n):
    for tg in fez:
        qc = classifier(n, 2, n)
        out, info = _call(mods["a"], qc, tg)
        ref, _ = _call(mods["a21"], qc, tg)
        assert info["path"] == "formula" and _sig(out) == _sig(ref)


def test_item42_switch_off_is_a21(mods, fez):
    a = mods["a"]
    qc = classifier(12, 2, 5)
    a.EMBED_RINGS = False
    try:
        out, info = _call(a, qc, fez[0])
    finally:
        a.EMBED_RINGS = True
    ref, _ = _call(mods["a21"], qc, fez[0])
    assert info["path"] == "formula" and _sig(out) == _sig(ref)


def test_item43_model_template_picks_by_the_same_rule(mods, fez):
    a, pc = mods["a"], mods["pc"]
    t, _ = fez
    tpl = a.model_template(classifier(12, 2, 6), t)
    assert type(tpl).__name__ == "ModelTemplate" and len(tpl.layout) == 12
    for seed in (7, 8):
        qc = classifier(12, 2, seed)
        out, _ = _call(a, qc, t)
        b = tpl.bind(qc)
        assert _sig(b) == _sig(out) and pc._exact93(qc, b) is True
    assert type(a.model_template(classifier(6, 2, 6), t)).__name__ == "RingTemplate"
    tc = a.model_template(chain_cz(8), t)
    assert type(tc).__name__ == "ModelTemplate" and pc._exact93(chain_cz(8, seed=9), tc.bind(chain_cz(8, seed=9))) is True


def test_item43_refusals(mods, fez):
    import networkx as nx
    from qiskit import QuantumCircuit
    a = mods["a"]
    t, _ = fez
    g = nx.random_regular_graph(3, 10, seed=1)
    qc = QuantumCircuit(10)
    qc.h(range(10))
    for u, v in sorted(g.edges()):
        qc.rzz(0.7, u, v)
    qc.measure_all()
    assert a.model_template(qc, t) is None  # out of scope: no template
    tpl = a.model_template(classifier(12, 2, 6), t)
    with pytest.raises(ValueError):
        tpl.bind(classifier(12, 3, 6))  # another structure
