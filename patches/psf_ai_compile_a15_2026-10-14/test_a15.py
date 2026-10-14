"""Tests for prototype psf_ai_compile 2026-10-14.a15 (items 23-26: psf_compile's constructions of a few deterministic
starting points, chosen by duration; item 27: a14's searched output as the last candidate), on psf_compile prototype
2026-10-14.c49.

Run from the repository root:  python -m pytest patches/psf_ai_compile_a15_2026-10-14/test_a15.py -q
"""
import contextlib
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
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c49 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c49_2026-10-14", "psf_compile.py"), "psf_compile")
    c49.WARN_WITHOUT_TARGET = False
    a15 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a15")
    return dict(c49=c49, a15=a15, a14=a15.a14)


@pytest.fixture(scope="module")
def fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez()


def _basis(t):
    return [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]


def _classifier(n, layers, seed):
    """A PennyLane-style layered classifier: angle encoding, then per layer ry/rz on every qubit and a CZ ring."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(rng.uniform(0, math.pi), q)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-math.pi, math.pi), q)
            qc.rz(rng.uniform(-math.pi, math.pi), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    return qc


def _unitary_circuit(n, k, seed):
    """Two-qubit `unitary` gates on random pairs, as PennyLane tapes arrive."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_unitary
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for j in range(k):
        a, b = rng.sample(range(n), 2)
        qc.unitary(random_unitary(4, seed=seed * 100 + j), [a, b])
    return qc


def _cases():
    return [("clf4_2", _classifier(4, 2, 1)), ("clf6_3", _classifier(6, 3, 2)), ("clf4_6", _classifier(4, 6, 3)),
            ("unitary5", _unitary_circuit(5, 6, 4))]


def _call(a15, qc, t, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return a15.compile_for_model_circuit(qc, t.build_coupling_map(), _basis(t), target=t, **kw)


def test_version(mods):
    assert mods["a15"].AI_COMPILE_VERSION == "2026-10-14.a15"


def test_exact_and_never_longer_than_the_input_compile(mods, fez):
    c49, a15 = mods["c49"], mods["a15"]
    t = fez.target
    for name, qc in _cases():
        out, info = _call(a15, qc, t, return_info=True)
        assert info["path"] == "small-constructed", name
        assert c49._implements(qc, out), name
        with contextlib.redirect_stdout(io.StringIO()):
            ref = c49.compile_for_hardware(qc, target=t, basis_gates=_basis(t), entangling_basis="cx",
                                           layout_search=True, seed_transpiler=0)
        assert not c49._better(ref, out, t), name  # the input's own compile is a candidate
        with contextlib.redirect_stdout(io.StringIO()):
            searched = mods["a14"].compile_for_model_circuit(qc, t.build_coupling_map(), _basis(t), target=t)
        assert not c49._better(searched, out, t), name  # item 27: a14's output is a candidate too


def test_same_output_every_call(mods, fez):
    from c29_identity import value_sig
    a15 = mods["a15"]
    for _, qc in _cases()[:2]:
        assert len({value_sig(_call(a15, qc, fez.target)) for _ in range(3)}) == 1


def test_no_error_value_decides(mods, fez):
    """Rescaling every reported error (below the failed threshold) leaves the choice among the same candidates
    unchanged: _better reads durations and gate counts only."""
    import copy
    from qiskit.transpiler import InstructionProperties
    c49, a15 = mods["c49"], mods["a15"]
    t1 = fez.target
    t2 = copy.deepcopy(t1)
    rng = random.Random(9)
    for name in ("cz", "sx", "x", "measure"):
        for qa, p in t2[name].items():
            if qa is not None and p is not None and p.error is not None and p.error < 0.5:
                t2.update_instruction_properties(name, qa, InstructionProperties(
                    duration=p.duration, error=min(0.4, p.error * rng.uniform(0.2, 5.0))))
    qc = _classifier(4, 3, 7)
    outs = []
    for sname, circ in a15._starts(qc):
        with contextlib.redirect_stdout(io.StringIO()):
            outs.append(c49.compile_for_hardware(circ, target=t1, basis_gates=_basis(t1), entangling_basis="cx",
                                                 layout_search=True, seed_transpiler=0))
    for a in outs:
        for b in outs:
            assert c49._better(a, b, t1) == c49._better(a, b, t2)


def test_failed_qubits_are_never_used(mods, fez):
    import copy
    from qiskit.transpiler import InstructionProperties
    c49, a15 = mods["c49"], mods["a15"]
    t = copy.deepcopy(fez.target)
    bad = (0, 1, 2, 3, 16, 17)
    for q in bad:
        p = t["sx"][(q,)]
        t.update_instruction_properties("sx", (q,), InstructionProperties(duration=p.duration, error=1.0))
    for name, qc in _cases():
        out = _call(a15, qc, t)
        pos = {q: i for i, q in enumerate(out.qubits)}
        used = {pos[q] for i in out.data if i.operation.name != "barrier" for q in i.qubits}
        assert not used & set(bad), name
        assert c49._implements(qc, out), name


def test_without_target_is_a14(mods, fez):
    from c29_identity import value_sig
    a15, a14 = mods["a15"], mods["a14"]
    t = fez.target
    qc = _classifier(4, 2, 11)
    with contextlib.redirect_stdout(io.StringIO()):
        x = a15.compile_for_model_circuit(qc, t.build_coupling_map(), _basis(t))
        y = a14.compile_for_model_circuit(qc, t.build_coupling_map(), _basis(t))
    assert value_sig(x) == value_sig(y)


def test_switch_off_is_a14(mods, fez):
    from c29_identity import value_sig
    a15, a14 = mods["a15"], mods["a14"]
    t = fez.target
    qc = _classifier(4, 2, 12)
    a15.CONSTRUCT_BY_DURATION = False
    try:
        x = _call(a15, qc, t)
    finally:
        a15.CONSTRUCT_BY_DURATION = True
    with contextlib.redirect_stdout(io.StringIO()):
        y = a14.compile_for_model_circuit(qc, t.build_coupling_map(), _basis(t), target=t)
    assert value_sig(x) == value_sig(y)


def test_constructions_only(mods, fez):
    """With SEARCH_FALLBACK = False the output is one of the constructions, still exact."""
    c49, a15 = mods["c49"], mods["a15"]
    a15.SEARCH_FALLBACK = False
    try:
        for name, qc in _cases()[:2]:
            out, info = _call(a15, qc, fez.target, return_info=True)
            assert info["chosen"] != "search" and "search" not in info["candidates"], name
            assert c49._implements(qc, out), name
    finally:
        a15.SEARCH_FALLBACK = True
