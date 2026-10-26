"""Tests of AI front end a26 (item 50, the ring folded onto a device cycle), from the PSF-Zero root with rust/psfbind
(ring r3) built:

    python -m pytest -q patches/psf_ai_compile_a26_2026-10-26/test_a26.py
"""
import contextlib
import copy
import importlib.util
import io
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit.transpiler import InstructionProperties  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

warnings.simplefilter("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mods():
    with contextlib.redirect_stdout(io.StringIO()):
        pc = _load(os.path.join(ROOT, "psf_compile.py"), "psf_compile")
        pc.WARN_WITHOUT_TARGET = False
        a25 = _load(os.path.join(ROOT, "patches", "psf_ai_compile_a25_2026-10-25", "psf_ai_compile.py"), "ai25_t26")
        a26 = _load(os.path.join(ROOT, "patches", "psf_ai_compile_a26_2026-10-26", "psf_ai_compile.py"), "ai26_test")
    try:
        a26._psfbind()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"rust/psfbind not built: {e}")
    return a25, a26


@pytest.fixture(scope="module")
def target():
    return fake_provider.FakeMarrakesh().target


def ring(n, L, seed, order=None):
    """TOQB's ring classifier shape: per layer ry, rz on every qubit, then the CZs of one cycle; measured."""
    rng = np.random.default_rng(seed)
    order = list(range(n)) if order is None else order
    qc = QuantumCircuit(n, n)
    for _ in range(L):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q)
            qc.rz(rng.uniform(-3, 3), q)
        for i in range(n):
            qc.cz(order[i], order[(i + 1) % n])
    for q in range(n):
        qc.ry(rng.uniform(-3, 3), q)
    qc.measure(range(n), range(n))
    return qc


def p1_all(c):
    act = sorted({c.find_bit(q).index for i in c.data for q in i.qubits})
    idx = {q: k for k, q in enumerate(act)}
    s, meas = QuantumCircuit(len(act)), {}
    for i in c.data:
        if i.operation.name == "measure":
            meas[c.find_bit(i.clbits[0]).index] = idx[c.find_bit(i.qubits[0]).index]
        elif i.operation.name not in ("barrier", "delay"):
            s.append(i.operation, [idx[c.find_bit(q).index] for q in i.qubits])
    sv = Statevector(s)
    return [float(sv.probabilities([meas[k]])[1]) for k in sorted(meas)]


def q2(c):
    return sum(1 for i in c.data if len(i.qubits) == 2)


@pytest.mark.parametrize("n,kind", [(13, "pend"), (16, "pend"), (18, "holes"), (10, "holes")])
def test_folds_exactly(mods, target, n, kind):
    _, a26 = mods
    qc = ring(n, 2, n)
    tp = a26.model_template(qc, target)
    assert isinstance(tp, a26.FoldTemplate) and tp.schedule == kind
    c = tp.bind(qc)
    assert max(abs(a - b) for a, b in zip(p1_all(c), p1_all(qc))) < 1e-9
    if kind == "pend":
        assert q2(c) == 2 * (2 * n - tp.cycle)
    qc2 = ring(n, 2, 999)  # same structure, other angles: bound, not rebuilt
    assert max(abs(a - b) for a, b in zip(p1_all(tp.bind(qc2)), p1_all(qc2))) < 1e-9
    d = a26.compile_for_model_circuit(qc, target.build_coupling_map(), ["cz", "rz", "sx", "x"], target=target)
    assert [(i.operation.name, [d.find_bit(q).index for q in i.qubits]) for i in d.data] == \
        [(i.operation.name, [c.find_bit(q).index for q in i.qubits]) for i in c.data]


def test_any_cycle_order(mods, target):
    _, a26 = mods
    order = [0, 5, 2, 9, 12, 1, 7, 3, 11, 4, 10, 6, 8]
    qc = ring(13, 2, 3, order)
    tp = a26.model_template(qc, target)
    assert isinstance(tp, a26.FoldTemplate)
    assert max(abs(a - b) for a, b in zip(p1_all(tp.bind(qc)), p1_all(qc))) < 1e-9


def test_switch_off_and_small_rings_are_a25(mods, target):
    a25, a26 = mods
    for n, off in ((13, True), (6, False), (5, False)):
        qc = ring(n, 2, n)
        if off:
            os.environ["PSF_RING_FOLD"] = "0"
        try:
            t25, t26 = a25.model_template(qc, target), a26.model_template(qc, target)
        finally:
            os.environ.pop("PSF_RING_FOLD", None)
        assert not isinstance(t26, a26.FoldTemplate)
        c25, c26 = t25.bind(qc), t26.bind(qc)
        assert [(i.operation.name, [c25.find_bit(q).index for q in i.qubits]) for i in c25.data] == \
            [(i.operation.name, [c26.find_bit(q).index for q in i.qubits]) for i in c26.data]


def test_failed_coupler_avoided(mods, target):
    _, a26 = mods
    qc = ring(13, 2, 5)
    tp = a26.model_template(qc, target)
    c = tp.bind(qc)
    a, b = next(tuple(c.find_bit(q).index for q in i.qubits) for i in c.data if i.operation.name == "cz")
    t2 = copy.deepcopy(target)
    for qa in ((a, b), (b, a)):
        if qa in t2["cz"]:
            p = t2["cz"][qa]
            t2.update_instruction_properties("cz", qa, InstructionProperties(duration=p.duration, error=1.0))
    tp2 = a26.model_template(qc, t2)
    c2 = tp2.bind(qc) if tp2 is not None else None
    if c2 is not None:
        assert all({c2.find_bit(q).index for q in i.qubits} != {a, b} for i in c2.data if len(i.qubits) == 2)
        if isinstance(tp2, a26.FoldTemplate):
            assert max(abs(x - y) for x, y in zip(p1_all(c2), p1_all(qc))) < 1e-9


def test_not_a_ring_is_a25(mods, target):
    a25, a26 = mods
    qc = QuantumCircuit(5, 5)
    qc.h(0)
    for j in range(4):
        qc.cx(j, j + 1)
    qc.measure(range(5), range(5))
    with contextlib.redirect_stdout(io.StringIO()):
        o25 = a25.compile_for_model_circuit(qc, target.build_coupling_map(), ["cz", "rz", "sx", "x"], target=target)
        o26 = a26.compile_for_model_circuit(qc, target.build_coupling_map(), ["cz", "rz", "sx", "x"], target=target)
    assert [(i.operation.name, [o25.find_bit(q).index for q in i.qubits]) for i in o25.data] == \
        [(i.operation.name, [o26.find_bit(q).index for q in i.qubits]) for i in o26.data]
