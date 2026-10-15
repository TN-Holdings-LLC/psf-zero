"""Tests for item 100 (workplace proposal c63 = c62 + the compile template, planned binding); the reference is c62. Run from the repository root (bundle q), with
C63 / C62 pointing to the files (defaults: next to this test, and ../c62/psf_compile.py):
    python -m pytest <dir>/test_c63.py -q
"""
import contextlib
import importlib.util
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
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def m():
    _load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    mod = _load(os.environ.get("C63", os.path.join(HERE, "psf_compile.py")), "psf_c63_test")
    mod.WARN_WITHOUT_TARGET = False
    return mod


@pytest.fixture(scope="module")
def m60():
    mod = _load(os.environ.get("C62", os.path.join(HERE, "..", "c62", "psf_compile.py")), "psf_c62_test63")
    mod.WARN_WITHOUT_TARGET = False
    return mod


@pytest.fixture(scope="module")
def targets():
    import copy
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.transpiler import InstructionProperties
    from qiskit_ibm_runtime.fake_provider import FakeFez
    t = FakeFez().target
    f = copy.deepcopy(t)
    f.add_instruction(RZZGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                                 for qa, p in t["cz"].items()})
    f.add_instruction(RXGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                               for qa, p in t["sx"].items()})
    return t, f


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _probs(c):
    from qiskit.quantum_info import Statevector
    meas, body = {}, []
    for ins in c.data:
        qs = [c.find_bit(q).index for q in ins.qubits]
        if ins.operation.name == "measure":
            meas[c.find_bit(ins.clbits[0]).index] = qs[0]
        elif ins.operation.name != "barrier":
            body.append((ins.operation, qs))
    act = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(act)}
    from qiskit import QuantumCircuit, transpile
    small = QuantumCircuit(len(act))
    for op, qs in body:
        small.append(op, [pos[q] for q in qs])
    small = transpile(small, basis_gates=["u", "cx", "rz", "ry", "rx", "rzz", "cz", "h", "sx", "x"],
                      optimization_level=0)
    p = Statevector(small).probabilities_dict(qargs=[pos[meas[k]] for k in sorted(meas)])
    return p


def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def _chain(n, layers=3, seed=0):
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q)
            qc.rz(rng.uniform(-3, 3), q)
        for q in range(n - 1):
            qc.cz(q, q + 1)
    qc.measure_all()
    return qc


def _tfim(n, seed=0, steps=3, prefix=True):
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for q in range(n if prefix else 0):
        qc.ry(0.3 + 0.1 * q, q)
    for _ in range(steps):
        for q in range(n - 1):
            qc.rzz(rng.uniform(-1, 1), q, q + 1)
        for q in range(n):
            qc.rx(rng.uniform(-1, 1), q)
    qc.measure_all()
    return qc


def test_version(m):
    assert m.VERSION.startswith("2026-10-15.c63")


def test_canonical_identity(m):
    from scipy.linalg import expm
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1])
    rng = np.random.default_rng(100)
    for _ in range(300):
        a, b, c = rng.uniform(-10, 10, 3)
        U = expm(1j * (a * np.kron(X, X) + b * np.kron(Y, Y) + c * np.kron(Z, Z)))
        assert m._n99_same(U, m._n99_kron_steps(m._t100_canonical(0, 1, a, b, c), 0, 1))


def test_native_template_identical_to_c62(m, m60, targets):
    for tg in targets:
        qc = _chain(8)
        tpl = m.compile_template(qc, tg)
        assert tpl.kind == "native"
        with contextlib.redirect_stdout(io.StringIO()):
            ref = m60.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True,
                                           seed_transpiler=0)
        assert _sig(tpl.bind(qc)) == _sig(ref)
        rng = random.Random(1)
        vals = [rng.uniform(-math.pi, math.pi) for _ in tpl.values_of(qc)]
        q2 = qc.copy_empty_like()
        it = iter(vals)
        for ins in qc.data:
            op = ins.operation
            if op.params:
                op = op.__class__(next(it))
            q2.append(op, ins.qubits, ins.clbits)
        with contextlib.redirect_stdout(io.StringIO()):
            ref2 = m60.compile_for_hardware(q2, target=tg, entangling_basis="cx", layout_search=True,
                                            seed_transpiler=0)
        assert _sig(tpl.bind_values(vals)) == _sig(ref2)


def test_formula_templates_exact(m, targets):
    for tg in targets:
        qc = _tfim(8)
        tpl = m.compile_template(qc, tg)
        assert tpl.kind in ("chain", "graph")
        assert tpl._gen.frame == ("rzz" in tg.operation_names)
        assert _tvd(_probs(tpl.bind(qc)), _probs(qc)) < 1e-9


def test_rebuild_on_sign_change_and_still_exact(m, targets):
    t, f = targets
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(6)  # from |0>: an rzz made negative needs an x before it (item 88), a run that was empty
    for q in range(5):
        qc.rzz(0.4 + 0.1 * q, q, q + 1)
    for q in range(6):
        qc.rx(0.3, q)
    qc.measure_all()
    tpl = m.compile_template(qc, f)
    q2 = qc.copy_empty_like()
    for ins in qc.data:
        op = ins.operation
        if op.name == "rzz":
            op = op.__class__(-float(op.params[0]))
        q2.append(op, ins.qubits, ins.clbits)
    before = m.TEMPLATE100_STATS["rebuilt"]
    out = tpl.bind(q2)
    assert m.TEMPLATE100_STATS["rebuilt"] == before + 1
    assert _tvd(_probs(out), _probs(q2)) < 1e-9


def test_structure_mismatch_raises(m, targets):
    tpl = m.compile_template(_chain(6), targets[0])
    with pytest.raises(ValueError):
        tpl.bind(_chain(6, layers=2))


def test_refusals(m, targets):
    with pytest.raises(m.OutOfScopeError):
        m.compile_template(_chain(6), None)
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(5)
    for a in range(5):
        for b in range(a + 1, 5):
            qc.cz(a, b)
    qc.measure_all()
    with pytest.raises(m.OutOfScopeError):
        m.compile_template(qc, targets[0])


def test_default_call_unchanged(m, m60, targets):
    for qc in (_chain(8), _tfim(8)):
        for tg in targets:
            with contextlib.redirect_stdout(io.StringIO()):
                a = m.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True,
                                           seed_transpiler=0)
                b = m60.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True,
                                             seed_transpiler=0)
            assert _sig(a) == _sig(b)
