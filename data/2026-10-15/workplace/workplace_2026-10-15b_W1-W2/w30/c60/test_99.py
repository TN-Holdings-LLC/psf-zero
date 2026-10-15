"""Tests for item 99 (workplace proposal c60): native two-qubit gate circuits on an embeddable graph, by formula.
Run from the repository root (bundle o), C60 pointing to the file (default: next to this test):
    python -m pytest <dir>/test_99.py -q
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
    mod = _load(os.environ.get("C60", os.path.join(HERE, "psf_compile.py")), "psf_c60_test99")
    mod.WARN_WITHOUT_TARGET = False
    return mod


@pytest.fixture(scope="module")
def c59():
    mod = _load(os.path.join(REPO, "patches", "psf_compile_c59_2026-10-15", "psf_compile.py"), "psf_c59_test99")
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


def _call(mod, qc, tg):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)


def _probs(c):
    from qiskit import QuantumCircuit, transpile
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
    small = QuantumCircuit(len(act))
    for op, qs in body:
        small.append(op, [pos[q] for q in qs])
    small = transpile(small, basis_gates=["u", "cx", "rzz", "cz", "rz", "sx", "x", "rx"], optimization_level=0)
    return Statevector(small).probabilities_dict(qargs=[pos[meas[k]] for k in sorted(meas)])


def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def _mixed(n=6, seed=7):
    """A chain with every kind of two-qubit gate item 99 handles, in both orientations."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
        qc.ry(rng.uniform(-3, 3), q)
    gates = ["cz", "cx", "rzz", "cp", "crz", "rxx", "ryy", "iswap", "swap", "crx", "cy"]
    for layer in range(2):
        for q in range(n - 1):
            g = gates[(q + layer * (n - 1)) % len(gates)]
            a, b = (q, q + 1) if (q + layer) % 2 else (q + 1, q)
            if g in ("rzz", "cp", "crz", "rxx", "ryy", "crx"):
                getattr(qc, g)(rng.uniform(-3, 3), a, b)
            else:
                getattr(qc, g)(a, b)
        for q in range(n):
            qc.rz(rng.uniform(-3, 3), q)
            qc.sx(q)
    qc.rzz(math.pi / 2, 0, 1)
    qc.rzz(-math.pi / 2, 2, 3)
    qc.rzz(math.pi, 4, 5)
    qc.measure_all()
    return qc


def test_version(m):
    assert m.VERSION.startswith("2026-10-15.c60") and m.NATIVE_BRICKWORK


def test_mixed_gates_exact_native(m, targets):
    for tg in targets:
        qc = _mixed()
        b = m.FORMULA97_STATS["native"]
        out = _call(m, qc, tg)
        assert m.FORMULA97_STATS["native"] == b + 1
        assert _tvd(_probs(out), _probs(qc)) < 1e-9
        names = set(tg.operation_names)
        assert all(i.operation.name in names for i in out.data)
        assert all(0 < float(i.operation.params[0]) <= math.pi / 2 + 1e-12 for i in out.data if i.operation.name == "rzz")


def test_cz_chain_depth_by_colouring(m, targets):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(10)
    for q in range(10):
        qc.h(q)
    for q in range(9):
        qc.cz(q, q + 1)
    qc.measure_all()
    out = _call(m, qc, targets[0])
    clock = {}
    for i in out.data:
        if i.operation.name == "cz":
            qs = [out.find_bit(x).index for x in i.qubits]
            s = max(clock.get(x, 0) for x in qs) + 1
            for x in qs:
                clock[x] = s
    assert max(clock.values()) == 2  # nine CZs in two layers, not a staircase of nine


def test_formula_outputs_unchanged_and_out_of_scope_refused(m, c59, targets):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(8)
    for q in range(8):
        qc.ry(0.2 * q, q)
    for _ in range(2):
        for q in range(7):
            qc.rzz(0.3 + 0.1 * q, q, q + 1)
        for q in range(8):
            qc.rx(0.4, q)
    qc.measure_all()
    for tg in targets:
        assert repr(_call(m, qc, tg).data) == repr(_call(c59, qc, tg).data)
    k5 = QuantumCircuit(5)
    for a in range(5):
        for b in range(a + 1, 5):
            k5.cz(a, b)
    with pytest.raises(m.OutOfScopeError):
        _call(m, k5, targets[0])


def test_deterministic_and_switch(m, targets):
    qc = _mixed(seed=3)
    a, b = _call(m, qc, targets[1]), _call(m, qc, targets[1])
    assert repr(a.data) == repr(b.data)
    m.NATIVE_BRICKWORK = False
    try:
        with pytest.raises(m.OutOfScopeError):
            _call(m, qc, targets[1])
    finally:
        m.NATIVE_BRICKWORK = True


def test_layout_recorded(m, targets):
    qc = _mixed()
    out = _call(m, qc, targets[0])
    lay = out.layout.initial_index_layout(filter_ancillas=True)
    assert len(lay) == qc.num_qubits and len(set(lay)) == qc.num_qubits
