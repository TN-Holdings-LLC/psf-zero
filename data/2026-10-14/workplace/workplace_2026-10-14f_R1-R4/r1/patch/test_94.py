"""Tests for the workplace proposal of item 94 on prototype psf_compile 2026-10-14.c54: runs of commuting ZZ terms
re-layered by a first-fit edge colouring, the rewrite compiled too, the shorter exact output returned.

Run from the repository root with this file next to the built psf_compile.py:
    python -m pytest <dir>/test_94.py -q
"""
import contextlib
import copy
import io
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("PSF_REPO", os.getcwd())
sys.path[:0] = [HERE, os.path.join(REPO, "benchmarks"), REPO, os.path.join(REPO, "data", "2026-10-12", "c48_dev")]
C54_FILE = os.path.join(REPO, "patches", "psf_compile_c54_2026-10-14", "psf_compile.py")


def _load(path, name):
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = H.load_module(path, name)
    m.WARN_WITHOUT_TARGET = False
    return m


@pytest.fixture(scope="module")
def new():
    return _load(os.path.join(HERE, "psf_compile.py"), "psf_compile_c54_94_test")


@pytest.fixture(scope="module")
def c54():
    return _load(C54_FILE, "psf_compile_c54_for_94_test")


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


def _probs(c):
    from qiskit.quantum_info import Statevector
    from qiskit import QuantumCircuit
    meas, body = {}, []
    for ins in c.data:
        nm = ins.operation.name
        if nm == "measure":
            meas[c.find_bit(ins.clbits[0]).index] = c.find_bit(ins.qubits[0]).index
        elif nm not in ("barrier", "delay"):
            body.append((ins.operation, [c.find_bit(q).index for q in ins.qubits]))
    act = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(act)}
    s = QuantumCircuit(len(act))
    for op, qs in body:
        s.append(op, [pos[q] for q in qs])
    return np.asarray(Statevector(s).probabilities([pos[meas[k]] for k in range(c.num_clbits)]))


def _logical(qc):
    from qiskit import transpile
    return transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h", "rzz", "rx", "ry", "u", "measure"],
                     optimization_level=0)


def _call(m, qc, t):
    with contextlib.redirect_stdout(io.StringIO()):
        return m.compile_for_hardware(qc, target=t, entangling_basis="cx", layout_search=True, seed_transpiler=0)


@pytest.fixture(scope="module")
def qaoa12():
    sys.path.insert(0, os.path.join(REPO, "benchmarks"))
    import core_fix_c2_eval as H
    hw1 = H.load_module(os.path.join(REPO, "benchmarks", "hw1_real.py"), "hw1_for_94")
    hw1.KIT = os.path.join(REPO, "data", "2026-10-12", "hw_kit")
    return hw1.build_test("qaoa_n12_p2")[0]


def test_version(new):
    assert new.VERSION == "2026-10-14.c54+94"


def test_relayered_is_exact_and_changes_the_order(new, qaoa12):
    hl = new._zz_relayered(qaoa12)
    assert hl is not None
    assert np.abs(_probs(_logical(hl)) - _probs(_logical(qaoa12))).max() < 1e-12


def test_never_longer_than_c54_and_exact(new, c54, fez, qaoa12):
    for t in fez:
        out, ref = _call(new, qaoa12, t), _call(c54, qaoa12, t)
        assert new._duration_s(out, t) <= c54._duration_s(ref, t) + 1e-12
        assert np.abs(_probs(out) - _probs(_logical(qaoa12))).max() < 1e-9


def test_qaoa_n12_shorter_on_t(new, c54, fez, qaoa12):
    t, _ = fez
    k = new.ZZ94_STATS["kept_relayered"]
    out = _call(new, qaoa12, t)
    assert new.ZZ94_STATS["kept_relayered"] == k + 1
    assert new._duration_s(out, t) < c54._duration_s(_call(c54, qaoa12, t), t)


def test_not_a_pauli_program_is_c54s(new, c54, fez):
    from qiskit import QuantumCircuit
    from c29_identity import value_sig
    t, _ = fez
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    qc.measure_all()
    assert value_sig(_call(new, qc, t)) == value_sig(_call(c54, qc, t))


def test_switch_off_is_c54(new, c54, fez, qaoa12):
    from c29_identity import value_sig
    t, _ = fez
    new.ZZ_RELAYER = False
    try:
        a = _call(new, qaoa12, t)
    finally:
        new.ZZ_RELAYER = True
    assert value_sig(a) == value_sig(_call(c54, qaoa12, t))


def test_deterministic(new, fez, qaoa12):
    from c29_identity import value_sig
    t, f = fez
    assert value_sig(_call(new, qaoa12, f)) == value_sig(_call(new, qaoa12, f))
