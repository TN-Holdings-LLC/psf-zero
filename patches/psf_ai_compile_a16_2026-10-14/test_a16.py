"""Tests for the workplace proposal psf_ai_compile 2026-10-14.a16 (a15 + item 28: N2's ring formula as a candidate;
item 29: single-qubit runs as one rx pulse when the Target offers rx), on psf_compile c51 + items 87-88.

Run from the repository root, with the a16 folder at patches/psf_ai_compile_a16_2026-10-14/ and the built
psf_compile at patches/psf_compile_c51_87_88_2026-10-14/psf_compile.py (or PSF_COMPILE pointing to it):
    python -m pytest patches/psf_ai_compile_a16_2026-10-14/test_a16.py -q
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
PSF = os.environ.get("PSF_COMPILE", os.path.join(REPO, "patches", "psf_compile_c53_2026-10-14", "psf_compile.py"))  # home: c53 = c52 + items 87-88 (Addendum 483)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(PSF, "psf_compile")
    pc.WARN_WITHOUT_TARGET = False
    a16 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a16")
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
    assert mods["a16"].AI_COMPILE_VERSION == "2026-10-14.a16"


def test_ring_read(mods):
    a16 = mods["a16"]
    got = a16._ring_read(classifier(5, 3, 1))
    assert got is not None and got[0] == [0, 1, 2, 3, 4] and len(got[1]) == 3 and len(got[3]) == 5
    assert a16._ring_read(classifier(6, 2, 2, shuffle=True)) is not None  # the CZs of a layer in any order
    from qiskit import QuantumCircuit
    chain = QuantumCircuit(4)
    for q in range(3):
        chain.cz(q, q + 1)
    assert a16._ring_read(chain) is None  # not a cycle
    two = classifier(4, 1, 3, measure=False)
    two.cz(0, 2)
    two.cz(2, 1)
    two.cz(1, 3)
    two.cz(3, 0)  # a different cycle in the second layer
    assert a16._ring_read(two) is None
    assert a16._ring_read(classifier(3, 2, 4)) is None  # n < 4


def test_formula_cz_counts_and_exact(mods, fez):
    """7 CZ per layer at n = 4 and 15 at n = 6 (N2), exact, chosen over a15's candidates and a14's search"""
    a16 = mods["a16"]
    t, _ = fez
    for n, L, per in ((4, 8, 7), (6, 6, 15)):
        qc = classifier(n, L, 10 + n)
        out, info = _call(a16, qc, t)
        assert info["chosen"] == "ring"
        assert _two_q(out) == per * L
        assert same_distribution(out, qc)
        assert mods["pc"]._isa_ok(out, t)


def test_not_longer_than_a15(mods, fez):
    a16, a15 = mods["a16"], mods["a15"]
    t, _ = fez
    for n, L in ((4, 1), (4, 5), (6, 3)):
        qc = classifier(n, L, 20 + L)
        out, _ = _call(a16, qc, t)
        with contextlib.redirect_stdout(io.StringIO()):
            ref = a15.compile_for_model_circuit(qc, t.build_coupling_map(), _basis(t), target=t)
        assert mods["pc"]._duration_s(out, t) <= mods["pc"]._duration_s(ref, t) + 1e-12


def test_rx_one_pulse_with_fractional_target(mods, fez):
    a16, pc = mods["a16"], mods["pc"]
    t, ft = fez
    qc = classifier(4, 6, 30)
    out, info = _call(a16, qc, ft)
    names = {i.operation.name for i in out.data}
    assert "sx" not in names and "rx" in names
    assert same_distribution(out, qc)
    assert pc._isa_ok(out, ft)
    plain, _ = _call(a16, qc, t)
    assert pc._duration_s(out, ft) < pc._duration_s(plain, t)


def test_unmeasured_even_layers_implement_the_unitary(mods, fez):
    a16, pc = mods["a16"], mods["pc"]
    t, _ = fez
    qc = classifier(4, 4, 40, measure=False)
    ring = a16._ring_candidate(qc, t, _basis(t), 0)
    assert ring is not None
    assert pc._implements(qc, ring)
    k = a16.A16_STATS["ring_odd_unmeasured"]
    assert a16._ring_candidate(classifier(4, 3, 41, measure=False), t, _basis(t), 0) is None
    assert a16.A16_STATS["ring_odd_unmeasured"] == k + 1


def test_switches_off_give_a15(mods, fez):
    a16, a15 = mods["a16"], mods["a15"]
    t, _ = fez
    from c29_identity import value_sig
    qc = classifier(4, 3, 50)
    a16.RING_FORMULA, a16.RX_ONE_PULSE = False, False
    try:
        out, _ = _call(a16, qc, t)
    finally:
        a16.RING_FORMULA, a16.RX_ONE_PULSE = True, True
    with contextlib.redirect_stdout(io.StringIO()):
        ref = a15.compile_for_model_circuit(qc, t.build_coupling_map(), _basis(t), target=t)
    assert value_sig(out) == value_sig(ref)


def test_deterministic(mods, fez):
    from c29_identity import value_sig
    a16 = mods["a16"]
    _, ft = fez
    qc = classifier(6, 4, 60)
    assert value_sig(_call(a16, qc, ft)[0]) == value_sig(_call(a16, qc, ft)[0])


def test_failed_qubit_avoided(mods, fez):
    """a qubit of the line item 87's rule would take, reported failed, is cut first"""
    from qiskit.transpiler import InstructionProperties
    a16, pc = mods["a16"], mods["pc"]
    t, _ = fez
    qc = classifier(4, 4, 70)
    out, _ = _call(a16, qc, t)
    used = sorted({out.find_bit(q).index for i in out.data for q in i.qubits if i.operation.name != "barrier"})
    bad = copy.deepcopy(t)
    p = bad["sx"][(used[1],)]
    bad.update_instruction_properties("sx", (used[1],), InstructionProperties(duration=p.duration, error=1.0))
    out2, info = _call(a16, qc, bad)
    used2 = {out2.find_bit(q).index for i in out2.data for q in i.qubits if i.operation.name != "barrier"}
    assert used[1] not in used2
    assert same_distribution(out2, qc)


def test_out_of_range_rzz_folded(mods, fez):
    """item 29: a candidate with rzz angles outside 0 < theta <= pi/2 is folded (same action) before it is compared"""
    from qiskit import QuantumCircuit, transpile
    a16, pc = mods["a16"], mods["pc"]
    _, ft = fez
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rzz(-0.4, 0, 1)
    qc.rzz(2.6, 0, 1)
    qc.rx(0.3, 1)
    c = transpile(qc, target=ft, initial_layout=[0, 1], optimization_level=0)
    assert any(float(i.operation.params[0]) <= 0 or float(i.operation.params[0]) > math.pi / 2
               for i in c.data if i.operation.name == "rzz")
    k = a16.A16_STATS["rzz_folded"]
    new = a16._executable(c, ft)
    assert a16.A16_STATS["rzz_folded"] == k + 1
    assert all(0 < float(i.operation.params[0]) <= math.pi / 2 + 1e-12 for i in new.data if i.operation.name == "rzz")
    assert pc._implements(qc, new)
