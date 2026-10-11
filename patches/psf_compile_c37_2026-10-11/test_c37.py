"""Tests for candidate psf_compile 2026-10-11.c37 (changelog items 65-67): release 2026-10-11.1 with a last relabelling
scored on every operation's error, readout included (item 65), the Pauli parts of single-qubit runs carried through
two-qubit Clifford gates (item 66), and several routing tries kept by ESP (item 67). All are on with the device's
Target and off without one.

Run from the repository root:  python -m pytest patches/psf_compile_c37_2026-10-11/test_c37.py -q
"""
import contextlib
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.1", "psf_compile.py"), "psf_compile_rel_for_c37_test")
    c37 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c37_test")
    rel.WARN_WITHOUT_TARGET = c37.WARN_WITHOUT_TARGET = False
    return dict(rel=rel, c37=c37)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


@pytest.fixture(scope="module")
def small():
    """A 7-qubit line device with cz, rz, sx, x and measure, and errors that differ from qubit to qubit, so that
    whole-circuit operators can be compared."""
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap, InstructionProperties
    b = GenericBackendV2(7, basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_line(7), seed=65)
    rng = random.Random(65)
    for q in range(7):
        b.target.update_instruction_properties("measure", (q,), InstructionProperties(error=rng.uniform(0.005, 0.1)))
        for g in ("sx", "x"):
            b.target.update_instruction_properties(g, (q,), InstructionProperties(error=rng.uniform(1e-4, 1e-3)))
    return b


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits(n=6, seed=65_100, measure=None):
    from qiskit.circuit.random import random_circuit
    return [random_circuit(4 + k % 3, 8 + 2 * k, max_operands=2,
                           measure=(k % 2 == 0) if measure is None else measure, seed=seed + k) for k in range(n)]


def _call(mod, qc, **kw):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return mod.compile_for_hardware(qc, entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)


def test_version(mods):
    assert mods["c37"].VERSION == "2026-10-11.c37"
    assert mods["rel"].VERSION == "2026-10-11.1"


def test_items_off_is_the_release(mods, torino):
    """With items 65-67 off and the device given, the release's call, by value."""
    for qc in _circuits():
        assert _sig(_call(mods["c37"], qc, backend=torino, final_relabel=False, final_push=False, route_tries=1)) == \
            _sig(_call(mods["rel"], qc, backend=torino))


def test_without_the_device_unchanged(mods, torino):
    """Without a Target, the release's default call, by value."""
    cm = torino.target.build_coupling_map()
    basis = [g for g in torino.target.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    for qc in _circuits(seed=65_200):
        assert _sig(_call(mods["c37"], qc, coupling_map=cm, basis_gates=basis)) == \
            _sig(_call(mods["rel"], qc, coupling_map=cm, basis_gates=basis))


def test_needs_the_target(mods, torino):
    cm = torino.target.build_coupling_map()
    for kw in (dict(final_push=True), dict(final_relabel=True), dict(route_tries=2)):
        with pytest.raises(ValueError):
            _call(mods["c37"], _circuits(n=1)[0], coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"], **kw)
    with pytest.raises(ValueError):
        _call(mods["c37"], _circuits(n=1)[0], backend=torino, route_tries=0)


def test_no_failed_element_with_the_device(mods, torino):
    edges, qubits = mods["c37"]._failed_elements(torino.target, 0.5)
    for qc in _circuits(seed=65_300):
        assert not mods["c37"]._uses_failed(_call(mods["c37"], qc, backend=torino), edges, qubits)


def test_same_operator_on_a_small_device(mods, small):
    """The output implements the input (layout and final permutation accounted for) on a 7-qubit device."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    c37 = mods["c37"]
    before = dict(c37.FINAL_STATS)
    for qc in _circuits(n=8, seed=65_400, measure=False):
        out = _call(c37, qc, backend=small)
        op = Operator.from_circuit(out)  # on the input's qubits and the ancillas, which must be left alone
        ext = QuantumCircuit(op.num_qubits)
        ext.compose(qc, range(qc.num_qubits), inplace=True)
        assert op.equiv(Operator(ext))
    assert c37.FINAL_STATS["push_applied"] > before["push_applied"]


def test_push_is_exact_with_global_phase(mods, torino):
    """Item 66 alone, on random circuits of the device's gates: the same matrix, global phase included."""
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    from qiskit.quantum_info import Operator
    c37 = mods["c37"]
    rng = random.Random(66)
    applied = 0
    for _ in range(150):
        qc = QuantumCircuit(4, global_phase=rng.random())
        for _ in range(rng.randint(5, 40)):
            k = rng.random()
            a, b = rng.sample(range(4), 2)
            if k < 0.3:
                qc.cz(a, b)
            elif k < 0.45:
                qc.x(a)
            elif k < 0.6:
                qc.sx(a)
            elif k < 0.8:
                qc.rz(rng.choice([np.pi, np.pi / 2, rng.random() * 6]), a)
            else:
                qc.x(a)
                qc.rz(np.pi, a)
        n0 = c37.FINAL_STATS["push_applied"]
        out = dag_to_circuit(c37._PushPaulis(torino.target, 0.5).run(circuit_to_dag(qc)))
        applied += c37.FINAL_STATS["push_applied"] - n0
        assert np.allclose(Operator(out).data, Operator(qc).data, atol=1e-8)
    assert applied > 50


def test_push_does_not_raise_the_one_qubit_cost(mods, torino):
    """Kept only when the single-qubit gates' summed -ln(1 - error) falls; two-qubit gates and measurements equal."""
    c37 = mods["c37"]
    for qc in _circuits(seed=65_500):
        a = _call(c37, qc, backend=torino, final_relabel=False, route_tries=1)
        b = _call(c37, qc, backend=torino, final_relabel=False, final_push=False, route_tries=1)
        ops_a, ops_b = a.count_ops(), b.count_ops()
        assert ops_a.get("cz", 0) == ops_b.get("cz", 0) and ops_a.get("measure", 0) == ops_b.get("measure", 0)
        p = c37._PushPaulis(torino.target, 0.5)
        from qiskit.converters import circuit_to_dag
        assert p._cost_1q(circuit_to_dag(a)) <= p._cost_1q(circuit_to_dag(b)) + 1e-12


def test_relabel_places_readout_without_two_qubit_gates(mods, torino):
    """Item 65: a circuit of single-qubit gates and measurements is measured on the device's best readout qubits,
    where the release (item 33 gives up on two or more such qubits) keeps qubits 0-3."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4, 4)
    for q in range(4):
        qc.h(q)
        qc.measure(q, q)
    tgt = torino.target
    out = _call(mods["c37"], qc, backend=torino)
    measured = sorted(out.find_bit(i.qubits[0]).index for i in out.data if i.operation.name == "measure")
    edges, failed = mods["c37"]._failed_elements(tgt, 0.5)
    ro = sorted(tgt["measure"][(q,)].error for q in range(tgt.num_qubits)
                if q not in failed)
    got = sum(tgt["measure"][(q,)].error for q in measured)
    assert got <= sum(ro[:4]) + 0.01  # the best four up to the single-qubit gates' share
    rel = _call(mods["rel"], qc, backend=torino)
    assert sum(tgt["measure"][(rel.find_bit(i.qubits[0]).index,)].error for i in rel.data
               if i.operation.name == "measure") > got


def test_route_tries_never_worse(mods, torino):
    """Item 67: the kept output is clean and its -ln ESP is not above the single try's."""
    c37 = mods["c37"]
    edges, qubits = c37._failed_elements(torino.target, 0.5)
    n0 = c37.TRIES_STATS["tries"]
    for qc in _circuits(n=4, seed=65_600):
        many = _call(c37, qc, backend=torino)
        one = _call(c37, qc, backend=torino, route_tries=1)
        assert not c37._uses_failed(many, edges, qubits)
        assert c37._esp_cost(many, torino.target) <= c37._esp_cost(one, torino.target) + 1e-12
    assert c37.TRIES_STATS["tries"] - n0 == 4 * 7
