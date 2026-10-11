"""Tests for candidate psf_compile 2026-10-11.c38 (changelog item 68): release 2026-10-11.1 with a Pauli evolution
whose terms all commute compiled by STEINER6's construction (Addendum 444).

Run from the repository root:  python -m pytest patches/psf_compile_c38_2026-10-11/test_c38.py -q
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
    rel = H.load_module(os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.1", "psf_compile.py"), "psf_compile_rel_for_c38_test")
    c38 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c38_test")
    rel.WARN_WITHOUT_TARGET = c38.WARN_WITHOUT_TARGET = False
    return dict(rel=rel, c38=c38)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


@pytest.fixture(scope="module")
def small():
    """A 7-qubit heavy-hex-like device (a line with one branch) with cz and errors that differ from qubit to qubit."""
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap, InstructionProperties
    cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 6]])
    cm.make_symmetric()
    b = GenericBackendV2(7, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cm, seed=68)
    rng = random.Random(68)
    for q in range(7):
        b.target.update_instruction_properties("measure", (q,), InstructionProperties(error=rng.uniform(0.005, 0.1)))
        for g in ("sx", "x"):
            b.target.update_instruction_properties(g, (q,), InstructionProperties(error=rng.uniform(1e-4, 1e-3)))
    for qa in b.target["cz"]:
        b.target.update_instruction_properties("cz", qa, InstructionProperties(error=rng.uniform(2e-3, 2e-2)))
    return b


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _call(mod, qc, **kw):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return mod.compile_for_hardware(qc, entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)


def _evolution(n, terms, seed, measure=False, time=0.7):
    """One PauliEvolutionGate of `terms` (labels) with random real coefficients."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    rng = random.Random(seed)
    op = SparsePauliOp.from_list([(t, rng.uniform(-1, 1)) for t in terms])
    qc = QuantumCircuit(n, n if measure else 0)
    qc.append(PauliEvolutionGate(op, time=time), range(n))
    if measure:
        qc.measure(range(n), range(n))
    return qc


def _label(n, paulis):
    lb = ["I"] * n
    for q, p in paulis.items():
        lb[n - 1 - q] = p
    return "".join(lb)


def _diagonal(n, seed, k3=True):
    """Z, ZZ (random pairs) and, if k3, ZZZ terms: all commute."""
    rng = random.Random(seed)
    terms = [_label(n, {q: "Z"}) for q in range(n)]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    terms += [_label(n, {a: "Z", b: "Z"}) for a, b in rng.sample(pairs, min(len(pairs), 2 * n))]
    if k3:
        for _ in range(n // 2):
            a, b, c = rng.sample(range(n), 3)
            terms.append(_label(n, {a: "Z", b: "Z", c: "Z"}))
    return list(dict.fromkeys(terms))


def _commuting_xy(n, seed):
    """XX and YY on the same pairs, and ZZ on others' complement: a commuting set with basis changes."""
    rng = random.Random(seed)
    pairs = [(2 * i, 2 * i + 1) for i in range(n // 2)]
    terms = []
    for a, b in pairs:
        terms += [_label(n, {a: "X", b: "X"}), _label(n, {a: "Y", b: "Y"}), _label(n, {a: "Z", b: "Z"})]
    rng.shuffle(terms)
    return terms


def test_version(mods):
    assert mods["c38"].VERSION == "2026-10-11.c38"
    assert mods["rel"].VERSION == "2026-10-11.1"


def test_commuting_evolution_is_exact_with_its_layout(mods, small):
    """The output, read through its layout, implements the input on the input's qubits and leaves the ancillas alone."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    c38 = mods["c38"]
    n0 = c38.STEINER_STATS["used"]
    cases = [(_diagonal(n, 680 + n), n) for n in (3, 4, 5, 6)] + [(_commuting_xy(n, 690 + n), n) for n in (4, 6)]
    for k, (terms, n) in enumerate(cases):
        qc = _evolution(n, terms, 700 + k)
        out = _call(c38, qc, backend=small)
        op = Operator.from_circuit(out)
        ext = QuantumCircuit(op.num_qubits)
        ext.compose(qc, range(n), inplace=True)
        assert op.equiv(Operator(ext)), k
    assert c38.STEINER_STATS["used"] - n0 == len(cases)


def test_measurements_follow_the_final_placement(mods, small):
    c38 = mods["c38"]
    qc = _evolution(5, _diagonal(5, 711), 712, measure=True)
    out = _call(c38, qc, backend=small)
    final = out.layout.final_index_layout()  # logical qubit -> physical qubit at the end
    got = {out.find_bit(i.clbits[0]).index: out.find_bit(i.qubits[0]).index for i in out.data
           if i.operation.name == "measure"}
    assert got == {q: final[q] for q in range(5)}


def test_other_inputs_are_the_release(mods, torino):
    """Non-commuting evolutions and ordinary circuits: the release's outputs, by value."""
    from qiskit.circuit.random import random_circuit
    c38, rel = mods["c38"], mods["rel"]
    n0 = dict(c38.STEINER_STATS)
    tfim = _evolution(6, [_label(6, {q: "X"}) for q in range(6)] + [_label(6, {q: "Z", q + 1: "Z"}) for q in range(5)],
                      720)
    cases = [tfim] + [random_circuit(4 + k, 8, max_operands=2, measure=k % 2 == 0, seed=6800 + k) for k in range(4)]
    for qc in cases:
        assert _sig(_call(c38, qc, backend=torino)) == _sig(_call(rel, qc, backend=torino))
    assert c38.STEINER_STATS["used"] == n0["used"]
    assert c38.STEINER_STATS["not_commuting"] == n0["not_commuting"] + 1


def test_off_without_target_or_with_the_callers_map(mods, torino):
    c38, rel = mods["c38"], mods["rel"]
    qc = _evolution(6, _diagonal(6, 730), 731)
    cm = torino.target.build_coupling_map()
    basis = ["cz", "rz", "sx", "x"]
    assert _sig(_call(c38, qc, coupling_map=cm, basis_gates=basis)) == \
        _sig(_call(rel, qc, coupling_map=cm, basis_gates=basis))
    assert _sig(_call(c38, qc, backend=torino, coupling_map=cm)) == _sig(_call(rel, qc, backend=torino, coupling_map=cm))
    assert _sig(_call(c38, qc, backend=torino, steiner=False)) == _sig(_call(rel, qc, backend=torino))
    with pytest.raises(ValueError):
        _call(c38, qc, coupling_map=cm, basis_gates=basis, steiner=True)


def test_no_failed_element_and_the_device_gates_only(mods, torino):
    c38 = mods["c38"]
    tgt = torino.target
    edges, bad = c38._failed_elements(tgt, 0.5)
    n0 = c38.STEINER_STATS["used"]
    for k, n in enumerate((8, 12, 16)):
        out = _call(c38, _evolution(n, _diagonal(n, 740 + k), 750 + k, measure=True), backend=torino)
        assert not c38._uses_failed(out, edges, bad)
        for ins in out.data:
            if ins.operation.name in ("barrier", "delay"):
                continue
            tgt[ins.operation.name][tuple(out.find_bit(q).index for q in ins.qubits)]  # KeyError if not on the device
    assert c38.STEINER_STATS["used"] - n0 == 3
