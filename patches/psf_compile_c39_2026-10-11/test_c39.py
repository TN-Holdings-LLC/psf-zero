"""Tests for candidate psf_compile 2026-10-11.c39 (changelog items 69-70): release 2026-10-11.2 with STEINER6's output
kept only where its ESP on the device is above the usual path's, and a 2000-term limit for item 68.

Run from the repository root:  python -m pytest patches/psf_compile_c39_2026-10-11/test_c39.py -q
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
    rel = H.load_module(os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.2", "psf_compile.py"), "psf_compile_rel_for_c39_test")
    c38 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c39_test")
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
    assert mods["c38"].VERSION == "2026-10-11.c39"
    assert mods["rel"].VERSION == "2026-10-11.2"


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
    assert c38.STEINER_STATS["used"] - n0 == len(cases)  # STEINER6 ran on each (item 69 then chose)


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
    assert _sig(_call(c38, qc, backend=torino, steiner=False)) == _sig(_call(rel, qc, backend=torino, steiner=False))
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


def _tsp_unary(k, seed):
    """A tsp-type Hamiltonian in unary encoding: qubit (city c, step t) = c * k + t; one-hot penalties as ZZ within each
    city's row and each step's column, distances as ZZ between consecutive steps; Z fields. Dense, two-local,
    diagonal: the type item 68 lost on (Addendum 445)."""
    rng = random.Random(seed)
    n = k * k
    terms = set()
    for c in range(k):
        for t in range(k):
            terms.add(_label(n, {c * k + t: "Z"}))
            for t2 in range(t + 1, k):
                terms.add(_label(n, {c * k + t: "Z", c * k + t2: "Z"}))
                terms.add(_label(n, {t * k + c: "Z", t2 * k + c: "Z"}))
    for a in range(k):
        for b in range(k):
            if a != b:
                for t in range(k):
                    if rng.random() < 0.8:
                        terms.add(_label(n, {a * k + t: "Z", b * k + (t + 1) % k: "Z"}))
    return sorted(terms)


def test_item_69_never_worse_than_the_usual_path(mods, torino):
    """The returned circuit is the lower-cost of STEINER6's and the usual path's, by the device's errors."""
    c39, rel = mods["c38"], mods["rel"]
    tgt = torino.target
    cases = [_evolution(16, _tsp_unary(4, 7400), 7401), _evolution(12, _diagonal(12, 7402), 7403),
             _evolution(20, _diagonal(20, 7404, k3=False), 7405)]
    k0 = dict(c39.STEINER_STATS)
    for qc in cases:
        out = _call(c39, qc, backend=torino)
        usual = _call(rel, qc, backend=torino, steiner=False)
        steiner = _call(rel, qc, backend=torino)  # release 2026-10-11.2: item 68 alone
        best = min(c39._st_cost(usual, tgt), c39._st_cost(steiner, tgt))
        assert abs(c39._st_cost(out, tgt) - best) < 1e-9
        assert c39._st_cost(out, tgt) <= c39._st_cost(usual, tgt) + 1e-12
    k1 = c39.STEINER_STATS
    assert (k1["chosen"] - k0["chosen"]) + (k1["kept_usual"] - k0["kept_usual"]) == len(cases)


def test_item_70_large_inputs_take_the_usual_path(mods, torino):
    """More than 2000 terms: the usual path's output, by value, and the size limit counted."""
    c39, rel = mods["c38"], mods["rel"]
    n = 66
    terms = [_label(n, {a: "Z", b: "Z"}) for a in range(n) for b in range(a + 1, n)]  # 2145 commuting terms
    qc = _evolution(n, terms, 7500)
    k0 = c39.STEINER_STATS["too_large"]
    assert _sig(_call(c39, qc, backend=torino)) == _sig(_call(rel, qc, backend=torino, steiner=False))
    assert c39.STEINER_STATS["too_large"] == k0 + 1
