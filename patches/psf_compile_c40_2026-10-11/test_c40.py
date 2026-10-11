"""Tests for prototype psf_compile 2026-10-11.c40 (changelog items 71-72 on c39): STEINER6 tried on every circuit it can
read as Pauli rotations (commuting or not; product formulas of any order and repetitions; sequences of evolutions and
rotation gates; QAOAAnsatz; a leading single-qubit layer; final measurements), with exact merging of repeated terms,
and item 69's choice by ESP.

Run from the repository root:  python -m pytest patches/psf_compile_c40_2026-10-11/test_c40.py -q
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
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_for_c40_test")
    c40 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c40_test")
    rel.WARN_WITHOUT_TARGET = c40.WARN_WITHOUT_TARGET = False
    return dict(rel=rel, c40=c40)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


@pytest.fixture(scope="module")
def small():
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap, InstructionProperties
    cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 6]])
    cm.make_symmetric()
    b = GenericBackendV2(7, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cm, seed=72)
    rng = random.Random(72)
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


def _label(n, paulis):
    lb = ["I"] * n
    for q, p in paulis.items():
        lb[n - 1 - q] = p
    return "".join(lb)


def _op(n, seed, kinds="XYZ", k_max=3, count=None):
    from qiskit.quantum_info import SparsePauliOp
    rng = random.Random(seed)
    terms = {}
    for _ in range(count or 3 * n):
        k = rng.randint(1, k_max)
        qs = rng.sample(range(n), k)
        terms[_label(n, {q: rng.choice(kinds) for q in qs})] = rng.uniform(-1, 1)
    return SparsePauliOp.from_list(sorted(terms.items()))


def _evo(op, n, synthesis=None, time=0.6, measure=False):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    qc = QuantumCircuit(n, n if measure else 0)
    qc.append(PauliEvolutionGate(op, time=time, synthesis=synthesis), range(n))
    if measure:
        qc.measure(range(n), range(n))
    return qc


def _qaoa(n, seed, reps=2):
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.5] or [(0, 1)]
    cost = SparsePauliOp.from_list([(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a, b in pairs])
    qa = QAOAAnsatz(cost, reps=reps)
    return qa.assign_parameters([rng.uniform(0, 3) for _ in range(qa.num_parameters)])


def _rotations(n, seed):
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for _ in range(4 * n):
        g = rng.choice(["rz", "rx", "ry", "rzz", "rxx", "ryy"])
        if g in ("rz", "rx", "ry"):
            getattr(qc, g)(rng.uniform(-2, 2), rng.randrange(n))
        else:
            a, b = rng.sample(range(n), 2)
            getattr(qc, g)(rng.uniform(-2, 2), a, b)
    return qc


def _exact(out, qc):
    """The output implements the product formula Qiskit compiles for the input (the evolutions' definitions, not their
    exact exponentials, which differ when terms do not commute)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    op = Operator.from_circuit(out)
    ref = qc.remove_final_measurements(inplace=False) if qc.num_clbits else qc
    ref = ref.decompose(reps=4)
    ext = QuantumCircuit(op.num_qubits)
    ext.compose(ref, range(qc.num_qubits), inplace=True)
    return op.equiv(Operator(ext))


def _cases():
    from qiskit.synthesis import LieTrotter, SuzukiTrotter
    out = [("noncommuting", _evo(_op(5, 1), 5)),
           ("lie3", _evo(_op(4, 2), 4, LieTrotter(reps=3))),
           ("suzuki2", _evo(_op(4, 3), 4, SuzukiTrotter(order=2, reps=2))),
           ("lie_unordered", _evo(_op(5, 4), 5, LieTrotter(reps=2, preserve_order=False))),
           ("qaoa", _qaoa(5, 5)),
           ("rotations", _rotations(5, 6)),
           ("commuting_reps4", _evo(_op(5, 7, kinds="Z"), 5, LieTrotter(reps=4)))]
    return out


def test_version(mods):
    assert mods["c40"].VERSION == "2026-10-11.c40"


def test_steiner_construction_is_exact(mods, small):
    """STEINER6's own output (before item 69's choice), read through its layout, implements every kind of input."""
    c40 = mods["c40"]
    tgt = small.target
    for name, qc in _cases():
        st = c40._st_compile(qc, tgt, 0.5, 0, 300_000, 2_500)
        assert st is not None, name
        assert _exact(st, qc), name


def test_chosen_output_is_exact_and_never_worse(mods, small):
    c40 = mods["c40"]
    tgt = small.target
    for name, qc in _cases():
        out = _call(c40, qc, backend=small)
        assert _exact(out, qc), name
        usual = _call(c40, qc, backend=small, steiner=False)
        assert c40._st_cost(out, tgt) <= c40._st_cost(usual, tgt) + 1e-12, name


def test_merging(mods):
    """A commuting Hamiltonian over four Trotter steps is merged to its distinct terms; a non-commuting one is not."""
    from qiskit.synthesis import LieTrotter
    c40 = mods["c40"]
    op = _op(5, 7, kinds="Z")
    got = c40._st_parse(_evo(op, 5, LieTrotter(reps=4)))
    assert len(got[2]) == 4 * len(op) and len(c40._st_merge(got[2])) == len(op)
    op2 = _op(5, 8)
    got2 = c40._st_parse(_evo(op2, 5, LieTrotter(reps=2)))
    assert len(c40._st_merge(got2[2])) > len(op2)


def test_measurements_follow_the_final_placement(mods, small):
    c40 = mods["c40"]
    qc = _qaoa(5, 9)
    qc.measure_all()
    st = c40._st_compile(qc, small.target, 0.5, 0, 300_000, 2_500)
    final = st.layout.final_index_layout()
    got = {st.find_bit(i.clbits[0]).index: st.find_bit(i.qubits[0]).index for i in st.data
           if i.operation.name == "measure"}
    assert got == {q: final[q] for q in range(5)}


def test_unreadable_inputs_take_the_usual_path(mods, torino):
    """A single-qubit gate after the rotations, an unbound parameter, QDrift, ordinary circuits: as release 2026-10-11.2
    with steiner off, by value; and not read by item 72."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.random import random_circuit
    from qiskit.synthesis import QDrift
    c40, rel = mods["c40"], mods["rel"]
    a = _rotations(4, 10)
    a.h(0)
    b = QuantumCircuit(3)
    b.rzz(Parameter("t"), 0, 1)
    c = _evo(_op(4, 11), 4, QDrift(reps=3, seed=11))  # QDrift samples terms at random: a fixed seed
    for qc in (a, random_circuit(5, 8, max_operands=2, seed=12)):
        assert c40._st_parse(qc) is None
        assert _sig(_call(c40, qc, backend=torino)) == _sig(_call(rel, qc, backend=torino, steiner=False))
    assert c40._st_parse(b) is None
    assert c40._st_parse(c) is None  # QDrift: not read; its output is random from call to call, also in the release


def test_no_failed_element_on_a_device(mods, torino):
    c40 = mods["c40"]
    edges, bad = c40._failed_elements(torino.target, 0.5)
    for qc in (_qaoa(16, 13, reps=2), _evo(_op(12, 14, k_max=2), 12)):
        qc.measure_all()
        out = _call(c40, qc, backend=torino)
        assert not c40._uses_failed(out, edges, bad)
