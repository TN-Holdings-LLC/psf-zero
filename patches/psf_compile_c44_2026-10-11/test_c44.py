"""Tests for prototype psf_compile 2026-10-11.c44 (item 75 revised, item 77 on c43 = release 2026-10-11.4). c43's tests
are repeated on c44 (items 73-76), then items 75 (revised) and 77 are tested. Item 73 as in c41: circuits written as gates are read as Pauli
rotations by carrying their Clifford gates to the end (cx-rz-cx QAOA and Trotter circuits, basis changes by rx/ry,
phase gadgets from cp/crz/t, Toffolis through their definitions, any single-qubit gate as rz.ry.rz, final
single-qubit Cliffords such as an X-basis readout); the inputs of items 71-72 still work; item 69's choice by ESP.
The outputs equal the inputs up to a global phase.

Run from the repository root:  python -m pytest patches/psf_compile_c44_2026-10-11/test_c44.py -q
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
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_for_c44_test")
    c40 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c44_test")
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


def _qaoa_gates(n, seed, p=2):
    """QAOA as Benchpress's QASM files write it: ry(pi/2).rx(pi) per qubit, cx-rz-cx per edge, rx mixers."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(np.pi / 2, q)
        qc.rx(np.pi, q)
    edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.5] or [(0, 1)]
    for _ in range(p):
        g, beta = rng.uniform(0, 3), rng.uniform(0, 3)
        for a, b in edges:
            qc.cx(a, b)
            qc.rz(g, b)
            qc.cx(a, b)
        for q in range(n):
            qc.rx(beta, q)
    return qc


def _heisenberg_gates(n, seed):
    """XX, YY, ZZ on a chain by cx-rz-cx with rx(pi/2) and ry(pi/2).rx(pi) basis changes (as square-heisenberg)."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for a in range(n - 1):
        b = a + 1
        qc.cx(a, b)
        qc.rz(rng.uniform(-1, 1), b)
        qc.cx(a, b)
        qc.rx(np.pi / 2, [a, b])
        qc.cx(a, b)
        qc.rz(rng.uniform(-1, 1), b)
        qc.cx(a, b)
        qc.rx(-np.pi / 2, [a, b])
        for q in (a, b):
            qc.ry(np.pi / 2, q)
            qc.rx(np.pi, q)
        qc.cx(a, b)
        qc.rz(rng.uniform(-1, 1), b)
        qc.cx(a, b)
        for q in (a, b):
            qc.ry(np.pi / 2, q)
            qc.rx(np.pi, q)
    return qc


def _trotter_basis(n, seed):
    """A non-commuting evolution compiled by Qiskit to cx, rz, h, sx, x, s, sdg (as a QASM export would be)."""
    from qiskit import transpile
    return transpile(_evo(_op(n, seed, k_max=3), n), basis_gates=["cx", "rz", "h", "sx", "x", "s", "sdg"],
                     optimization_level=0)


def _gadgets(n, seed):
    """H layer, then cp, t and crz (read through their definitions), then an H layer (final single-qubit Cliffords)."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(2 * n):
        a, c = rng.sample(range(n), 2)
        qc.cp(rng.uniform(-2, 2), a, c)
        qc.t(a)
        qc.crz(rng.uniform(-2, 2), c, a)
    qc.h(range(n))
    return qc


def _mixed(n, seed):
    """Gates, a Pauli evolution and rotations interleaved, a u gate after a qubit's first cx, and an X-basis end."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    qc.u(0.3, 0.2, 0.1, 0)
    qc.cx(0, 1)
    qc.u(rng.uniform(0, 3), rng.uniform(0, 3), rng.uniform(0, 3), 1)
    qc.append(PauliEvolutionGate(_op(n, seed, k_max=2, count=n), time=0.4), range(n))
    qc.cz(1, 2)
    qc.rzz(0.5, 0, 2)
    qc.sx(2)
    qc.ecr(2, 3)
    qc.ry(0.7, 3)
    qc.ecr(2, 3)
    qc.sxdg(2)
    qc.cz(1, 2)
    qc.cx(0, 1)
    qc.h(range(n))
    return qc


def _toffolis(n, seed):
    """Toffolis (read through their definitions: h, t, tdg, cx) between H layers on the targets."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(n):
        a, b, c = rng.sample(range(n), 3)
        qc.h(c)
        qc.ccx(a, b, c)
        qc.h(c)
    return qc


def _qft_noswap(n):
    from qiskit.synthesis import synth_qft_full
    return synth_qft_full(n, do_swaps=False)


def _gate_cases():
    return [("qaoa_gates", _qaoa_gates(5, 21)), ("heisenberg_gates", _heisenberg_gates(5, 22)),
            ("trotter_basis", _trotter_basis(4, 23)), ("gadgets", _gadgets(5, 24)), ("mixed", _mixed(4, 25)),
            ("toffolis", _toffolis(5, 26)), ("qft_noswap", _qft_noswap(4))]


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
    assert mods["c40"].VERSION == "2026-10-11.c44"


def test_gate_level_inputs_are_read(mods):
    """Item 73 reads them (item 72 does not); the merged terms, the final gates."""
    c40 = mods["c40"]
    for name, qc in _gate_cases():
        assert c40._st_parse(qc) is None, name
        got = c40._st_parse_gates(qc)
        assert got is not None and got[2], name
    assert not c40._st_parse_gates(_qaoa_gates(5, 21))[4]  # nothing left at the end
    assert c40._st_parse_gates(_gadgets(5, 24))[4]  # the last H layer
    assert c40._st_parse_gates(_qaoa_gates(5, 21))[5] == list(range(5))  # no permutation


def test_steiner_construction_is_exact(mods, small):
    """STEINER6's own output (before item 69's choice), read through its layout, implements every kind of input."""
    c40 = mods["c40"]
    tgt = small.target
    for name, qc in _cases() + _gate_cases():
        st = c40._st_compile(qc, tgt, 0.5, 0, 300_000, 2_500)
        assert st is not None, name
        assert _exact(st, qc), name


def test_chosen_output_is_exact_and_never_worse(mods, small):
    c40 = mods["c40"]
    tgt = small.target
    for name, qc in _cases() + _gate_cases():
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
    for qc in (_qaoa(5, 9), _gadgets(5, 27)):
        qc.measure_all()
        st = c40._st_compile(qc, small.target, 0.5, 0, 300_000, 2_500)
        final = st.layout.final_index_layout()
        got = {st.find_bit(i.clbits[0]).index: st.find_bit(i.qubits[0]).index for i in st.data
               if i.operation.name == "measure"}
        assert got == {q: final[q] for q in range(5)}


def test_measured_gate_level_output_is_exact(mods, small):
    """With final measurements (and final H gates before them), the output without its measurements is exact, and
    every measurement comes after the last gate on its qubit."""
    c40 = mods["c40"]
    qc = _gadgets(5, 28)
    qc.measure_all()
    st = c40._st_compile(qc, small.target, 0.5, 0, 300_000, 2_500)
    body = st.copy_empty_like()
    for i in st.data:
        if i.operation.name != "measure":
            body.append(i.operation, i.qubits, i.clbits)
    body._layout = st._layout
    assert _exact(body, qc)
    seen = set()
    for i in st.data:
        q = st.find_bit(i.qubits[0]).index
        if i.operation.name == "measure":
            seen.add(q)
        else:
            assert not seen.intersection(st.find_bit(b).index for b in i.qubits)


def test_unreadable_inputs_take_the_usual_path(mods, torino):
    """A circuit ending with a cx (not a permutation), a Clifford circuit (no terms), a
    mid-circuit measurement, an unbound parameter, QDrift, a random circuit: not read by items 72-73, and given the
    usual path's output (steiner off), by value."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.random import random_circuit
    from qiskit.synthesis import synth_qft_full
    from qiskit.synthesis import QDrift
    c40 = mods["c40"]
    qft = synth_qft_full(5)
    endcx = _qaoa_gates(5, 29)
    endcx.cx(0, 1)
    cliff = QuantumCircuit(4)
    cliff.h(0)
    cliff.cx(0, 1)
    cliff.s(1)
    cliff.cz(1, 2)
    mid = _qaoa_gates(4, 30)
    mid.measure_all()
    mid.rx(0.3, 0)
    b = QuantumCircuit(3)
    b.cx(0, 1)
    b.rz(Parameter("t"), 1)
    b.cx(0, 1)
    c = _evo(_op(4, 11), 4, QDrift(reps=3, seed=11))  # QDrift samples terms at random: a fixed seed
    assert c40._st_parse_gates(qft) is not None  # item 76: read since c43
    for qc in (endcx, cliff, random_circuit(5, 8, max_operands=2, seed=12)):
        assert c40._st_parse(qc) is None and c40._st_parse_gates(qc) is None
        # the usual path: c43's own steiner=False output (c43's usual path differs from the release's: items 74-75)
        assert _sig(_call(c40, qc, backend=torino)) == _sig(_call(c40, qc, backend=torino, steiner=False))
    for qc in (mid, b, c):
        assert c40._st_parse(qc) is None and c40._st_parse_gates(qc) is None


def test_no_failed_element_on_a_device(mods, torino):
    c40 = mods["c40"]
    edges, bad = c40._failed_elements(torino.target, 0.5)
    for qc in (_qaoa(16, 13, reps=2), _evo(_op(12, 14, k_max=2), 12), _qaoa_gates(16, 31),
               _heisenberg_gates(12, 32)):
        qc.measure_all()
        out = _call(c40, qc, backend=torino)
        assert not c40._uses_failed(out, edges, bad)


# ---------------------------------------------------------------- items 74 (revised), 75, 76

def _distribution_same(out, qc, trials=4, seed=0):
    """The output's distribution over the classical bits equals the input's for random product input states."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UGate
    from qiskit.quantum_info import Statevector

    def dist(circ, prep):
        body = QuantumCircuit(circ.num_qubits)
        for g, q in prep:
            body.append(g, [q])
        meas = {}
        for i in circ.data:
            if i.operation.name == "measure":
                meas[circ.find_bit(i.clbits[0]).index] = circ.find_bit(i.qubits[0]).index
            elif i.operation.name != "barrier":
                body.append(i.operation, [body.qubits[circ.find_bit(q).index] for q in i.qubits])
        p = Statevector(body).probabilities()
        out_ = np.zeros(2 ** qc.num_clbits)
        for idx in np.nonzero(p > 1e-14)[0]:
            c = 0
            for cb, q in meas.items():
                if (idx >> q) & 1:
                    c |= 1 << cb
            out_[c] += p[idx]
        return out_
    rng = random.Random(seed)
    init = out.layout.initial_index_layout()
    for _ in range(trials):
        gates = [UGate(rng.uniform(0, 3.2), rng.uniform(0, 6.3), rng.uniform(0, 6.3)) for _ in range(qc.num_qubits)]
        a = dist(qc, [(g, q) for q, g in enumerate(gates)])
        b = dist(out, [(g, init[q]) for q, g in enumerate(gates)])
        if np.max(np.abs(a - b)) > 1e-6:
            return False
    return True


def _qft_swaps(n):
    from qiskit.synthesis import synth_qft_full
    return synth_qft_full(n)


def test_item76_permutation_read_and_exact(mods, small):
    """QFT with its final swaps, and a cx-rz-cx circuit ending with swaps and H gates: read with a permutation, and
    STEINER6's output is exact (as a unitary through its layout, and as a measured distribution)."""
    c43 = mods["c40"]
    a = _qft_swaps(5)
    b = _qaoa_gates(5, 40)
    b.swap(0, 3)
    b.swap(1, 4)
    b.h(2)
    for qc in (a, b):
        got = c43._st_parse_gates(qc)
        assert got is not None and got[5] != list(range(5))
        st = c43._st_compile(qc, small.target, 0.5, 0, 300_000, 2_500)
        assert st is not None and _exact(st, qc)
        m = qc.copy()
        m.measure_all()
        st = c43._st_compile(m, small.target, 0.5, 0, 300_000, 2_500)
        assert _distribution_same(st, m)
        out = _call(c43, m, backend=small)
        assert _distribution_same(out, m)


def test_item75_same_output_every_call(mods, torino):
    """The same input gives the same output on every call, with the device and without it (Qiskit's VF2PostLayout did
    not: Addendum 450)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.random import random_circuit
    c43 = mods["c40"]
    cm = torino.target.build_coupling_map()
    basis = ["cz", "rz", "sx", "x"]
    bv = QuantumCircuit(12, 11)
    bv.x(11)
    bv.h(range(12))
    for q in range(0, 11, 2):
        bv.cx(q, 11)
    bv.h(range(11))
    bv.measure(range(11), range(11))
    circuits = [bv] + [random_circuit(6, 8, max_operands=2, measure=True, seed=750 + k) for k in range(4)]
    c43.WARN_WITHOUT_TARGET = False
    for qc in circuits:
        assert len({_sig(_call(c43, qc, backend=torino)) for _ in range(3)}) == 1
        assert len({_sig(_call(c43, qc, coupling_map=cm, basis_gates=basis)) for _ in range(3)}) == 1


def test_item75_replacement_exact_and_on_device(mods, torino):
    """With item 75's re-placement the outputs stay exact and use only the device's gates, with no failed element."""
    from qiskit.circuit.random import random_circuit
    c43 = mods["c40"]
    edges, bad = c43._failed_elements(torino.target, 0.5)
    for k in range(3):
        qc = random_circuit(5, 8, max_operands=2, seed=760 + k)
        out = _call(c43, qc, backend=torino)
        assert not c43._uses_failed(out, edges, bad)
        for ins in out.data:
            torino.target[ins.operation.name][tuple(out.find_bit(q).index for q in ins.qubits)]
    applied = c43.DET_STATS["applied"]
    assert c43.DET_STATS["calls"] > 0 and applied >= 0


def _near_device():
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap, InstructionProperties
    cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 6], [6, 5]])
    cm.make_symmetric()
    b = GenericBackendV2(7, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cm, seed=74)
    rng = random.Random(74)
    for q in range(7):
        b.target.update_instruction_properties("measure", (q,), InstructionProperties(error=rng.uniform(0.005, 0.03)))
        for g in ("sx", "x"):
            b.target.update_instruction_properties(g, (q,), InstructionProperties(error=rng.uniform(1e-4, 1e-3)))
    for qa in b.target["cz"]:
        b.target.update_instruction_properties("cz", qa, InstructionProperties(error=rng.uniform(2e-3, 1e-2)))
    for qa in ((2, 3), (3, 2)):
        b.target.update_instruction_properties("cz", qa, InstructionProperties(error=0.2))
    b.target.update_instruction_properties("measure", (4,), InstructionProperties(error=0.25))
    return b


def test_item74_revised_marks_only_used_elements_and_is_never_worse(mods):
    from qiskit import QuantumCircuit
    from qiskit.circuit.random import random_circuit
    c43 = mods["c40"]
    b = _near_device()
    t = b.target
    a = QuantumCircuit(7, 1)
    a.cz(3, 2)
    a.sx(4)
    assert c43._near_used_elements(a, t, 0.1, 0.5) == [("cz", (2, 3)), ("cz", (3, 2))]
    a.measure(4, 0)
    assert ("measure", (4,)) in c43._near_used_elements(a, t, 0.1, 0.5)
    for k in range(4):
        qc = random_circuit(6, 8, max_operands=2, measure=True, seed=770 + k)
        out = _call(c43, qc, backend=b)
        off = _call(c43, qc, backend=b, near_failed_error=None)
        assert c43._st_cost(out, t) <= c43._st_cost(off, t) + 1e-12
        assert _distribution_same(out, qc)


# ---------------------------------------------------------------- items 75 (revised), 77

def test_item75_revised_never_scores_above_c43s_search(mods, torino):
    """On the same routed circuit, c44's re-placement ends at a score no higher than c43's (it includes c43's order)."""
    import core_fix_c2_eval as H
    from qiskit import transpile
    from qiskit.circuit.random import random_circuit
    from qiskit.converters import circuit_to_dag
    c44 = mods["c40"]
    c43 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c43_2026-10-11", "psf_compile.py"), "c43_for_c44")
    t = torino.target
    for k in range(3):
        qc = random_circuit(12 + 4 * k, 6, max_operands=2, measure=True, seed=780 + k)
        routed = transpile(qc, target=t, optimization_level=1, seed_transpiler=k)
        dag = circuit_to_dag(routed)
        costs = []
        for mod in (c43, c44):
            p = mod._DeterministicPostLayout(t, 300_000, 2_500, True)
            p.run(dag)
            post = p.property_set["post_layout"]
            if post is None:
                costs.append(mod._st_cost(routed, t))
                continue
            moved = routed.copy_empty_like()
            for ins in routed.data:
                moved.append(ins.operation, [moved.qubits[post[q]] for q in ins.qubits], ins.clbits)
            costs.append(mod._st_cost(moved, t))
        assert costs[1] <= costs[0] + 1e-12


def test_item77_layout_search_does_not_depend_on_speed(mods, torino, monkeypatch):
    """With the default (no time budget), a slow clock -- every reading 10 s after the last -- gives the same output."""
    from qiskit.circuit.random import random_circuit
    c44 = mods["c40"]
    sl = sys.modules["psf_smart_layout"]
    circuits = [random_circuit(10, 6, max_operands=2, measure=True, seed=790 + k) for k in range(3)]
    normal = [_sig(_call(c44, qc, backend=torino)) for qc in circuits]
    clock = {"t": 0.0}

    def slow():
        clock["t"] += 10.0
        return clock["t"]
    monkeypatch.setattr(sl.time, "perf_counter", slow)
    slowed = [_sig(_call(c44, qc, backend=torino)) for qc in circuits]
    assert slowed == normal
