"""Tests for prototype psf_compile 2026-10-12.c48 (item 83 on c47; c47 = item 82 on c45; c45 = items 78-80 on c44; c44 = items 75 revised and
77 on c43 = release 2026-10-11.4). c47's tests are repeated on c48 (items 73-82), then item 83 is tested. Item 73 as in c41: circuits written as gates are read as Pauli
rotations by carrying their Clifford gates to the end (cx-rz-cx QAOA and Trotter circuits, basis changes by rx/ry,
phase gadgets from cp/crz/t, Toffolis through their definitions, any single-qubit gate as rz.ry.rz, final
single-qubit Cliffords such as an X-basis readout); the inputs of items 71-72 still work; item 69's choice by ESP.
The outputs equal the inputs up to a global phase.

c48's tests run unchanged on c49 with its two items switched off (C49_SELECT_BY=esp, C49_CUT=0, the defaults here):
c49 must then behave as c48. Set C49_SELECT_BY=duration C49_CUT=1 to see which of c48's expectations c49's defaults change.
Run from the repository root:  python -m pytest patches/psf_compile_c49_2026-10-14/test_c48_on_c49.py -q
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
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_for_c48_test")
    c40 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c48_test")
    c40.SELECT_BY, c40.CUT_FAILED_QUBITS = os.environ.get("C49_SELECT_BY", "esp"), os.environ.get("C49_CUT", "0") == "1"
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
    assert mods["c40"].VERSION == "2026-10-14.c50"  # c48's tests on c50, items 84-85 switched off as for c49


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
    c43 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c43_2026-10-11", "psf_compile.py"), "c43_for_c47")
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


# ---------------------------------------------------------------- items 78-80

def _routed_untranslated(t, seed):
    """A circuit as Qiskit's routing stage leaves it on the device's qubits: swaps, cx and h, not yet translated."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    edges = sorted({tuple(sorted(q)) for q in t["cz"] if q is not None})
    qc = QuantumCircuit(t.num_qubits, 6)
    start = rng.randrange(len(edges))
    chain = [edges[start]]
    for a, b in edges:  # a few edges next to the first, for a small connected interaction graph
        if len(chain) < 5 and (a, b) not in chain and ({a, b} & {x for e in chain for x in e}):
            chain.append((a, b))
    for _ in range(3):
        for a, b in chain:
            qc.h(a)
            qc.cx(a, b)
            qc.swap(a, b)
    used = sorted({x for e in chain for x in e})[:6]
    for i, q in enumerate(used):
        qc.measure(q, i)
    return qc


def test_item78_routing_stage_is_scored_not_skipped(mods):
    """At the routing stage (non-strict) of item 80's compile, a circuit with swap, cx and h on a cz device is re-placed:
    c44 skipped it, as c45's own pipeline still does. The strict call still refuses names the device does not run."""
    import core_fix_c2_eval as H
    from qiskit.converters import circuit_to_dag
    from qiskit_ibm_runtime.fake_provider import FakeBoston
    c45 = mods["c40"]
    c44 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c44_2026-10-11", "psf_compile.py"), "c44_for_c48")
    t = FakeBoston().target
    for k in range(3):
        dag = circuit_to_dag(_routed_untranslated(t, 7800 + k))
        p44 = c44._DeterministicPostLayout(t, 300_000, 2_500, False)
        before = c44.DET_STATS["skipped"]
        p44.run(dag)
        assert c44.DET_STATS["skipped"] == before + 1
        p45 = c45._DeterministicPostLayout(t, 300_000, 2_500, False)
        before = c45.DET_STATS["skipped"]
        p45.run(dag)
        assert c45.DET_STATS["skipped"] == before + 1  # PSF-Zero's own pipeline: as c44
        p45 = c45._DeterministicPostLayout(t, 300_000, 2_500, False)
        p45.proxy = True  # as _DeterministicVF2(target, routing_stage=True) sets it, for item 80
        before = c45.DET_STATS["skipped"]
        p45.run(dag)
        assert c45.DET_STATS["skipped"] == before
        assert p45.property_set["VF2PostLayout_stop_reason"] is not None
        strict = c45._DeterministicPostLayout(t, 300_000, 2_500, True)
        before = c45.DET_STATS["skipped"]
        strict.run(dag)
        assert c45.DET_STATS["skipped"] == before + 1


def test_item78_scores_by_translated_cost(mods):
    """A swap costs three native two-qubit gates on its coupler, cx one, rz nothing, h one sx."""
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit_ibm_runtime.fake_provider import FakeBoston
    c45 = mods["c40"]
    assert c45._NATIVE2_COUNT["swap"] == 3 and c45._NATIVE2_COUNT["cx"] == 1
    assert c45._SX_COUNT["rz"] == 0 and c45._SX_COUNT["h"] == 1
    t = FakeBoston().target
    # one swap on the worst coupler: re-placed onto a better one
    worst = max((q for q in t["cz"] if q is not None), key=lambda q: t["cz"][q].error)
    qc = QuantumCircuit(t.num_qubits)
    qc.swap(*worst)
    p = c45._DeterministicPostLayout(t, 300_000, 2_500, False)
    p.proxy = True
    p.run(circuit_to_dag(qc))
    post = p.property_set["post_layout"]
    assert post is not None
    a, b = post[qc.qubits[worst[0]]], post[qc.qubits[worst[1]]]
    e = t["cz"][(a, b)] if (a, b) in t["cz"] else t["cz"][(b, a)]
    assert e.error < t["cz"][worst].error


def test_item79_target_alone_gives_device_gates(mods, torino, small):
    from qiskit.circuit.random import random_circuit
    c45 = mods["c40"]
    for b in (torino, small):
        t = b.target
        edges, bad = c45._failed_elements(t, 0.5)
        for k in range(3):
            qc = random_circuit(6, 8, max_operands=2, measure=True, seed=7900 + k)
            out = _call(c45, qc, target=t)
            for ins in out.data:
                t[ins.operation.name][tuple(out.find_bit(q).index for q in ins.qubits)]
            assert not c45._uses_failed(out, edges, bad)
            if b is small:
                assert _distribution_same(out, qc)


def test_item80_never_above_either_candidate_and_repeats(mods, torino):
    """With the device, the output's -ln ESP is at most PSF-Zero's own (level3_candidate=False) and at most Qiskit
    level 3's under item 75's re-placement; three calls give one output; no failed element; exact."""
    from qiskit.circuit.random import random_circuit
    from qiskit.transpiler import generate_preset_pass_manager
    c45 = mods["c40"]
    t = torino.target
    edges, bad = c45._failed_elements(t, 0.5)
    for k in range(4):
        qc = random_circuit(6 + 2 * k, 8, max_operands=2, measure=True, seed=8000 + k)
        outs = [_call(c45, qc, backend=torino) for _ in range(3)]
        assert len({_sig(o) for o in outs}) == 1
        out = outs[0]
        own = _call(c45, qc, backend=torino, level3_candidate=False)
        with c45._DeterministicVF2(t):
            pm = generate_preset_pass_manager(optimization_level=3, target=t, seed_transpiler=0)
        l3 = pm.run(qc)
        assert c45._st_cost(out, t) <= c45._st_cost(own, t) + 1e-12
        if not c45._uses_failed(l3, edges, bad):
            assert c45._st_cost(out, t) <= c45._st_cost(l3, t) + 1e-12
        assert not c45._uses_failed(out, edges, bad)


def test_item80_output_exact_on_a_small_device(mods, small):
    from qiskit.circuit.random import random_circuit
    c45 = mods["c40"]
    for k in range(4):
        qc = random_circuit(5 + k % 2, 10, max_operands=2, measure=True, seed=8050 + k)
        assert _distribution_same(_call(c45, qc, backend=small), qc)


def test_item80_auto_off_with_the_callers_map(mods, torino):
    from qiskit.circuit.random import random_circuit
    c45 = mods["c40"]
    cm = torino.target.build_coupling_map()
    qc = random_circuit(5, 6, max_operands=2, measure=True, seed=8100)
    before = c45.L3C_STATS["tried"]
    _call(c45, qc, backend=torino, coupling_map=cm)
    _call(c45, qc, backend=torino, level3_candidate=False)
    assert c45.L3C_STATS["tried"] == before
    _call(c45, qc, backend=torino)
    assert c45.L3C_STATS["tried"] == before + 1


def test_without_item80_the_output_is_c44s(mods, torino):
    """Items 78-79 do not change PSF-Zero's own pipeline with backend=: level3_candidate=False gives c44's output
    (on inputs item 82 does not apply to)."""
    import core_fix_c2_eval as H
    from qiskit.circuit.random import random_circuit
    c45 = mods["c40"]
    c44 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c44_2026-10-11", "psf_compile.py"), "c44_for_c48b")
    c44.WARN_WITHOUT_TARGET = False
    for k in range(4):
        qc = random_circuit(6 + 3 * k, 8, max_operands=2, measure=True, seed=8200 + k)
        assert _sig(_call(c45, qc, backend=torino, level3_candidate=False)) == _sig(_call(c44, qc, backend=torino))
    # (QAOA is left out here since c47: item 82 changes it)


def test_item80_routing_stage_scored_in_its_compile(mods):
    """Item 80's level-3 compile re-places at the routing stage (FakeBoston, a 30-qubit random circuit): the
    deterministic pass is called non-strict with the proxy and is not skipped."""
    from qiskit.circuit.random import random_circuit
    from qiskit_ibm_runtime.fake_provider import FakeBoston
    c45 = mods["c40"]
    b = FakeBoston()
    qc = random_circuit(30, 6, max_operands=2, measure=True, seed=8300)
    before = dict(c45.DET_STATS)
    with c45._DeterministicVF2(b.target, routing_stage=True):
        from qiskit.transpiler import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=3, target=b.target, seed_transpiler=0)
    pm.run(qc)
    assert c45.DET_STATS["calls"] > before["calls"] and c45.DET_STATS["skipped"] == before["skipped"]


# ---------------------------------------------------------------- item 82

def _qaoa3(n, p, seed):
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    rng = random.Random(seed)
    while True:
        stubs = [v for v in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        edges, ok = set(), True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok:
            break
    qa = QAOAAnsatz(SparsePauliOp.from_list([(_label(n, {a: "Z", b: "Z"}), rng.uniform(-1, 1)) for a, b in sorted(edges)]),
                    reps=p)
    qc = qa.assign_parameters([rng.uniform(0, 3) for _ in range(qa.num_parameters)])
    qc.measure_all()
    return qc


def _tfim(n, reps, seed):
    from qiskit.synthesis import SuzukiTrotter
    from qiskit.quantum_info import SparsePauliOp
    rng = random.Random(seed)
    terms = [(_label(n, {q: "Z", (q + 1) % n: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
    terms += [(_label(n, {q: "X"}), rng.uniform(-1, 1)) for q in range(n)]
    return _evo(SparsePauliOp.from_list(terms), n, SuzukiTrotter(order=2, reps=reps), measure=True)


def _trot(qc):
    """The reference for a Trotterised evolution: the circuit's own synthesis (a statevector of a PauliEvolutionGate
    is the exact exponential, not what the circuit's product formula does)."""
    from qiskit import transpile
    return transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h", "rzz", "rx", "ry", "u", "measure"],
                     optimization_level=0)


def test_item82_blocks_detected_only_where_exact(mods):
    c47 = mods["c40"]
    for qc in (_qaoa3(8, 3, 8200), _tfim(6, 3, 8201)):
        ev = c47._st_parse(qc)
        ev = ev + ([], None) if ev is not None else c47._st_parse_gates(qc)
        blocks = c47._st_mirror_blocks(c47._st_merge(ev[2]))
        assert blocks is not None and sum(1 for k, _ in blocks if k == "C") >= 3  # Suzuki: 4 ZZ layers after merging
    heis = _evo(_op(5, 8202, kinds="XYZ", k_max=2), 5, None)  # non-commuting multi-qubit terms: not layers
    ev = c47._st_parse(heis)
    assert c47._st_mirror_blocks(c47._st_merge(ev[2])) is None


def test_item82_exact_and_never_worse(mods, small, monkeypatch):
    c47 = mods["c40"]
    t = small.target
    for qc in (_qaoa3(6, 3, 8210), _qaoa3(6, 4, 8211), _tfim(6, 3, 8212)):
        before = c47.STEINER_STATS["mirror_tried"]
        out = c47._st_compile(qc, t, 0.5, 0, 300_000, 2_500)
        assert c47.STEINER_STATS["mirror_tried"] == before + 1
        assert out is not None and _distribution_same(out, _trot(qc))
        monkeypatch.setattr(c47, "_st_mirror_blocks", lambda triples: None)
        own = c47._st_compile(qc, t, 0.5, 0, 300_000, 2_500)
        monkeypatch.undo()
        assert c47._st_cost(out, t) <= c47._st_cost(own, t) + 1e-12


def test_item82_mirror_alone_is_exact(mods, small):
    """The mirrored construction itself (not only the one kept) gives the input's distribution."""
    c47 = mods["c40"]
    t = small.target
    for qc in (_qaoa3(6, 3, 8220), _tfim(5, 4, 8221)):
        ev = c47._st_parse(qc)
        ev = ev + ([], None) if ev is not None else c47._st_parse_gates(qc)
        triples = c47._st_merge(ev[2])
        blocks = c47._st_mirror_blocks(triples)
        orig = c47._st_construct
        mir = c47._st_mirror_construct(qc.num_qubits, triples, t, blocks)
        try:
            c47._st_construct = lambda n, tr, tg, me: mir
            c47._st_mirror_blocks_saved = c47._st_mirror_blocks
            c47._st_mirror_blocks = lambda tr: None
            out = c47._st_compile(qc, t, 0.5, 0, 300_000, 2_500)
        finally:
            c47._st_construct = orig
            c47._st_mirror_blocks = c47._st_mirror_blocks_saved
        assert out is not None and _distribution_same(out, _trot(qc))


def test_item82_same_output_every_call(mods, torino):
    c47 = mods["c40"]
    for qc in (_qaoa3(12, 3, 8230), _tfim(10, 3, 8231)):
        assert len({_sig(_call(c47, qc, backend=torino)) for _ in range(3)}) == 1


# ---------------------------------------------------------------- item 83

def _mover_for(mod, qc, target):
    ev = mod._st_parse(qc)
    ev = ev + ([], None) if ev is not None else mod._st_parse_gates(qc)
    triples = mod._st_merge(ev[2])
    m, _ = mod._st_construct(qc.num_qubits, triples, target, ev[1])
    return m


def _ops_unitary(ops, qubits):
    """The unitary of a mover's op list on the given physical qubits (in that order)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    pos = {p: i for i, p in enumerate(qubits)}
    qc = QuantumCircuit(len(qubits))
    for name, q, prm in ops:
        args = [pos[x] for x in q]
        if name == "rz":
            qc.rz(prm[0], args[0])
        else:
            getattr(qc, name)(*args)
    return Operator(qc)


def test_item83_blocks_cover_every_op_once(mods, small):
    c48 = mods["c40"]
    t = small.target
    for name, qc in _cases() + _gate_cases():
        m = _mover_for(c48, qc, t)
        assert sorted(i for b in m.st_blocks for i in b[1]) == list(range(len(m.ops))), name


def test_item83_same_gates_same_unitary(mods, small):
    """The reordered ops are the same ops (as a multiset) and the same unitary, on inputs whose terms do not commute
    as well as on commuting ones (SWAPs included: the small device needs them)."""
    c48 = mods["c40"]
    t = small.target
    checked = 0
    for name, qc in _cases() + _gate_cases() + [("tfim", _tfim(5, 2, 8300)), ("qaoa3", _qaoa3(6, 2, 8301))]:
        m = _mover_for(c48, qc, t)
        new = c48._st_reorder(m.ops, m.st_blocks)
        assert new is not None, name
        key = lambda o: (o[0], tuple(o[1]), tuple(float(x) for x in o[2]))
        assert sorted(map(key, new)) == sorted(map(key, m.ops)), name
        qs = sorted({x for _, q, _ in m.ops for x in q})
        assert _ops_unitary(new, qs).equiv(_ops_unitary(m.ops, qs)), name
        checked += new != m.ops
    assert checked >= 3  # the order did change on several inputs


def test_item83_random_paulis_with_swaps(mods, small):
    """Random weight-1..3 Pauli terms (X, Y, Z; most pairs anticommute somewhere) on the small device: the reorder
    keeps the unitary."""
    c48 = mods["c40"]
    t = small.target
    for seed in range(8310, 8316):
        rng = random.Random(seed)
        n = 6
        triples = []
        for _ in range(24):
            k = rng.randint(1, 3)
            qs = rng.sample(range(n), k)
            triples.append((_label(n, {q: rng.choice("XYZ") for q in qs}), rng.uniform(-1, 1), 0.7))
        m, _ = c48._st_construct(n, c48._st_merge(triples), t, [])
        new = c48._st_reorder(m.ops, m.st_blocks)
        qs = sorted({x for _, q, _ in m.ops for x in q})
        assert new is not None and _ops_unitary(new, qs).equiv(_ops_unitary(m.ops, qs)), seed


def test_item83_shortens_the_ising_staircase(mods, torino):
    """A transverse-field Ising Trotter circuit: the mover's order is a staircase; the reorder is much shorter, and the
    compiled output is exact and shorter on the device's durations."""
    c48 = mods["c40"]
    qc = _tfim(10, 2, 8320)
    m = _mover_for(c48, qc, torino.target)
    new = c48._st_reorder(m.ops, m.st_blocks)
    assert c48._st_asap(new) < 0.75 * c48._st_asap(m.ops)  # 46.2 -> 28.2 CX units (a ring: one bridged term)

    def duration(c):
        t = torino.target
        pos = {q: i for i, q in enumerate(c.qubits)}
        clock = {}
        for ins in c.data:
            if ins.operation.name in ("barrier", "measure"):
                continue
            qs = tuple(pos[q] for q in ins.qubits)
            pr = t[ins.operation.name].get(qs)
            s = max(clock.get(q, 0.0) for q in qs)
            for q in qs:
                clock[q] = s + ((pr.duration or 0.0) if pr is not None else 0.0)
        return max(clock.values())

    c48.ST_SCHEDULE = False
    try:
        off = _call(c48, qc, backend=torino, level3_candidate=False)
    finally:
        c48.ST_SCHEDULE = True
    on = _call(c48, qc, backend=torino, level3_candidate=False)
    assert duration(on) < 0.8 * duration(off)


def test_item83_compiled_output_exact(mods, small):
    c48 = mods["c40"]
    for qc in (_tfim(5, 2, 8321), _qaoa3(6, 2, 8322), _evo(_op(5, 8323), 5, measure=True)):
        out = c48._st_compile(qc, small.target, 0.5, 0, 300_000, 2_500)
        assert out is not None and _distribution_same(out, _trot(qc))


def test_item83_off_gives_c47s_output(mods, torino):
    import core_fix_c2_eval as H
    c48 = mods["c40"]
    c47 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c47_2026-10-12", "psf_compile.py"), "c47_for_c48")
    c47.WARN_WITHOUT_TARGET = False
    c48.ST_SCHEDULE = False
    try:
        for qc in (_qaoa3(10, 2, 8330), _tfim(8, 2, 8331), _qaoa_gates(6, 8332)):
            assert _sig(_call(c48, qc, backend=torino)) == _sig(_call(c47, qc, backend=torino))
    finally:
        c48.ST_SCHEDULE = True


def test_item83_kept_only_when_shorter(mods, small):
    """The order used by _st_compile is the reordered one only when its schedule is shorter (counted)."""
    c48 = mods["c40"]
    t = small.target
    from qiskit.quantum_info import SparsePauliOp
    chain = SparsePauliOp.from_list([(_label(5, {q: "Z", q + 1: "Z"}), 0.3 + 0.1 * q) for q in range(4)])
    before = dict(c48.STEINER_STATS)
    out = c48._st_compile(_evo(chain, 5), t, 0.5, 0, 300_000, 2_500)  # a staircase in the given order
    assert out is not None and c48.STEINER_STATS["reordered"] == before["reordered"] + 1
    before = dict(c48.STEINER_STATS)
    one = SparsePauliOp.from_list([(_label(5, {0: "Z", 1: "Z"}), 0.4)])
    out = c48._st_compile(_evo(one, 5), t, 0.5, 0, 300_000, 2_500)  # one term: nothing to gain
    assert out is not None and c48.STEINER_STATS["reorder_kept_order"] == before["reorder_kept_order"] + 1
    assert c48.STEINER_STATS["reordered"] == before["reordered"]


def test_item83_mirror_layers_untouched(mods, small):
    """Item 82's mirror keeps its layer list (the recorded blocks have their own name) and emits gates."""
    c48 = mods["c40"]
    t = small.target
    qc = _qaoa3(6, 3, 8350)
    ev = c48._st_parse(qc)
    ev = ev + ([], None) if ev is not None else c48._st_parse_gates(qc)
    triples = c48._st_merge(ev[2])
    blocks = c48._st_mirror_blocks(triples)
    mir, _ = c48._st_mirror_construct(qc.num_qubits, triples, t, blocks)
    assert mir.blocks == blocks
    assert sum(1 for o in mir.ops if o[0] == "cx") > 0
    assert sorted(i for b in mir.st_blocks for i in b[1]) == list(range(len(mir.ops)))


def test_item83_same_output_every_call(mods, torino):
    c48 = mods["c40"]
    for qc in (_qaoa3(12, 3, 8360), _tfim(10, 3, 8361)):
        assert len({_sig(_call(c48, qc, backend=torino)) for _ in range(3)}) == 1
