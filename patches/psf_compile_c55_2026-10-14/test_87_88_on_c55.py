"""Tests for the workplace proposal of items 87-88, merged by home onto prototype c52 as prototype 2026-10-14.c54
(Addendum 483). The workplace's file, with c51 replaced by c52 as the inner compiler compared against and the version
string updated; nothing else changed.

Item 87: a chain Pauli evolution built by formula as a brickwork (N1, renumbered). Item 88: with rzz and rx in the
Target, its terms emitted as rzz / rx with the signs carried in a Pauli X frame undone inside the circuit.
Targets: FakeFez's own (no rzz) and FakeFez's with rzz and rx added as in test_c51.py (a CZ's and an sx's durations).

Run from the repository root with this file next to the built psf_compile.py:
    python -m pytest <dir>/test_87_88.py -q
"""
import contextlib
import copy
import io
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("PSF_REPO", os.getcwd())
sys.path[:0] = [HERE, os.path.join(REPO, "benchmarks"), REPO]
N1_FILE = os.path.join(REPO, "data", "2026-10-14", "workplace", "workplace_2026-10-14b_N1-N2", "n1", "patch",
                       "psf_compile.py")
C51_FILE = os.path.join(REPO, "patches", "psf_compile_c52_2026-10-14", "psf_compile.py")  # c53: the inner is c52


def _load(path, name):
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = H.load_module(path, name)
    m.WARN_WITHOUT_TARGET = False
    m.ZZ_RELAYER = False  # c55: item 94 off (it compiles a rewrite too and changes these outputs; tested in test_94.py)
    return m


@pytest.fixture(scope="module")
def new():
    return _load(os.path.join(HERE, "psf_compile.py"), "psf_compile_c51_87_88_test")


@pytest.fixture(scope="module")
def c51():
    return _load(C51_FILE, "psf_compile_c51_for_87_test")


@pytest.fixture(scope="module")
def n1():
    return _load(N1_FILE, "psf_compile_c49n1_for_87_test")


def fractional(target):
    """test_c51's: a copy of `target` with rzz on every cz coupler and rx on every qubit (a CZ's and an sx's
    duration and error)."""
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
    b = FakeFez()
    return b.target, fractional(b.target)


def _lab(n, d):
    s = ["I"] * n
    for q, p in d.items():
        s[n - 1 - q] = p
    return "".join(s)


def _evo(n, terms, t, steps, measure, order=2):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter, SuzukiTrotter
    qc = QuantumCircuit(n)
    syn = SuzukiTrotter(order=2, reps=steps) if order == 2 else LieTrotter(reps=steps)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=t, synthesis=syn), range(n))
    if measure:
        qc.measure_all()
    return qc


def tfim(n, steps, measure=True):
    terms = [(_lab(n, {q: "Z", q + 1: "Z"}), 1.0) for q in range(n - 1)] + [(_lab(n, {q: "X"}), 1.0) for q in range(n)]
    return _evo(n, terms, 1.0, steps, measure)


def tfim_signs(n, steps, measure=True, seed=2):
    """random signs and sizes; some ZZ angles per step above pi/2 (folded by Z x Z), many negative (frame)"""
    rng = np.random.default_rng(seed)
    terms = [(_lab(n, {q: "Z", q + 1: "Z"}), float(rng.choice([-1, 1]) * rng.uniform(0.3, 2.5))) for q in range(n - 1)]
    terms += [(_lab(n, {q: "X"}), float(rng.choice([-1, 1]) * rng.uniform(0.2, 1.5))) for q in range(n)]
    return _evo(n, terms, 1.2, steps, measure)


def heis(n, steps, measure=True, seed=1):
    """XYZ chain with fields; terms listed even edges, then odd edges, then fields (a brickwork product formula)"""
    rng = np.random.default_rng(seed)
    terms = []
    for s in (0, 1):
        for q in range(s, n - 1, 2):
            for p in "XYZ":
                terms.append((_lab(n, {q: p, q + 1: p}), float(rng.choice([-1, 1]) * rng.uniform(0.5, 1.5))))
    terms += [(_lab(n, {q: "Z"}), float(rng.uniform(-1, 1))) for q in range(n)]
    return _evo(n, terms, 0.8, steps, measure, order=1)


def mixed(n, steps, measure=True, seed=3):
    """XY, YZ, ZX two-qubit terms on a chain (other two-qubit Paulis: basis change, one rzz)"""
    rng = np.random.default_rng(seed)
    pairs = ["XY", "YZ", "ZX", "XZ"]
    terms = [(_lab(n, {q: pairs[q % 4][0], q + 1: pairs[q % 4][1]}), float(rng.uniform(-2, 2))) for q in range(n - 1)]
    terms += [(_lab(n, {q: "X"}), float(rng.uniform(-1, 1))) for q in range(n)]
    return _evo(n, terms, 1.0, steps, measure)


def ring(n, steps):
    terms = [(_lab(n, {q: "Z", (q + 1) % n: "Z"}), 1.0) for q in range(n)] + [(_lab(n, {q: "X"}), 1.0) for q in range(n)]
    return _evo(n, terms, 1.0, steps, True)


def _call(mod, qc, target, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=target, seed_transpiler=0, **kw)


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _probs(circ):
    """exact outcome distribution over the circuit's clbits, simulated on the active qubits only"""
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


def _logical(qc):
    from qiskit import transpile
    return transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h", "rzz", "rx", "ry", "u", "measure"],
                     optimization_level=0)


def same_distribution(out, qc):
    return 0.5 * float(np.abs(_probs(out) - _probs(_logical(qc))).sum()) < 1e-9


def same_unitary_with_phase(out, qc):
    """the compiled circuit, restricted to the logical qubits' physical places (no other qubit touched, no routing),
    equals the logical circuit's unitary including the global phase"""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    n = qc.num_qubits
    lay = out.layout.initial_index_layout(filter_ancillas=True)
    fin = out.layout.final_index_layout(filter_ancillas=True)
    if lay != fin:
        return False
    pos = {p: i for i, p in enumerate(lay)}
    red = QuantumCircuit(n, global_phase=out.global_phase)
    for ins in out.data:
        qs = [out.find_bit(q).index for q in ins.qubits]
        if any(q not in pos for q in qs):
            return False
        red.append(ins.operation, [pos[q] for q in qs])
    return np.allclose(Operator(red).data, Operator(_logical(qc)).data, atol=1e-8)


def rzz_angles(c):
    return [float(i.operation.params[0]) for i in c.data if i.operation.name == "rzz"]


def in_range(c):
    return all(0 < a <= math.pi / 2 + 1e-12 for a in rzz_angles(c))


def test_version(new):
    assert new.VERSION == "2026-10-14.c55"


def test_tfim_uses_rzz_exact_in_range_isa_and_shorter_than_c51(new, c51, fez):
    _, ft = fez
    for n, s in ((8, 2), (12, 4)):
        qc = tfim(n, s)
        k = new.CHAIN_STATS["frame"]
        out = _call(new, qc, ft)
        assert new.CHAIN_STATS["frame"] == k + 1
        assert rzz_angles(out) and in_range(out)
        assert not any(i.operation.name in ("cz", "sx") for i in out.data)  # rzz and rx only (rz virtual, x)
        assert new._isa_ok(out, ft)
        assert same_distribution(out, qc)
        assert new._duration_s(out, ft) <= c51._duration_s(_call(c51, qc, ft), ft)


def test_unmeasured_circuits_are_the_same_unitary_with_phase(new, fez):
    _, ft = fez
    for qc in (tfim(6, 3, measure=False), tfim_signs(6, 3, measure=False), heis(5, 2, measure=False),
               mixed(5, 2, measure=False)):
        k = new.CHAIN_STATS["used"]
        out = new._chain_brickwork(qc, ft, 0.5, 0)
        assert out is not None and new.CHAIN_STATS["used"] == k + 1
        assert in_range(out)
        assert same_unitary_with_phase(out, qc)
        assert new._implements(_logical(qc), out)  # item 39's check (it reads gates, not PauliEvolutionGate)


def test_signs_and_large_angles_folded_and_framed_exactly(new, fez):
    _, ft = fez
    f0, x0 = new.CHAIN_STATS["frame_folded"], new.CHAIN_STATS["frame_x_end"]
    for measure in (True, False):
        qc = tfim_signs(8, 3, measure=measure)
        out = new._chain_brickwork(qc, ft, 0.5, 0)
        assert in_range(out)
        if measure:
            assert same_distribution(out, qc)
        else:
            assert same_unitary_with_phase(out, qc)
    assert new.CHAIN_STATS["frame_folded"] > f0  # angles above pi/2 occurred and were moved by Z x Z
    assert new.CHAIN_STATS["frame_x_end"] > x0  # the frame was undone at the end of unmeasured qubits


def test_frame_never_flips_measured_bits(new, fez):
    """every measurement writes its own clbit and nothing follows it on its qubit: no classical post-processing"""
    _, ft = fez
    out = new._chain_brickwork(tfim_signs(8, 3), ft, 0.5, 0)
    measured = set()
    for ins in out.data:
        qs = [out.find_bit(q).index for q in ins.qubits]
        if ins.operation.name == "measure":
            measured.add(qs[0])
        else:
            assert not measured & set(qs)
        assert ins.operation.name not in ("if_else", "while_loop", "switch_case")
    assert len(measured) == 8


def test_heisenberg_and_mixed_terms_exact(new, fez):
    _, ft = fez
    for qc in (heis(8, 2), mixed(8, 3)):
        out = new._chain_brickwork(qc, ft, 0.5, 0)
        assert out is not None and in_range(out)
        assert same_distribution(out, qc)


def test_frame_costs_no_second_rzz(new, fez):
    """item 88 emits one rzz per two-qubit term (no rzz(-theta) built from two)"""
    _, ft = fez
    qc = tfim_signs(10, 2, measure=False)
    out = new._chain_brickwork(qc, ft, 0.5, 0)
    # second-order Suzuki, 2 steps, merged half layers: 9 edges x 3 layers
    assert len(rzz_angles(out)) <= 9 * 3


def test_without_rzz_identical_to_n1(new, n1, fez):
    t, _ = fez
    for qc in (tfim(8, 2), tfim(12, 4), heis(6, 2)):
        assert _sig(_call(new, qc, t)) == _sig(_call(n1, qc, t))


def test_chain_frame_off_gives_cz_brickwork(new, fez):
    t, ft = fez
    new.CHAIN_FRAME = False
    try:
        out = new._chain_brickwork(tfim(8, 2), ft, 0.5, 0)
    finally:
        new.CHAIN_FRAME = True
    assert not rzz_angles(out)
    assert _sig(out) == _sig(new._chain_brickwork(tfim(8, 2), t, 0.5, 0))


def test_not_a_chain_is_c51s(new, c51, fez):
    _, ft = fez
    qc = ring(6, 2)
    assert _sig(_call(new, qc, ft)) == _sig(_call(c51, qc, ft))


def test_deterministic(new, fez):
    _, ft = fez
    qc = tfim_signs(10, 3)
    assert _sig(_call(new, qc, ft)) == _sig(_call(new, qc, ft))


def test_never_longer_than_c51(new, c51, fez):
    _, ft = fez
    for qc in (heis(8, 3), mixed(8, 2), tfim_signs(10, 4)):
        a, b = _call(new, qc, ft), _call(c51, qc, ft)
        assert new._duration_s(a, ft) <= new._duration_s(b, ft) + 1e-12
        assert in_range(a)


def test_n100_executable(new, fez):
    _, ft = fez
    out = _call(new, tfim(100, 10), ft)
    assert in_range(out) and new._isa_ok(out, ft)
    assert new._duration_s(out, ft) < 4e-6
