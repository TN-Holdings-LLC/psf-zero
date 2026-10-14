"""Tests for prototype psf_compile 2026-10-14.c57 (c56 + item 96 revised: handed-on rzz folded into IBM's range): item 96, the scope gate (a circuit whose two-qubit interaction graph
does not embed in the device's working couplers is handed to Qiskit's level 3 unchanged), and item 94 off by default.
c55's suites run on c57 as *_on_c57.py with the gate off.

Run from the repository root:  python -m pytest patches/psf_compile_c57_2026-10-14/test_c56.py -q
"""
import contextlib
import copy
import io
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "benchmarks"), REPO]


@pytest.fixture(scope="module")
def m():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    mod = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c56_test")
    mod.WARN_WITHOUT_TARGET = False
    return mod


@pytest.fixture(scope="module")
def fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez().target


def _call(mod, qc, target, **kw):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.compile_for_hardware(qc, target=target, seed_transpiler=0, **kw)


def _qk3(qc, target, seed=0):
    from qiskit.transpiler import generate_preset_pass_manager
    return generate_preset_pass_manager(3, target=target, seed_transpiler=seed).run(qc)


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _regular3(n, seed):
    """A random 3-regular QAOA layer (does not embed in heavy-hex: degree 3 everywhere, short cycles)."""
    import networkx as nx
    from qiskit import QuantumCircuit
    g = nx.random_regular_graph(3, n, seed=seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for a, b in sorted(g.edges()):
        qc.rzz(0.7, a, b)
    for q in range(n):
        qc.rx(0.4, q)
    qc.measure_all()
    return qc


def _chain(n, steps=2):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(0.2 + 0.05 * q, q)
    for _ in range(steps):
        for q in range(n - 1):
            qc.rzz(0.3 + 0.01 * q, q, q + 1)
        for q in range(n):
            qc.rx(0.25, q)
    qc.measure_all()
    return qc


def test_version_and_defaults(m):
    assert m.VERSION == "2026-10-14.c57"
    assert m.SCOPE_GATE is True and m.ZZ_RELAYER is False


def test_scope_reading(m, fez):
    """chains embed; a 3-regular graph and a three-qubit operation do not"""
    from qiskit import QuantumCircuit
    e, r = m._scope96_edges(_chain(8))
    assert r is None and e == [(q, q + 1) for q in range(7)] and m._scope96_embeds(e, 8, fez, 0.5)
    e, r = m._scope96_edges(_regular3(10, 1))
    assert r is None and len(e) == 15 and not m._scope96_embeds(e, 10, fez, 0.5)
    three = QuantumCircuit(3)
    three.ccx(0, 1, 2)
    assert m._scope96_edges(three) == (None, "multi_qubit_operation")
    one = QuantumCircuit(3)
    one.h(range(3))
    assert m._scope96_edges(one) == ([], None)


def test_out_of_scope_is_qiskit_level3_unchanged(m, fez):
    for n, seed in ((10, 1), (14, 2)):
        qc = _regular3(n, seed)
        before = m.SCOPE96_STATS["handed"]
        out = _call(m, qc, fez)
        assert m.SCOPE96_STATS["handed"] == before + 1
        assert _sig(out) == _sig(_qk3(qc, fez))


def test_in_scope_is_c55s_path(m, fez):
    qc = _chain(12)
    out = _call(m, qc, fez)
    m.SCOPE_GATE = False
    try:
        ref = _call(m, qc, fez)
    finally:
        m.SCOPE_GATE = True
    assert _sig(out) == _sig(ref)


def test_switch_off_hands_nothing(m, fez):
    qc = _regular3(10, 3)
    m.SCOPE_GATE = False
    try:
        before = m.SCOPE96_STATS["calls"]
        out = _call(m, qc, fez)
        assert m.SCOPE96_STATS["calls"] == before
    finally:
        m.SCOPE_GATE = True
    assert out.num_qubits == fez.num_qubits  # c55's own output, compiled without the gate


def test_without_target_unchanged(m, fez):
    """no Target: item 96 does not apply"""
    qc = _regular3(10, 4)
    before = m.SCOPE96_STATS["calls"]
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.compile_for_hardware(qc, coupling_map=fez.build_coupling_map(), basis_gates=["cz", "rz", "sx", "x"],
                               seed_transpiler=0)
    assert m.SCOPE96_STATS["calls"] == before


def test_handed_output_avoids_failed_qubits(m, fez):
    """a qubit Qiskit's output uses, reported failed: the handed-on output avoids it and stays exact"""
    from qiskit.transpiler import InstructionProperties
    qc = _regular3(10, 5)
    plain = _qk3(qc, fez)
    used = sorted({plain.find_bit(q).index for i in plain.data if i.operation.name not in ("barrier", "measure")
                   for q in i.qubits})
    bad = copy.deepcopy(fez)
    q = used[len(used) // 2]
    bad.update_instruction_properties("sx", (q,), InstructionProperties(duration=bad["sx"][(q,)].duration, error=0.9))
    out = _call(m, qc, bad)
    fe, fq = m._failed_elements(bad, 0.5)
    assert q in fq and not m._uses_failed(out, fe, fq)
    assert m._exact93(qc, out) is True


def test_inexact_handoff_warns(m, fez, monkeypatch):
    """a handed-on output that is not exact is returned with a warning and counted"""
    qc = _regular3(10, 6)
    monkeypatch.setattr(m, "_hand_off96", lambda *a, **k: (_qk3(_regular3(10, 7), fez), "qiskit"))
    before = m.SCOPE96_STATS["handoff_inexact"]
    with contextlib.redirect_stdout(io.StringIO()), pytest.warns(RuntimeWarning, match="item 96"):
        m.compile_for_hardware(qc, target=fez, seed_transpiler=0)
    assert m.SCOPE96_STATS["handoff_inexact"] == before + 1


def test_deterministic(m, fez):
    for qc in (_regular3(12, 8), _chain(10)):
        assert _sig(_call(m, qc, fez)) == _sig(_call(m, qc, fez))


def test_default_seed_is_zero(m, fez):
    """no seed given: the hand-off uses seed 0, so the output is reproducible"""
    qc = _regular3(10, 9)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = m.compile_for_hardware(qc, target=fez)
    assert _sig(out) == _sig(_qk3(qc, fez, seed=0))


def _fractional(target):
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.transpiler import InstructionProperties
    t = copy.deepcopy(target)
    t.add_instruction(RZZGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                                 for qa, p in target["cz"].items()})
    t.add_instruction(RXGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                               for qa, p in target["sx"].items()})
    return t


def test_handed_on_rzz_folded_on_a_fractional_target(m, fez):
    """item 96 revised (c57): out of scope on a Target with rzz, the output is Qiskit level 3's folded into IBM's
    range by item 86's _fold, native and exact"""
    ft = _fractional(fez)
    for n, seed in ((10, 11), (14, 12)):
        qc = _regular3(n, seed)
        raw = _qk3(qc, ft)
        out = _call(m, qc, ft)
        assert m._rzz_in_range(out) and m._isa_ok(out, ft)
        assert _sig(out) == _sig(m._fold(raw, ft))
        assert m._exact93(qc, out) is True


def test_plain_target_hand_off_unchanged_by_the_revision(m, fez):
    """without rzz in the Target the revision changes nothing: Qiskit level 3's output as is"""
    qc = _regular3(12, 13)
    assert _sig(_call(m, qc, fez)) == _sig(_qk3(qc, fez))
