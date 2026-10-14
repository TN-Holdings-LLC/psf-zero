"""Tests for prototype psf_compile 2026-10-14.c49 (items 84 and 85 on c48).

c48's own 36 tests run on c49 in test_c48_on_c49.py with the two items switched off (c49 must then behave as c48).
Here c49 runs with its defaults:
  item 84: every choice between two compiled circuits is made by the as-soon-as-possible duration on the device's
           reported gate durations, then the two-qubit count; no error value decides;
  item 85: the device's failed qubits are cut out before compiling.
The three c48 tests whose expectations item 84 changes by design (outputs never worse by ESP; item 80 never above
either candidate by ESP; item 83 off equal to c47) are restated here by duration.

c49's tests run on c50: none of their Targets offers rzz, so item 86 never acts and c50 must behave as c49.
Run from the repository root:  python -m pytest patches/psf_compile_c55_2026-10-14/test_c49_on_c55.py -q
"""
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "benchmarks"), REPO]

from test_c48_on_c49_for_c55 import (_call, _cases, _exact, _gate_cases, _qaoa3, _qaoa_gates, _sig,  # noqa: E402
                             _tfim)


@pytest.fixture(scope="module")
def c49():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c54_as_c49_test")
    m.WARN_WITHOUT_TARGET = False
    m.ZZ_RELAYER = False  # c55: item 94 off (it compiles a rewrite too and changes these outputs; tested in test_94.py)
    assert m.SELECT_BY == "duration" and m.CUT_FAILED_QUBITS is True  # the defaults under test
    return m


@pytest.fixture(scope="module")
def c48():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = H.load_module(os.path.join(REPO, "patches", "psf_compile_c48_2026-10-12", "psf_compile.py"),
                      "psf_compile_c48_for_c49_test")
    m.WARN_WITHOUT_TARGET = False
    return m


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _small(failed=(), seed=72):
    """A 7-qubit device (a chain 0-5 with a branch 2-6), cz/rz/sx/x with durations; the qubits in `failed` are
    reported at sx error 1 (failed by item 58's rule)."""
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap, InstructionProperties
    cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 6]])
    cm.make_symmetric()
    b = GenericBackendV2(7, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cm, seed=seed)
    rng = random.Random(seed)
    t = b.target
    for q in range(7):
        t.update_instruction_properties("measure", (q,), InstructionProperties(
            duration=t["measure"][(q,)].duration, error=rng.uniform(0.005, 0.1)))
        for g in ("sx", "x"):
            e = 1.0 if q in failed else rng.uniform(1e-4, 1e-3)
            t.update_instruction_properties(g, (q,), InstructionProperties(duration=t[g][(q,)].duration, error=e))
    for qa in t["cz"]:
        t.update_instruction_properties("cz", qa, InstructionProperties(duration=t["cz"][qa].duration,
                                                                        error=rng.uniform(2e-3, 2e-2)))
    return b


def _used_qubits(c):
    pos = {q: i for i, q in enumerate(c.qubits)}
    return {pos[q] for ins in c.data if ins.operation.name not in ("barrier",) for q in ins.qubits}


# ----------------------------------------------------------------------------------------------- item 84

def test_version(c49):
    assert c49.VERSION == "2026-10-14.c55"  # c49's tests on c50: no Target here offers rzz


def test_duration_by_hand(c49):
    """_duration_s is the as-soon-as-possible makespan on the Target's durations (rz 0, barriers 0)."""
    from qiskit import QuantumCircuit
    b = _small()
    t = b.target
    qc = QuantumCircuit(7)
    qc.sx(0)
    qc.cz(0, 1)
    qc.rz(0.3, 1)
    qc.barrier()
    qc.sx(1)
    qc.x(3)
    d = lambda g, q: t[g][q].duration  # noqa: E731
    want = max(d("sx", (0,)) + d("cz", (0, 1)) + d("sx", (1,)), d("x", (3,)))
    assert abs(c49._duration_s(qc, t) - want) < 1e-15
    assert c49._duration_cx(qc) == pytest.approx(0.3 + 1.0 + 0.3)


def test_better_rules(c49):
    """Shorter wins; equal durations: fewer two-qubit gates; a full tie keeps the second (the caller's default);
    an operation the device does not offer loses whatever its duration."""
    from qiskit import QuantumCircuit
    t = _small().target
    short, long_ = QuantumCircuit(7), QuantumCircuit(7)
    short.cz(0, 1)
    long_.cz(0, 1)
    long_.cz(1, 2)
    assert c49._better(short, long_, t) and not c49._better(long_, short, t)
    a, b = QuantumCircuit(7), QuantumCircuit(7)  # same duration, b one more cz in parallel
    a.cz(0, 1)
    b.cz(0, 1)
    b.cz(3, 4)
    if abs(c49._duration_s(a, t) - c49._duration_s(b, t)) <= 1e-12:
        assert c49._better(a, b, t) and not c49._better(b, a, t)
    assert not c49._better(a, a.copy(), t)
    bad = QuantumCircuit(7)
    bad.cz(0, 5)  # no coupler 0-5: not offered
    assert not c49._better(bad, long_, t) and c49._better(long_, bad, t)


def test_no_error_value_decides(c49, torino):
    """Item 84: the choice between two compiled circuits does not change when every reported error of the device is
    rescaled (failed elements aside). (Errors still shape the candidates themselves: items 60, 75, STEINER6.)"""
    import copy
    from qiskit.transpiler import InstructionProperties
    t1 = torino.target
    t2 = copy.deepcopy(t1)
    rng = random.Random(5)
    for name in ("cz", "sx", "x", "measure"):
        for qa, p in t2[name].items():
            if qa is not None and p is not None and p.error is not None and p.error < 0.5:
                t2.update_instruction_properties(name, qa, InstructionProperties(
                    duration=p.duration, error=min(0.4, p.error * rng.uniform(0.2, 5.0))))
    outs = []
    for qc in (_qaoa_gates(6, 8370), _tfim(8, 2, 8371)):
        outs.append(_call(c49, qc, backend=torino))
        outs.append(_call(c49, qc, backend=torino, level3_candidate=False, steiner=False))
    for a in outs:
        for b in outs:
            if a.num_qubits == b.num_qubits:
                assert c49._better(a, b, t1) == c49._better(a, b, t2)


def test_chosen_output_is_exact_and_never_longer(c49):
    """c48's test_chosen_output_is_exact_and_never_worse, by duration: STEINER6's output is returned only when it is
    not longer than the usual path's."""
    b = _small()
    t = b.target
    for name, qc in _cases() + _gate_cases():
        out = _call(c49, qc, backend=b)
        assert _exact(out, qc), name
        usual = _call(c49, qc, backend=b, steiner=False)
        assert c49._duration_s(out, t) <= c49._duration_s(usual, t) + 1e-12, name


def test_item80_by_duration_never_longer_and_repeats(c49, torino):
    """c48's test_item80_never_above_either_candidate_and_repeats, by duration: the output is never longer than
    PSF-Zero's own (level3_candidate=False); three calls give one output; no failed element."""
    t = torino.target
    edges, bad = c49._failed_elements(t, 0.5)
    for qc in (_qaoa_gates(6, 8340), _tfim(8, 2, 8341), _qaoa3(10, 2, 8342)):
        outs = [_call(c49, qc, backend=torino) for _ in range(3)]
        assert len({_sig(o) for o in outs}) == 1
        own = _call(c49, qc, backend=torino, level3_candidate=False)
        assert c49._duration_s(outs[0], t) <= c49._duration_s(own, t) + 1e-12
        assert not c49._uses_failed(outs[0], edges, bad)


def test_esp_switch_restores_c48(c49, c48, torino):
    """SELECT_BY = "esp" and CUT_FAILED_QUBITS = False give c48's outputs."""
    c49.SELECT_BY, c49.CUT_FAILED_QUBITS = "esp", False
    c49.DROP_DIAGONAL_BEFORE_MEASURE = c49.MAP_FREE_DEFAULTS = False  # c52: items 89-90 off as well
    c49.PLACE_SCORE = "esp"
    try:
        for qc in (_qaoa3(10, 2, 8330), _tfim(8, 2, 8331), _qaoa_gates(6, 8332)):
            assert _sig(_call(c49, qc, backend=torino)) == _sig(_call(c48, qc, backend=torino))
    finally:
        c49.SELECT_BY, c49.CUT_FAILED_QUBITS = "duration", True
        c49.DROP_DIAGONAL_BEFORE_MEASURE = c49.MAP_FREE_DEFAULTS = True
        c49.PLACE_SCORE = "duration"


def test_ising_still_short(c49, torino):
    """Item 83's gain survives item 84: a TFIM chain stays at most half of c48 with item 83 off."""
    t = torino.target
    qc = _tfim(12, 4, 8350)
    on = _call(c49, qc, backend=torino)
    c49.ST_SCHEDULE = False
    c49.GRAPH_BRICKWORK = False  # c54: item 91 embeds this 12-ring and would replace the "off" arm (workplace Q1 note)
    try:
        off = _call(c49, qc, backend=torino)
    finally:
        c49.ST_SCHEDULE = True
        c49.GRAPH_BRICKWORK = True
    assert c49._duration_s(on, t) <= 0.5 * c49._duration_s(off, t)


# ----------------------------------------------------------------------------------------------- item 85

def _idle_qubit_circuit(n=6, measure=True):
    """n qubits; qubit n-1 has no two-qubit gate (only an x), the others a cz chain (as SCALE1's R circuits)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for q in range(n - 1):
        qc.h(q)
    for q in range(n - 2):
        qc.cz(q, q + 1)
    qc.x(n - 1)
    if measure:
        qc.measure_all()
    return qc


@pytest.mark.parametrize("failed", [(6,), (0,), (5,), (0, 6)])
def test_failed_qubits_never_used_and_exact(c49, failed):
    b = _small(failed=failed)
    qc = _idle_qubit_circuit(7 - len(failed), measure=False)
    out = _call(c49, qc, backend=b)
    assert not (_used_qubits(out) & set(failed))
    assert out.num_qubits == 7
    assert _exact(out, qc)
    assert c49.CUT_STATS["cut"] >= 1


def test_measured_circuit_reads_the_right_bits(c49):
    """With failed qubit 6 cut, a measured circuit's measurements map each input qubit to its own clbit: the ideal
    outcome of a basis-state preparation is read back (sampled exactly)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    b = _small(failed=(6,))
    qc = QuantumCircuit(6, 6)
    pattern = [1, 0, 1, 1, 0, 1]
    for q, v in enumerate(pattern):
        if v:
            qc.x(q)
    qc.cz(0, 1)
    qc.cz(2, 3)
    qc.measure(range(6), range(6))
    out = _call(c49, qc, backend=b)
    assert not (_used_qubits(out) & {6})
    sv_circ = out.remove_final_measurements(inplace=False)
    probs = Statevector(sv_circ).probabilities_dict()
    meas = {}
    pos = {q: i for i, q in enumerate(out.qubits)}
    cpos = {c: i for i, c in enumerate(out.clbits)}
    for ins in out.data:
        if ins.operation.name == "measure":
            meas[cpos[ins.clbits[0]]] = pos[ins.qubits[0]]
    (state, p), = [(k, v) for k, v in probs.items() if v > 1e-9]
    bits = [int(state[::-1][meas[c]]) for c in range(6)]
    assert p == pytest.approx(1.0) and bits == pattern


def test_no_failed_qubit_same_as_without_item85(c49, torino):
    """A device without failed qubits is compiled as with CUT_FAILED_QUBITS = False."""
    b = _small()
    for qc in (_idle_qubit_circuit(6), _qaoa_gates(5, 8360)):
        on = _call(c49, qc, backend=b)
        c49.CUT_FAILED_QUBITS = False
        try:
            off = _call(c49, qc, backend=b)
        finally:
            c49.CUT_FAILED_QUBITS = True
        assert _sig(on) == _sig(off)


def test_too_few_working_qubits(c49):
    b = _small(failed=(0, 6))
    qc = _idle_qubit_circuit(6)
    with pytest.raises(c49.FailedElementsError, match="item 85"):
        _call(c49, qc, backend=b)


def test_item85_same_output_every_call(c49):
    b = _small(failed=(6,))
    qc = _idle_qubit_circuit(6)
    assert len({_sig(_call(c49, qc, backend=b)) for _ in range(3)}) == 1


def test_working_target_copies_properties(c49):
    b = _small(failed=(3,))
    t = b.target
    sub, keep = c49._working_target(t, {3})
    assert keep == [0, 1, 2, 4, 5, 6] and sub.num_qubits == 6
    idx = {q: i for i, q in enumerate(keep)}
    for name in ("cz", "sx", "measure"):
        for qa, p in t[name].items():
            if all(q in idx for q in qa):
                sp = sub[name][tuple(idx[q] for q in qa)]
                assert sp.error == p.error and sp.duration == p.duration
            else:
                assert tuple(idx.get(q, -1) for q in qa) not in sub[name]


def test_no_placement_on_working_qubits_raises_failed_elements(c49):
    """Failed qubit 2 splits the device into {0, 1}, {3, 4, 5}, {6}: a four-qubit cz chain has no placement. c49
    raises FailedElementsError (as c48 does), not a routing error."""
    from qiskit import QuantumCircuit
    b = _small(failed=(2,))
    qc = QuantumCircuit(6)
    for q in range(6):
        qc.h(q)
    for a in range(3):
        qc.cz(a, a + 1)
    with pytest.raises(c49.FailedElementsError, match="item 85"):
        _call(c49, qc, backend=b)
