"""Tests for prototype psf_compile 2026-10-14.c54 (item 86 as revised in c51: IBM Heron's native fractional gates rzz and rx).

c49's 18 tests and c48's 36 run on c50 in test_c49_on_c55.py and test_c48_on_c49_for_c55.py (none of their Targets
offers rzz, so item 86 never acts there). Here the Targets are FakeFez's and a small device's with rzz and rx added,
with a CZ's and an sx's durations (as reported for ibm_marrakesh on 2026-10-13: rzz 68 ns, rx 36 ns).

Run from the repository root:  python -m pytest patches/psf_compile_c55_2026-10-14/test_c50.py -q
"""
import contextlib
import copy
import io
import math
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "benchmarks"), REPO]

from test_c48_on_c49_for_c55 import _qaoa_gates, _sig, _tfim  # noqa: E402
from test_c49_on_c55 import _small  # noqa: E402


@pytest.fixture(scope="module")
def c50():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c54_as_c51_test")
    m.WARN_WITHOUT_TARGET = False
    m.DROP_DIAGONAL_BEFORE_MEASURE = m.MAP_FREE_DEFAULTS = False  # c51's tests on c52 with items 89-90 off
    m.PLACE_SCORE = "esp"
    m.CHAIN_BRICKWORK = m.GRAPH_BRICKWORK = m.EXACT_CHECK = m.ZZ_RELAYER = False  # c55: items 87-88, 91, 93 and 94 off as well (c51's semantics)
    return m


@pytest.fixture(scope="module")
def c49():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = H.load_module(os.path.join(REPO, "patches", "psf_compile_c49_2026-10-14", "psf_compile.py"),
                      "psf_compile_c49_for_c50_test")
    m.WARN_WITHOUT_TARGET = False
    return m


def fractional(target):
    """A copy of `target` with rzz on every cz coupler (a CZ's duration and error) and rx on every qubit (an sx's)."""
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


def _same_distribution(out, qc):
    """The compiled circuit's exact outcome distribution equals the logical circuit's (its own product formula), within
    1e-6, computed on the active qubits only (hw1_real.measured_distribution)."""
    import core_fix_c2_eval as H
    hw1 = sys.modules.get("hw1_for_c50") or H.load_module(os.path.join(REPO, "benchmarks", "hw1_real.py"), "hw1_for_c50")
    if not any(i.operation.name == "measure" for i in qc.data):
        qc = qc.copy()
        qc.measure_all()
        out = out.copy()
        out.measure_all() if not any(i.operation.name == "measure" for i in out.data) else None
    d0, _ = hw1.measured_distribution(hw1._unrolled(qc))
    d1, _ = hw1.measured_distribution(out)
    return d1 is not None and max(abs(d0.get(k, 0) - d1.get(k, 0)) for k in set(d0) | set(d1)) < 1e-6


def _call(mod, qc, target, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=target, entangling_basis="cx", layout_search=True,
                                        seed_transpiler=0, **kw)


def _rzz_angles(c):
    return [float(i.operation.params[0]) for i in c.data if i.operation.name == "rzz"]


def test_version(c50):
    assert c50.VERSION == "2026-10-14.c55"


def test_without_rzz_the_output_is_c49s(c50, c49, fez):
    t, _ = fez
    for qc in (_tfim(8, 2, 9001), _qaoa_gates(6, 9002)):
        assert _sig(_call(c50, qc, t)) == _sig(_call(c49, qc, t))


def test_ising_uses_rzz_is_exact_shorter_and_in_range(c50, fez):
    t, ft = fez
    qc = _tfim(10, 3, 9003)
    plain = _call(c50, qc, t)
    out = _call(c50, qc, ft)
    assert _same_distribution(out, qc)
    ang = _rzz_angles(out)
    assert ang and all(0 < a <= math.pi / 2 + 1e-12 for a in ang)
    assert c50._duration_s(out, ft) < c50._duration_s(plain, t)  # a ring: SWAPs remain; chains are measured in C50-VAL
    assert c50._isa_ok(out, ft)


def test_never_longer_and_exact_on_small_circuits(c50, fez):
    t, ft = fez
    for qc in (_qaoa_gates(6, 9004), _tfim(6, 2, 9005), _qaoa_gates(5, 9006, p=3)):
        plain = _call(c50, qc, t)
        out = _call(c50, qc, ft)
        if any(i.operation.name == "measure" for i in qc.data):
            assert _same_distribution(out, qc)
        else:
            assert c50._implements(qc, out)
        assert c50._duration_s(out, ft) <= c50._duration_s(plain, ft) + 1e-12


def test_measured_circuit_keeps_its_distribution(c50, fez):
    """A measured TFIM chain: the re-synthesised circuit gives the logical circuit's outcome distribution."""
    import numpy as np
    from qiskit.quantum_info import Statevector
    t, ft = fez
    qc = _tfim(6, 2, 9007)
    qc.measure_all()
    out = _call(c50, qc, ft)
    assert "rzz" in out.count_ops()

    def dist(c):
        body = c.remove_final_measurements(inplace=False)
        meas = {}
        for ins in c.data:
            if ins.operation.name == "measure":
                meas[c.find_bit(ins.clbits[0]).index] = c.find_bit(ins.qubits[0]).index
        act = sorted({body.find_bit(q).index for i in body.data for q in i.qubits} | set(meas.values()))
        pos = {q: k for k, q in enumerate(act)}
        from qiskit import QuantumCircuit
        small = QuantumCircuit(len(act))
        for ins in body.data:
            if ins.operation.name != "barrier":
                small.append(ins.operation, [pos[body.find_bit(q).index] for q in ins.qubits])
        p = Statevector(small).probabilities()
        d = {}
        for idx in np.nonzero(p > 1e-14)[0]:
            key = "".join("1" if (int(idx) >> pos[meas[cb]]) & 1 else "0" for cb in reversed(range(len(meas))))
            d[key] = d.get(key, 0.0) + float(p[idx])
        return d

    ref = qc.decompose(reps=4)
    d0, d1 = dist(ref), dist(out)
    assert max(abs(d0.get(k, 0) - d1.get(k, 0)) for k in set(d0) | set(d1)) < 1e-6


def test_no_error_value_decides(c50, fez):
    """Item 86's re-synthesis does not change when every reported error is rescaled."""
    from qiskit.transpiler import InstructionProperties
    t, ft = fez
    ft2 = copy.deepcopy(ft)
    rng = random.Random(4)
    for name in ("cz", "rzz", "sx", "rx", "x", "measure"):
        for qa, p in ft2[name].items():
            if qa is not None and p is not None and p.error is not None and p.error < 0.5:
                ft2.update_instruction_properties(name, qa, InstructionProperties(
                    duration=p.duration, error=min(0.4, p.error * rng.uniform(0.2, 5.0))))
    base = _call(c50, _tfim(8, 2, 9008), t)
    assert _sig(c50._fractional(base, ft)) == _sig(c50._fractional(base, ft2))


def test_switch_off_and_determinism(c50, c49, fez):
    """Three calls give one output; with FRACTIONAL = False the output is c49's on the same Target with its rzz angles
    folded (c51: an unfolded rzz does not run, so folding is not switched off)."""
    t, ft = fez
    qc = _tfim(8, 2, 9009)
    outs = {_sig(_call(c50, qc, ft)) for _ in range(3)}
    assert len(outs) == 1
    c50.FRACTIONAL = False
    try:
        off = _call(c50, qc, ft)
    finally:
        c50.FRACTIONAL = "auto"
    ref = _call(c49, qc, ft)
    if not c50._rzz_in_range(ref):
        ref = c50._fold(ref, ft)
    assert _sig(off) == _sig(ref)
    assert c50._rzz_in_range(off)


def test_every_output_executable(c50, fez):
    """C50-VAL's refuted V6 (Addendum 480): every circuit returned on a Target with rzz has its rzz angles in
    (0, pi/2], with the re-synthesis on and off, and is native to the Target; measured circuits keep their
    distribution."""
    t, ft = fez
    for qc in (_tfim(8, 2, 9011), _qaoa_gates(6, 9012), _qaoa_gates(5, 9013, p=3)):
        for flag in ("auto", False):
            c50.FRACTIONAL = flag
            try:
                out = _call(c50, qc, ft)
            finally:
                c50.FRACTIONAL = "auto"
            assert c50._rzz_in_range(out) and c50._isa_ok(out, ft)
            if any(i.operation.name == "measure" for i in qc.data):
                assert _same_distribution(out, qc)
            else:
                assert c50._implements(qc, out)


def test_failed_qubits_still_cut(c50):
    """Item 85 still applies under item 86: no gate on a failed qubit, and the output is exact."""
    b = _small(failed=(6,))
    ft = fractional(b.target)
    qc = _tfim(6, 2, 9010)
    out = _call(c50, qc, ft)
    pos = {q: i for i, q in enumerate(out.qubits)}
    assert not any(pos[q] == 6 for i in out.data if i.operation.name != "barrier" for q in i.qubits)
    assert _same_distribution(out, qc)
