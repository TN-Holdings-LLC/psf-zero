"""Tests for prototype psf_compile 2026-10-14.c54 (item 89: a diagonal Clifford before the measurements dropped in
item 73's reader; item 90: estimate-based defaults off, item 75 scored by duration).

c51's tests run on c52 in test_c51_on_c52.py, test_c49_on_c57.py and test_c48_on_c49_for_c57.py (items 89-90 switched
off where they compare with earlier prototypes' outputs).

Run from the repository root:  python -m pytest patches/psf_compile_c57_2026-10-14/test_c52.py -q
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

from test_c48_on_c49_for_c57 import _qaoa_gates, _sig  # noqa: E402


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c52 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c54_as_c52_test")
    c51 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c51_2026-10-14", "psf_compile.py"),
                        "psf_compile_c51_for_c52_test")
    hw1 = H.load_module(os.path.join(REPO, "benchmarks", "hw1_real.py"), "hw1_for_c52_test")
    c52.WARN_WITHOUT_TARGET = c51.WARN_WITHOUT_TARGET = False
    c52.SCOPE_GATE = False  # c56: item 96 (scope gate) off; these are the older suites' semantics (tested in test_c56.py)
    return dict(c52=c52, c51=c51, hw1=hw1)


@pytest.fixture(scope="module")
def fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez()


def _call(mod, qc, backend, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True,
                                        seed_transpiler=0, **kw)


def _classifier(n, layers, seed, measure=True):
    """W3's model: angle encoding, then per layer ry/rz on every qubit and a CZ ring."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(rng.uniform(0, math.pi), q)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q)
            qc.rz(rng.uniform(-3, 3), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    if measure:
        qc.measure_all()
    return qc


def _same_distribution(hw1, out, qc):
    p0, _ = hw1.measured_distribution(hw1._unrolled(qc))
    p1, _ = hw1.measured_distribution(out)
    return max(abs(p0.get(k, 0) - p1.get(k, 0)) for k in set(p0) | set(p1)) < 1e-6


def test_version(mods):
    assert mods["c52"].VERSION == "2026-10-14.c57"


@pytest.mark.parametrize("n,L", [(4, 3), (6, 5), (6, 9)])
def test_item89_odd_layers_read_exact_and_shorter(mods, fez, n, L):
    c52 = mods["c52"]
    qc = _classifier(n, L, 100 * n + L)
    before = c52.STEINER_STATS["diag_dropped"]
    out = _call(c52, qc, fez)
    assert c52.STEINER_STATS["diag_dropped"] > before
    assert _same_distribution(mods["hw1"], out, qc)
    c52.DROP_DIAGONAL_BEFORE_MEASURE = False
    try:
        off = _call(c52, qc, fez)
    finally:
        c52.DROP_DIAGONAL_BEFORE_MEASURE = True
    assert c52._duration_s(out, fez.target) < c52._duration_s(off, fez.target)


def test_item89_needs_every_qubit_measured(mods):
    """Without final measurements the diagonal Clifford changes the unitary: the reader still refuses."""
    c52 = mods["c52"]
    qc = _classifier(4, 3, 403, measure=False)
    ev = c52._st_parse_gates(qc)
    assert ev is None


def test_item89_partial_measurement_refused(mods):
    c52 = mods["c52"]
    qc = _classifier(4, 3, 404, measure=False)
    from qiskit import ClassicalRegister
    cr = ClassicalRegister(2)
    qc.add_register(cr)
    qc.measure([0, 1], cr)
    assert c52._st_parse_gates(qc) is None


def test_item90_defaults(mods, fez):
    """MAP_FREE_DEFAULTS: no re-placement by errors and routing at level 1 in the default call."""
    c52 = mods["c52"]
    assert c52.MAP_FREE_DEFAULTS is True and c52.PLACE_SCORE == "duration"
    qc = _qaoa_gates(6, 9101)
    out = _call(c52, qc, fez)
    explicit = _call(c52, qc, fez, placement_refine=False, routing_optimization_level=1)
    assert _sig(out) == _sig(explicit)


def test_item90_errors_do_not_move_the_usual_path(mods, fez):
    """With every reported error rescaled (all kept below item 74's 0.1 and the failed threshold), the usual path's
    output (no STEINER6, no Qiskit candidate) is unchanged: item 75's score reads durations only."""
    from qiskit.transpiler import InstructionProperties
    c52 = mods["c52"]
    t2 = copy.deepcopy(fez.target)
    rng = random.Random(6)
    for name in ("cz", "sx", "x", "measure"):
        for qa, p in t2[name].items():
            if qa is not None and p is not None and p.error is not None and p.error < 0.05:
                t2.update_instruction_properties(name, qa, InstructionProperties(
                    duration=p.duration, error=min(0.09, p.error * rng.uniform(0.2, 5.0))))
    for qc in (_qaoa_gates(6, 9102), _qaoa_gates(8, 9103, p=1)):
        kw = dict(steiner=False, level3_candidate=False, near_failed_error=None)
        with contextlib.redirect_stdout(io.StringIO()):
            a = c52.compile_for_hardware(qc, target=fez.target, entangling_basis="cx", layout_search=True,
                                         seed_transpiler=0, **kw)
            b = c52.compile_for_hardware(qc, target=t2, entangling_basis="cx", layout_search=True,
                                         seed_transpiler=0, **kw)
        assert _sig(a) == _sig(b)


def test_switches_restore_c51(mods, fez):
    c52, c51 = mods["c52"], mods["c51"]
    c52.DROP_DIAGONAL_BEFORE_MEASURE = c52.MAP_FREE_DEFAULTS = False
    c52.PLACE_SCORE = "esp"
    try:
        for qc in (_classifier(4, 3, 9104), _qaoa_gates(6, 9105)):
            assert _sig(_call(c52, qc, fez)) == _sig(_call(c51, qc, fez))
    finally:
        c52.DROP_DIAGONAL_BEFORE_MEASURE = c52.MAP_FREE_DEFAULTS = True
        c52.PLACE_SCORE = "duration"


def test_same_output_every_call(mods, fez):
    c52 = mods["c52"]
    qc = _classifier(6, 5, 9106)
    assert len({_sig(_call(c52, qc, fez)) for _ in range(3)}) == 1
