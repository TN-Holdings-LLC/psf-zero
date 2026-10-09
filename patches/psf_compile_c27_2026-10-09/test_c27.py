"""Tests for candidate psf_compile 2026-10-09.c27 (changelog item 54: a work budget shared by the estimates and
exactness checks of one compile_for_hardware call) against candidate 2026-10-09.c26, on which it is based.

Run from the repository root:  python -m pytest patches/psf_compile_c27_2026-10-09/test_c27.py -q
"""
import inspect
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
COSTS = ("excitation_cost", "hybrid_cost", "pauli_cost", "kraus_cost")


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c26 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c26_2026-10-09", "psf_compile.py"),
                        "psf_compile_c26_for_c27_test")
    c27 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c27_test")
    return dict(c26=c26, c27=c27)


@pytest.fixture(scope="module")
def cases():
    from qiskit import transpile
    from qiskit.circuit.random import random_circuit
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend = FakeTorino()
    out = []
    for k in range(40):
        n = 2 + k % 9
        qc = random_circuit(n, 2 + (k * 7) % 12, max_operands=2, measure=False, seed=54_000 + k)
        a = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=k)
        b = transpile(qc, backend=backend, optimization_level=2, seed_transpiler=k + 1)
        out.append((qc, a, b))
    return backend.target, out


def _calls(target, case):
    qc, a, b = case
    return [(f, (a, target)) for f in COSTS] + [("_implements", (qc, a)), ("_implements", (qc, b)),
                                                ("_same_action", (a, b))]


def test_versions(mods):
    assert mods["c27"].VERSION == "2026-10-09.c27" and mods["c26"].VERSION == "2026-10-09.c26"


def test_no_budget_outside_compile(mods, cases):
    """Called directly (no compile_for_hardware around them), the functions have no budget and return c26's values."""
    c26, c27 = mods["c26"], mods["c27"]
    target, cs = cases
    assert c27._WORK == {"depth": 0, "left": None}
    for case in cs:
        for f, args in _calls(target, case):
            assert getattr(c27, f)(*args) == getattr(c26, f)(*args), f
    assert c27._WORK["left"] is None


def test_large_budget_same_values_and_exact_draws(mods, cases):
    """With a budget large enough, every value is c26's, and exactly the counted work is drawn, in integers."""
    c26, c27 = mods["c26"], mods["c27"]
    target, cs = cases
    zero = []
    for i, case in enumerate(cs):
        for f, args in _calls(target, case):
            c27._WORK["left"] = start = 10 ** 15
            try:
                assert getattr(c27, f)(*args) == getattr(c26, f)(*args), f
                drawn = start - c27._WORK["left"]
            finally:
                c27._WORK["left"] = None
            expected = _expected_work(c27, f, args)
            assert isinstance(drawn, int) and drawn == expected, (i, f, drawn, expected)
            if drawn == 0:
                zero.append((i, f))
    print(f"\ncalls not made anyway (over 16 touched qubits), so not counted (case, function): {zero}")


def _expected_work(c27, f, args):
    """The work item 54 should count, computed independently of the module's code paths; 0 for a call that cannot
    be made anyway (more than RESYNTH_MAX_QUBITS touched qubits), which is not counted."""
    def shape(c):
        ops = [tuple(c.find_bit(x).index for x in ins.qubits) for ins in c.data
               if ins.operation.name not in ("barrier", "measure", "delay")]
        return len(ops), sum(len(q) >= 2 for q in ops), {i for q in ops for i in q}
    top = c27.RESYNTH_MAX_QUBITS
    if f in COSTS:
        n_all, n_multi, t = shape(args[0])
        if len(t) > top:
            return 0
        k = max(len(t), 1)
        amps = (n_multi if f in ("excitation_cost", "hybrid_cost") else n_all) * 2 ** k
        return c27.WORK_PER_OP * n_all + c27.WORK_PER_AMP[f] * amps
    if f == "_same_action":
        a, b = shape(args[0]), shape(args[1])
        k = len(a[2] | b[2])
        if k > top:
            return 0
        ops = 2 * (a[0] + b[0])
        return c27.WORK_PER_OP * ops + c27.WORK_PER_AMP[f] * ops * 2 ** k
    qc, out = args
    a, b = shape(qc), shape(out)
    n = qc.num_qubits
    lay = out.layout
    init = set(lay.initial_index_layout(filter_ancillas=True)[:n]) if lay is not None else set(range(n))
    fin = set(lay.final_index_layout(filter_ancillas=True)[:n]) if lay is not None else set(range(n))
    k = len(b[2] | init | fin)
    if n > top or k > top:
        return 0
    return c27.WORK_PER_OP * 2 * (a[0] + b[0]) + c27.WORK_PER_AMP[f] * 2 * (a[0] * 2 ** n + b[0] * 2 ** k)


def test_budget_exhausted_refuses(mods, cases):
    """With no budget left, estimates cannot be made (None) and checks cannot be made (False); nothing is drawn."""
    c27 = mods["c27"]
    target, cs = cases
    before = dict(c27.WORK_STATS)
    n = 0
    for case in cs[:5]:
        for f, args in _calls(target, case):
            c27._WORK["left"] = 0
            try:
                r = getattr(c27, f)(*args)
                assert c27._WORK["left"] == 0
            finally:
                c27._WORK["left"] = None
            assert r is (None if f in COSTS else False), f
            n += 1
    assert c27.WORK_STATS["refused"] - before["refused"] == n


def test_draw_is_all_or_nothing(mods):
    c27 = mods["c27"]
    c27._WORK["left"] = 100_000
    try:
        assert c27._work_draw("_same_action", 2, 4) is True  # 2 * 20,000 + 5 * 4 = 40,020
        assert c27._WORK["left"] == 100_000 - 40_020
        assert c27._work_draw("_same_action", 3, 0) is False  # 60,000 > 59,980: nothing drawn
        assert c27._WORK["left"] == 59_980
    finally:
        c27._WORK["left"] = None


def _small_compile_args():
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend = FakeTorino()
    qc = QuantumCircuit(4)
    for k in range(6):
        qc.cx(k % 4, (k + 1) % 4)
        qc.rz(0.1 * (k + 1), k % 4)
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, placement_refine=True, final_resynthesis="select",
              compare_level3=True, compare_floor=True, candidate_score="hybrid")
    return qc, kw


def test_compile_opens_and_closes_the_budget(mods):
    c27 = mods["c27"]
    qc, kw = _small_compile_args()
    seen = []
    orig = c27._work_draw

    def spy(fn, ops, amps):
        seen.append((c27._WORK["depth"], c27._WORK["left"]))
        return orig(fn, ops, amps)

    c27._work_draw = spy
    try:
        c27.compile_for_hardware(qc, **kw)
    finally:
        c27._work_draw = orig
    assert seen, "the recommended call made no estimate or check"
    assert all(d >= 1 and left is not None for d, left in seen)
    assert seen[0][1] <= int(c27.WORK_BUDGET_S * c27.WORK_UNITS_PER_S)
    assert c27._WORK == {"depth": 0, "left": None}


def test_zero_and_no_budget(mods):
    """work_budget_s=0 refuses every estimate and check made under compile_for_hardware; it still compiles."""
    c27 = mods["c27"]
    qc, kw = _small_compile_args()
    before = c27.WORK_STATS["refused"]
    made = c27.WORK_STATS["made"]
    out = c27.compile_for_hardware(qc, work_budget_s=0, **kw)
    assert out.num_qubits > 0
    assert c27.WORK_STATS["refused"] > before and c27.WORK_STATS["made"] == made
    before = c27.WORK_STATS["refused"]
    c27.compile_for_hardware(qc, work_budget_s=None, **kw)
    assert c27.WORK_STATS["refused"] == before  # no budget: nothing refused, nothing counted
    assert c27._WORK == {"depth": 0, "left": None}


def test_budget_closed_after_an_error(mods):
    c27 = mods["c27"]
    qc, kw = _small_compile_args()
    with pytest.raises(ValueError):
        c27.compile_for_hardware(qc, **dict(kw, candidate_score="nope"))
    assert c27._WORK == {"depth": 0, "left": None}


@pytest.mark.parametrize("bad", [-1.0, True, "10", float("nan")])
def test_bad_budget(mods, bad):
    c27 = mods["c27"]
    qc, kw = _small_compile_args()
    with pytest.raises(ValueError):
        c27.compile_for_hardware(qc, work_budget_s=bad, **kw)
    assert c27._WORK == {"depth": 0, "left": None}


def test_signature_and_doc(mods):
    c27 = mods["c27"]
    sig = inspect.signature(c27.compile_for_hardware)
    assert sig.parameters["work_budget_s"].default == 10.0
    assert "work_budget_s" in c27.compile_for_hardware.__doc__
