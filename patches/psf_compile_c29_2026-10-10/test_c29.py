"""Tests for candidate psf_compile 2026-10-10.c29 (changelog item 56: no estimate for a candidate whose exactness check
cannot be made) against candidate 2026-10-09.c26, on which it is based.

Item 56 is meant to change no returned circuit. These tests check (1) that its counts are sound -- they say "cannot"
only where `_ops_of` / `_implements` / `_same_action` would give up or return False -- and (2) that the functions it
changes return what c26's return, with EXACT_MAX_OPS lowered in both modules so that the skips happen on small
circuits.

Run from the repository root:  python -m pytest patches/psf_compile_c29_2026-10-10/test_c29.py -q
"""
import contextlib
import inspect
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
MAX_ERROR = 0.5


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c26 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c26_2026-10-09", "psf_compile.py"),
                        "psf_compile_c26_for_c29_test")
    c29 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c29_test")
    return dict(c26=c26, c29=c29, layout=lay)


@contextlib.contextmanager
def limit(mods, n):
    """EXACT_MAX_OPS set to n in both compile modules (read at call time by _ops_of and item 56's counts)."""
    old = {k: mods[k].EXACT_MAX_OPS for k in ("c26", "c29")}
    for k in old:
        mods[k].EXACT_MAX_OPS = n
    try:
        yield
    finally:
        for k, v in old.items():
            mods[k].EXACT_MAX_OPS = v


@pytest.fixture(scope="module")
def cases():
    from qiskit import transpile
    from qiskit.circuit.random import random_circuit
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend = FakeTorino()
    out = []
    for k in range(40):
        n = 2 + k % 11
        qc = random_circuit(n, 3 + (k * 5) % 14, max_operands=3, measure=False, seed=56_000 + k)
        a = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=k)
        b = transpile(qc, backend=backend, optimization_level=2, seed_transpiler=k + 1)
        wrong = a.copy()
        touched = sorted({a.find_bit(x).index for i in a.data for x in i.qubits})
        wrong.x(touched[0] if touched else 0)
        out.append((qc, a, b, wrong))
    return backend.target, out


def _wide(n_qubits, body=None):
    """A gate on `n_qubits` (more than EXACT_MAX_GATE_QUBITS) whose definition is `body` (empty if None)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    d = QuantumCircuit(n_qubits)
    if body is not None:
        body(d)
    g = Gate(f"wide{n_qubits}", n_qubits, [])
    g.definition = d
    return g


def _pairs(cs):
    for qc, a, b, wrong in cs:
        yield qc, a
        yield qc, b
        yield qc, wrong


def test_versions(mods):
    assert mods["c29"].VERSION == "2026-10-10.c29" and mods["c26"].VERSION == "2026-10-09.c26"


def test_signature_unchanged(mods):
    assert (inspect.signature(mods["c29"].compile_for_hardware)
            == inspect.signature(mods["c26"].compile_for_hardware))


def test_count_is_a_sound_bound(mods, cases):
    """More than EXACT_MAX_OPS narrow instructions => _ops_of gives up; the narrow qubits are among _ops_of's."""
    c29 = mods["c29"]
    _, cs = cases
    seen = 0
    for qc, a, b, wrong in cs:
        for c in (qc, a, b, wrong):
            for n in (3, 10, 40, 200, 10 ** 9):
                with limit(mods, n):
                    ops = c29._ops_of(c)
                    if c29._n_narrow(c) > n:
                        assert ops is None
                        seen += 1
                    if ops is not None:
                        assert c29._narrow_touched(c) <= {i for _, q in ops for i in q}
    assert seen > 0


def test_cannot_implement_is_sound(mods, cases):
    """_cannot_implement(qc, out) => _implements(qc, out) is False; both outcomes occur."""
    c29 = mods["c29"]
    _, cs = cases
    yes = no = 0
    for qc, out in _pairs(cs):
        for n in (8, 30, 120, 10 ** 9):
            with limit(mods, n):
                if c29._cannot_implement(qc, out):
                    assert c29._implements(qc, out) is False
                    yes += 1
                else:
                    no += 1
    assert yes > 0 and no > 0


def test_wide_instructions_are_not_counted(mods):
    """An instruction wider than EXACT_MAX_GATE_QUBITS is expanded by _ops_of, possibly into nothing: counting it (as
    exploratory candidate c28 did) can say "cannot" where the check can be made. c29 does not count it."""
    from qiskit import QuantumCircuit
    c29 = mods["c29"]
    w = c29.EXACT_MAX_GATE_QUBITS + 1
    qc = QuantumCircuit(w)
    for _ in range(10):
        qc.rz(0.1, 0)
    for _ in range(5):
        qc.append(_wide(w), range(w))
    with limit(mods, 10):
        assert sum(1 for i in qc.data if i.operation.name not in c29._EXACT_SKIP) > 10
        assert c29._n_narrow(qc) == 10 and c29._ops_of(qc) is not None
        assert c29._checkable_logical(qc) is True  # the input can still be checked
    big = 17
    logical = QuantumCircuit(2)
    logical.h(0)
    out = QuantumCircuit(big)
    out.h(0)
    out.append(_wide(big), range(big))  # touches 17 qubits, but its definition is empty
    assert c29._implements(logical, out) is True
    assert c29._cannot_implement(logical, out) is False


def test_choose_lazy_returns_c26s_choice(mods, cases):
    """With some candidates over the (lowered) limit, _choose_lazy returns the same circuit object as c26's."""
    target, cs = cases
    c26, c29 = mods["c26"], mods["c29"]
    before = c29.FEASIBILITY_STATS["candidate"]
    for qc, a, b, wrong in cs[:25]:
        others = [("floor", b), ("level3", wrong), ("floor", a)]
        sizes = sorted({c29._n_narrow(c) for c in (a, b, wrong)})
        for n in [sizes[0] - 1] + [(x + y) // 2 for x, y in zip(sizes, sizes[1:])] + [10 ** 9]:
            for score in ("hybrid", "excitation"):
                with limit(mods, n):
                    r26 = c26._choose_lazy(qc, a, others, target, score)
                    r29 = c29._choose_lazy(qc, a, others, target, score)
                assert r26 is r29, (n, score)
    assert c29.FEASIBILITY_STATS["candidate"] > before


def _same(x, y, given):
    """Both are `given` itself, or neither is and they are equal circuits with equal layouts."""
    if x is given or y is given:
        return x is given and y is given
    return x == y and x.layout == y.layout


def test_compare_level3_returns_c26s_choice(mods, cases):
    target, cs = cases
    c26, c29 = mods["c26"], mods["c29"]
    for qc, a, b, wrong in cs[:20]:
        for n in (5, max(c29._n_narrow(a) - 1, 1), 10 ** 9):
            with limit(mods, n):
                r26 = c26._compare_level3(qc, a, target, MAX_ERROR, 0)
                r29 = c29._compare_level3(qc, a, target, MAX_ERROR, 0)
            assert _same(r26, r29, a), n


def test_resynthesise_returns_c26s_result(mods, cases):
    target, cs = cases
    c26, c29 = mods["c26"], mods["c29"]
    before = c29.FEASIBILITY_STATS["resynthesis"]
    for qc, a, b, wrong in cs[:20]:
        for n in (5, 10 ** 9):
            for mode in ("select", True):
                with limit(mods, n):
                    r26 = c26._resynthesise(a, mode, target, MAX_ERROR)
                    r29 = c29._resynthesise(a, mode, target, MAX_ERROR)
                assert _same(r26, r29, a), (n, mode)
    assert c29.FEASIBILITY_STATS["resynthesis"] > before


def test_whole_compile_same_where_c26_repeats_itself(mods):
    """The recommended call on small circuits, the layout search's clock virtual (as C25-ID2), with the limit lowered
    so that item 56 skips: c29's output equals c26's, compared as C29-ID compares (`c29_identity.value_sig`:
    instructions, parameters, global phase and the logical qubits' initial and final layouts), wherever c26 run twice
    gives one output. The ancillas' assignment in the layout is not compared: within one process it changes between
    runs of c26 itself, with the same instructions and logical layouts (the diagnostic of 2026-10-10, Addendum 422)."""
    from qiskit.circuit.random import random_circuit
    from c29_identity import value_sig
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    import c25_identity2 as C2
    backend = FakeTorino()
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, **C2.RECOMMENDED)

    def run(mod, qc):
        mods["layout"].time = C2._VirtualTime()
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            return mod.compile_for_hardware(qc, **kw)

    scored = 0
    before = sum(mods["c29"].FEASIBILITY_STATS.values())
    for k in range(5):
        qc = random_circuit(3 + k, 4 + k, max_operands=2, measure=False, seed=56_100 + k)
        for n in (12, 10 ** 9):
            with limit(mods, n):
                x, x2, y = run(mods["c26"], qc), run(mods["c26"], qc), run(mods["c29"], qc)
            if value_sig(x) == value_sig(x2):
                assert value_sig(y) == value_sig(x), (k, n)
                scored += 1
    assert scored >= 6
    assert sum(mods["c29"].FEASIBILITY_STATS.values()) > before


def test_feasibility_stats_keys(mods):
    assert set(mods["c29"].FEASIBILITY_STATS) == {"candidate", "level3", "input", "resynthesis"}
