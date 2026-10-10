"""Tests for candidate psf_compile 2026-10-10.c30 (changelog item 57b: the estimates' and checks' per-gate loops in
psf_zero_core57) against candidate 2026-10-10.c29, on which it is based. They need psf_zero_core57 built and
installed (patches/psf_zero_core57_2026-10-10: `maturin develop --release` in that folder).

Item 57b's values agree with Python's to rounding, not bit for bit; the tests require 1e-12 (relative) and the same
None cases, the same check results, and c29's values exactly when the core is not used.

Run from the repository root:  python -m pytest patches/psf_compile_c30_2026-10-10/test_c30.py -q
"""
import contextlib
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
REL_TOL = 1e-12
COSTS = ("excitation_cost", "hybrid_cost")


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c29 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c29_2026-10-10", "psf_compile.py"),
                        "psf_compile_c29_for_c30_test")
    c30 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c30_test")
    return dict(c29=c29, c30=c30, layout=lay)


@contextlib.contextmanager
def no_core(mod):
    core = mod._CORE57
    mod._CORE57 = None
    try:
        yield
    finally:
        mod._CORE57 = core


@pytest.fixture(scope="module")
def cases():
    from qiskit import transpile
    from qiskit.circuit.random import random_circuit
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend = FakeTorino()
    out = []
    for k in range(60):
        n = 2 + k % 13
        qc = random_circuit(n, 3 + (k * 7) % 20, max_operands=3, measure=k % 4 == 0, seed=57_200 + k)
        a = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=k)
        b = transpile(qc, backend=backend, optimization_level=2, seed_transpiler=k + 1)
        out.append((qc.remove_final_measurements(inplace=False), a, b))
    return backend.target, out


def _close(x, y):
    if x is None or y is None:
        return x is None and y is None
    return x == y or abs(x - y) <= REL_TOL * max(abs(x), abs(y))


def test_versions_and_core(mods):
    assert mods["c30"].VERSION == "2026-10-10.c30" and mods["c29"].VERSION == "2026-10-10.c29"
    assert mods["c30"]._CORE57 is not None, "psf_zero_core57 is not installed: build patches/psf_zero_core57_2026-10-10"
    assert mods["c30"].CORE57_VERSION == "2026-10-10.c30"


def test_estimates_within_tolerance_of_python(mods, cases):
    target, cs = cases
    c29, c30 = mods["c29"], mods["c30"]
    before = c30.CORE57_STATS["rust"]
    worst = 0.0
    for _, a, b in cs:
        for c in (a, b):
            for f in COSTS:
                x, y = getattr(c29, f)(c, target), getattr(c30, f)(c, target)
                assert _close(x, y), (f, x, y)
                if x is not None and y is not None and x != y:
                    worst = max(worst, abs(x - y) / max(abs(x), abs(y)))
    assert c30.CORE57_STATS["rust"] > before
    print(f"\nlargest relative difference: {worst:.2e}")


def test_python_path_is_c29_exactly(mods, cases):
    target, cs = cases
    c29, c30 = mods["c29"], mods["c30"]
    with no_core(c30):
        for _, a, b in cs[:20]:
            for f in COSTS:
                assert getattr(c29, f)(a, target) == getattr(c30, f)(a, target), f


def test_checks_return_c29s_results(mods, cases):
    _, cs = cases
    c29, c30 = mods["c29"], mods["c30"]
    before = c30.CORE57_STATS["rust"]
    for qc, a, b in cs:
        wrong = a.copy()
        touched = sorted({a.find_bit(x).index for i in a.data for x in i.qubits})
        wrong.x(touched[0] if touched else 0)
        for c in (a, b, wrong):
            assert c29._implements(qc, c) == c30._implements(qc, c)
        assert c29._same_action(a, b) == c30._same_action(a, b)
    assert c30.CORE57_STATS["rust"] > before


def test_apply_ops_amplitudes(mods, cases):
    c29, c30 = mods["c29"], mods["c30"]
    _, cs = cases
    rng = np.random.default_rng(57)
    for _, a, _ in cs[:30]:
        ops = c29._ops_of(a)
        if not ops:
            continue
        touched = sorted({i for _, q in ops for i in q})
        if len(touched) > 12:
            continue
        pos = {p: j for j, p in enumerate(touched)}
        psi = rng.normal(size=(2,) * len(touched)) + 1j * rng.normal(size=(2,) * len(touched))
        x, y = c29._apply_ops(psi, ops, pos), c30._apply_ops(psi, ops, pos)
        assert x.shape == y.shape and np.max(np.abs(x - y)) <= REL_TOL * max(1.0, float(np.max(np.abs(x))))


def test_wide_instruction_falls_back_to_python(mods):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCXGate
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    target = FakeTorino().target
    c29, c30 = mods["c29"], mods["c30"]
    qc = QuantumCircuit(8)
    qc.h(0)
    qc.append(MCXGate(6), range(7))  # seven qubits: the core takes at most six
    qc.cx(0, 7)
    before = dict(c30.CORE57_STATS)
    for f in COSTS:
        assert getattr(c29, f)(qc, target) == getattr(c30, f)(qc, target)
    assert c30.CORE57_STATS["python"] == before["python"] + 2 and c30.CORE57_STATS["rust"] == before["rust"]


def test_malformed_buffer_raises(mods):
    core = mods["c30"]._CORE57
    with pytest.raises(ValueError):
        core.estimate57(b"PF57")
    with pytest.raises(ValueError):
        core.apply_ops57(b"")


def test_whole_compile_same_where_c29_repeats_itself(mods):
    """The recommended call on small circuits, the layout search's clock virtual, compared as C29-ID compares."""
    from qiskit.circuit.random import random_circuit
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    import c25_identity2 as C2
    from c29_identity import value_sig
    backend = FakeTorino()
    basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0, target=backend.target, **C2.RECOMMENDED)

    def run(mod, qc):
        mods["layout"].time = C2._VirtualTime()
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            return mod.compile_for_hardware(qc, **kw)

    scored = 0
    for k in range(6):
        qc = random_circuit(3 + k, 5 + 2 * k, max_operands=2, measure=False, seed=57_300 + k)
        x, x2, y = run(mods["c29"], qc), run(mods["c29"], qc), run(mods["c30"], qc)
        if value_sig(x) == value_sig(x2):
            assert value_sig(y) == value_sig(x), k
            scored += 1
    assert scored >= 4
