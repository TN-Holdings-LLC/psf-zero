"""Tests for candidate psf_compile 2026-10-07.c21 (changelog item 48: instructions on three or more qubits expanded
before PSF-Zero's own pipeline) against candidate 2026-10-07.c20 (patches/psf_compile_c20_2026-10-07/psf_compile.py),
on which it is based.

Run from the repository root:  python -m pytest patches/psf_compile_c21_2026-10-07/test_c21.py -q -s
"""
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
DEVICES = ("FakeTorino", "FakeHanoiV2", "FakeKingston")


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c20 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c20_2026-10-07", "psf_compile.py"),
                        "psf_compile_c20_c21_test")
    c21 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c21_test")
    return dict(c20=c20, c21=c21)


def target(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)().target


def call(mod, qc, dev, recommended=False):
    t = target(dev)
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in t.operation_names]
    kw = dict(target=t, **RECOMMENDED) if recommended else {}
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                        entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def q2(c):
    return sum(1 for i in c.data if len(i.qubits) == 2 and i.operation.name != "barrier")


def valid(c, dev):
    t = target(dev)
    for i in c.data:
        if i.operation.name in ("barrier", "measure", "delay"):
            continue
        q = tuple(c.find_bit(b).index for b in i.qubits)
        if i.operation.name not in t.operation_names or q not in t[i.operation.name]:
            return False
    return True


def family(name, n, seed):
    """SKIP's generator (benchmarks/skip_eval.py): ring and brick are two-qubit gates only; pauli is one
    PauliEvolutionGate per term and qft one QFT instruction, both on more than two qubits."""
    import skip_eval
    return skip_eval.family_circuit(name, n, np.random.default_rng(seed))


def toffolis(n, seed):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(6):
        a, b, c = (int(x) for x in rng.choice(n, 3, replace=False))
        qc.h(a)
        qc.ccx(a, b, c)
        qc.rz(float(rng.uniform(0, 6)), c)
    return qc


def test_versions(mods):
    assert mods["c21"].VERSION == "2026-10-07.c21"
    assert mods["c20"].VERSION == "2026-10-07.c20"


@pytest.mark.parametrize("dev", DEVICES)
def test_two_qubit_inputs_unchanged(mods, dev):
    """No instruction on more than two qubits (a measure_all barrier aside): c20's circuit, default and recommended."""
    for name, n, seed in (("ring", 8, 1), ("ring", 14, 2), ("brick", 10, 3), ("brick", 24, 4)):  # <= 27 qubits
        qc = family(name, n, 48_000_000 + seed)
        if seed % 2:
            qc.measure_all()
        assert not mods["c21"]._has_wide(qc)
        assert sig(call(mods["c21"], qc, dev)) == sig(call(mods["c20"], qc, dev)), (name, n)
        if n <= 16:
            assert sig(call(mods["c21"], qc, dev, True)) == sig(call(mods["c20"], qc, dev, True)), (name, n)


@pytest.mark.parametrize("dev", DEVICES)
def test_wide_inputs_default_call(mods, dev):
    """Inputs with instructions on more than two qubits: c21's default call is valid, implements the input wherever
    c20's does (item 39's check, up to 16 qubits), and its two-qubit count is printed against c20's."""
    rows = []
    cases = [("pauli", 8, 11), ("pauli", 12, 12), ("qft", 8, 13), ("qft", 12, 14), ("pauli", 24, 15), ("qft", 24, 16)]
    for name, n, seed in cases:
        qc = family(name, n, 48_100_000 + seed)
        assert mods["c21"]._has_wide(qc)
        o20, o21 = call(mods["c20"], qc, dev), call(mods["c21"], qc, dev)
        assert valid(o21, dev)
        if n <= 16:
            assert mods["c21"]._implements(qc, o21) or not mods["c21"]._implements(qc, o20), (name, n)
        rows.append((name, n, q2(o20), q2(o21)))
    for k, n in enumerate((5, 7, 9)):
        qc = toffolis(n, 48_200_000 + k)
        o20, o21 = call(mods["c20"], qc, dev), call(mods["c21"], qc, dev)
        assert valid(o21, dev)
        assert mods["c21"]._implements(qc, o21) or not mods["c21"]._implements(qc, o20), ("ccx", n)
        rows.append(("ccx", n, q2(o20), q2(o21)))
    print(f"\n{dev}: two-qubit gates c20 -> c21: " + ", ".join(f"{a}{n} {x}->{y}" for a, n, x, y in rows))


def test_wide_inputs_recommended_call(mods):
    """The recommended call on inputs with wide instructions: valid, implements the input."""
    for name, n, seed in (("pauli", 10, 21), ("qft", 10, 22)):
        qc = family(name, n, 48_300_000 + seed)
        qc.measure_all()
        for dev in ("FakeTorino", "FakeHanoiV2"):
            o = call(mods["c21"], qc, dev, True)
            assert valid(o, dev)
            assert mods["c21"]._implements(qc, o)  # measurements are skipped by the check, as in the call itself


def test_unexpandable_instruction(mods):
    """A three-qubit instruction without a definition: c21 warns, counts it, and behaves as c20 (same circuit, or the
    same exception)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.append(Gate("opaque3", 3, []), [0, 1, 2])
    qc.cx(2, 3)
    before = mods["c21"].UNROLL_STATS["failed"]
    res = {}
    for m in ("c20", "c21"):
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                res[m] = ("ok", sig(call(mods[m], qc, "FakeTorino")))
        except Exception as exc:  # noqa: BLE001 -- the type is what is compared
            res[m] = ("raised", type(exc).__name__)
    show = {m: res[m][1] if res[m][0] == "raised" else "ok" for m in res}
    print(f"\nopaque three-qubit gate: c20 {show['c20']}, c21 {show['c21']}")
    assert res["c21"] == res["c20"]
    assert mods["c21"].UNROLL_STATS["failed"] == before + 1
