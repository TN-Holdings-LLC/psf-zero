"""Tests for item 97 (workplace proposal on prototype c57): symmetric two-qubit instructions written smaller index first
before c55's path compiles an in-scope circuit.

Run from the repository root, with C57_97 pointing to the patched file (default: next to this test):
    python -m pytest <dir>/test_97.py -q
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings

import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("PSF_REPO", os.getcwd())
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def m():
    _load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    mod = _load(os.environ.get("C57_97", os.path.join(HERE, "psf_compile.py")), "psf_c57_97_test")
    mod.WARN_WITHOUT_TARGET = False
    return mod


@pytest.fixture(scope="module")
def c57():
    mod = _load(os.path.join(REPO, "patches", "psf_compile_c57_2026-10-14", "psf_compile.py"), "psf_c57_test97")
    mod.WARN_WITHOUT_TARGET = False
    return mod


@pytest.fixture(scope="module")
def fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez().target


def _call(mod, qc, t):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=t, entangling_basis="cx", layout_search=True, seed_transpiler=0)


def _sig(c):
    return repr([(i.operation.name, [c.find_bit(q).index for q in i.qubits],
                  [round(float(p), 12) for p in i.operation.params]) for i in c.data])


def _chain(n, flip):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(0.3 + 0.1 * q, q)
    for s in range(3):
        for q in range(n - 1):
            a, b = (q + 1, q) if flip(q, s) else (q, q + 1)
            qc.rzz(0.2 + 0.05 * q, a, b)
        for q in range(n):
            qc.rx(0.35, q)
    qc.measure_all()
    return qc


def test_version_and_switch(m):
    assert m.VERSION.startswith("2026-10-14.c57+97") and m.ORIENT97 is True


def test_rule_is_exact_and_selective(m):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Operator, SparsePauliOp
    qc = QuantumCircuit(4)
    qc.rzz(0.3, 1, 0)
    qc.cz(3, 2)
    qc.rxx(0.2, 2, 1)
    qc.ryy(0.1, 3, 0)
    qc.cp(0.4, 2, 0)
    qc.cx(3, 0)
    qc.crz(0.4, 3, 1)
    qc.append(PauliEvolutionGate(SparsePauliOp(["ZZ"], [0.5]), time=0.7), [2, 0])
    qc.append(PauliEvolutionGate(SparsePauliOp(["ZX"], [0.5]), time=0.7), [3, 1])
    out = m._orient97(qc)
    assert Operator(out).equiv(Operator(qc))
    got = [(i.operation.name, [out.find_bit(q).index for q in i.qubits]) for i in out.data]
    assert got == [("rzz", [0, 1]), ("cz", [2, 3]), ("rxx", [1, 2]), ("ryy", [0, 3]), ("cp", [0, 2]),
                   ("cx", [3, 0]), ("crz", [3, 1]), ("PauliEvolution", [0, 2]), ("PauliEvolution", [3, 1])]


def test_unchanged_input_is_returned_as_is(m):
    qc = _chain(5, lambda q, s: False)
    assert m._orient97(qc) is qc


def test_three_orientations_one_output(m, fez):
    import random
    rng = random.Random(97)
    flips = {(q, s): rng.random() < 0.5 for q in range(7) for s in range(3)}
    outs = [_call(m, _chain(8, f), fez) for f in (lambda q, s: False, lambda q, s: True, lambda q, s: flips[(q, s)])]
    assert len({_sig(o) for o in outs}) == 1


def test_in_scope_equals_c57_on_the_canonical_input(m, c57, fez):
    qc = _chain(8, lambda q, s: (q + s) % 2 == 1)
    assert _sig(_call(m, qc, fez)) == _sig(_call(c57, m._orient97(qc), fez))


def test_out_of_scope_untouched(m, c57, fez):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(5)
    qc.h(range(5))
    for a in range(5):
        for b in range(a + 1, 5):
            qc.rzz(0.1 * (a + b), b, a)  # K5 does not embed: handed to Qiskit level 3 as written
    qc.measure_all()
    from qiskit.transpiler import generate_preset_pass_manager
    out = _call(m, qc, fez)
    assert _sig(out) == _sig(_call(c57, qc, fez))
    assert _sig(out) == _sig(generate_preset_pass_manager(3, target=fez, seed_transpiler=0).run(qc))


def test_switch_off_restores_c57(m, c57, fez):
    qc = _chain(8, lambda q, s: True)
    m.ORIENT97 = False
    try:
        assert _sig(_call(m, qc, fez)) == _sig(_call(c57, qc, fez))
    finally:
        m.ORIENT97 = True
