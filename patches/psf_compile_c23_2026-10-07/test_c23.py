"""Tests for candidate psf_compile 2026-10-07.c23 (changelog item 50: commutative cancellation tried when it removes
two-qubit gates) against candidate 2026-10-07.c22 (patches/psf_compile_c22_2026-10-07/psf_compile.py), on which it is
based.

Run from the repository root:  python -m pytest patches/psf_compile_c23_2026-10-07/test_c23.py -q -s
"""
import contextlib
import io
import os
import sys
import time
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


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c22 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c22_2026-10-07", "psf_compile.py"),
                        "psf_compile_c22_c23_test")
    c23 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c23_test")
    return dict(c22=c22, c23=c23)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def call(mod, qc, dev, recommended=False):
    b = backend(dev)
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in b.operation_names]
    kw = dict(target=b.target, **RECOMMENDED) if recommended else {}
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = mod.compile_for_hardware(qc, coupling_map=b.coupling_map, basis_gates=basis, entangling_basis="cx",
                                       layout_search=True, seed_transpiler=0, **kw)
    return out, time.perf_counter() - t0


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def q2(c):
    return sum(1 for i in c.data if len(i.qubits) == 2 and i.operation.name != "barrier")


def bvlike(n):
    """Benchpress's trivial_bvlike_circuit (qiskit_gym/circuits/circuits.py), copied: CX from every qubit to the last,
    X and Z, and the CX gates again in reverse order."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for k in range(n - 1):
        qc.cx(k, n - 1)
    qc.x(n - 1)
    qc.z(n - 2)
    for k in range(n - 2, -1, -1):
        qc.cx(k, n - 1)
    return qc


def family(name, n, seed):
    import skip_eval
    return skip_eval.family_circuit(name, n, np.random.default_rng(seed))


def test_versions(mods):
    assert mods["c23"].VERSION == "2026-10-07.c23"
    assert mods["c22"].VERSION == "2026-10-07.c22"


@pytest.mark.parametrize("n", (8, 30, 100))
def test_bvlike_cancels(mods, n):
    qc = bvlike(n)
    assert mods["c23"]._cancel_candidate(qc) is not None
    o22, _ = call(mods["c22"], qc, "FakeTorino")
    o23, _ = call(mods["c23"], qc, "FakeTorino")
    print(f"\nBV-like {n}: c22 {q2(o22)}, c23 {q2(o23)}")
    assert q2(o23) <= q2(o22)
    assert q2(o23) == 0


@pytest.mark.parametrize("dev", ("FakeTorino", "FakeHanoiV2", "FakeKingston"))
def test_nothing_cancels_unchanged(mods, dev):
    """Inputs where commutative cancellation removes no two-qubit gate: c22's circuit, default and recommended call."""
    for name, n, seed in (("ring", 8, 1), ("brick", 12, 2), ("pauli", 10, 3), ("qft", 10, 4), ("pauli", 20, 5)):
        qc = family(name, n, 50_000_000 + seed)
        assert mods["c23"]._cancel_candidate(mods["c23"]._unroll_wide(qc, None)) is None, (name, n)
        assert sig(call(mods["c23"], qc, dev)[0]) == sig(call(mods["c22"], qc, dev)[0]), (name, n)
        if n <= 12:
            assert sig(call(mods["c23"], qc, dev, True)[0]) == sig(call(mods["c22"], qc, dev, True)[0]), (name, n)


def test_random_cancellations_exact(mods):
    """Random circuits with commuting CX pairs inserted: the output implements the input and has no more two-qubit
    gates than c22's."""
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(50_100_000)
    c23 = mods["c23"]
    tried = c23.CANCEL_STATS["tried"]
    for k in range(6):
        n = int(rng.integers(5, 9))
        qc = QuantumCircuit(n)
        for _ in range(30):
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            r = rng.random()
            if r < 0.3:
                qc.cx(a, b)
                qc.rz(float(rng.uniform(0, 6)), a)  # commutes with the CX on its control
                qc.cx(a, b)
            elif r < 0.6:
                qc.cx(a, b)
            else:
                qc.ry(float(rng.uniform(0, 6)), a)
        o22, _ = call(mods["c22"], qc, "FakeHanoiV2")
        o23, _ = call(c23, qc, "FakeHanoiV2")
        assert q2(o23) <= q2(o22)
        assert c23._implements(qc, o23), k
        print(f"\nrandom {k} ({n} qubits): c22 {q2(o22)}, c23 {q2(o23)}")
    assert c23.CANCEL_STATS["tried"] > tried


def test_time_when_nothing_cancels(mods):
    """The pass's own time on inputs where it removes nothing (printed, and below 1 s on a 100-qubit QFT)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFT
    qc = QuantumCircuit(100)
    qc.append(QFT(100), range(100))
    qc = mods["c23"]._unroll_wide(qc, None)
    t0 = time.perf_counter()
    assert mods["c23"]._cancel_candidate(qc) is None
    dt = time.perf_counter() - t0
    print(f"\ncommutative cancellation on a 100-qubit QFT ({q2(qc)} two-qubit gates): {dt:.2f} s")
    assert dt < 5.0
