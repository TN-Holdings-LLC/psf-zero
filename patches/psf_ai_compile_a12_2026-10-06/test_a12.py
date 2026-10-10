"""Tests for candidate psf_ai_compile 2026-10-06.a12 (changelog item 17: faster re-placement, same result) against the
adopted front end a11 (benchmarks/psf_ai_compile_a11.py since a12's adoption), on the current release.

Run from the repository root:  python -m pytest patches/psf_ai_compile_a12_2026-10-06/test_a12.py -q
"""
import contextlib
import io
import os
import random
import sys
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    a11 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a11.py"), "psf_ai_compile_a11_for_a12_test")  # a11 (frozen at a12's adoption)
    a12 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a12_test")
    return dict(rel=rel, a11=a11, a12=a12)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def comp(A, qc, dev):
    t = backend(dev).target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return A.compile_for_model_circuit(qc, t.build_coupling_map(), basis, target=t)


def full_sig(c):
    """Instructions (name, qubits, clbits, parameters), global phase, and where the logical qubits start and end.
    The device positions given to ancilla qubits in the layout are left out: they do not change the circuit."""
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def circs(measured):
    from qiskit import QuantumCircuit
    out = []
    rng = np.random.default_rng(17)
    for n in (3, 4, 5, 6):
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.ry(float(rng.uniform(-1, 1)), i)
        qc.cx(n - 1, 0)
        if measured:
            qc.measure_all()
        out.append(qc)
    return out


def test_versions(mods):
    assert mods["a12"].AI_COMPILE_VERSION == "2026-10-06.a12"
    assert mods["a11"].AI_COMPILE_VERSION == "2026-10-05.a11"
    assert mods["rel"].VERSION == "2026-10-11.1"


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeBrussels"])
def test_cached_score_is_the_same_float(mods, dev):
    """On random placements of compiled circuits the cached score equals `_score_weights` exactly (inf included)."""
    a12 = mods["a12"]
    t = backend(dev).target
    rng = random.Random(1)
    for qc in circs(True):
        out = comp(mods["a11"], qc, dev)
        w = a12._state_weights(out)
        used = sorted({out.find_bit(q).index for i in out.data for q in i.qubits})
        cache = {}
        for _ in range(50):
            phys = rng.sample(range(t.num_qubits), len(used))
            mp = dict(zip(used, phys))
            assert a12._score_weights_cached(w, t, mp, cache) == a12._score_weights(w, t, mp)


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston", "FakeAuckland", "FakeBrussels"])
def test_same_circuit_as_a11(mods, dev):
    for m in (False, True):
        for qc in circs(m):
            assert full_sig(comp(mods["a12"], qc, dev)) == full_sig(comp(mods["a11"], qc, dev))
