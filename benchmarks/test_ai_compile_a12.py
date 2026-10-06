"""Tests for the AI front end psf_ai_compile 2026-10-06.a12 (adopted 2026-10-06; benchmarks/psf_ai_compile.py; item 17:
faster re-placement with the same result), on the current release. The candidate's own tests are
patches/psf_ai_compile_a12_2026-10-06/test_a12.py; a11 is benchmarks/psf_ai_compile_a11.py.

Run from the repository root:  python -m pytest benchmarks/test_ai_compile_a12.py -q
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
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    sys.path.insert(0, p)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    a11 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a11.py"), "psf_ai_compile_a11_frozen_a12_test")
    a12 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
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
    """Instructions, global phase, and where the logical qubits start and end (as SPEED compared them)."""
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def code(path):
    """The module's code without its docstring and version line (the only lines changed at adoption)."""
    t = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    return [ln for ln in t.split('"""', 2)[2].split("\n") if not ln.startswith("AI_COMPILE_VERSION")]


def circs():
    from qiskit import QuantumCircuit
    out = []
    rng = np.random.default_rng(1206)
    for n in (3, 4, 5):
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.ry(float(rng.uniform(-1, 1)), i)
        qc.cx(n - 1, 0)
        out.append(qc)
        m = qc.copy()
        m.measure_all()
        out.append(m)
    return out


def test_versions(mods):
    assert mods["a12"].AI_COMPILE_VERSION == "2026-10-06.a12"
    assert mods["a11"].AI_COMPILE_VERSION == "2026-10-05.a11"
    assert mods["rel"].VERSION == "2026-10-06.1"
    assert mods["a12"].pc is mods["rel"] and mods["a11"].pc is mods["rel"]


def test_adopted_file_is_the_candidate():
    assert code(os.path.join(REPO, "benchmarks", "psf_ai_compile.py")) == \
        code(os.path.join(REPO, "patches", "psf_ai_compile_a12_2026-10-06", "psf_ai_compile.py"))


@pytest.mark.parametrize("dev", ["FakeKingston", "FakeOsaka", "FakeHanoiV2"])
def test_same_circuit_as_a11(mods, dev):
    for qc in circs():
        assert full_sig(comp(mods["a12"], qc, dev)) == full_sig(comp(mods["a11"], qc, dev))
