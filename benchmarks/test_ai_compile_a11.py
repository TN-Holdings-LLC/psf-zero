"""Tests for the AI front end psf_ai_compile 2026-10-05.a11 (adopted 2026-10-06; kept as benchmarks/psf_ai_compile_a11.py
since a12's adoption on 2026-10-06; items
15-16: readout in the state-aware estimate, gate direction kept on directional devices), on the current release. The
candidate's own tests are patches/psf_ai_compile_a11_2026-10-05/test_a11.py; a9 is benchmarks/psf_ai_compile_a9.py.

Run from the repository root:  python -m pytest benchmarks/test_ai_compile_a11.py -q
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
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
for p in (os.path.join(WORK, "readout"), os.path.join(WORK, "depth1"), os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_a11_release_test")
    sys.modules["psf_smart_layout"] = lay
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    a9 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a9.py"), "psf_ai_compile_a9_frozen_a11_test")
    a11 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a11.py"), "psf_ai_compile")  # a11, frozen at a12's adoption
    cand = H.load_module(os.path.join(REPO, "patches", "psf_ai_compile_a11_2026-10-05", "psf_ai_compile.py"),
                         "psf_ai_compile_a11_candidate_release_test")
    import readout_eval as RE
    return dict(rel=rel, a9=a9, a11=a11, cand=cand, RE=RE)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def comp(A, qc, dev):
    t = backend(dev).target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return A.compile_for_model_circuit(qc, t.build_coupling_map(), basis, target=t)


def circs():
    from qiskit import QuantumCircuit
    out = []
    rng = np.random.default_rng(61)
    for n in (3, 4, 5):
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.ry(float(rng.uniform(-1, 1)), i)
        qc.cx(n - 1, 0)
        out.append(qc)
    return out


def code(path):
    """The module's code without its docstring and version line (the only lines changed at adoption)."""
    t = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    body = t.split('"""', 2)[2]
    return [ln for ln in body.split("\n") if not ln.startswith("AI_COMPILE_VERSION")]


def test_versions(mods):
    assert mods["a11"].AI_COMPILE_VERSION == "2026-10-05.a11"
    assert mods["a9"].AI_COMPILE_VERSION == "2026-10-05.a9"
    assert mods["rel"].VERSION == "2026-10-11.1"
    assert mods["a11"].pc is mods["rel"] and mods["a9"].pc is mods["rel"]


def test_adopted_file_is_the_candidate(mods):
    assert code(os.path.join(REPO, "benchmarks", "psf_ai_compile_a11.py")) == \
        code(os.path.join(REPO, "patches", "psf_ai_compile_a11_2026-10-05", "psf_ai_compile.py"))


@pytest.mark.parametrize("dev", ["FakeBrussels", "FakeOsaka"])
def test_ecr_outputs_on_target_and_exact(mods, dev):
    a11, RE = mods["a11"], mods["RE"]
    t = backend(dev).target
    for qc0 in circs():
        for m in (False, True):
            qc = qc0.copy()
            if m:
                qc.measure_all()
            out = comp(a11, qc, dev)
            assert a11._off_target_2q(out, t) == 0
            assert RE.state_infid(qc0, RE.strip_measure(out)) <= 1e-6


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeAuckland"])
def test_unmeasured_is_a9(mods, dev):
    """Without measurements, on devices whose couplers work both ways, a11 returns a9's circuit."""
    RE = mods["RE"]
    for qc0 in circs():
        assert RE.sig(comp(mods["a11"], qc0, dev)) == RE.sig(comp(mods["a9"], qc0, dev))


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston"])
def test_measured_outputs_exact(mods, dev):
    a11, RE = mods["a11"], mods["RE"]
    for qc0 in circs():
        qc = qc0.copy()
        qc.measure_all()
        assert RE.state_infid(qc0, RE.strip_measure(comp(a11, qc, dev))) <= 1e-6
