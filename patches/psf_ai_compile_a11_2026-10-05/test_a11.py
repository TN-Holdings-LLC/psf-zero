"""Tests for candidate psf_ai_compile 2026-10-05.a11 (workplace; items 15 and 16) on top of the adopted front end a9,
adapted from the workplace test test_a11_direction.py (data/2026-10-05/workplace/ecr/). The release underneath is
candidate psf_compile 2026-10-05.c14, registered as `psf_compile` before the front ends are loaded, as the workplace
ran it and as it would be if both are adopted. a10 is data/2026-10-05/workplace/model_ro2/a10/psf_ai_compile.py.

Run from the repository root:  python -m pytest patches/psf_ai_compile_a11_2026-10-05/test_a11.py -q
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
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
for p in (os.path.join(WORK, "readout"), os.path.join(WORK, "depth1"), os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_a11_test")
    sys.modules["psf_smart_layout"] = lay
    c14 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c14_2026-10-05", "psf_compile.py"), "psf_compile")
    a9 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a9.py"), "psf_ai_compile_a9_for_a11_test")  # a9 (frozen at a11's adoption)
    a10 = H.load_module(os.path.join(WORK, "model_ro2", "a10", "psf_ai_compile.py"), "psf_ai_compile_a10_for_a11_test")
    a11 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a11_test")
    import readout_eval as RE
    return dict(c14=c14, a9=a9, a10=a10, a11=a11, RE=RE)


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
    rng = np.random.default_rng(16)
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


def one_way_ecr(t):
    return next(iter(q for q in t["ecr"] if q is not None and tuple(q[::-1]) not in t["ecr"]))


def test_versions(mods):
    assert mods["a11"].AI_COMPILE_VERSION == "2026-10-05.a11"
    assert mods["a10"].AI_COMPILE_VERSION == "2026-10-05.a10"
    assert mods["a9"].AI_COMPILE_VERSION == "2026-10-05.a9"
    assert mods["c14"].VERSION == "2026-10-05.c14"
    assert mods["a11"].pc is mods["c14"] and mods["a9"].pc is mods["c14"]


def test_estimate_rejects_unsupported_direction(mods):
    from qiskit import QuantumCircuit
    a10, a11 = mods["a10"], mods["a11"]
    t = backend("FakeBrussels").target
    a, b = one_way_ecr(t)
    c = QuantumCircuit(t.num_qubits)
    c.append(t.operation_from_name("ecr"), [b, a])
    assert a11.state_aware_cost(c, t) == float("inf")
    c2 = QuantumCircuit(t.num_qubits)
    c2.append(t.operation_from_name("ecr"), [a, b])
    assert a11.state_aware_cost(c2, t) == a10.state_aware_cost(c2, t)


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
def test_bidirectional_devices(mods, dev):
    """With measurements a11 is a10 (item 16 changes nothing where every direction exists); without measurements it
    is a9 (item 15 changes nothing without measurements)."""
    RE = mods["RE"]
    for qc0 in circs():
        qc = qc0.copy()
        qc.measure_all()
        assert RE.sig(comp(mods["a11"], qc, dev)) == RE.sig(comp(mods["a10"], qc, dev))
        assert RE.sig(comp(mods["a11"], qc0, dev)) == RE.sig(comp(mods["a9"], qc0, dev))


def test_backstop_fixes_a_reversed_gate(mods):
    from qiskit import QuantumCircuit
    a11, c14 = mods["a11"], mods["c14"]
    t = backend("FakeBrussels").target
    a, b = one_way_ecr(t)
    c = QuantumCircuit(t.num_qubits)
    c.sx(a)
    c.append(t.operation_from_name("ecr"), [b, a])
    c.rz(0.3, b)
    called = []
    out = a11._keep_direction(c, t, lambda: called.append(1))
    assert not called and a11._off_target_2q(out, t) == 0 and c14._same_action(c, out)


def test_a9_defect_on_ecr_reproduces(mods):
    """The defect a11 fixes: a9 returns ecr gates in an unsupported direction on FakeBrussels (workplace exploration:
    224 over 56 circuits). At least one of these small circuits shows it."""
    t = backend("FakeBrussels").target
    off = 0
    for qc0 in circs():
        qc = qc0.copy()
        qc.measure_all()
        off += mods["a11"]._off_target_2q(comp(mods["a9"], qc, "FakeBrussels"), t)
    assert off >= 1
