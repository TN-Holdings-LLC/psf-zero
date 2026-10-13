"""Tests for candidate psf_ai_compile 2026-10-13.a14 (items 19-22: device-aware candidates, idle term, duration-diverse
candidates, failed-element guarantee), on release 2026-10-11.4.

Run from the repository root:  python -m pytest patches/psf_ai_compile_a14_2026-10-13/test_a14.py -q
"""
import contextlib
import copy
import io
import os
import sys
import warnings

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
    a12 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
    a14 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a14")
    return dict(rel=rel, a12=a12, a14=a14)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def basis_of(t):
    return [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]


def comp(A, qc, t, with_target=True):
    with contextlib.redirect_stdout(io.StringIO()):
        return A.compile_for_model_circuit(qc, t.build_coupling_map(), basis_of(t), target=t if with_target else None)


def circs():
    import test_ai_compile_a12 as T
    return T.circs()


def test_versions(mods):
    assert mods["a14"].AI_COMPILE_VERSION == "2026-10-13.a14"
    assert mods["a12"].AI_COMPILE_VERSION == "2026-10-06.a12"
    assert mods["rel"].VERSION == "2026-10-11.4"
    assert mods["a14"].pc is mods["rel"]


def test_without_target_same_as_a12(mods):
    import test_ai_compile_a12 as T
    t = backend("FakeKingston").target
    for qc in circs():
        assert T.full_sig(comp(mods["a14"], qc, t, False)) == T.full_sig(comp(mods["a12"], qc, t, False))


def test_idle_rate_terms(mods):
    f = mods["a14"].idle_loss_rate_terms
    assert f(0.0, 1.0) == (0.0, 0.0)          # |0>: nothing
    assert f(0.0, -1.0) == (1.0, 0.0)         # |1>: t/T1
    a, b = f(0.25, 0.0)                       # |+>: t/(4 T1) + t/T_phi * 1/2 = t/(2 T2)
    assert abs(a - 0.25) < 1e-15 and abs(b - 0.5) < 1e-15


def test_idle_schedule_and_cost(mods):
    from qiskit import QuantumCircuit
    a14 = mods["a14"]
    t = backend("FakeKingston").target
    qc = QuantumCircuit(2)
    qc.x(0)
    for _ in range(10):
        qc.sx(1)
    iw, T = a14.idle_weights(qc, t)
    d0, d1 = t["x"][(0,)].duration, t["sx"][(1,)].duration
    assert abs(T - 10 * d1) < 1e-15
    assert abs(iw[0][0] - (T - d0)) < 1e-15 and abs(iw[0][1]) < 1e-15   # |1> waits T - d0
    assert all(abs(v) < 1e-12 for v in iw[1])                             # q1 never waits
    t1 = t.qubit_properties[0].t1
    assert abs(a14.idle_cost(iw, t, {0: 0, 1: 1}) - (T - d0) / t1) < 1e-15
    untouched = QuantumCircuit(2)
    untouched.sx(1)
    iw2, _ = a14.idle_weights(untouched, t)
    assert 0 not in iw2


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston", "FakeBrussels"])
def test_exact_on_target_no_failed(mods, dev):
    rel, a14 = mods["rel"], mods["a14"]
    t = backend(dev).target
    for qc in circs():
        out = comp(a14, qc, t)
        assert rel._implements(qc, out)
        assert not a14._uses_failed_target(out, t)
        assert a14._off_target_2q(out, t) == 0


def test_estimate_not_worse_than_a12(mods):
    a12, a14 = mods["a12"], mods["a14"]
    t = backend("FakeKingston").target
    for qc in circs():
        e14 = a14.estimate_a14(comp(a14, qc, t), t)
        e12 = a14.estimate_a14(comp(a12, qc, t), t)
        assert e14 <= e12 * 1.01 + 1e-12


def test_failed_coupler_avoided(mods):
    from qiskit.transpiler import InstructionProperties
    a14 = mods["a14"]
    t = copy.deepcopy(backend("FakeKingston").target)
    qc = circs()[4]  # 5 qubits
    out = comp(a14, qc, t)
    pos = {q: i for i, q in enumerate(out.qubits)}
    used = [tuple(pos[q] for q in i.qubits) for i in out.data if len(i.qubits) == 2]
    a, b = used[0]
    for qa in ((a, b), (b, a)):
        if qa in t["cz"]:
            t.update_instruction_properties("cz", qa, InstructionProperties(duration=t["cz"][qa].duration, error=1.0))
    out2 = comp(a14, qc, t)
    assert not a14._uses_failed_target(out2, t)
    assert mods["rel"]._implements(qc, out2)
