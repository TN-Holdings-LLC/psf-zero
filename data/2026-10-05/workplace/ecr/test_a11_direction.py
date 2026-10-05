"""Tests for candidate psf_ai_compile 2026-10-05.a11 (item 16). Run in this folder (PYTHONPATH with the Rust core):
    python -m pytest test_a11_direction.py -q
psf_compile here is candidate c13; psf_ai_compile_a10.py is a10."""
import contextlib, importlib.util, io, os, sys, warnings
import numpy as np
import pytest
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import psf_compile as pc  # noqa: E402  (c13)
import readout_eval as RE  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit_ibm_runtime import fake_provider as fp  # noqa: E402


def load(p, n):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, p))
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m


A10, A11 = load("psf_ai_compile_a10.py", "t_a10"), load("psf_ai_compile.py", "t_a11")
DEV = {d: getattr(fp, d)() for d in ("FakeBrussels", "FakeOsaka", "FakeTorino", "FakeAuckland")}


def comp(A, qc, dev):
    t = DEV[dev].target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return A.compile_for_model_circuit(qc, t.build_coupling_map(), basis, target=t)


def circs():
    out = []
    rng = np.random.default_rng(16)
    for n in (3, 4, 5):
        qc = QuantumCircuit(n); qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.ry(float(rng.uniform(-1, 1)), i)
        qc.cx(n - 1, 0)
        out.append(qc)
    return out


def test_version():
    assert A11.AI_COMPILE_VERSION == "2026-10-05.a11" and A10.AI_COMPILE_VERSION == "2026-10-05.a10"


def test_estimate_rejects_unsupported_direction():
    t = DEV["FakeBrussels"].target
    (a, b) = next(iter(q for q in t["ecr"] if q is not None and tuple(q[::-1]) not in t["ecr"]))
    c = QuantumCircuit(t.num_qubits); c.append(t.operation_from_name("ecr"), [b, a])
    assert A11.state_aware_cost(c, t) == float("inf")
    c2 = QuantumCircuit(t.num_qubits); c2.append(t.operation_from_name("ecr"), [a, b])
    assert A11.state_aware_cost(c2, t) == A10.state_aware_cost(c2, t)


@pytest.mark.parametrize("dev", ["FakeBrussels", "FakeOsaka"])
def test_ecr_outputs_on_target_and_exact(dev):
    t = DEV[dev].target
    for qc0 in circs():
        for m in (False, True):
            qc = qc0.copy()
            if m:
                qc.measure_all()
            out = comp(A11, qc, dev)
            assert A11._off_target_2q(out, t) == 0
            assert RE.state_infid(qc0, RE.strip_measure(out)) <= 1e-6


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeAuckland"])
def test_bidirectional_devices_unchanged(dev):
    for qc0 in circs():
        qc = qc0.copy(); qc.measure_all()
        assert RE.sig(comp(A11, qc, dev)) == RE.sig(comp(A10, qc, dev))


def test_backstop_fixes_a_reversed_gate():
    t = DEV["FakeBrussels"].target
    (a, b) = next(iter(q for q in t["ecr"] if q is not None and tuple(q[::-1]) not in t["ecr"]))
    c = QuantumCircuit(t.num_qubits); c.sx(a); c.append(t.operation_from_name("ecr"), [b, a]); c.rz(0.3, b)
    called = []
    out = A11._keep_direction(c, t, lambda: called.append(1))
    assert not called and A11._off_target_2q(out, t) == 0 and pc._same_action(c, out)
