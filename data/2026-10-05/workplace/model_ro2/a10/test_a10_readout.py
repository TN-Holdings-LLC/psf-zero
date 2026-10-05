"""Tests for candidate psf_ai_compile 2026-10-05.a10 (item 15). Run in this folder (PYTHONPATH with the Rust core):
    python -m pytest test_a10_readout.py -q
psf_compile here is candidate c13; psf_ai_compile_a9.py is the a9 of exact_stage_v1."""
import contextlib, importlib.util, io, os, sys, warnings
import numpy as np
import pytest
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import psf_compile as pc  # noqa: E402,F401  (c13)
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit_ibm_runtime import fake_provider as fp  # noqa: E402


def load(p, n):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, p))
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m


A9, A10 = load("psf_ai_compile_a9.py", "t_a9"), load("psf_ai_compile.py", "t_a10")
DEV = {d: getattr(fp, d)() for d in ("FakeTorino", "FakeKingston", "FakeAuckland")}


def comp(A, qc, dev):
    t = DEV[dev].target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return A.compile_for_model_circuit(qc, t.build_coupling_map(), basis, target=t)


def ghz(n, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n); qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i in range(n):
        qc.ry(float(rng.uniform(-1, 1)), i)
    return qc


def w3(seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(3); qc.ry(2 * np.arccos(1 / np.sqrt(3)), 0); qc.ch(0, 1); qc.cx(1, 2); qc.cx(0, 1); qc.x(0)
    qc.rz(float(rng.uniform(-1, 1)), 2)
    return qc


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [round(float(p), 12) for p in i.operation.params])
            for i in c.data]


def meas_err(out, t):
    return sum(t["measure"][(out.find_bit(i.qubits[0]).index,)].error for i in out.data if i.operation.name == "measure")


def test_version():
    assert A10.AI_COMPILE_VERSION == "2026-10-05.a10" and A9.AI_COMPILE_VERSION == "2026-10-05.a9"
    assert pc.VERSION == "2026-10-05.c13"


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeAuckland"])
def test_unmeasured_identical_to_a9(dev):
    for qc in (ghz(4, 1), w3(2), ghz(5, 3)):
        assert sig(comp(A10, qc, dev)) == sig(comp(A9, qc, dev))


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston"])
def test_estimate_adds_measure_errors(dev):
    t = DEV[dev].target
    qc = ghz(4, 4); qc.measure_all()
    out = comp(A9, qc, dev)
    d = A10.state_aware_cost(out, t) - A9.state_aware_cost(out, t)
    assert abs(d - meas_err(out, t)) < 1e-12


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston", "FakeAuckland"])
def test_measured_output_exact_and_measures_final_positions(dev):
    qc = ghz(4, 5)
    m = qc.copy(); m.measure_all()
    out = comp(A10, m, dev)
    fin = list(out.layout.final_index_layout(filter_ancillas=True))
    for ins in out.data:
        if ins.operation.name == "measure":
            assert out.find_bit(ins.qubits[0]).index == fin[out.find_bit(ins.clbits[0]).index]
    # noiseless distribution of the measured qubits equals the ideal one
    act = sorted({out.find_bit(b).index for i in out.data for b in i.qubits})
    idx = {p: k for k, p in enumerate(act)}
    red = QuantumCircuit(len(act))
    for ins in out.data:
        if ins.operation.name not in ("measure", "barrier"):
            red.append(ins.operation, [idx[out.find_bit(b).index] for b in ins.qubits])
    pr = Statevector(red).probabilities_dict(qargs=[idx[p] for p in fin])
    ideal = Statevector(qc).probabilities_dict()
    keys = set(pr) | set(ideal)
    assert sum(abs(pr.get(k, 0) - ideal.get(k, 0)) for k in keys) / 2 <= 1e-6


def test_readout_never_worse_and_better_somewhere():
    t = DEV["FakeTorino"].target
    better = 0
    for qc in (ghz(3, 11), ghz(4, 12), ghz(5, 13), w3(14)):
        m = qc.copy(); m.measure_all()
        e9, e10 = meas_err(comp(A9, m, "FakeTorino"), t), meas_err(comp(A10, m, "FakeTorino"), t)
        assert e10 <= e9 + 1e-12
        better += e10 < e9 - 1e-12
    assert better >= 1
