"""Tests for candidate psf_compile 2026-10-05.c14 (item 41). Run in this folder (PYTHONPATH with the Rust core):
    python -m pytest test_c14_direction.py -q
C13_PATH: candidate c13 (default ../c13/psf_compile.py)."""
import contextlib, importlib.util, io, os, sys, warnings
import numpy as np
import pytest
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from qiskit import QuantumCircuit  # noqa: E402
from qiskit_ibm_runtime import fake_provider as fp  # noqa: E402
import readout_eval as RE  # noqa: E402


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m
    s.loader.exec_module(m); return m


P14 = load(os.path.join(HERE, "psf_compile.py"), "t_c14")
P13 = load(os.environ.get("C13_PATH", os.path.join(HERE, "..", "c13", "psf_compile.py")), "t_c13")
DEV = {d: getattr(fp, d)() for d in ("FakeHanoiV2", "FakeGeneva", "FakeTorino", "FakeAuckland")}


def call(P, qc, dev, **kw):
    t = DEV[dev].target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return P.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0, target=t, placement_refine=True, **kw)


def one_way(t):
    for (a, b), p in t["cx"].items():
        r = t["cx"].get((b, a))
        if p is not None and p.error is not None and p.error >= 0.5 and r is not None and r.error is not None and r.error < 0.5:
            return a, b
    raise AssertionError("no one-way failed coupler")


def test_versions():
    assert P14.VERSION == "2026-10-05.c14" and P13.VERSION == "2026-10-05.c13"


def test_uses_failed_is_direction_aware():
    t = DEV["FakeHanoiV2"].target
    a, b = one_way(t)                      # (a, b) failed, (b, a) healthy
    edges, qubits = P14._failed_elements(t, 0.5)
    ok = QuantumCircuit(t.num_qubits); ok.cx(b, a)
    bad = QuantumCircuit(t.num_qubits); bad.cx(a, b)
    sym = QuantumCircuit(t.num_qubits); sym.cz(b, a)
    assert not P14._uses_failed(ok, edges, qubits) and P13._uses_failed(ok, edges, qubits)
    assert P14._uses_failed(bad, edges, qubits)
    assert P14._uses_failed(sym, edges, qubits)


def test_failed_qubit_still_flagged():
    t = DEV["FakeTorino"].target
    edges, qubits = P14._failed_elements(t, 0.5)
    e = next(iter(edges))
    c = QuantumCircuit(t.num_qubits); c.cz(*e)
    assert P14._uses_failed(c, edges, qubits)


def ring(n, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            qc.ry(float(rng.uniform(-1, 1)), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    return qc


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeAuckland"])
def test_unchanged_where_failures_are_two_way(dev):
    for n, seed in ((6, 1), (4, 2), (6, 3)):
        qc = ring(n, seed)
        for kw in ({}, dict(final_resynthesis="select", compare_level3=True, compare_floor=True, candidate_score="hybrid")):
            assert RE.sig(call(P14, qc, dev, **kw)) == RE.sig(call(P13, qc, dev, **kw))


@pytest.mark.parametrize("dev", ["FakeHanoiV2", "FakeGeneva"])
def test_outputs_exact_and_free_of_failed_directions(dev):
    t = DEV[dev].target
    edges, qubits = P14._failed_elements(t, 0.5)
    for n, seed in ((6, 4), (4, 5), (6, 6)):
        qc = ring(n, seed)
        out = call(P14, qc, dev, final_resynthesis="select", compare_level3=True, compare_floor=True, candidate_score="hybrid")
        assert not P14._uses_failed(out, edges, qubits)
        assert RE.state_infid(qc, out) <= 1e-6
