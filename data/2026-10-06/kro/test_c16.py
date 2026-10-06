"""Tests for candidate psf_compile 2026-10-06.c16 (changelog item 43: readout of measured qubits in `kraus_cost`). The
release is the current one, 2026-10-06.1 (psf_compile.py).

Run from the repository root:  python -m pytest patches/psf_compile_c16_2026-10-06/test_c16.py -q
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
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

FULL = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_c16_test")
    sys.modules["psf_smart_layout"] = lay
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c16_test")
    c16 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c16_test")
    return dict(rel=rel, c16=c16)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def call(P, qc, be, score):
    t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return P.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0, target=t, candidate_score=score, **FULL)


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [round(float(p), 12) for p in i.operation.params])
            for i in c.data]


def ring(n, seed, measured=False):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            qc.ry(float(rng.uniform(-1, 1)), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    if measured:
        qc.measure_all()
    return qc


def ghz(n, measured=False):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    if measured:
        qc.measure_all()
    return qc


def state_infid(qc, out):
    """1 - |<psi|phi>|^2 between qc's output and the compiled circuit's (measurements removed) on its final-layout
    qubits, other touched qubits back in |0>."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix, state_fidelity
    n = qc.num_qubits
    bare = qc.copy()
    bare.remove_final_measurements(inplace=True)
    fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
    act = sorted({out.find_bit(b).index for i in out.data for b in i.qubits} | set(fin))
    idx = {p: k for k, p in enumerate(act)}
    red = QuantumCircuit(len(act))
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        red.append(ins.operation, [idx[out.find_bit(b).index] for b in ins.qubits])
    ref = QuantumCircuit(len(act))
    ref.compose(bare, qubits=[idx[p] for p in fin], inplace=True)
    return float(1 - abs(Statevector(ref).inner(Statevector(red))) ** 2)


def test_versions(mods):
    assert mods["c16"].VERSION == "2026-10-06.c16"
    assert mods["rel"].VERSION == "2026-10-06.1"


@pytest.mark.parametrize("dev", ["FakeAlgiers", "FakeTorino"])
def test_kraus_cost_is_the_release_plus_readout(mods, dev):
    rel, c16 = mods["rel"], mods["c16"]
    be = backend(dev)
    for qc in (ghz(4, True), ring(5, 1, True), ring(4, 2)):
        out = call(rel, qc, be, "hybrid")
        want = rel.kraus_cost(out, be.target) + rel.readout_cost(out, be.target)
        assert abs(c16.kraus_cost(out, be.target) - want) <= 1e-12 * max(1.0, want)


@pytest.mark.parametrize("dev", ["FakeAuckland", "FakeKingston"])
def test_unmeasured_kraus_and_hybrid_unchanged(mods, dev):
    """Without measurements `kraus` gives the release's circuit; `hybrid` is the release's in every case."""
    be = backend(dev)
    for qc in (ghz(4), ring(5, 3)):
        assert sig(call(mods["c16"], qc, be, "kraus")) == sig(call(mods["rel"], qc, be, "kraus"))
    for qc in (ghz(4, True), ring(5, 4, True)):
        assert sig(call(mods["c16"], qc, be, "hybrid")) == sig(call(mods["rel"], qc, be, "hybrid"))


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeHanoiV2"])
def test_measured_kraus_outputs_exact_and_measured(mods, dev):
    be = backend(dev)
    for qc in (ghz(4, True), ring(5, 5, True), ring(6, 6, True)):
        out = call(mods["c16"], qc, be, "kraus")
        assert state_infid(qc, out) <= 1e-6
        assert sum(1 for i in out.data if i.operation.name == "measure") == qc.num_qubits


def test_readout_term_moves_a_choice(mods):
    """Somewhere among these measured circuits the readout term changes `kraus`'s choice; where it does, c16's circuit
    has the lower c16 estimate (both are among the same candidates)."""
    rel, c16 = mods["rel"], mods["c16"]
    be = backend("FakeTorino")
    diff = 0
    for seed in range(12):
        qc = ring(4, 10 + seed, True)
        a, b = call(rel, qc, be, "kraus"), call(c16, qc, be, "kraus")
        if sig(a) != sig(b):
            diff += 1
            assert c16.kraus_cost(b, be.target) <= c16.kraus_cost(a, be.target) + 1e-12
    assert diff >= 1


def test_unknown_score_rejected(mods):
    with pytest.raises(ValueError):
        call(mods["c16"], ring(4, 0), backend("FakeTorino"), "nope")
