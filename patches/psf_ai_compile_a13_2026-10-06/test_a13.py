"""Tests for candidate psf_ai_compile 2026-10-06.a13 (changelog item 18: the release's whole recommended call above
SMALL_MAX_QUBITS) against the adopted front end a12 (benchmarks/psf_ai_compile.py), on the current release.

Run from the repository root:  python -m pytest patches/psf_ai_compile_a13_2026-10-06/test_a13.py -q
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

RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    a12 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile_a12_for_a13_test")
    a13 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a13_test")
    return dict(rel=rel, a12=a12, a13=a13)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def comp(A, qc, dev):
    t = backend(dev).target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return A.compile_for_model_circuit(qc, t.build_coupling_map(), basis, target=t)


def release_call(rel, qc, dev):
    t = backend(dev).target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return rel.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                        entangling_basis="cx", layout_search=True, seed_transpiler=0, target=t,
                                        **RECOMMENDED)


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def ring(n, seed, measured=False):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            qc.ry(float(rng.uniform(-1, 1)), q)
        for q in range(n - 1):
            qc.cz(q, q + 1)
    if measured:
        qc.measure_all()
    return qc


def state_infid(qc, out):
    """1 - |<psi|phi>|^2 on the touched qubits, measurements removed, the other touched qubits back in |0>."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
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
    assert mods["a13"].AI_COMPILE_VERSION == "2026-10-06.a13"
    assert mods["a12"].AI_COMPILE_VERSION == "2026-10-06.a12"
    assert mods["rel"].VERSION == "2026-10-06.2"
    assert mods["a13"].FAST_PATH_RECOMMENDED == RECOMMENDED


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeAuckland"])
def test_small_circuits_unchanged(mods, dev):
    for qc in (ring(4, 1), ring(6, 2, True), ring(8, 3, True)):
        assert sig(comp(mods["a13"], qc, dev)) == sig(comp(mods["a12"], qc, dev))


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston"])
def test_large_circuits_are_the_recommended_call(mods, dev):
    for qc in (ring(9, 4, True), ring(10, 5)):
        out = comp(mods["a13"], qc, dev)
        assert sig(out) == sig(release_call(mods["rel"], qc, dev))
        assert state_infid(qc, out) <= 1e-6
