"""Tests for candidate psf_ai_compile 2026-10-02.a6 (a5 with release psf_compile 2026-10-02.2's exact re-placement
inside every compile).

Run from the repository root:  python -m pytest patches/psf_ai_compile_a6_2026-10-02/test_ai6.py -q
"""
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
sys.path.insert(0, REPO)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_ai6_test")
    sys.modules["psf_smart_layout"] = lay
    a5 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
    a6 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a6_test")
    prev = H.load_module(os.path.join(REPO, "patches", "psf_compile_c3_2026-10-02", "psf_compile.py"),
                         "psf_compile_prev_ai6_test")
    return dict(rel=rel, a5=a5, a6=a6, prev=prev)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def ring(n, layers=2, seed=0):
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
            qc.rz(float(rng.uniform(-math.pi, math.pi)), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    return qc


def model_like(seed=0):
    """A small circuit as a model writes it: a SWAP, a redundant CX pair and a 2-qubit unitary."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.swap(1, 2)
    qc.cx(2, 3)
    qc.cx(2, 3)
    qc.cx(2, 3)
    qc.append(UnitaryGate(random_unitary(4, seed=seed)), [0, 3])
    return qc


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [float(p) for p in i.operation.params])
            for i in c.data]


def compact_fidelity(qc, out):
    """|<ref|out>|^2 on the touched physical qubits only (as in test_c3_prune.py)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    n = qc.num_qubits
    fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
    used = sorted({out.find_bit(q).index for ins in out.data for q in ins.qubits} | set(fin))
    pos = {p: i for i, p in enumerate(used)}
    comp = QuantumCircuit(len(used))
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure"):
            continue
        comp.append(ins.operation, [pos[out.find_bit(q).index] for q in ins.qubits])
    ref = QuantumCircuit(len(used))
    ref.compose(qc, qubits=[pos[p] for p in fin], inplace=True)
    return abs(Statevector(ref).inner(Statevector(comp))) ** 2


def nat(tgt):
    return [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]


def test_version(mods):
    assert mods["a6"].AI_COMPILE_VERSION == "2026-10-02.a6"
    assert mods["rel"].VERSION == "2026-10-06.4"  # current release (2026-10-02.2 when this candidate was evaluated)


def test_without_target_identical_to_a5(mods):
    for name in ("FakeTorino", "FakeAuckland"):
        tgt = backend(name).target
        cm = tgt.build_coupling_map()
        for qc in (ring(4, seed=5), model_like(seed=6)):
            assert sig(mods["a6"].compile_for_model_circuit(qc, cm, nat(tgt))) == \
                sig(mods["a5"].compile_for_model_circuit(qc, cm, nat(tgt))), name


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeTorino", "FakeKingston"])
def test_with_target_exact_and_avoids_failed(mods, name):
    a6, rel = mods["a6"], mods["rel"]
    tgt = backend(name).target
    cm = tgt.build_coupling_map()
    edges, qubits = rel._failed_elements(tgt, 0.5)
    for qc in (ring(4, seed=1), ring(6, seed=2), model_like(seed=3)):
        for sa in (True, False):
            out = a6.compile_for_model_circuit(qc, cm, nat(tgt), target=tgt, state_aware_placement=sa)
            assert compact_fidelity(qc, out) > 1 - 1e-6, (name, sa)
            assert not rel._uses_failed(out, edges, qubits), (name, sa)


def test_needs_release_with_placement_refine(mods, monkeypatch):
    tgt = backend("FakeTorino").target
    monkeypatch.setattr(mods["a6"], "pc", mods["prev"])
    with pytest.raises(RuntimeError):
        mods["a6"].compile_for_model_circuit(ring(4), tgt.build_coupling_map(), nat(tgt), target=tgt)
