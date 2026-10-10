"""Tests for candidate psf_compile 2026-10-02.c4 (changelog item 32: error-aware layout through the device Target).

Run from the repository root:  python -m pytest patches/psf_compile_c4_2026-10-02/test_c4_layout.py -q
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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_c4_test")
    sys.modules["psf_smart_layout"] = lay
    return (H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c4_test"),
            H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel1002_test"))


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def failed_edges(tgt, gate):
    return {tuple(q) for q, p in tgt[gate].items() if p is not None and p.error is not None and p.error >= 0.5}


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


def compact_fidelity(qc, out):
    """|<ref|out>|^2 on the touched physical qubits only (a full 133/156-qubit Operator is impossible):
    `ref` is the logical circuit placed on the final-layout positions of its qubits, ancillas in |0>."""
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


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [float(p) for p in i.operation.params])
            for i in c.data]


def test_version(mods):
    assert mods[0].VERSION == "2026-10-02.c4"
    assert mods[1].VERSION == "2026-10-10.1"  # current release (2026-10-02.1 when this candidate was evaluated)


def test_default_identical_to_release(mods):
    """error_aware_layout=False (default): identical to release 2026-10-02.1, with and without target."""
    c4, rel = mods
    tgt = backend("FakeTorino").target
    cm = tgt.build_coupling_map()
    for n in (4, 6):
        qc = ring(n, seed=20 + n)
        kw = dict(coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"], entangling_basis="cx", layout_search=True,
                  seed_transpiler=0)
        assert sig(c4.compile_for_hardware(qc, **kw)) == sig(rel.compile_for_hardware(qc, **kw))
        assert sig(c4.compile_for_hardware(qc, target=tgt, **kw)) == sig(rel.compile_for_hardware(qc, target=tgt, **kw))


def test_error_aware_exact_isa_and_no_failed(mods):
    c4 = mods[0]
    for name in ("FakeTorino", "FakeKingston"):
        tgt = backend(name).target
        cm = tgt.build_coupling_map()
        g = next(x for x in ("cz", "ecr", "cx") if x in tgt.operation_names)
        edges, qubits = c4._failed_elements(tgt, 0.5)
        basis = [x for x in ("cx", "cz", "rz", "sx", "x") if x in tgt.operation_names]
        for n in (4, 6):
            qc = ring(n, seed=30 + n)
            out = c4.compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx",
                                          layout_search=True, seed_transpiler=0, target=tgt, error_aware_layout=True)
            assert not c4._uses_failed(out, edges, qubits)
            assert compact_fidelity(qc, out) > 1 - 1e-6
            for ins in out.data:
                idx = tuple(out.find_bit(q).index for q in ins.qubits)
                assert tgt.instruction_supported(ins.operation.name, idx), (name, ins.operation.name, idx)


def test_error_aware_differs_only_when_asked(mods):
    """Without target, error_aware_layout has no effect (there is nothing to route against)."""
    c4, rel = mods
    cm = backend("FakeAuckland").target.build_coupling_map()
    qc = ring(4, seed=3)
    kw = dict(coupling_map=cm, basis_gates=["cx", "rz", "sx", "x"], entangling_basis="cx", layout_search=True,
              seed_transpiler=0)
    assert sig(c4.compile_for_hardware(qc, error_aware_layout=True, **kw)) == sig(rel.compile_for_hardware(qc, **kw))
