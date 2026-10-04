"""Tests for release psf_compile 2026-10-02.1 (changelog item 31: avoidance of failed couplers and qubits), adapted
from the candidate's tests (patches/psf_compile_c3_2026-10-02/test_c3_prune.py). The candidate's test also checks
that output without `target` equals the previous release gate for gate.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_02.py -q
"""
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
sys.path.insert(0, REPO)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_c3_test")
    sys.modules["psf_smart_layout"] = lay
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_1002_test")
    return rel, rel


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


def test_version(mods):
    assert mods[0].VERSION == "2026-10-04.1"  # current release (this file was written for 2026-10-02.1)


def test_prune_keeps_size_and_removes_exactly_failed(mods):
    c3 = mods[0]
    for name in ("FakeAuckland", "FakeTorino", "FakeKingston"):
        tgt = backend(name).target
        cm = tgt.build_coupling_map()
        g = next(x for x in ("cz", "ecr", "cx") if x in tgt.operation_names)
        bad_q = {q for q, e in c3.qubit_errors_from_target(tgt).items() if e >= 0.5}
        out = c3.prune_coupling_map(cm, tgt)
        assert out.size() == cm.size()
        kept, orig = set(out.get_edges()), set(cm.get_edges())
        assert kept <= orig
        expect = {e for e in orig if not (e in failed_edges(tgt, g) or e[::-1] in failed_edges(tgt, g)
                                          or e[0] in bad_q or e[1] in bad_q)}
        assert kept == expect, name
    assert set(c3.prune_coupling_map(backend("FakeAuckland").target.build_coupling_map(),
                                     backend("FakeAuckland").target).get_edges()) == \
        set(backend("FakeAuckland").target.build_coupling_map().get_edges())


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


def test_torino_ring_avoids_failed_edges_and_is_exact(mods):
    c3 = mods[0]
    tgt = backend("FakeTorino").target
    cm = tgt.build_coupling_map()
    bad = failed_edges(tgt, "cz")
    before = c3.PRUNE_STATS["recompiled"]
    for n in (4, 6):
        qc = ring(n)
        out = c3.compile_for_hardware(qc, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"], entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0, target=tgt)
        used = {tuple(out.find_bit(q).index for q in ins.qubits) for ins in out.data if ins.operation.num_qubits == 2}
        assert not any(e in bad or e[::-1] in bad for e in used)
        assert compact_fidelity(qc, out) > 1 - 1e-6
    assert c3.PRUNE_STATS["recompiled"] >= before


def test_unaffected_output_identical_with_target(mods):
    """Where the release uses no failed element, passing target must not change a single gate."""
    c3, rel = mods
    for name in ("FakeAuckland", "FakeTorino"):
        tgt = backend(name).target
        cm = tgt.build_coupling_map()
        edges, qubits = c3._failed_elements(tgt, 0.5)
        basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
        for n in (4, 5):
            qc = ring(n, seed=10 + n)
            kw = dict(coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True, seed_transpiler=0)
            b = rel.compile_for_hardware(qc, **kw)
            if c3._uses_failed(b, edges, qubits):
                continue
            a = c3.compile_for_hardware(qc, target=tgt, **kw)
            sig = lambda c: [(i.operation.name, [c.find_bit(q).index for q in i.qubits],
                              [float(p) for p in i.operation.params]) for i in c.data]
            assert sig(a) == sig(b), (name, n)


def test_without_target_identical_to_release(mods):
    c3, rel = mods
    cm = backend("FakeTorino").target.build_coupling_map()
    for n in (4, 6):
        qc = ring(n, seed=n)
        kw = dict(coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"], entangling_basis="cx", layout_search=True,
                  seed_transpiler=0)
        a, b = c3.compile_for_hardware(qc, **kw), rel.compile_for_hardware(qc, **kw)
        assert [(i.operation.name, [a.find_bit(q).index for q in i.qubits], [float(p) for p in i.operation.params])
                for i in a.data] == \
               [(i.operation.name, [b.find_bit(q).index for q in i.qubits], [float(p) for p in i.operation.params])
                for i in b.data]
