"""Tests for candidate psf_compile 2026-10-03.c6 (changelog item 34: floor-aware re-placement score). Helpers are copied
from benchmarks/test_release_2026_10_02_2.py.

Run from the repository root:  python -m pytest patches/psf_compile_c6_2026-10-03/test_c6_floor.py -q
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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_c6_test")
    sys.modules["psf_smart_layout"] = lay
    return (H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c6_test"),
            H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c6_test"))


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


def chain(n, seed=0):
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
    for q in range(n - 1):
        qc.cx(q, q + 1)
        qc.rz(float(rng.uniform(-math.pi, math.pi)), q + 1)
    return qc


def kw_for(tgt):
    return dict(coupling_map=tgt.build_coupling_map(),
                basis_gates=[g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names],
                entangling_basis="cx", layout_search=True, seed_transpiler=0)


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [float(p) for p in i.operation.params])
            for i in c.data]


def s_rep(out, tgt):
    """Sum of -log(1 - reported error) over the gates as placed."""
    s = 0.0
    for ins in out.data:
        q = tuple(out.find_bit(b).index for b in ins.qubits)
        name = ins.operation.name
        if name in ("barrier", "measure") or name not in tgt.operation_names:
            continue
        props = tgt[name].get(q) or tgt[name].get(q[::-1])
        e = props.error if props is not None and props.error is not None else 0.0
        s += -math.log(max(1.0 - e, 1e-300))
    return s


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


def test_version(mods):
    assert mods[0].VERSION == "2026-10-03.c6"
    assert mods[1].VERSION == "2026-10-10.2"  # current release (2026-10-02.2 when this candidate was evaluated)


def test_default_identical_to_release(mods):
    c6, rel = mods
    for name in ("FakeTorino", "FakeAuckland"):
        tgt = backend(name).target
        for qc in (ring(4, seed=4), ring(6, seed=6), chain(5, seed=5)):
            kw = kw_for(tgt)
            for extra in ({}, {"target": tgt}, {"target": tgt, "placement_refine": True}):
                assert sig(c6.compile_for_hardware(qc, **kw, **extra)) == sig(rel.compile_for_hardware(qc, **kw, **extra)), \
                    (name, extra.keys())


def test_bad_placement_score_raises(mods):
    tgt = backend("FakeTorino").target
    with pytest.raises(ValueError):
        mods[0].compile_for_hardware(ring(4), target=tgt, placement_refine=True, placement_score="x", **kw_for(tgt))


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeHanoiV2", "FakeTorino"])
def test_floor_target_bounds_and_original_unchanged(mods, name):
    c6 = mods[0]
    tgt = backend(name).target
    before = {(n, q): tgt[n][q].error for n in tgt.operation_names for q in tgt[n]
              if q is not None and tgt[n][q] is not None and tgt[n][q].error is not None}
    ft = c6.floor_aware_target(tgt)
    raised = 0
    for (n, q), e in before.items():
        assert tgt[n][q].error == e
        fe = ft[n][q].error
        if n in ("measure", "delay", "reset", "barrier"):
            assert fe == e
            continue
        fl = c6.decoherence_floor(tgt, list(q), tgt[n][q].duration)
        assert abs(fe - max(e, fl)) <= 1e-15, (n, q, e, fl, fe)
        raised += fe > e
    if name in ("FakeAuckland", "FakeHanoiV2"):
        assert raised > 0, name  # cx devices with below-floor reported errors (Addendum 293)


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeHanoiV2", "FakeTorino", "FakeKingston"])
def test_floor_refine_exact_relabel_only_and_avoids_failed(mods, name):
    c6 = mods[0]
    tgt = backend(name).target
    edges, qubits = c6._failed_elements(tgt, 0.5)
    for qc in (ring(4, seed=1), ring(6, seed=2), chain(5, seed=3)):
        kw = kw_for(tgt)
        a = c6.compile_for_hardware(qc, target=tgt, placement_refine=True, **kw)
        b = c6.compile_for_hardware(qc, target=tgt, placement_refine=True, placement_score="floor", **kw)
        assert compact_fidelity(qc, b) > 1 - 1e-6
        assert sorted(i.operation.name for i in a.data) == sorted(i.operation.name for i in b.data)
        # direction-aware: the gate as placed must not be a direction the Target reports as failed (Addendum 319, s. 4)
        for ins in b.data:
            q = tuple(b.find_bit(x).index for x in ins.qubits)
            assert not any(i in qubits for i in q)
            if len(q) == 2:
                assert q not in edges, (name, ins.operation.name, q)
