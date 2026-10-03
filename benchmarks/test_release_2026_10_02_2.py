"""Tests for release psf_compile 2026-10-02.2 (changelog item 33: exact error-weighted re-placement), adapted from the
candidate's tests (patches/psf_compile_c5_2026-10-02/test_c5_placement.py). The previous release, 2026-10-02.1, is
represented by its candidate's file (patches/psf_compile_c3_2026-10-02/psf_compile.py), which differs from it only
in the version lines.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_02_2.py -q
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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_rel1002b_test")
    sys.modules["psf_smart_layout"] = lay
    return (H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel1002b_test"),
            H.load_module(os.path.join(REPO, "patches", "psf_compile_c3_2026-10-02", "psf_compile.py"),
                          "psf_compile_prev1002_test"))


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
    assert mods[0].VERSION == "2026-10-03.2"  # current release (this file was written for 2026-10-02.2)
    assert mods[1].VERSION == "2026-10-02.c3"  # the code of release 2026-10-02.1


def test_default_identical_to_release(mods):
    c5, rel = mods
    for name in ("FakeTorino", "FakeAuckland"):
        tgt = backend(name).target
        for n in (4, 6):
            qc = ring(n, seed=n)
            kw = kw_for(tgt)
            assert sig(c5.compile_for_hardware(qc, **kw)) == sig(rel.compile_for_hardware(qc, **kw)), (name, n)
            assert sig(c5.compile_for_hardware(qc, target=tgt, **kw)) == \
                sig(rel.compile_for_hardware(qc, target=tgt, **kw)), (name, n)


def test_refine_needs_target(mods):
    tgt = backend("FakeTorino").target
    with pytest.raises(ValueError):
        mods[0].compile_for_hardware(ring(4), placement_refine=True, **kw_for(tgt))


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeTorino", "FakeKingston"])
def test_refine_exact_relabel_only_and_not_worse(mods, name):
    """Re-placement keeps the output exact, changes no gate count, never scores worse, and avoids failed
    elements. The reference is the compile without `target` (the release's output before item 31): the
    refined call runs that same compile and then re-places it. Where item 31's backstop still recompiles the
    refined call, the result is a different routing and only exactness and avoidance are checked."""
    c5 = mods[0]
    tgt = backend(name).target
    edges, qubits = c5._failed_elements(tgt, 0.5)
    applied0, compared = c5.REFINE_STATS["applied"], 0
    for qc in (ring(4, seed=1), ring(6, seed=2), chain(5, seed=3)):
        kw = kw_for(tgt)
        ref = c5.compile_for_hardware(qc, **kw)
        r0 = c5.PRUNE_STATS["recompiled"]
        b = c5.compile_for_hardware(qc, target=tgt, placement_refine=True, **kw)
        assert compact_fidelity(qc, b) > 1 - 1e-6
        assert not c5._uses_failed(b, edges, qubits)
        if c5.PRUNE_STATS["recompiled"] > r0:
            continue
        compared += 1
        assert sorted(i.operation.name for i in ref.data) == sorted(i.operation.name for i in b.data)
        assert s_rep(b, tgt) <= s_rep(ref, tgt) + 1e-12
    assert compared >= 2
    assert c5.REFINE_STATS["applied"] >= applied0
