"""Tests for release psf_compile 2026-10-03.2 (changelog item 36: choice against Qiskit level 3 by excitation_cost),
adapted from the candidate's tests (patches/psf_compile_c9_2026-10-03/test_c9_compare.py). The previous release,
2026-10-03.1, is represented by its candidate's file (patches/psf_compile_c8_2026-10-03/psf_compile.py), which differs
from it only in the version lines.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_03_2.py -q
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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_rel10032_test")
    sys.modules["psf_smart_layout"] = lay
    return (H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel10032_test"),
            H.load_module(os.path.join(REPO, "patches", "psf_compile_c8_2026-10-03", "psf_compile.py"),
                          "psf_compile_prev10032_test"))


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


def xxz_chain(n=6, steps=2, seed=0):
    """An open XXZ Trotter chain like HOLD2's F3 (Addendum 322's case)."""
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    jx, jy = rng.uniform(0.5, 1.5, 2)
    jz = rng.uniform(0.2, 1.0) * jx
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for parity in (0, 1):
            for a in range(parity, n - 1, 2):
                qc.rxx(0.2 * jx, a, a + 1)
                qc.ryy(0.2 * jy, a, a + 1)
                qc.rzz(0.2 * jz, a, a + 1)
        for q in range(n):
            qc.rz(float(rng.uniform(-0.2, 0.2)), q)
    return qc


def xxz_ring(n=6, steps=2, seed=0):
    """A periodic XXZ Trotter ring like HOLD3's F3 periodic cell (Addendum 326's case)."""
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    jx, jy = rng.uniform(0.5, 1.5, 2)
    jz = rng.uniform(0.2, 1.0) * jx
    bonds = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for parity in (0, 1):
            for a, b in bonds:
                if a % 2 == parity:
                    qc.rxx(0.2 * jx, a, b)
                    qc.ryy(0.2 * jy, a, b)
                    qc.rzz(0.2 * jz, a, b)
    return qc


def l3(qc, tgt):
    from qiskit import transpile
    return transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)


def test_version(mods):
    assert mods[0].VERSION == "2026-10-03.2"
    assert mods[1].VERSION == "2026-10-03.c8"  # the code of release 2026-10-03.1


def test_default_identical_to_release(mods):
    new, prev = mods
    for name in ("FakeTorino", "FakeAuckland"):
        tgt = backend(name).target
        for qc in (ring(4, seed=4), ring(6, seed=6), chain(5, seed=5)):
            kw = kw_for(tgt)
            for extra in ({}, {"target": tgt}, {"target": tgt, "placement_refine": True},
                          {"target": tgt, "placement_refine": True, "final_resynthesis": "select"}):
                assert sig(new.compile_for_hardware(qc, **kw, **extra)) == sig(prev.compile_for_hardware(qc, **kw, **extra)), \
                    (name, extra.keys())


def test_needs_target(mods):
    tgt = backend("FakeTorino").target
    with pytest.raises(ValueError):
        mods[0].compile_for_hardware(ring(4), compare_level3=True, **kw_for(tgt))


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeHanoiV2", "FakeGeneva", "FakeTorino", "FakeKingston"])
def test_compare_returns_lower_estimate_exact_and_safe(mods, name):
    """The result is the release's circuit or level 3's, whichever has the lower excitation_cost; exact; on the
    target; no failed qubit or failed direction (direction-aware)."""
    new, prev = mods
    tgt = backend(name).target
    edges, qubits = new._failed_elements(tgt, 0.5)
    kw = kw_for(tgt)
    for qc in (ring(4, seed=1), ring(6, seed=2), chain(5, seed=3), xxz_ring(seed=4)):
        a = prev.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select", **kw)
        b = l3(qc, tgt)
        c = new.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select",
                                    compare_level3=True, **kw)
        assert compact_fidelity(qc, c) > 1 - 1e-6, name
        ea, eb = new.excitation_cost(a, tgt), new.excitation_cost(b, tgt)
        if new._acceptable(b, tgt, 0.5) and eb < ea:
            assert sig(c) == sig(b), name
        else:
            assert sig(c) == sig(a), name
        for ins in c.data:
            q = tuple(c.find_bit(x).index for x in ins.qubits)
            assert ins.operation.name in tgt.operation_names and q in tgt[ins.operation.name], (name, ins.operation.name, q)
            assert not any(i in qubits for i in q)
            if len(q) == 2:
                assert q not in edges, (name, ins.operation.name, q)


def test_level3_chosen_on_periodic_ring_cx(mods):
    """Addendum 326's main case: on a cx device the estimate prefers level 3 on periodic XXZ rings."""
    new = mods[0]
    tgt = backend("FakeAuckland").target
    kw = kw_for(tgt)
    chosen = 0
    for s in range(3):
        qc = xxz_ring(seed=20 + s)
        c = new.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select",
                                    compare_level3=True, **kw)
        chosen += sig(c) == sig(l3(qc, tgt))
    assert chosen >= 2, chosen


def test_acceptable_is_direction_aware(mods):
    """FakeHanoiV2 reports cx(5, 8) failed and cx(8, 5) healthy (Addendum 319, section 4)."""
    from qiskit import QuantumCircuit
    new = mods[0]
    tgt = backend("FakeHanoiV2").target
    if not (tgt["cx"][(5, 8)].error >= 0.5 and tgt["cx"][(8, 5)].error < 0.5):
        pytest.skip("this snapshot no longer has the one-way failure")
    bad, good = QuantumCircuit(tgt.num_qubits), QuantumCircuit(tgt.num_qubits)
    bad.cx(5, 8)
    good.cx(8, 5)
    assert not new._acceptable(bad, tgt, 0.5)
    assert new._acceptable(good, tgt, 0.5)
