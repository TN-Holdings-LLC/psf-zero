"""Tests for release psf_compile 2026-10-03.3 (changelog item 37: floor-placed candidate and choice by a state-aware
Pauli estimate), adapted from the candidate's tests (patches/psf_compile_c10_2026-10-03/test_c10_floor_pauli.py). The
previous release, 2026-10-03.2, is represented by its candidate's file (patches/psf_compile_c9_2026-10-03/psf_compile.py),
which differs from it only in the version lines.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_03_3.py -q
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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_rel10033_test")
    sys.modules["psf_smart_layout"] = lay
    return (H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel10033_test"),
            H.load_module(os.path.join(REPO, "patches", "psf_compile_c9_2026-10-03", "psf_compile.py"),
                          "psf_compile_prev10033_test"))


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


def ghz(n=6, seed=0):
    """A GHZ chain with random final rotations, like HOLD4's F5 cell (Addendum 330's case)."""
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
        qc.rz(float(rng.uniform(-math.pi, math.pi)), q)
    return qc


REC = dict(placement_refine=True, final_resynthesis="select", compare_level3=True)


def test_version(mods):
    assert mods[0].VERSION == "2026-10-05.1"  # current release (this file was written for 2026-10-03.3)
    assert mods[1].VERSION == "2026-10-03.c9"  # the code of release 2026-10-03.2


def test_default_identical_to_release(mods):
    new, prev = mods
    for name in ("FakeTorino", "FakeAuckland"):
        tgt = backend(name).target
        for qc in (ring(4, seed=4), ring(6, seed=6), chain(5, seed=5)):
            kw = kw_for(tgt)
            for extra in ({}, {"target": tgt}, {"target": tgt, "placement_refine": True}, dict(REC, target=tgt)):
                assert sig(new.compile_for_hardware(qc, **kw, **extra)) == sig(prev.compile_for_hardware(qc, **kw, **extra)), \
                    (name, extra.keys())


def test_bad_arguments_raise(mods):
    tgt = backend("FakeTorino").target
    kw = kw_for(tgt)
    with pytest.raises(ValueError):
        mods[0].compile_for_hardware(ring(4), compare_floor=True, **kw)
    with pytest.raises(ValueError):
        mods[0].compile_for_hardware(ring(4), target=tgt, candidate_score="other", **kw)


def test_pauli_cost_matches_statevector(mods):
    """pauli_cost's numpy state and Pauli expectations against Qiskit's Statevector, on a compiled GHZ chain."""
    from qiskit.quantum_info import Pauli, Statevector
    new = mods[0]
    tgt = backend("FakeGeneva").target
    out = new.compile_for_hardware(ghz(seed=1), target=tgt, placement_refine=True, **kw_for(tgt))
    ops = [(i.operation, tuple(out.find_bit(b).index for b in i.qubits)) for i in out.data
           if i.operation.name not in ("barrier", "measure", "delay")]
    active = sorted({i for _, q in ops for i in q})
    pos = {p: j for j, p in enumerate(active)}
    sv = Statevector.from_label("0" * len(active))
    ref = 0.0
    for op, q in ops:
        sv = sv.evolve(op, qargs=[pos[i] for i in q])
        props = tgt[op.name][q]
        t, e, f = props.duration or 0.0, props.error or 0.0, 1.0
        for i in q:
            if not t:
                continue
            t1 = tgt.qubit_properties[i].t1
            t2 = min(tgt.qubit_properties[i].t2, 2 * t1)
            px = (1 - math.exp(-t / t1)) / 4
            pz = max((1 - math.exp(-t / t2)) / 2 - px, 0.0)
            for lab, p in (("X", px), ("Y", px), ("Z", pz)):
                ev = float(sv.expectation_value(Pauli(lab), [pos[i]]).real)
                ref += p * (1 - ev * ev)
            f *= (1 + 2 * math.exp(-t / t2) + math.exp(-t / t1)) / 4
        d = 2 ** len(q)
        ref += max(e - (1 - (d * f + 1) / (d + 1)), 0.0) * (d + 1) / d
    assert abs(new.pauli_cost(out, tgt) - ref) <= 1e-9 * max(1.0, ref)


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeHanoiV2", "FakeGeneva", "FakeTorino", "FakeKingston"])
def test_full_choice_exact_safe_and_not_worse_by_estimate(mods, name):
    """With compare_floor and candidate_score="pauli": exact, on the target, no failed qubit or direction, and no
    higher pauli_cost than the release's own circuit or level 3's."""
    new, prev = mods
    tgt = backend(name).target
    edges, qubits = new._failed_elements(tgt, 0.5)
    kw = kw_for(tgt)
    for qc in (ghz(seed=1), ring(6, seed=2), chain(5, seed=3), xxz_ring(seed=4)):
        c = new.compile_for_hardware(qc, target=tgt, compare_floor=True, candidate_score="pauli", **REC, **kw)
        assert compact_fidelity(qc, c) > 1 - 1e-6, name
        a = prev.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select", **kw)
        pc = new.pauli_cost(c, tgt)
        assert pc <= new.pauli_cost(a, tgt) + 1e-12
        b = l3(qc, tgt)
        if new._acceptable(b, tgt, 0.5):
            assert pc <= new.pauli_cost(b, tgt) + 1e-12
        for ins in c.data:
            q = tuple(c.find_bit(x).index for x in ins.qubits)
            assert ins.operation.name in tgt.operation_names and q in tgt[ins.operation.name], (name, ins.operation.name, q)
            assert not any(i in qubits for i in q)
            if len(q) == 2:
                assert q not in edges, (name, ins.operation.name, q)


def test_floor_candidate_used_on_ghz_geneva(mods):
    """Addendum 330's main case: on FakeGeneva the floor-placed circuit is chosen for 8-qubit GHZ chains (in Addendum
    330 the floor placement differed from the release's on FakeGeneva only for n = 8; for n = 4 and 6 it was the same
    placement, and a tie keeps the release's circuit)."""
    new = mods[0]
    tgt = backend("FakeGeneva").target
    kw = kw_for(tgt)
    before = new.COMPARE_STATS["floor"]
    for s in range(3):
        new.compile_for_hardware(ghz(n=8, seed=30 + s), target=tgt, compare_floor=True, candidate_score="pauli", **REC,
                                 **kw)
    assert new.COMPARE_STATS["floor"] - before >= 2


def test_floor_target_bounds(mods):
    new = mods[0]
    tgt = backend("FakeAuckland").target
    ft = new.floor_aware_target(tgt)
    for n in tgt.operation_names:
        if n in ("measure", "delay", "reset", "barrier"):
            continue
        for q, p in tgt[n].items():
            if q is None or p is None or p.error is None:
                continue
            assert ft[n][q].error == max(p.error, new.decoherence_floor(tgt, list(q), p.duration))
