"""Tests for candidate psf_compile 2026-10-04.c11 (changelog item 38: choice among the candidates of items 36-37 by
`hybrid_cost`, an estimate with amplitude damping and pure dephasing). Helpers are copied from
patches/psf_compile_c10_2026-10-03/test_c10_floor_pauli.py.

Run from the repository root:  python -m pytest patches/psf_compile_c11_2026-10-04/test_c11_hybrid.py -q
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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_c11_test")
    sys.modules["psf_smart_layout"] = lay
    return (H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c11_test"),
            H.load_module(os.path.join(REPO, "patches", "psf_compile_release_2026-10-10.2", "psf_compile.py"), "psf_compile_rel_c11_test"))


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


FULL = dict(REC, compare_floor=True, candidate_score="hybrid")


def test_version(mods):
    assert mods[0].VERSION == "2026-10-04.c11"
    assert mods[1].VERSION == "2026-10-10.2"  # current release (2026-10-03.3 when this candidate was evaluated)


def test_other_options_identical_to_release(mods):
    """Every call without candidate_score="hybrid" gives the release's circuit."""
    c11, rel = mods
    for name in ("FakeTorino", "FakeAuckland"):
        tgt = backend(name).target
        for qc in (ring(4, seed=4), ring(6, seed=6), chain(5, seed=5)):
            kw = kw_for(tgt)
            for extra in ({}, {"target": tgt}, dict(REC, target=tgt),
                          dict(REC, target=tgt, compare_floor=True, candidate_score="pauli")):
                assert sig(c11.compile_for_hardware(qc, **kw, **extra)) == sig(rel.compile_for_hardware(qc, **kw, **extra)), \
                    (name, extra.keys())


def test_bad_arguments_raise(mods):
    tgt = backend("FakeTorino").target
    kw = kw_for(tgt)
    with pytest.raises(ValueError):
        mods[0].compile_for_hardware(ring(4), target=tgt, candidate_score="other", **kw)
    with pytest.raises(ValueError):
        mods[0].compile_for_hardware(ring(4), compare_floor=True, candidate_score="hybrid", **kw)


def test_hybrid_cost_matches_statevector(mods):
    """hybrid_cost's numpy state against Qiskit's Statevector: P(1) before each gate, <Z> after it."""
    from qiskit.quantum_info import Pauli, Statevector
    c11 = mods[0]
    for name, qc in (("FakeGeneva", ghz(seed=1)), ("FakeTorino", xxz_chain(seed=2))):
        tgt = backend(name).target
        out = c11.compile_for_hardware(qc, target=tgt, placement_refine=True, **kw_for(tgt))
        ops = [(i.operation, tuple(out.find_bit(b).index for b in i.qubits)) for i in out.data
               if i.operation.name not in ("barrier", "measure", "delay")]
        active = sorted({i for _, q in ops for i in q})
        pos = {p: j for j, p in enumerate(active)}
        sv = Statevector.from_label("0" * len(active))
        ref = 0.0
        for op, q in ops:
            props = tgt[op.name][q]
            t, e, f = props.duration or 0.0, props.error or 0.0, 1.0
            for i in q:
                if t:
                    ref += t / tgt.qubit_properties[i].t1 * float(sv.probabilities([pos[i]])[1])
            sv = sv.evolve(op, qargs=[pos[i] for i in q])
            for i in q:
                if not t:
                    continue
                t1 = tgt.qubit_properties[i].t1
                t2 = min(tgt.qubit_properties[i].t2, 2 * t1)
                rate = max(1 / t2 - 1 / (2 * t1), 0.0)
                ez = float(sv.expectation_value(Pauli("Z"), [pos[i]]).real)
                ref += (1 - math.exp(-t * rate)) / 2 * (1 - ez * ez)
                f *= (1 + 2 * math.exp(-t / t2) + math.exp(-t / t1)) / 4
            d = 2 ** len(q)
            ref += max(e - (1 - (d * f + 1) / (d + 1)), 0.0) * (d + 1) / d
        assert abs(c11.hybrid_cost(out, tgt) - ref) <= 1e-9 * max(1.0, ref), name


def test_hybrid_cost_is_the_diagnosis_estimate(mods):
    """The candidate's hybrid_cost is the estimate the HYBRID diagnosis (Addendum 335) measured."""
    import core_fix_c2_eval as H
    c11 = mods[0]
    diag = H.load_module(os.path.join(REPO, "data", "2026-10-04", "hybrid", "diag", "hybrid_diag.py"), "hybrid_diag_c11_test")
    for name in ("FakeAuckland", "FakeKingston"):
        tgt = backend(name).target
        for qc in (ghz(seed=3), xxz_ring(seed=4), ring(6, seed=5)):
            for out in (c11.compile_for_hardware(qc, target=tgt, placement_refine=True, **kw_for(tgt)), l3(qc, tgt)):
                a, b = c11.hybrid_cost(out, tgt), diag.hybrid_terms(out, tgt)[0]
                assert abs(a - b) <= 1e-12 * max(1.0, b), name


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeHanoiV2", "FakeGeneva", "FakeTorino", "FakeKingston"])
def test_full_choice_exact_safe_and_not_worse_by_estimate(mods, name):
    """With the intended call: exact, on the target, no failed qubit or direction, and no higher hybrid_cost than the
    release's own circuit or level 3's."""
    c11, rel = mods
    tgt = backend(name).target
    edges, qubits = c11._failed_elements(tgt, 0.5)
    kw = kw_for(tgt)
    for qc in (ghz(seed=1), ring(6, seed=2), chain(5, seed=3), xxz_chain(seed=4)):
        c = c11.compile_for_hardware(qc, target=tgt, **FULL, **kw)
        assert compact_fidelity(qc, c) > 1 - 1e-6, name
        a = rel.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select", **kw)
        hc = c11.hybrid_cost(c, tgt)
        assert hc <= c11.hybrid_cost(a, tgt) + 1e-12
        b = l3(qc, tgt)
        if c11._acceptable(b, tgt, 0.5):
            assert hc <= c11.hybrid_cost(b, tgt) + 1e-12
        for ins in c.data:
            q = tuple(c.find_bit(x).index for x in ins.qubits)
            assert ins.operation.name in tgt.operation_names and q in tgt[ins.operation.name], (name, ins.operation.name, q)
            assert not any(i in qubits for i in q)
            if len(q) == 2:
                assert q not in edges, (name, ins.operation.name, q)
