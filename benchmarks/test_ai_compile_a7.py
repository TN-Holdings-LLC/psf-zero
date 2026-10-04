"""Tests for the AI front end psf_ai_compile 2026-10-02.a7 (adopted 2026-10-02; kept as benchmarks/psf_ai_compile_a7.py
since a8 was adopted on 2026-10-04), adapted from the
candidate's tests (patches/psf_ai_compile_a7_2026-10-02/test_a7.py); a5 is benchmarks/psf_ai_compile_a5.py.

Run from the repository root:  python -m pytest benchmarks/test_ai_compile_a7.py -q
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
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_a7_release_test")
    sys.modules["psf_smart_layout"] = lay
    a5 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a5.py"), "psf_ai_compile_a5_frozen_test")
    a7 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a7.py"), "psf_ai_compile")
    import gap_eval as G
    return dict(rel=rel, a5=a5, a7=a7, G=G)


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
    assert mods["a7"].AI_COMPILE_VERSION == "2026-10-02.a7"
    assert mods["a5"].AI_COMPILE_VERSION == "2026-10-01.a5"


def test_without_target_identical_to_a5(mods):
    for name in ("FakeTorino", "FakeAuckland"):
        tgt = backend(name).target
        cm = tgt.build_coupling_map()
        for qc in (ring(4, seed=5), ring(5, seed=6)):
            assert sig(mods["a7"].compile_for_model_circuit(qc, cm, nat(tgt))) == \
                sig(mods["a5"].compile_for_model_circuit(qc, cm, nat(tgt))), name


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeTorino", "FakeKingston"])
def test_with_target_exact_avoids_failed_and_not_worse_by_estimate(mods, name):
    """On the smoke F3 chains (seeds disjoint from the scored ones): exact, no failed element, and a7's own estimate is
    no worse than a5's or than level 3's output, since both are among its candidates."""
    a5, a7, rel = mods["a5"], mods["a7"], mods["rel"]
    from qiskit import transpile
    tgt = backend(name).target
    cm = tgt.build_coupling_map()
    edges, qubits = rel._failed_elements(tgt, 0.5)
    for _, qc in mods["G"].family("F3", True):
        out, info = a7.compile_for_model_circuit(qc, cm, nat(tgt), target=tgt, return_info=True)
        assert compact_fidelity(qc, out) > 1 - 1e-6
        assert not rel._uses_failed(out, edges, qubits)
        assert info["chosen"] in ("PSF", "L3T")
        assert any(t[0] == "L3T" for t in info["tried"])
        e7 = a7.state_aware_cost(out, tgt)
        e5 = a5.state_aware_cost(a5.compile_for_model_circuit(qc, cm, nat(tgt), target=tgt), tgt)
        el = a7.state_aware_cost(transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0), tgt)
        assert e7 <= e5 + 1e-9 and e7 <= el + 1e-9, (name, e7, e5, el)
