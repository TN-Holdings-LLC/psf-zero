"""Tests for candidate psf_ai_compile 2026-10-04.a8 (item 13: above SMALL_MAX_QUBITS with a target, the release's
recommended call instead of a target-blind compile). Helpers are copied from
patches/psf_compile_c11_2026-10-04/test_c11_hybrid.py.

Run from the repository root:  python -m pytest patches/psf_ai_compile_a8_2026-10-04/test_a8.py -q
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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_a8_test")
    sys.modules["psf_smart_layout"] = lay
    a7 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a7.py"), "psf_ai_compile")  # a7 (frozen at a8's adoption)
    a8 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a8_test")
    return dict(rel=rel, a7=a7, a8=a8)


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


def nat(tgt):
    return [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]


REC = dict(placement_refine=True, final_resynthesis="select", compare_level3=True)


def test_version(mods):
    assert mods["a8"].AI_COMPILE_VERSION == "2026-10-04.a8"
    assert mods["a7"].AI_COMPILE_VERSION == "2026-10-02.a7"
    assert mods["rel"].VERSION == "2026-10-06.1"  # current release (2026-10-03.3 when this candidate was evaluated)


def test_small_circuits_identical_to_a7(mods):
    for name in ("FakeAuckland", "FakeTorino"):
        tgt = backend(name).target
        cm = tgt.build_coupling_map()
        for qc in (ring(6, seed=1), chain(5, seed=2)):
            for kw in ({}, {"target": tgt}):
                a = mods["a7"].compile_for_model_circuit(qc, cm, nat(tgt), **kw)
                b = mods["a8"].compile_for_model_circuit(qc, cm, nat(tgt), **kw)
                assert sig(a) == sig(b), (name, kw.keys())


def test_large_without_target_identical_to_a7(mods):
    tgt = backend("FakeTorino").target
    cm = tgt.build_coupling_map()
    for qc in (ring(10, seed=3), ghz(n=9, seed=4)):
        a, ia = mods["a7"].compile_for_model_circuit(qc, cm, nat(tgt), return_info=True)
        b, ib = mods["a8"].compile_for_model_circuit(qc, cm, nat(tgt), return_info=True)
        assert sig(a) == sig(b) and ia["path"] == ib["path"] == "fast"


@pytest.mark.parametrize("name", ["FakeHanoiV2", "FakeAlgiers", "FakeTorino", "FakeAachen"])
def test_large_with_target_is_the_release_call_exact_and_safe(mods, name):
    """Above SMALL_MAX_QUBITS with a target: exactly the release's recommended call, exact, on the target, and no
    failed qubit or direction (devices with failed elements; a7 used them on all four in WIDE)."""
    rel = mods["rel"]
    tgt = backend(name).target
    cm = tgt.build_coupling_map()
    edges, qubits = rel._failed_elements(tgt, 0.5)
    for qc in (ring(10, seed=5), ghz(n=9, seed=6), chain(10, seed=7)):
        out, info = mods["a8"].compile_for_model_circuit(qc, cm, nat(tgt), target=tgt, return_info=True)
        assert info["path"] == "fast-target"
        ref = rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat(tgt), entangling_basis="cx",
                                       layout_search=True, seed_transpiler=0, target=tgt, **REC)
        assert sig(out) == sig(ref), name
        assert compact_fidelity(qc, out) > 1 - 1e-6, name
        for ins in out.data:
            q = tuple(out.find_bit(x).index for x in ins.qubits)
            assert ins.operation.name in tgt.operation_names and q in tgt[ins.operation.name], (name, ins.operation.name, q)
            assert not any(i in qubits for i in q)
            if len(q) == 2:
                assert q not in edges, (name, ins.operation.name, q)
