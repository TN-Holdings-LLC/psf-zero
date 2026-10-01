"""Tests for the 2026-10-01 candidates: psf_compile.py 2026-10-01.c2 and psf_smart_layout.py 2026-10-01.c2.

compile c2: cost-aware consolidation of short blocks, permutation elision, absorption of routing SWAPs
(all on by default only for entangling_basis="cx"). layout c2: exact packing search for disjoint
2- and 3-qubit paths."""
import os
import sys
import warnings

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import psf_compile as pc  # noqa: E402
import psf_smart_layout as psl  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import SwapGate, UnitaryGate  # noqa: E402
from qiskit.quantum_info import Operator, Statevector, random_statevector  # noqa: E402

warnings.simplefilter("ignore")


def _backend(name):
    from qiskit_ibm_runtime import fake_provider
    tgt = getattr(fake_provider, name)().target
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    return tgt.build_coupling_map(), nat


def _two_q(c):
    return sum(1 for i in c.data if len(i.qubits) == 2)


def _routed_equiv(qc, out, seed=3):
    """Statevector check on the physical qubits that out touches, read at the final layout."""
    n = qc.num_qubits
    init = out.layout.initial_index_layout(filter_ancillas=True)[:n]
    fin = out.layout.final_index_layout(filter_ancillas=True)[:n]
    touched = set(init) | set(fin)
    for inst in out.data:
        touched |= {out.find_bit(q).index for q in inst.qubits}
    others = sorted(touched - set(init))
    idx = {p: v for v, p in enumerate(init)}
    idx.update({p: n + j for j, p in enumerate(others)})
    m = n + len(others)
    small = QuantumCircuit(m)
    for inst in out.data:
        small.append(inst.operation, [idx[out.find_bit(q).index] for q in inst.qubits])
    psi = random_statevector(2 ** n, seed=seed)
    full = Statevector.from_label("0" * (m - n)).tensor(psi) if m > n else psi
    res = full.evolve(small).data.reshape([2] * m)
    # axis k of res is qubit m-1-k; bring virtual v (now at small index idx[fin[v]]) to position v
    axes = [m - 1 - idx[fin[v]] for v in reversed(range(n))]
    rest = [a for a in range(m) if a not in axes]
    vec = np.transpose(res, rest + axes).reshape(2 ** (m - n), 2 ** n)
    phi = psi.evolve(qc).data
    return float(np.sum(np.abs(vec @ phi.conj()) ** 2))


def test_versions():
    assert pc.VERSION == "2026-10-01.c2"
    assert psl.LAYOUT_VERSION.startswith("2026-10-01.c2")


def test_short_cheaper_block_is_consolidated_in_cx_basis_only():
    qc = QuantumCircuit(2)
    qc.cry(1.23, 0, 1)
    qc.cx(1, 0)          # 3 CX as written, Weyl optimum 2
    out = pc.compile(qc, entangling_basis="cx")
    assert out.count_ops().get("cx", 0) == 2
    assert Operator(out).equiv(Operator(qc))
    can = pc.compile(qc, entangling_basis="canonical")
    assert [i.operation.name for i in can.data] == ["cry", "cx"]


def test_block_unitary_matches_operator():
    from qiskit.circuit.random import random_circuit
    from qiskit.converters import circuit_to_dag
    for seed in range(40):
        qc = random_circuit(2, 6, max_operands=2, seed=seed)
        nodes = list(circuit_to_dag(qc).topological_op_nodes())
        if any(not hasattr(n.op, "to_matrix") for n in nodes):
            continue
        for order in (list(qc.qubits), list(qc.qubits)[::-1]):
            sub = QuantumCircuit(2)
            for n in nodes:
                sub.append(n.op, [order.index(q) for q in n.qargs])
            assert np.allclose(pc._block_unitary_4x4(nodes, order), Operator(sub).data, atol=1e-12)


@pytest.mark.parametrize("as_unitary", [False, True])
def test_swap_is_elided_with_final_layout(as_unitary):
    cm, nat = _backend("FakeAuckland")
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ry(0.4, 1)
    if as_unitary:
        qc.append(UnitaryGate(SwapGate().to_matrix()), [0, 2])
    else:
        qc.swap(0, 2)
    out = pc.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                  layout_search=True, seed_transpiler=0)
    assert _two_q(out) == 0
    assert _routed_equiv(qc, out) > 1 - 1e-9


def test_routing_swap_absorbed():
    cm, nat = _backend("FakeAuckland")
    qc = QuantumCircuit(3)     # a triangle: heavy-hex needs one routing SWAP
    qc.h(0)
    qc.cry(1.1, 0, 1)
    qc.cry(0.7, 1, 2)
    qc.crz(0.5, 0, 2)
    rel = pc.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                  layout_search=True, seed_transpiler=0, post_routing_resynthesis=False)
    new = pc.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                  layout_search=True, seed_transpiler=0)
    assert _two_q(new) <= _two_q(rel)
    assert _routed_equiv(qc, new) > 1 - 1e-9


def test_canonical_path_unchanged_by_auto():
    cm, nat = _backend("FakeAuckland")
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.swap(1, 2)
    qc.cp(0.3, 0, 2)
    a = pc.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="canonical",
                                layout_search=True, seed_transpiler=0)
    b = pc.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="canonical",
                                layout_search=True, seed_transpiler=0, elide_permutations=False,
                                post_routing_resynthesis=False)
    assert [(i.operation.name, [a.find_bit(q).index for q in i.qubits]) for i in a.data] == \
           [(i.operation.name, [b.find_bit(q).index for q in i.qubits]) for i in b.data]


def test_bad_auto_value_rejected():
    cm, nat = _backend("FakeAuckland")
    with pytest.raises(ValueError):
        pc.compile_for_hardware(QuantumCircuit(2), coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                elide_permutations="yes")


def test_packing_places_nine_ghz3_on_auckland():
    cm, _ = _backend("FakeAuckland")
    triples = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(9)]
    pairs = []
    assert psl.short_path_layout(cm, pairs, triples) is None      # the c1 shortcut misses this one
    lm = psl.packing_layout(cm, pairs, triples, time_budget_s=5.0)
    assert lm is not None and len(set(lm.values())) == 27
    edges = {tuple(sorted(e)) for e in cm.get_edges()}
    for a, b, c in triples:
        assert tuple(sorted((lm[a], lm[b]))) in edges and tuple(sorted((lm[b], lm[c]))) in edges


def test_packing_reports_infeasible():
    cm, _ = _backend("FakeAuckland")       # maximum matching 10, so 11 disjoint edges cannot exist
    pairs = [(2 * i, 2 * i + 1) for i in range(11)]
    assert psl.packing_layout(cm, pairs, [], time_budget_s=5.0) is None
