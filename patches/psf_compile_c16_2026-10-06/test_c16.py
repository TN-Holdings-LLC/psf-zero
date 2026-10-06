"""Tests for candidate psf_compile 2026-10-06.c16 (changelog item 43: item 31's recompile keeps the first output when
no placement avoids the failed elements) against the release 2026-10-06.1 (psf_compile.py).

Run from the repository root:  python -m pytest patches/psf_compile_c16_2026-10-06/test_c16.py -q
"""
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c16_test")
    c16 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c16_test")
    return dict(rel=rel, c16=c16)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def rec_call(mod, qc, dev):
    t = backend(dev).target
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in t.operation_names]
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                        entangling_basis="cx", layout_search=True, target=t, seed_transpiler=0,
                                        **RECOMMENDED)


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def ring(n, seed, measured=False):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            qc.ry(float(rng.uniform(-1, 1)), q)
        for q in range(n - 1):
            qc.cz(q, q + 1)
    if measured:
        qc.measure_all()
    return qc


def full_kingston(spare):
    """The Qiskit circuit of PL-REDO's lap 1 (pl_heavyhex_chain.initial_tape + tape_to_qiskit), built without
    PennyLane: family T on FakeKingston, Haar-random 2-qubit unitaries, seed 1000 * spare, qubits reversed as
    tape_to_qiskit does."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_unitary
    import full_heavyhex_cliff as fh
    f = fh.graph_facts(backend("FakeKingston").target)
    blocks = fh.layout_blocks("T", spare, f["nq"], f["matching"])
    n = sum(len(b) for b in blocks)
    rng = np.random.default_rng(1000 * spare)
    qc = QuantumCircuit(n)

    def ru():
        return random_unitary(4, seed=int(rng.integers(0, 2**31))).data

    for b in blocks:
        if len(b) == 2:
            for _ in range(20):
                qc.unitary(ru(), [b[1], b[0]])
        else:
            for _ in range(10):
                qc.unitary(ru(), [b[1], b[0]])
            for _ in range(10):
                qc.unitary(ru(), [b[2], b[1]])
    return qc, blocks, fh.expected_twoq(blocks)


def block_ops(circ, blocks):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    owner = {q: k for k, b in enumerate(blocks) for q in b}
    subs = [QuantumCircuit(len(b)) for b in blocks]
    for ins in circ.data:
        qs = [circ.find_bit(q).index for q in ins.qubits]
        k = owner[qs[0]]
        assert all(owner[q] == k for q in qs)
        subs[k].append(ins.operation, [blocks[k].index(q) for q in qs])
    return [Operator(s).data for s in subs]


def aligned(a, b):
    t = np.trace(a.conj().T @ b)
    return float(np.linalg.norm(b - (t / abs(t)) * a))


def test_versions(mods):
    assert mods["c16"].VERSION == "2026-10-06.c16"
    assert mods["rel"].VERSION == "2026-10-06.1"
    assert mods["c16"].PRUNE_STATS["unavoidable"] == 0


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeHanoiV2", "FakeGeneva", "FakeKingston"])
def test_unchanged_where_the_release_succeeds(mods, dev):
    for qc in (ring(5, 1), ring(7, 2, True), ring(12, 3, True)):
        assert sig(rec_call(mods["c16"], qc, dev)) == sig(rec_call(mods["rel"], qc, dev))
    assert mods["c16"].PRUNE_STATS["unavoidable"] == 0


def test_full_device_circuit(mods):
    """The release raises on PL-REDO's spare-0 circuit; c16 returns a swap-free circuit that implements every block."""
    import pl_heavyhex_chain as plc
    from qiskit.transpiler.exceptions import TranspilerError
    qc, blocks, swapfree = full_kingston(0)
    with pytest.raises(TranspilerError):
        rec_call(mods["rel"], qc, "FakeKingston")
    before = mods["c16"].PRUNE_STATS["unavoidable"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = rec_call(mods["c16"], qc, "FakeKingston")
    assert mods["c16"].PRUNE_STATS["unavoidable"] > before
    assert any(issubclass(x.category, RuntimeWarning) and "item 43" in str(x.message) for x in w)
    logical, _ = plc.back_to_logical(out, qc.num_qubits, blocks)
    assert sum(1 for i in out.data if len(i.qubits) == 2) == swapfree
    for a, b in zip(block_ops(qc, blocks), block_ops(logical, blocks)):
        assert aligned(a, b) <= 1e-10
