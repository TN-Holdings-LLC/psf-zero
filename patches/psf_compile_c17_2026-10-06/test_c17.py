"""Tests for candidate psf_compile 2026-10-06.c17 (changelog item 45: alternatives that item 39 cannot check are not
built) against the release 2026-10-06.3 (psf_compile.py).

Run from the repository root:  python -m pytest patches/psf_compile_c17_2026-10-06/test_c17.py -q -s
"""
import contextlib
import io
import os
import sys
import time
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
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c17_test")
    c17 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c17_test")
    return dict(rel=rel, c17=c17)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def timed_call(mod, qc, dev, **options):
    t = backend(dev).target
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in t.operation_names]
    kw = dict(RECOMMENDED, **options)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = mod.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                       entangling_basis="cx", layout_search=True, target=t, seed_transpiler=0, **kw)
    return out, time.perf_counter() - t0


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


def blocks_circuit(dev, spare, seed):
    """PL-GPU-REDO's circuit family T (Addendum 375) on `dev` at `spare`, without PennyLane: Haar-random two-qubit
    unitaries on pair and triple blocks of a full heavy-hex layout."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_unitary
    import full_heavyhex_cliff as fh
    f = fh.graph_facts(backend(dev).target)
    blocks = fh.layout_blocks("T", spare, f["nq"], f["matching"])
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(sum(len(b) for b in blocks))
    for b in blocks:
        pairs = [(b[1], b[0])] if len(b) == 2 else [(b[1], b[0]), (b[2], b[1])]
        for a, c in pairs:
            for _ in range(10):
                qc.unitary(random_unitary(4, seed=int(rng.integers(0, 2**31))).data, [a, c])
    return qc


def hamiltonian48():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    n = 48
    terms = [("ZZ", [q, q + 1], 0.5) for q in range(n - 1)] + [("X", [q], 0.7) for q in range(n)]
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_sparse_list(terms, num_qubits=n), time=1.0), range(n))
    return qc


def test_versions(mods):
    assert mods["c17"].VERSION == "2026-10-06.c17"
    assert mods["rel"].VERSION == "2026-10-06.3"
    assert mods["c17"].SKIP_STATS == {"resynthesis": 0, "floor": 0, "level3": 0}


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeHanoiV2", "FakeGeneva", "FakeKingston"])
def test_unchanged_and_nothing_skipped_up_to_16_qubits(mods, dev):
    before = dict(mods["c17"].SKIP_STATS)
    for qc in (ring(5, 1), ring(7, 2, True), ring(12, 3, True), ring(16, 4)):
        assert sig(timed_call(mods["c17"], qc, dev)[0]) == sig(timed_call(mods["rel"], qc, dev)[0])
    assert mods["c17"].SKIP_STATS["floor"] == before["floor"]
    assert mods["c17"].SKIP_STATS["level3"] == before["level3"]


@pytest.mark.parametrize("options", [{}, dict(compare_floor=False, candidate_score="excitation"),
                                     dict(final_resynthesis=True)])
def test_unchanged_above_16_qubits(mods, options):
    for qc, dev in ((ring(20, 5, True), "FakeTorino"), (ring(24, 6), "FakeKingston"),
                    (blocks_circuit("FakeAuckland", 4, 7), "FakeAuckland")):
        assert sig(timed_call(mods["c17"], qc, dev, **options)[0]) == sig(timed_call(mods["rel"], qc, dev,
                                                                                    **options)[0])


@pytest.mark.parametrize("make,dev", [("auckland0", "FakeAuckland"), ("ham48", "FakeTorino")])
def test_same_circuit_in_less_time(mods, make, dev):
    """PL-GPU-REDO's full-occupancy case (G9) and a 48-qubit HamLib-like input: the same circuit, faster."""
    qc = blocks_circuit("FakeAuckland", 0, 8) if make == "auckland0" else hamiltonian48()
    timed_call(mods["c17"], qc, dev)  # warm-up (imports, caches)
    timed_call(mods["rel"], qc, dev)
    before = dict(mods["c17"].SKIP_STATS)
    out17, t17 = timed_call(mods["c17"], qc, dev)
    outr, tr = timed_call(mods["rel"], qc, dev)
    skipped = {k: mods["c17"].SKIP_STATS[k] - before[k] for k in before}
    print(f"{make}: {qc.num_qubits} qubits on {dev}; release {tr:.2f} s, c17 {t17:.2f} s; skipped {skipped}")
    assert sig(out17) == sig(outr)
    assert skipped["floor"] == 1 and skipped["level3"] == 1
    assert t17 < (0.5 if make == "auckland0" else 0.8) * tr
