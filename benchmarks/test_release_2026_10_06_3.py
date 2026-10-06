"""Tests for release psf_compile 2026-10-06.3: candidate 2026-10-06.c18 of Addendum 377 (changelog item 44: item 39's
checks never turn a wide instruction into a matrix). The candidate's own test is
patches/psf_compile_c18_2026-10-06/test_c18.py; BP-PROBE's run 3 (Addendum 377) ran it on Benchpress's HamLib tests.
The previous release, 2026-10-06.2, is kept unchanged in patches/psf_compile_release_2026-10-06.2/psf_compile.py.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_06_3.py -q
"""
import contextlib
import hashlib
import io
import os
import sys
import time
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

REL = os.path.join(REPO, "psf_compile.py")
C18 = os.path.join(REPO, "patches", "psf_compile_c18_2026-10-06", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-06.2", "psf_compile.py")
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip() for ln in f.read().replace("\r\n", "\n").split("\n")]


def nsha(path):
    ls = lines(path)
    while ls and not ls[-1]:
        ls.pop()
    return hashlib.sha256("\n".join(ls).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    return dict(rel=H.load_module(REL, "psf_compile_rel10063_test"),
                prev=H.load_module(PREV, "psf_compile_prev10063_test"))


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-06.3"
    assert mods["prev"].VERSION == "2026-10-06.2"
    assert mods["rel"].EXACT_MAX_GATE_QUBITS == 6


def test_file_is_c18_except_the_version_lines():
    a, b = lines(REL), lines(C18)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-06.3 -- release") and diff[0][1].startswith("VERSION: 2026-10-06.c18")
    assert diff[1][0].startswith('VERSION = "2026-10-06.3"') and diff[1][1].startswith('VERSION = "2026-10-06.c18"')


def test_previous_release_kept_unchanged():
    assert nsha(PREV) == "1c3dfb0853c2fcd28eacbcdd14abc4c328cf5f40ac8a0a771fbfd70a183dfb84"


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


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def call(m, qc, dev):
    from qiskit_ibm_runtime import fake_provider
    t = getattr(fake_provider, dev)().target
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in t.operation_names]
    with contextlib.redirect_stdout(io.StringIO()):
        return m.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                      entangling_basis="cx", layout_search=True, target=t, seed_transpiler=0,
                                      **RECOMMENDED), t


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeHanoiV2", "FakeGeneva", "FakeKingston"])
def test_same_output_as_the_previous_release(mods, dev):
    for qc in (ring(6, 11), ring(9, 12, True)):
        assert sig(call(mods["rel"], qc, dev)[0]) == sig(call(mods["prev"], qc, dev)[0])


def test_wide_instruction_does_not_abort(mods):
    """2026-10-06.2 aborts the process here (Addendum 377); the release returns a valid circuit."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    n = 48
    terms = [("ZZ", [q, q + 1], 0.5) for q in range(n - 1)] + [("X", [q], 0.7) for q in range(n)]
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_sparse_list(terms, num_qubits=n), time=1.0), range(n))
    t0 = time.perf_counter()
    out, t = call(mods["rel"], qc, "FakeTorino")
    assert time.perf_counter() - t0 < 120
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure"):
            continue
        q = tuple(out.find_bit(b).index for b in ins.qubits)
        assert ins.operation.name in t.operation_names and q in t[ins.operation.name]
