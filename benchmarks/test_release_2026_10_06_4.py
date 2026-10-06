"""Tests for release psf_compile 2026-10-06.4: candidate 2026-10-06.c17 of Addendum 379 (changelog item 45:
alternatives that item 39 cannot check are not built). The candidate's own test is
patches/psf_compile_c17_2026-10-06/test_c17.py; SKIP (Addenda 379-380) tested it on 294 circuits on six devices.
The previous release, 2026-10-06.3, is kept unchanged in patches/psf_compile_release_2026-10-06.3/psf_compile.py.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_06_4.py -q
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
C17 = os.path.join(REPO, "patches", "psf_compile_c17_2026-10-06", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-06.3", "psf_compile.py")
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
    return dict(rel=H.load_module(REL, "psf_compile_rel10064_test"),
                prev=H.load_module(PREV, "psf_compile_prev10064_test"))


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-06.4"
    assert mods["prev"].VERSION == "2026-10-06.3"
    assert mods["rel"].SKIP_STATS == {"resynthesis": 0, "floor": 0, "level3": 0}


def test_file_is_c17_except_the_version_lines():
    a, b = lines(REL), lines(C17)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-06.4 -- release") and diff[0][1].startswith("VERSION: 2026-10-06.c17")
    assert diff[1][0].startswith('VERSION = "2026-10-06.4"') and diff[1][1].startswith('VERSION = "2026-10-06.c17"')


def test_previous_release_kept_unchanged():
    assert nsha(PREV) == "2a49f611fa99b6849afc4aef287aa4c03803aac2d2d1bf32840b4dc9dc80acb5"


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
    for qc in (ring(6, 11), ring(9, 12, True), ring(20, 13, True)):
        assert sig(call(mods["rel"], qc, dev)[0]) == sig(call(mods["prev"], qc, dev)[0])


def test_wide_instruction_skips_what_it_cannot_check(mods):
    """2026-10-06.2 aborts the process here (Addendum 377); the release returns a valid circuit and builds neither
    level 3 nor the floor (item 45)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    n = 48
    terms = [("ZZ", [q, q + 1], 0.5) for q in range(n - 1)] + [("X", [q], 0.7) for q in range(n)]
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_sparse_list(terms, num_qubits=n), time=1.0), range(n))
    before = dict(mods["rel"].SKIP_STATS)
    t0 = time.perf_counter()
    out, t = call(mods["rel"], qc, "FakeTorino")
    assert time.perf_counter() - t0 < 120
    assert mods["rel"].SKIP_STATS["level3"] == before["level3"] + 1
    assert mods["rel"].SKIP_STATS["floor"] == before["floor"] + 1
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure"):
            continue
        q = tuple(out.find_bit(b).index for b in ins.qubits)
        assert ins.operation.name in t.operation_names and q in t[ins.operation.name]
