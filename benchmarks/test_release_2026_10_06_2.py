"""Tests for release psf_compile 2026-10-06.2: candidate 2026-10-06.c16 of Addendum 372 (changelog item 43: item 31's
recompile keeps the first output when no placement avoids the failed elements). The candidate's own test is
patches/psf_compile_c16_2026-10-06/test_c16.py; PL-REDO (Addenda 372-373) tested it in the PennyLane loop. The previous
release, 2026-10-06.1, is kept unchanged in patches/psf_compile_release_2026-10-06.1/psf_compile.py.

Since 2026-10-06.3 (Addendum 378) 2026-10-06.2 is loaded from its kept copy,
patches/psf_compile_release_2026-10-06.2/psf_compile.py.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_06_2.py -q
"""
import contextlib
import hashlib
import io
import os
import sys
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

# kept copy since 2026-10-06.3 (Addendum 378)
REL = os.path.join(REPO, "patches", "psf_compile_release_2026-10-06.2", "psf_compile.py")
C16 = os.path.join(REPO, "patches", "psf_compile_c16_2026-10-06", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-06.1", "psf_compile.py")
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
    return dict(rel=H.load_module(REL, "psf_compile_rel10062_test"),
                prev=H.load_module(PREV, "psf_compile_prev10062_test"))


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-06.2"
    assert mods["prev"].VERSION == "2026-10-06.1"
    assert "unavoidable" in mods["rel"].PRUNE_STATS


def test_file_is_c16_except_the_version_lines():
    a, b = lines(REL), lines(C16)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-06.2 -- release") and diff[0][1].startswith("VERSION: 2026-10-06.c16")
    assert diff[1][0].startswith('VERSION = "2026-10-06.2"') and diff[1][1].startswith('VERSION = "2026-10-06.c16"')


def test_previous_release_kept_unchanged():
    assert nsha(PREV) == "bf4630d6356d8e288902fc1cf5460a0929b7fe6b385fa0f1d6faf8a8971d9246"


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


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeHanoiV2", "FakeGeneva", "FakeKingston"])
def test_same_output_as_the_previous_release(mods, dev):
    from qiskit_ibm_runtime import fake_provider
    t = getattr(fake_provider, dev)().target
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in t.operation_names]

    def call(m, qc):
        with contextlib.redirect_stdout(io.StringIO()):
            return m.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                          entangling_basis="cx", layout_search=True, target=t, seed_transpiler=0,
                                          **RECOMMENDED)

    for qc in (ring(6, 11), ring(9, 12, True)):
        assert sig(call(mods["rel"], qc)) == sig(call(mods["prev"], qc))
