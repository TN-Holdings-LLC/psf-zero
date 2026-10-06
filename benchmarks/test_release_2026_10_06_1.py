"""Tests for release psf_compile 2026-10-06.1: candidate 2026-10-05.c14 (changelog items 40-41: readout of measured
qubits in the choice, direction-aware failed-element check) plus candidate 2026-10-06.c15's item 42 (opt-in
`candidate_score="kraus"`). The candidates' own tests are patches/psf_compile_c14_2026-10-05/test_c14.py and
patches/psf_compile_c15_2026-10-06/test_c15.py. The previous release, 2026-10-05.1, is represented by its candidate's
file (patches/psf_compile_c12_2026-10-05/psf_compile.py), which differs from it only in the version lines.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_06_1.py -q
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
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

FULL = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_rel10061_test")
    sys.modules["psf_smart_layout"] = lay
    load = lambda *p: H.load_module(os.path.join(REPO, *p[:-1]), p[-1])
    return dict(rel=load("psf_compile.py", "psf_compile_rel10061_test"),
                prev=load("patches", "psf_compile_c12_2026-10-05", "psf_compile.py", "psf_compile_prev10061_test"),
                c14=load("patches", "psf_compile_c14_2026-10-05", "psf_compile.py", "psf_compile_c14_rel10061_test"),
                c15=load("patches", "psf_compile_c15_2026-10-06", "psf_compile.py", "psf_compile_c15_rel10061_test"))


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def call(P, qc, be, score="hybrid"):
    t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return P.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0, target=t, candidate_score=score, **FULL)


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [round(float(p), 12) for p in i.operation.params])
            for i in c.data]


def ring(n, seed, measured=False):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            qc.ry(float(rng.uniform(-1, 1)), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    if measured:
        qc.measure_all()
    return qc


def ghz(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-06.1"
    assert mods["prev"].VERSION == "2026-10-05.c12"
    assert mods["c14"].VERSION == "2026-10-05.c14"
    assert mods["c15"].VERSION == "2026-10-06.c15"


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston", "FakeHanoiV2"])
def test_hybrid_is_c14(mods, dev):
    """With the recommended call the release is c14, with and without measurements."""
    be = backend(dev)
    for qc in (ring(4, 1), ring(4, 1, measured=True), ring(6, 2, measured=True), ghz(5)):
        assert sig(call(mods["rel"], qc, be)) == sig(call(mods["c14"], qc, be))


@pytest.mark.parametrize("dev", ["FakeAuckland", "FakeTorino"])
def test_unmeasured_is_the_previous_release(mods, dev):
    be = backend(dev)
    for qc in (ring(4, 3), ring(6, 4), ghz(4)):
        assert sig(call(mods["rel"], qc, be)) == sig(call(mods["prev"], qc, be))


@pytest.mark.parametrize("dev", ["FakeAlgiers", "FakeMarrakesh"])
def test_kraus_is_c15(mods, dev):
    """`candidate_score="kraus"` gives c15's circuit, and kraus_cost is c15's to the last bit."""
    be = backend(dev)
    for qc in (ghz(4), ring(4, 5), ring(5, 6)):
        out = call(mods["rel"], qc, be, "kraus")
        assert sig(out) == sig(call(mods["c15"], qc, be, "kraus"))
        assert mods["rel"].kraus_cost(out, be.target) == mods["c15"].kraus_cost(out, be.target)


def test_kraus_ignores_measurements(mods):
    """kraus_cost skips `measure` (it has no readout term; changelog item 42), so a measured circuit gets the
    estimate of the same circuit without measurements."""
    be = backend("FakeTorino")
    out = call(mods["rel"], ring(4, 7, measured=True), be)
    bare = out.copy()
    bare.remove_final_measurements(inplace=True)
    assert mods["rel"].kraus_cost(out, be.target) == pytest.approx(mods["rel"].kraus_cost(bare, be.target), rel=1e-12)
    assert mods["rel"].readout_cost(out, be.target) > 0


def test_unknown_score_rejected(mods):
    with pytest.raises(ValueError):
        call(mods["rel"], ring(4, 0), backend("FakeTorino"), "nope")
