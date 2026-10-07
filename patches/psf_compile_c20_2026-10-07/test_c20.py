"""Tests for candidate psf_compile 2026-10-07.c20 (changelog item 47: item 39's checks only where they can change the
output) against candidate 2026-10-07.c19 (patches/psf_compile_c19_2026-10-07/psf_compile.py), on which it is based.

Run from the repository root:  python -m pytest patches/psf_compile_c20_2026-10-07/test_c20.py -q -s
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
DEVICES = ("FakeTorino", "FakeHanoiV2", "FakeGeneva", "FakeKingston")
STATS = ("COMPARE_STATS", "RESYNTH_STATS", "EXACT_STATS")


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c19 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c19_2026-10-07", "psf_compile.py"),
                        "psf_compile_c19_c20_test")
    c20 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c20_test")
    return dict(c19=c19, c20=c20)


def target(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)().target


def timed_call(mod, qc, dev, **kw):
    t = target(dev)
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in t.operation_names]
    opts = dict(RECOMMENDED, **kw)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = mod.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                       entangling_basis="cx", layout_search=True, target=t, seed_transpiler=0, **opts)
    return out, time.perf_counter() - t0


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def family(name, n, seed):
    """SKIP's generator (benchmarks/skip_eval.py, Addendum 379), at seeds not used by SKIP, FUSE or test_c19."""
    import skip_eval
    return skip_eval.family_circuit(name, n, np.random.default_rng(seed))


def checks(mod):
    return mod.EXACT_STATS["checked"]


def test_versions(mods):
    assert mods["c20"].VERSION == "2026-10-07.c20"
    assert mods["c19"].VERSION == "2026-10-07.c19"


@pytest.mark.parametrize("dev", DEVICES)
def test_outputs_unchanged(mods, dev):
    """The recommended call returns c19's circuit; c20 makes no more checks than c19."""
    n19 = n20 = 0
    for name, n, seed in (("ring", 6, 1), ("ring", 12, 2), ("brick", 12, 3), ("pauli", 12, 4), ("qft", 10, 5),
                          ("brick", 8, 6)):
        qc = family(name, n, 47_000_000 + seed)
        if seed % 2:
            qc.measure_all()
        a19, a20 = checks(mods["c19"]), checks(mods["c20"])
        o19 = timed_call(mods["c19"], qc, dev)[0]
        o20 = timed_call(mods["c20"], qc, dev)[0]
        n19 += checks(mods["c19"]) - a19
        n20 += checks(mods["c20"]) - a20
        assert sig(o20) == sig(o19), (name, n)
    print(f"{dev}: item 39 checks c19 {n19}, c20 {n20}")
    assert n20 <= n19


@pytest.mark.parametrize("score", ("excitation", "pauli", "kraus"))
def test_other_scores_and_paths(mods, score):
    """Other candidate scores; with "excitation" and no floor, item 36's own comparison (_compare_level3)."""
    for kw in (dict(candidate_score=score), dict(candidate_score=score, compare_floor=False)):
        for name, n, seed in (("ring", 10, 11), ("qft", 8, 12), ("pauli", 10, 13)):
            qc = family(name, n, 47_100_000 + seed)
            qc.measure_all()
            for dev in ("FakeTorino", "FakeHanoiV2"):
                assert sig(timed_call(mods["c20"], qc, dev, **kw)[0]) == \
                    sig(timed_call(mods["c19"], qc, dev, **kw)[0]), (kw, name, dev)


@pytest.mark.parametrize("verdict", (False, True))
def test_checks_forced(mods, monkeypatch, verdict):
    """With every item 39 check forced to fail (or to pass) in both modules, the outputs still agree: what c20 does
    not check is never what decides the output."""
    for m in (mods["c19"], mods["c20"]):
        monkeypatch.setattr(m, "_implements", lambda qc, out, tol=1e-6, _v=verdict: _v)
        monkeypatch.setattr(m, "_same_action", lambda ref, new, tol=1e-6, _v=verdict: _v)
    for name, n, seed in (("ring", 8, 21), ("brick", 10, 22), ("qft", 8, 23)):
        qc = family(name, n, 47_200_000 + seed)
        qc.measure_all()
        for dev in ("FakeTorino", "FakeKingston"):
            for kw in (dict(), dict(candidate_score="excitation", compare_floor=False)):
                assert sig(timed_call(mods["c20"], qc, dev, **kw)[0]) == \
                    sig(timed_call(mods["c19"], qc, dev, **kw)[0]), (verdict, name, dev, kw)


def test_smoke_tie_case(mods):
    """FUSE's smoke circuit 3 on FakeKingston (Addendum 383): an exact tie with level 3's circuit keeps the release's."""
    qc = family("ring", 16, 82_500_003)
    qc.measure_all()
    assert sig(timed_call(mods["c20"], qc, "FakeKingston")[0]) == sig(timed_call(mods["c19"], qc, "FakeKingston")[0])


def test_faster_at_16_qubits(mods):
    """A 16-qubit Hamiltonian of SKIP's family on FakeTorino: the same circuit; the times are printed, not asserted."""
    qc = family("pauli", 16, 47_300_000)
    qc.measure_all()
    before = {m: {s: dict(getattr(mods[m], s)) for s in STATS} for m in mods}
    o20, t20 = timed_call(mods["c20"], qc, "FakeTorino")
    o19, t19 = timed_call(mods["c19"], qc, "FakeTorino")
    for m in ("c19", "c20"):
        moved = {s: {k: getattr(mods[m], s)[k] - before[m][s].get(k, 0) for k in getattr(mods[m], s)
                     if getattr(mods[m], s)[k] != before[m][s].get(k, 0)} for s in STATS}
        print(f"{m}: {moved}")
    print(f"pauli16 on FakeTorino: c19 {t19:.1f} s, c20 {t20:.1f} s ({t20 / t19:.2f})")
    assert sig(o20) == sig(o19)
