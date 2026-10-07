"""Tests for candidate psf_compile 2026-10-07.c22 (changelog item 49: the estimates follow single-qubit gates on 2x2
reduced states) against candidate 2026-10-07.c21 (patches/psf_compile_c21_2026-10-07/psf_compile.py), on which it is
based.

Run from the repository root:  python -m pytest patches/psf_compile_c22_2026-10-07/test_c22.py -q -s
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


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c21 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c21_2026-10-07", "psf_compile.py"),
                        "psf_compile_c21_c22_test")
    c22 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c22_test")
    return dict(c21=c21, c22=c22)


def target(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)().target


def timed_call(mod, qc, dev):
    t = target(dev)
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in t.operation_names]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = mod.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                       entangling_basis="cx", layout_search=True, target=t, seed_transpiler=0,
                                       **RECOMMENDED)
    return out, time.perf_counter() - t0


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def family(name, n, seed):
    """SKIP's generator (benchmarks/skip_eval.py), at seeds not used by SKIP, FUSE or the earlier tests."""
    import skip_eval
    return skip_eval.family_circuit(name, n, np.random.default_rng(seed))


def test_versions(mods):
    assert mods["c22"].VERSION == "2026-10-07.c22"
    assert mods["c21"].VERSION == "2026-10-07.c21"


def test_rho1_is_the_reduced_state(mods):
    rng = np.random.default_rng(49)
    psi = rng.normal(size=(2,) * 7) + 1j * rng.normal(size=(2,) * 7)
    psi /= np.linalg.norm(psi)
    for ax in range(7):
        m = np.moveaxis(psi, ax, 0).reshape(2, -1)
        assert np.allclose(mods["c22"]._rho1(psi, ax), m @ m.conj().T, atol=1e-14)


@pytest.mark.parametrize("dev", DEVICES)
def test_estimates_agree(mods, dev):
    """On compiled circuits (and their re-synthesis), both estimates agree with c21's to 1e-12 (relative)."""
    c21, c22 = mods["c21"], mods["c22"]
    t = target(dev)
    worst = 0.0
    for name, n, seed in (("ring", 8, 1), ("brick", 12, 2), ("pauli", 10, 3), ("qft", 12, 4), ("pauli", 14, 5)):
        qc = family(name, n, 49_000_000 + seed)
        qc.measure_all()
        out, _ = timed_call(c21, qc, dev)
        for circ in (out, c21._resynthesis_candidate(out, t, 0.5)):
            for f in ("excitation_cost", "hybrid_cost"):
                a, b = getattr(c21, f)(circ, t), getattr(c22, f)(circ, t)
                rel = abs(a - b) / max(abs(a), 1e-300)
                worst = max(worst, rel)
                assert rel <= 1e-12, (f, name, n, a, b)
    print(f"\n{dev}: worst relative difference of the estimates {worst:.1e}")


@pytest.mark.parametrize("dev", DEVICES)
def test_outputs_unchanged(mods, dev):
    for name, n, seed in (("ring", 6, 11), ("ring", 12, 12), ("brick", 12, 13), ("pauli", 12, 14), ("qft", 10, 15),
                          ("brick", 8, 16)):
        qc = family(name, n, 49_100_000 + seed)
        if seed % 2:
            qc.measure_all()
        assert sig(timed_call(mods["c22"], qc, dev)[0]) == sig(timed_call(mods["c21"], qc, dev)[0]), (name, n)


def test_smoke_tie_case(mods):
    """FUSE's smoke circuit 3 on FakeKingston (Addendum 383), the exact tie with level 3's circuit."""
    qc = family("ring", 16, 82_500_003)
    qc.measure_all()
    assert sig(timed_call(mods["c22"], qc, "FakeKingston")[0]) == sig(timed_call(mods["c21"], qc, "FakeKingston")[0])


def test_faster_at_16_qubits(mods):
    """A 16-qubit Hamiltonian and a 16-qubit QFT of SKIP's families on FakeTorino: the same circuit; times printed."""
    for name, seed in (("pauli", 49_200_000), ("qft", 49_200_001)):
        qc = family(name, 16, seed)
        qc.measure_all()
        o22, t22 = timed_call(mods["c22"], qc, "FakeTorino")
        o21, t21 = timed_call(mods["c21"], qc, "FakeTorino")
        print(f"\n{name}16 on FakeTorino: c21 {t21:.1f} s, c22 {t22:.1f} s ({t22 / t21:.2f})")
        assert sig(o22) == sig(o21)
