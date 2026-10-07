"""Tests for candidate psf_compile 2026-10-07.c19 (changelog item 46: fewer, larger matrices in the state-vector
estimates and checks) against the release 2026-10-06.4 (psf_compile.py).

Run from the repository root:  python -m pytest patches/psf_compile_c19_2026-10-07/test_c19.py -q -s
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
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c19_test")
    c19 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c19_test")
    return dict(rel=rel, c19=c19)


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
    """SKIP's generator (benchmarks/skip_eval.py, Addendum 379), at seeds not used by SKIP."""
    import skip_eval
    return skip_eval.family_circuit(name, n, np.random.default_rng(seed))


def test_versions(mods):
    assert mods["c19"].VERSION == "2026-10-07.c19"
    assert mods["rel"].VERSION == "2026-10-06.4"


def test_fused_ops_have_the_same_action(mods):
    c19 = mods["c19"]
    rng = np.random.default_rng(46)

    def ru(d):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
        return q

    n, worst, shrink = 9, 0.0, []
    for _ in range(20):
        ops = []
        for _ in range(200):
            r = rng.random()
            k = 1 if r < 0.6 else (2 if r < 0.9 else 3)
            ops.append((ru(2 ** k), tuple(int(x) for x in rng.choice(n, k, replace=False))))
        psi = np.array(1.0 + 0j)
        for _ in range(n):
            v = rng.normal(size=2) + 1j * rng.normal(size=2)
            psi = np.multiply.outer(psi, v / np.linalg.norm(v))
        pos = {i: i for i in range(n)}
        fused = c19._fuse_ops(ops)
        a = c19._apply_ops(psi, ops, pos).ravel()
        b = c19._apply_ops(psi, fused, pos).ravel()
        worst = max(worst, 1.0 - abs(np.vdot(a, b)) ** 2)
        shrink.append(len(fused) / len(ops))
    print(f"fusion: worst state infidelity {worst:.1e}, ops kept {min(shrink):.2f}-{max(shrink):.2f}")
    assert worst <= 1e-12


def test_populations_are_the_reduced_diagonal(mods):
    rng = np.random.default_rng(7)
    psi = rng.normal(size=(2,) * 8) + 1j * rng.normal(size=(2,) * 8)
    psi /= np.linalg.norm(psi)
    for ax in range(8):
        m = np.moveaxis(psi, ax, 0).reshape(2, -1)
        rho = m @ m.conj().T
        p0, p1 = mods["c19"]._populations(psi, ax)
        assert abs(p0 - rho[0, 0].real) <= 1e-14 and abs(p1 - rho[1, 1].real) <= 1e-14


@pytest.mark.parametrize("dev", DEVICES)
def test_estimates_and_checks_agree(mods, dev):
    """On compiled circuits, the two estimates agree to rounding and the checks give the same answers, including on
    a corrupted circuit."""
    from qiskit import QuantumCircuit
    rel, c19 = mods["rel"], mods["c19"]
    t = target(dev)
    for name, n, seed in (("ring", 8, 1), ("brick", 10, 2), ("pauli", 10, 3), ("qft", 8, 4)):
        qc = family(name, n, 46_000_000 + seed)
        out, _ = timed_call(rel, qc, dev)
        for f in ("excitation_cost", "hybrid_cost"):
            a, b = getattr(rel, f)(out, t), getattr(c19, f)(out, t)
            assert abs(a - b) <= 1e-12 * max(1.0, abs(a)), (f, name, a, b)
        assert c19._implements(qc, out) == rel._implements(qc, out)
        bad = out.copy()
        q0 = bad.layout.final_index_layout(filter_ancillas=True)[0]
        extra = QuantumCircuit(bad.num_qubits)
        extra.rz(0.1, q0)
        bad.compose(extra, inplace=True)
        bad._layout = out.layout
        assert rel._implements(qc, bad) is False and c19._implements(qc, bad) is False
        assert c19._same_action(out, out) is True and c19._same_action(out, bad) is False


@pytest.mark.parametrize("dev", DEVICES)
def test_outputs_unchanged(mods, dev):
    for name, n, seed in (("ring", 5, 11), ("ring", 12, 12), ("brick", 12, 13), ("pauli", 12, 14), ("qft", 10, 15)):
        qc = family(name, n, 46_100_000 + seed)
        if seed % 2:
            qc.measure_all()
        assert sig(timed_call(mods["c19"], qc, dev)[0]) == sig(timed_call(mods["rel"], qc, dev)[0]), (name, n)


def test_ties_keep_the_earlier_candidate(mods):
    c19 = mods["c19"]
    assert not c19._lower(0.2323375484657434, 0.23233754846574342)   # within rounding: a tie
    assert c19._lower(0.1, 0.2) and not c19._lower(0.2, 0.1)


def test_smoke_tie_case(mods):
    """FUSE's smoke circuit 3 on FakeKingston (Addendum 383): 2026-10-06.4 keeps its own circuit on an exact tie with
    level 3's (they differ by two rz gates); the first version of c19 broke the tie by 2e-17. Now the same circuit."""
    qc = family("ring", 16, 82_500_003)
    qc.measure_all()
    assert sig(timed_call(mods["c19"], qc, "FakeKingston")[0]) == sig(timed_call(mods["rel"], qc, "FakeKingston")[0])


def test_faster_at_16_qubits(mods):
    """A 16-qubit Hamiltonian of SKIP's family on FakeTorino: the same circuit, in less time."""
    qc = family("pauli", 16, 46_200_000)
    qc.measure_all()
    o19, t19 = timed_call(mods["c19"], qc, "FakeTorino")
    orl, trl = timed_call(mods["rel"], qc, "FakeTorino")
    print(f"pauli16 on FakeTorino: release {trl:.1f} s, c19 {t19:.1f} s ({t19 / trl:.2f})")
    assert sig(o19) == sig(orl)
    assert t19 < 0.7 * trl
