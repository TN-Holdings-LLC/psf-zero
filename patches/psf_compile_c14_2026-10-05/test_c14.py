"""Tests for candidate psf_compile 2026-10-05.c14 (workplace; changelog items 40 and 41) on top of release
2026-10-05.1, adapted from the workplace tests test_c13_readout.py and test_c14_direction.py
(data/2026-10-05/workplace/readout/c13/, data/2026-10-05/workplace/c14/). The workplace compared c14 with c13 and c13
with c12; release 2026-10-05.1 is c12 with only its version lines changed, so the comparisons are made with the
release.

Run from the repository root:  python -m pytest patches/psf_compile_c14_2026-10-05/test_c14.py -q
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
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
for p in (os.path.join(WORK, "readout"), os.path.join(WORK, "depth1"), os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_c14_test")
    sys.modules["psf_smart_layout"] = lay
    rel = H.load_module(os.path.join(REPO, "patches", "psf_compile_c12_2026-10-05", "psf_compile.py"), "psf_compile_rel_c14_test")  # 2026-10-05.1 (c12's file differs from it only in the version lines); psf_compile.py is 2026-10-06.1 since this candidate's adoption
    c14 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c14_test")
    import depth_eval as DE
    import readout_eval as RE
    return dict(rel=rel, c14=c14, DE=DE, RE=RE)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


FULL = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
            candidate_score="hybrid")


def call(P, qc, be, **kw):
    t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return P.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0, target=t, **kw)


def classifier(DE, n=6, L=4, seed=5):
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 1, DE.n_params(n, L))
    return DE.circuit(rng.uniform(-1, 1, n), th, n, L)


def ring(n, seed):
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            qc.ry(float(rng.uniform(-1, 1)), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    return qc


def one_way(t):
    for (a, b), p in t["cx"].items():
        r = t["cx"].get((b, a))
        if p is not None and p.error is not None and p.error >= 0.5 and r is not None and r.error is not None \
                and r.error < 0.5:
            return a, b
    raise AssertionError("no one-way failed coupler")


def test_versions(mods):
    assert mods["c14"].VERSION == "2026-10-05.c14"
    assert mods["rel"].VERSION == "2026-10-05.c12"  # the previous release, 2026-10-05.1, as its candidate's file


def test_readout_cost_sums_measured_qubits_once(mods):
    from qiskit import QuantumCircuit
    c14 = mods["c14"]
    t = backend("FakeTorino").target
    c = QuantumCircuit(t.num_qubits, 3)
    c.x(5)
    c.measure(5, 0)
    c.measure(5, 1)
    c.measure(7, 2)
    assert abs(c14.readout_cost(c, t) - t["measure"][(5,)].error - t["measure"][(7,)].error) < 1e-15
    c2 = QuantumCircuit(t.num_qubits)
    c2.x(5)
    assert c14.readout_cost(c2, t) == 0.0


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeAuckland"])
def test_hybrid_cost_adds_only_readout(mods, dev):
    rel, c14, RE = mods["rel"], mods["c14"], mods["RE"]
    be = backend(dev)
    out = call(rel, RE.measured(classifier(mods["DE"])), be, **FULL)
    h_rel, h14 = rel.hybrid_cost(out, be.target), c14.hybrid_cost(out, be.target)
    assert abs(h14 - h_rel - c14.readout_cost(out, be.target)) < 1e-12
    un = RE.strip_measure(out)
    assert c14.hybrid_cost(un, be.target) == rel.hybrid_cost(un, be.target)


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston", "FakeAuckland"])
def test_unmeasured_identical_to_release(mods, dev):
    be = backend(dev)
    for seed in (1, 2, 3):
        qc = classifier(mods["DE"], n=6, L=2, seed=seed)
        assert mods["RE"].sig(call(mods["c14"], qc, be, **FULL)) == mods["RE"].sig(call(mods["rel"], qc, be, **FULL))


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeKingston", "FakeAuckland"])
def test_measured_output_exact_and_measured_where_logical0_ends(mods, dev):
    RE = mods["RE"]
    qc = classifier(mods["DE"], n=6, L=4, seed=7)
    out = call(mods["c14"], RE.measured(qc), backend(dev), **FULL)
    fin0 = out.layout.final_index_layout(filter_ancillas=True)[0]
    mq = [out.find_bit(g.qubits[0]).index for g in out.data if g.operation.name == "measure"]
    assert mq == [fin0]
    assert RE.state_infid(qc, RE.strip_measure(out)) <= 1e-6


def test_readout_term_never_worse_and_sometimes_better(mods):
    """FakeKingston, 6-qubit classifiers (the workplace's development case): c14 never puts the output on a
    worse-readout qubit than the release does, and does better on at least one."""
    RE = mods["RE"]
    be = backend("FakeKingston")
    t = be.target
    better = 0
    for seed in range(6):
        m = RE.measured(classifier(mods["DE"], n=6, L=4, seed=100 + seed))
        e = []
        for P in (mods["rel"], mods["c14"]):
            out = call(P, m, be, **FULL)
            e.append(t["measure"][(out.layout.final_index_layout(filter_ancillas=True)[0],)].error)
        assert e[1] <= e[0] + 1e-15
        better += e[1] < e[0]
    assert better >= 1


def test_uses_failed_is_direction_aware(mods):
    from qiskit import QuantumCircuit
    rel, c14 = mods["rel"], mods["c14"]
    t = backend("FakeHanoiV2").target
    a, b = one_way(t)  # (a, b) failed, (b, a) healthy
    edges, qubits = c14._failed_elements(t, 0.5)
    ok = QuantumCircuit(t.num_qubits)
    ok.cx(b, a)
    bad = QuantumCircuit(t.num_qubits)
    bad.cx(a, b)
    sym = QuantumCircuit(t.num_qubits)
    sym.cz(b, a)
    assert not c14._uses_failed(ok, edges, qubits) and rel._uses_failed(ok, edges, qubits)
    assert c14._uses_failed(bad, edges, qubits)
    assert c14._uses_failed(sym, edges, qubits)


@pytest.mark.parametrize("dev", ["FakeHanoiV2", "FakeGeneva"])
def test_outputs_exact_and_free_of_failed_directions(mods, dev):
    c14, RE = mods["c14"], mods["RE"]
    t = backend(dev).target
    edges, qubits = c14._failed_elements(t, 0.5)
    for n, seed in ((6, 4), (4, 5)):
        qc = ring(n, seed)
        out = call(c14, qc, backend(dev), **FULL)
        assert not c14._uses_failed(out, edges, qubits)
        assert RE.state_infid(qc, out) <= 1e-6
