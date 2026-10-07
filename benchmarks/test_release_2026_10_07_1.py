"""Tests for release psf_compile 2026-10-07.1: candidate 2026-10-07.c23 of Addendum 394, which carries changelog
items 46-50 (accepted in Addenda 385, 388, 392 and 395). The candidates' own tests are
patches/psf_compile_c19_2026-10-07/test_c19.py to patches/psf_compile_c23_2026-10-07/test_c23.py; the
pre-registered tests FUSE, TRACK, BP-MOCK, BP-MOCK2 and CANCEL (Addenda 383-395) measured them.
The previous release, 2026-10-06.4, is kept unchanged in patches/psf_compile_release_2026-10-06.4/psf_compile.py.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_07_1.py -q -s
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

REL = os.path.join(REPO, "psf_compile.py")
C23 = os.path.join(REPO, "patches", "psf_compile_c23_2026-10-07", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-06.4", "psf_compile.py")
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
    return dict(rel=H.load_module(REL, "psf_compile_rel10071_test"),
                prev=H.load_module(PREV, "psf_compile_prev10071_test"))


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def call(m, qc, dev, recommended=False):
    b = backend(dev)
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in b.operation_names]
    kw = dict(target=b.target, **RECOMMENDED) if recommended else {}
    with contextlib.redirect_stdout(io.StringIO()):
        return m.compile_for_hardware(qc, coupling_map=b.coupling_map, basis_gates=basis, entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0, **kw), b


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def q2(c):
    return sum(1 for i in c.data if len(i.qubits) == 2 and i.operation.name != "barrier")


def valid(c, b):
    """Every gate is in the device's basis and every two-qubit gate sits on a coupler (either direction)."""
    basis = {g for g in ("cx", "cz", "rz", "sx", "x") if g in b.operation_names} | {"barrier", "measure"}
    edges = {tuple(e) for e in b.coupling_map.get_edges()}
    for ins in c.data:
        if ins.operation.name not in basis:
            return False
        if len(ins.qubits) == 2 and ins.operation.name != "barrier":
            a, d = (c.find_bit(q).index for q in ins.qubits)
            if (a, d) not in edges and (d, a) not in edges:
                return False
    return True


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


def bvlike(n):
    """Benchpress's trivial_bvlike_circuit, as in test_c23.py."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for k in range(n - 1):
        qc.cx(k, n - 1)
    qc.x(n - 1)
    qc.z(n - 2)
    for k in range(n - 2, -1, -1):
        qc.cx(k, n - 1)
    return qc


def expanded(qc):
    from qiskit import transpile
    return transpile(qc, basis_gates=["u", "cx"], optimization_level=0)


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-07.1"
    assert mods["prev"].VERSION == "2026-10-06.4"
    assert set(mods["rel"].CANCEL_STATS) == {"tried", "cancelled_kept", "original_kept", "failed"}
    assert set(mods["rel"].UNROLL_STATS) == {"unrolled", "failed"}
    assert mods["rel"].ESTIMATE_TIE_TOL == 1e-12


def test_file_is_c23_except_the_version_lines():
    a, b = lines(REL), lines(C23)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-07.1 -- release")
    assert diff[0][1].startswith("VERSION: 2026-10-07.c23")
    assert diff[1][0].startswith('VERSION = "2026-10-07.1"') and diff[1][1].startswith('VERSION = "2026-10-07.c23"')


def test_candidate_and_previous_release_kept_unchanged():
    assert nsha(C23) == "568de9e691796e4efb330dff3f2ed61aa1cca4cc888286f9727854cf8c744420"
    assert nsha(PREV) == "69fe51d2d503638ceb4d067a0d86a5b27c38586694ec84996dea5e7ab6dab7aa"


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeHanoiV2", "FakeGeneva", "FakeKingston"])
def test_same_output_as_the_previous_release(mods, dev):
    """Inputs with no instruction on three or more qubits and no two-qubit gate that cancels: 2026-10-06.4's circuit,
    with the recommended call (items 46, 47 and 49 change its time, not its circuit) and with the default call."""
    for qc in (ring(6, 11), ring(9, 12, True), ring(20, 13, True)):
        assert sig(call(mods["rel"], qc, dev, True)[0]) == sig(call(mods["prev"], qc, dev, True)[0])
        assert sig(call(mods["rel"], qc, dev)[0]) == sig(call(mods["prev"], qc, dev)[0])


@pytest.mark.parametrize("n", (8, 30))
def test_bvlike_cancels(mods, n):
    """Item 50: Benchpress's BV-like circuit has no two-qubit gate left; 2026-10-06.4 kept them."""
    qc = bvlike(n)
    kept = mods["rel"].CANCEL_STATS["cancelled_kept"]
    out, b = call(mods["rel"], qc, "FakeTorino")
    old, _ = call(mods["prev"], qc, "FakeTorino")
    print(f"\nBV-like {n}: 2026-10-06.4 {q2(old)}, 2026-10-07.1 {q2(out)}")
    assert q2(out) == 0 and q2(old) > 0
    assert mods["rel"].CANCEL_STATS["cancelled_kept"] == kept + 1
    assert valid(out, b)
    if n <= 16:  # _implements refuses (returns False for) circuits of more than 16 qubits (item 45)
        assert mods["rel"]._implements(qc, out)


def test_three_qubit_instructions_expanded(mods):
    """Item 48: instructions on three or more qubits are expanded first; the output is valid and exact."""
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(50_200_000)
    n = 6
    qc = QuantumCircuit(n)
    for _ in range(6):
        a, b, c = (int(x) for x in rng.choice(n, 3, replace=False))
        qc.h(a)
        qc.ccx(a, b, c)
        qc.cx(c, a)
        qc.ry(float(rng.uniform(0, 6)), b)
    before = mods["rel"].UNROLL_STATS["unrolled"]
    out, b = call(mods["rel"], qc, "FakeHanoiV2")
    old, _ = call(mods["prev"], qc, "FakeHanoiV2")
    print(f"\nccx circuit: 2026-10-06.4 {q2(old)}, 2026-10-07.1 {q2(out)}")
    assert mods["rel"].UNROLL_STATS["unrolled"] > before
    assert valid(out, b)
    assert mods["rel"]._implements(expanded(qc), out)


def test_random_cancellations_exact(mods):
    """Item 50 on random circuits with commuting CX pairs: valid, exact, and no more two-qubit gates than
    2026-10-06.4."""
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(50_300_000)
    for k in range(3):
        n = int(rng.integers(5, 8))
        qc = QuantumCircuit(n)
        for _ in range(24):
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            r = rng.random()
            if r < 0.3:
                qc.cx(a, b)
                qc.rz(float(rng.uniform(0, 6)), a)
                qc.cx(a, b)
            elif r < 0.6:
                qc.cx(a, b)
            else:
                qc.ry(float(rng.uniform(0, 6)), a)
        out, dev = call(mods["rel"], qc, "FakeHanoiV2")
        old, _ = call(mods["prev"], qc, "FakeHanoiV2")
        print(f"\nrandom {k} ({n} qubits): 2026-10-06.4 {q2(old)}, 2026-10-07.1 {q2(out)}")
        assert valid(out, dev) and mods["rel"]._implements(qc, out), k
        assert q2(out) <= q2(old), k
