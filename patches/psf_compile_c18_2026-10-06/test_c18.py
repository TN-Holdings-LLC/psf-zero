"""Tests for candidate psf_compile 2026-10-06.c18 (changelog item 44: item 39's checks never turn a wide instruction
into a matrix) against the release 2026-10-06.2 (psf_compile.py).

Run from the repository root:  python -m pytest patches/psf_compile_c18_2026-10-06/test_c18.py -q
"""
import contextlib
import io
import os
import subprocess
import sys
import textwrap
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
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c18_test")
    c18 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c18_test")
    return dict(rel=rel, c18=c18)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def rec_call(mod, qc, dev):
    t = backend(dev).target
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in t.operation_names]
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                        entangling_basis="cx", layout_search=True, target=t, seed_transpiler=0,
                                        **RECOMMENDED)


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


def hamiltonian(n, kind, seed=0):
    """A PauliEvolutionGate on all n qubits, as Benchpress builds HamLib tests (one gate, time 1).
    kind "ising": ZZ on neighbours + X on every qubit (non-commuting); "zz": ZZ and Z only (commuting);
    "random": 4n random Pauli strings of weight 2-5 (non-commuting)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    rng = np.random.default_rng(seed)
    terms = []
    if kind in ("ising", "zz"):
        for q in range(n - 1):
            terms.append(("ZZ", [q, q + 1], float(rng.uniform(0.2, 1.0))))
        for q in range(n):
            terms.append(("X" if kind == "ising" else "Z", [q], float(rng.uniform(0.2, 1.0))))
    else:
        for _ in range(4 * n):
            w = int(rng.integers(2, 6))
            qs = sorted(rng.choice(n, size=w, replace=False).tolist())
            terms.append(("".join(rng.choice(list("XYZ"), size=w)), qs, float(rng.uniform(-1, 1))))
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_sparse_list(terms, num_qubits=n), time=1.0), range(n))
    return qc


def product_state(n, seed):
    rng = np.random.default_rng(seed)
    psi = np.array(1.0 + 0j)
    for _ in range(n):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi = np.multiply.outer(psi, v / np.linalg.norm(v))
    return psi


def infidelity_to_operator(mod, qc, ops, seed=7):
    """State infidelity between Qiskit's Operator(qc) and `ops` on a random product state (mod's conventions)."""
    from qiskit.quantum_info import Operator
    n = qc.num_qubits
    psi = product_state(n, seed)
    pos = {q: q for q in range(n)}
    a = mod._apply_ops(psi, [(Operator(qc).data, tuple(range(n)))], pos).ravel()
    b = mod._apply_ops(psi, ops, pos).ravel()
    return 1.0 - abs(np.vdot(a, b)) ** 2


def test_versions(mods):
    assert mods["c18"].VERSION == "2026-10-06.c18"
    assert mods["rel"].VERSION == "2026-10-06.2"
    assert mods["c18"].EXACT_MAX_GATE_QUBITS == 6


def test_release_defect_reproduced():
    """Documents the defect: the release's _implements aborts the process on a 48-qubit PauliEvolutionGate."""
    code = textwrap.dedent(f"""
        import sys, warnings
        warnings.simplefilter("ignore")
        sys.path[:0] = [{os.path.join(REPO, 'benchmarks')!r}, {REPO!r}, {HERE!r}]
        import test_c18 as T
        import psf_compile as rel
        assert rel.VERSION == "2026-10-06.2"
        qc = T.hamiltonian(48, "ising")
        print("returned", rel._implements(qc, qc), flush=True)
    """)
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO)
    print(f"release: returncode {p.returncode}, {time.perf_counter() - t0:.1f} s, stderr tail {p.stderr[-200:]!r}")
    assert p.returncode != 0 and "returned" not in p.stdout


def test_wide_gate_is_not_made_a_matrix(mods):
    c18 = mods["c18"]
    qc = hamiltonian(48, "ising")
    before = dict(c18.EXACT_STATS)
    t0 = time.perf_counter()
    assert c18._implements(qc, qc) is False
    assert c18._same_action(qc, qc) is False
    assert time.perf_counter() - t0 < 5
    assert c18.EXACT_STATS["not_checkable"] == before["not_checkable"] + 2
    ops = c18._ops_of(hamiltonian(12, "random"))
    assert ops and max(m.shape[0] for m, _ in ops) <= 2 ** 6


@pytest.mark.parametrize("make", ["zz9", "qft8", "mcx7"])
def test_definition_agrees_with_the_matrix_where_both_are_exact(mods, make):
    """Expanding through the definition gives the gate's exact action (Qiskit's Operator) when the definition is
    exact."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCXGate, QFTGate
    if make == "zz9":
        qc = hamiltonian(9, "zz")
    elif make == "qft8":
        qc = QuantumCircuit(8)
        qc.append(QFTGate(8), range(8))
    else:
        qc = QuantumCircuit(7)
        qc.h(range(6))
        qc.append(MCXGate(6), range(7))
    b = mods["c18"]._ops_of(qc)
    assert b and max(m.shape[0] for m, _ in b) <= 2 ** 6
    assert infidelity_to_operator(mods["c18"], qc, b) <= 1e-10


def test_product_formula_is_the_reference_for_non_commuting_terms(mods):
    """Reported, not a pass condition: how far the exact exponential (Operator) is from the definition (product
    formula)."""
    qc = hamiltonian(8, "ising")
    d = infidelity_to_operator(mods["c18"], qc, mods["c18"]._ops_of(qc))
    print(f"8-qubit Ising, time 1: state infidelity exact exponential vs product formula = {d:.3e}")
    assert d >= 0.0


def test_matrices_unchanged_for_narrow_instructions(mods):
    from qiskit.circuit.random import random_circuit
    for seed in range(5):
        qc = random_circuit(6, 12, max_operands=3, seed=seed)
        a, b = mods["rel"]._ops_of(qc), mods["c18"]._ops_of(qc)
        assert (a is None) == (b is None)
        if a is None:
            continue
        assert len(a) == len(b)
        for (ma, qa), (mb, qb) in zip(a, b):
            assert qa == qb and np.array_equal(ma, mb)


@pytest.mark.parametrize("dev", ["FakeTorino", "FakeHanoiV2", "FakeGeneva", "FakeKingston"])
def test_outputs_unchanged_on_narrow_circuits(mods, dev):
    for qc in (ring(5, 1), ring(7, 2, True), ring(12, 3, True)):
        assert sig(rec_call(mods["c18"], qc, dev)) == sig(rec_call(mods["rel"], qc, dev))


@pytest.mark.parametrize("n,kind", [(14, "random"), (48, "ising")])
def test_recommended_call_finishes_on_hamiltonians(mods, n, kind):
    """The release times out (14 qubits) or aborts (48 qubits) on Benchpress's HamLib inputs; c18 finishes with a
    circuit in the device's gate set on its couplings."""
    t = backend("FakeTorino").target
    qc = hamiltonian(n, kind)
    t0 = time.perf_counter()
    out = rec_call(mods["c18"], qc, "FakeTorino")
    dt = time.perf_counter() - t0
    print(f"c18 recommended call, {n} qubits ({kind}): {dt:.1f} s, "
          f"cz {out.count_ops().get('cz', 0)}, EXACT_STATS {mods['c18'].EXACT_STATS}")
    assert dt < 300
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure"):
            continue
        q = tuple(out.find_bit(b).index for b in ins.qubits)
        assert ins.operation.name in t.operation_names and q in t[ins.operation.name]
