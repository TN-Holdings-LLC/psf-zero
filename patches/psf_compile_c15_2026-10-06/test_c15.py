"""Tests for candidate psf_compile 2026-10-06.c15 (changelog item 42: candidate_score="kraus", Addendum 339's
`kraus_pur`). The previous release is the current one, 2026-10-05.1 (psf_compile.py).

Run from the repository root:  python -m pytest patches/psf_compile_c15_2026-10-06/test_c15.py -q
"""
import contextlib
import io
import math
import os
import sys
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

FULL = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_c15_test")
    sys.modules["psf_smart_layout"] = lay
    rel = H.load_module(os.path.join(REPO, "patches", "psf_compile_c12_2026-10-05", "psf_compile.py"), "psf_compile_rel_c15_test")  # 2026-10-05.1 (c12's file differs from it only in the version lines); psf_compile.py is 2026-10-06.1 since this candidate's adoption
    c15 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c15_test")
    diag = H.load_module(os.path.join(REPO, "data", "2026-10-04", "h4", "diag", "h4_diag.py"), "h4_diag_c15_test")
    hold6 = H.load_module(os.path.join(REPO, "benchmarks", "hold6_eval.py"), "hold6_eval_c15_test")
    return dict(rel=rel, c15=c15, diag=diag, hold6=hold6)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def call(P, qc, be, score):
    t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    with contextlib.redirect_stdout(io.StringIO()):
        return P.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0, target=t, candidate_score=score, **FULL)


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [round(float(p), 12) for p in i.operation.params])
            for i in c.data]


def state_infid(qc, out):
    """1 - fidelity of the compiled circuit's output on its final-layout qubits with the logical circuit's output."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
    ideal = Statevector(qc)
    fin = list(out.layout.final_index_layout(filter_ancillas=True))
    active = sorted({out.find_bit(b).index for ins in out.data for b in ins.qubits} | set(fin))
    idx = {p: i for i, p in enumerate(active)}
    red = QuantumCircuit(len(active))
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        red.append(ins.operation, [idx[out.find_bit(b).index] for b in ins.qubits])
    sv = Statevector(red)
    keep = [idx[p] for p in fin]
    trace_out = [i for i in range(len(active)) if i not in keep]
    rho = DensityMatrix(partial_trace(sv, trace_out) if trace_out else sv)
    order = sorted(keep)
    perm = [order.index(k) for k in keep]
    n = len(keep)
    t = rho.data.reshape([2] * (2 * n))
    t = np.transpose(t, [n - 1 - perm[v] for v in range(n)][::-1] + [2 * n - 1 - perm[v] for v in range(n)][::-1])
    return float(1 - state_fidelity(DensityMatrix(t.reshape(2 ** n, 2 ** n)), ideal))


def circuits(mods, fam, n, count):
    out = []
    for params, qc in mods["hold6"].family(fam, smoke=False):
        if params["n"] == n:
            out.append(qc)
        if len(out) == count:
            break
    return out


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


def test_versions(mods):
    assert mods["c15"].VERSION == "2026-10-06.c15"
    assert mods["rel"].VERSION == "2026-10-05.c12"  # the previous release, 2026-10-05.1, as its candidate's file


@pytest.mark.parametrize("dev", ["FakeAlgiers", "FakeTorino"])
def test_kraus_cost_matches_addendum_339(mods, dev):
    """kraus_cost equals h4_diag.terms(...)['kraus_pur'] on compiled circuits."""
    be = backend(dev)
    for qc in circuits(mods, "F5", 4, 2) + circuits(mods, "F3", 6, 1) + [ring(6, 3)]:
        out = call(mods["rel"], qc, be, "hybrid")
        k = mods["c15"].kraus_cost(out, be.target)
        est, _ = mods["diag"].terms(out, be.target)
        assert abs(k - est["kraus_pur"]) <= 1e-12 * max(1.0, abs(k)), (k, est["kraus_pur"])


@pytest.mark.parametrize("dev", ["FakeAuckland", "FakeTorino"])
def test_hybrid_unchanged(mods, dev):
    be = backend(dev)
    for qc in circuits(mods, "F5", 4, 1) + circuits(mods, "F3", 6, 1) + [ring(6, 1)]:
        assert sig(call(mods["c15"], qc, be, "hybrid")) == sig(call(mods["rel"], qc, be, "hybrid"))


@pytest.mark.parametrize("dev", ["FakeAlgiers", "FakeMarrakesh"])
def test_kraus_outputs_exact(mods, dev):
    be = backend(dev)
    for qc in circuits(mods, "F5", 4, 1) + circuits(mods, "F1", 6, 1) + circuits(mods, "F6", 5, 1):
        assert state_infid(qc, call(mods["c15"], qc, be, "kraus")) <= 1e-6


def test_h4_case_takes_the_floor_candidate(mods):
    """Addendum 339: on FakeAlgiers 4-qubit GHZ chains (HOLD6's F5, n = 4) hybrid keeps the release's circuit and
    kraus_pur takes the floor-placed one."""
    c15 = mods["c15"]
    be = backend("FakeAlgiers")
    for qc in circuits(mods, "F5", 4, 3):
        picks = []
        for score in ("hybrid", "kraus"):
            before = dict(c15.COMPARE_STATS)
            call(c15, qc, be, score)
            picks.append([k for k in ("psf", "floor", "level3") if c15.COMPARE_STATS[k] > before[k]])
        assert picks == [["psf"], ["floor"]], picks


def test_unknown_score_rejected(mods):
    with pytest.raises(ValueError):
        call(mods["c15"], ring(4, 0), backend("FakeTorino"), "nope")
