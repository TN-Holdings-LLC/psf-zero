"""Tests for candidate psf_compile 2026-10-05.c12 (changelog item 39: equivalence check of Qiskit-made circuits).
Helpers are copied from patches/psf_compile_c11_2026-10-04/test_c11_hybrid.py; `with_prep` and `state_infid` are the
workplace probe's (Addendum 340), independent of the candidate's own checks.

Run from the repository root:  python -m pytest patches/psf_compile_c12_2026-10-05/test_c12_exact.py -q
"""
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
sys.path.insert(0, REPO)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_c12_test")
    sys.modules["psf_smart_layout"] = lay
    return (H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c12_test"),
            H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c12_test"))


def qiskit_17057_present():
    """True if Qiskit's CX-basis synthesis shows issue #17057 in this environment, on the issue's own input
    (exp(i(0.6 XX + 0.3 YY + c ZZ)), c = 1e-7). The failure hinges on a rounding-level offset: it appeared in the
    Linux environments at home and at the workplace, and not in a Windows environment (Addendum 357). Where it does
    not appear, Qiskit makes no wrong circuit for the checks to refuse, so the refusal counts are asserted only where
    it does; exactness is asserted everywhere."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import CXGate
    from qiskit.quantum_info import Operator, average_gate_fidelity
    from qiskit.synthesis import TwoQubitBasisDecomposer
    core = QuantumCircuit(2)
    core.rxx(-1.2, 0, 1)
    core.ryy(-0.6, 0, 1)
    core.rzz(-2e-7, 0, 1)
    u = Operator(core)
    # The decomposer itself (data/2026-10-06/windows/check17057.py): in a Linux environment where it fails (7.0e-2),
    # transpile at level 1 was found exact (2.2e-16), so transpile alone does not show the failure (Addendum 369).
    dec = Operator(TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")(u.data))
    out = Operator(transpile(core, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1))
    return max(1 - average_gate_fidelity(dec, u), 1 - average_gate_fidelity(out, u)) > 1e-6


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def ring(n, layers=2, seed=0):
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
            qc.rz(float(rng.uniform(-math.pi, math.pi)), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    return qc


def chain(n, seed=0):
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
    for q in range(n - 1):
        qc.cx(q, q + 1)
        qc.rz(float(rng.uniform(-math.pi, math.pi)), q + 1)
    return qc


def kw_for(tgt):
    return dict(coupling_map=tgt.build_coupling_map(),
                basis_gates=[g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names],
                entangling_basis="cx", layout_search=True, seed_transpiler=0)


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [float(p) for p in i.operation.params])
            for i in c.data]


def xxz_chain(n=6, steps=2, seed=0):
    """An open XXZ Trotter chain like HOLD2's F3 (Addendum 322's case)."""
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    jx, jy = rng.uniform(0.5, 1.5, 2)
    jz = rng.uniform(0.2, 1.0) * jx
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for parity in (0, 1):
            for a in range(parity, n - 1, 2):
                qc.rxx(0.2 * jx, a, a + 1)
                qc.ryy(0.2 * jy, a, a + 1)
                qc.rzz(0.2 * jz, a, a + 1)
        for q in range(n):
            qc.rz(float(rng.uniform(-0.2, 0.2)), q)
    return qc


def xxz_ring(n=6, steps=2, seed=0):
    """A periodic XXZ Trotter ring like HOLD3's F3 periodic cell (Addendum 326's case)."""
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    jx, jy = rng.uniform(0.5, 1.5, 2)
    jz = rng.uniform(0.2, 1.0) * jx
    bonds = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for parity in (0, 1):
            for a, b in bonds:
                if a % 2 == parity:
                    qc.rxx(0.2 * jx, a, b)
                    qc.ryy(0.2 * jy, a, b)
                    qc.rzz(0.2 * jz, a, b)
    return qc


def l3(qc, tgt):
    from qiskit import transpile
    return transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)


def ghz(n=6, seed=0):
    """A GHZ chain with random final rotations, like HOLD4's F5 cell (Addendum 330's case)."""
    import numpy as np
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
        qc.rz(float(rng.uniform(-math.pi, math.pi)), q)
    return qc


REC = dict(placement_refine=True, final_resynthesis="select", compare_level3=True)


FULL = dict(REC, compare_floor=True, candidate_score="hybrid")


def with_prep(qc, seed):
    """`qc` with a seeded random single-qubit layer prepended, so that a missing rotation shows in the output state
    (the workplace probe's construction, Addendum 340)."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    out = QuantumCircuit(qc.num_qubits)
    for q in range(qc.num_qubits):
        out.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
    out.compose(qc, inplace=True)
    return out


def state_infid(qc, out):
    """Noiseless output-state infidelity of the compiled circuit `out` against the logical circuit `qc` (from |0...0>),
    on the touched physical qubits, after undoing the final layout. The workplace probe's function (Addendum 340),
    written with Qiskit's Statevector and partial_trace, independently of psf_compile's item-39 checks."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
    ideal = Statevector(qc)
    active = sorted({out.find_bit(b).index for ins in out.data for b in ins.qubits})
    fin = out.layout.final_index_layout(filter_ancillas=True) if out.layout is not None else list(range(qc.num_qubits))
    active = sorted(set(active) | set(fin))
    idx = {p: i for i, p in enumerate(active)}
    red = QuantumCircuit(len(active))
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        red.append(ins.operation, [idx[out.find_bit(b).index] for b in ins.qubits])
    sv = Statevector(red)
    keep = [idx[p] for p in fin]
    trace_out = [i for i in range(len(active)) if i not in keep]
    rho = partial_trace(sv, trace_out) if trace_out else sv
    order = sorted(keep)
    perm = [order.index(k) for k in keep]
    rho = DensityMatrix(rho)
    n = len(keep)
    t = rho.data.reshape([2] * (2 * n))
    row_axes = [n - 1 - perm[v] for v in range(n)][::-1]
    col_axes = [2 * n - 1 - perm[v] for v in range(n)][::-1]
    t = np.transpose(t, row_axes + col_axes).reshape(2 ** n, 2 ** n)
    return float(1 - state_fidelity(DensityMatrix(t), ideal))


def near_boundary(kind, n, count):
    """B17's generators (benchmarks/b17_practice_eval.py), first `count` circuits, each with a random layer prepended."""
    import b17_practice_eval as B
    out = []
    if kind == "W2":
        for k, (p, qc) in enumerate(B.circuits("W2", n, smoke=False)):
            if k >= count:
                break
            out.append(with_prep(qc, 8000 + k))
    else:
        for p, qc in B.circuits("W1", n, smoke=False):
            if (p["dt"], p["r"]) in ((1e-3, 1e-4), (1e-2, 1e-5)) and p["seed"] < count:
                out.append(with_prep(qc, 7000 + p["seed"]))
    return out


REC = dict(placement_refine=True, final_resynthesis="select", compare_level3=True)
FULL = dict(REC, compare_floor=True, candidate_score="hybrid")


def test_version(mods):
    assert mods[0].VERSION == "2026-10-05.c12"
    assert mods[1].VERSION == "2026-10-07.1"  # current release (2026-10-04.1 when this candidate was evaluated)


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeTorino"])
def test_identical_to_release_where_qiskit_is_exact(mods, name):
    """On ordinary circuits every Qiskit-made candidate is exact, so every call gives the release's circuit."""
    c12, rel = mods
    tgt = backend(name).target
    kw = kw_for(tgt)
    for qc in (ring(6, seed=1), ghz(seed=2), chain(5, seed=3), xxz_chain(seed=4)):
        for extra in ({}, {"target": tgt}, dict(REC, target=tgt), dict(FULL, target=tgt),
                      dict(target=tgt, placement_refine=True, final_resynthesis=True)):
            assert sig(c12.compile_for_hardware(qc, **kw, **extra)) == sig(rel.compile_for_hardware(qc, **kw, **extra)), \
                (name, extra.keys())


def test_checks_accept_exact_and_refuse_tampered(mods):
    from qiskit import transpile
    c12 = mods[0]
    tgt = backend("FakeAuckland").target
    qc = xxz_ring(seed=5)
    for out in (c12.compile_for_hardware(qc, target=tgt, **FULL, **kw_for(tgt)), l3(qc, tgt)):
        assert c12._implements(qc, out)
        assert c12._same_action(out, out.copy())
        bad = out.copy()
        q = bad.layout.final_index_layout(filter_ancillas=True)[0]
        bad.rz(0.1, q)
        assert not c12._implements(qc, bad)
        assert not c12._same_action(out, bad)


def test_recommended_call_exact_on_near_boundary_cx(mods):
    """The workplace probe's W2 cells on FakeAuckland (Addendum 340: release 2026-10-04.1 wrong on 6 of 10): the
    candidate is exact on all of them, and its check refused at least one Qiskit-made circuit."""
    c12, rel = mods
    tgt = backend("FakeAuckland").target
    kw = kw_for(tgt)
    before = dict(c12.EXACT_STATS)
    worst = 0.0
    for n in (4, 6):
        for qc in near_boundary("W2", n, 5):
            worst = max(worst, state_infid(qc, c12.compile_for_hardware(qc, target=tgt, **FULL, **kw)))
    assert worst <= 1e-6, worst
    refused = sum(c12.EXACT_STATS[k] - before[k] for k in ("refused_resynthesis", "refused_floor", "refused_level3"))
    if qiskit_17057_present():  # Addendum 357: only where Qiskit makes a wrong circuit to refuse
        assert refused >= 1


def test_resynthesis_refused_on_near_boundary_trotter(mods):
    """Item 35 with final_resynthesis=True on B17's near-boundary Trotter cells (wrong on 20 of 20 in the probe): the
    candidate keeps the guarded circuit and is exact."""
    c12 = mods[0]
    tgt = backend("FakeAuckland").target
    kw = kw_for(tgt)
    before = c12.EXACT_STATS["refused_resynthesis"]
    for qc in near_boundary("W1", 4, 2):
        out = c12.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis=True, **kw)
        assert state_infid(qc, out) <= 1e-6
    if qiskit_17057_present():  # Addendum 357
        assert c12.EXACT_STATS["refused_resynthesis"] - before >= 1


def test_cz_device_unchanged_on_near_boundary(mods):
    """#17057 is in the CX/ZSX path only: on FakeTorino the candidate gives the release's circuits, all exact."""
    c12, rel = mods
    tgt = backend("FakeTorino").target
    kw = kw_for(tgt)
    for qc in near_boundary("W2", 4, 3):
        a = c12.compile_for_hardware(qc, target=tgt, **FULL, **kw)
        assert sig(a) == sig(rel.compile_for_hardware(qc, target=tgt, **FULL, **kw))
        assert state_infid(qc, a) <= 1e-6
