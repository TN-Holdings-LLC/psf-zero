"""Tests for candidate psf_ai_compile 2026-10-05.a9 (item 14: level 3's output used only if the release confirms it).
The release used is candidate psf_compile 2026-10-05.c12 (item 39), registered as `psf_compile` before the front ends
are loaded, as it would be after both are adopted. `with_prep` and `state_infid` are the workplace probe's
(Addendum 340).

Run from the repository root:  python -m pytest patches/psf_ai_compile_a9_2026-10-05/test_a9.py -q
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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_a9_test")
    sys.modules["psf_smart_layout"] = lay
    c12 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c12_2026-10-05", "psf_compile.py"), "psf_compile")
    a8 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a8.py"), "psf_ai_compile_a8_for_a9_test")  # a8 (frozen at a9's adoption)
    a9 = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a9_test")
    return dict(c12=c12, a8=a8, a9=a9)


def qiskit_17057_present():
    """True if Qiskit's CX-basis synthesis shows issue #17057 in this environment, on the issue's own input
    (exp(i(0.6 XX + 0.3 YY + c ZZ)), c = 1e-7). The failure hinges on a rounding-level offset: it appeared in the
    Linux environments at home and at the workplace, and not in a Windows environment (Addendum 357). Where it does
    not appear, Qiskit makes no wrong circuit for the checks to refuse, so the refusal counts are asserted only where
    it does; exactness is asserted everywhere."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator, average_gate_fidelity
    core = QuantumCircuit(2)
    core.rxx(-1.2, 0, 1)
    core.ryy(-0.6, 0, 1)
    core.rzz(-2e-7, 0, 1)
    out = transpile(core, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
    return 1 - average_gate_fidelity(Operator(out), Operator(core)) > 1e-6


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


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [float(p) for p in i.operation.params])
            for i in c.data]


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


def nat(tgt):
    return [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]


def test_version(mods):
    assert mods["a9"].AI_COMPILE_VERSION == "2026-10-05.a9"
    assert mods["a8"].AI_COMPILE_VERSION == "2026-10-04.a8"
    assert mods["c12"].VERSION == "2026-10-05.c12"
    assert mods["a9"].pc is mods["c12"] and mods["a8"].pc is mods["c12"]


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeTorino"])
def test_identical_to_a8_where_level3_is_exact(mods, name):
    tgt = backend(name).target
    cm = tgt.build_coupling_map()
    for qc in (ring(6, seed=1), chain(5, seed=2), ghz(seed=3)):
        for kw in ({}, {"target": tgt}):
            a = mods["a8"].compile_for_model_circuit(qc, cm, nat(tgt), **kw)
            b = mods["a9"].compile_for_model_circuit(qc, cm, nat(tgt), **kw)
            assert sig(a) == sig(b), (name, kw.keys())


def test_exact_on_near_boundary_unitaries_cx(mods):
    """The kind of circuit a model writes (explicit two-qubit `unitary` gates) near the boundary, on FakeAuckland:
    a9 is exact, and it refused level 3's output at least once."""
    import b17_practice_eval as B
    a9 = mods["a9"]
    tgt = backend("FakeAuckland").target
    cm = tgt.build_coupling_map()
    before = a9.L3T_CHECK_STATS["refused"]
    worst = 0.0
    for n in (4, 6):
        for k, (p, qc) in enumerate(B.circuits("W2", n, smoke=False)):
            if k >= 5:
                break
            qc = with_prep(qc, 8000 + k)
            worst = max(worst, state_infid(qc, a9.compile_for_model_circuit(qc, cm, nat(tgt), target=tgt)))
    assert worst <= 1e-6, worst
    if qiskit_17057_present():  # Addendum 357: only where Qiskit makes a wrong circuit to refuse
        assert a9.L3T_CHECK_STATS["refused"] - before >= 1
