"""Tests for candidate psf_compile 2026-10-03.c8 (changelog item 35: final two-qubit re-synthesis by Qiskit, always or
selected by an excitation-aware estimate). Helpers
are copied from patches/psf_compile_c6_2026-10-03/test_c6_floor.py (themselves from test_release_2026_10_02_2.py).

Run from the repository root:  python -m pytest patches/psf_compile_c8_2026-10-03/test_c8_resynth.py -q
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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_c8_test")
    sys.modules["psf_smart_layout"] = lay
    return (H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c8_test"),
            H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c8_test"))


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


def s_rep(out, tgt):
    """Sum of -log(1 - reported error) over the gates as placed."""
    s = 0.0
    for ins in out.data:
        q = tuple(out.find_bit(b).index for b in ins.qubits)
        name = ins.operation.name
        if name in ("barrier", "measure") or name not in tgt.operation_names:
            continue
        props = tgt[name].get(q) or tgt[name].get(q[::-1])
        e = props.error if props is not None and props.error is not None else 0.0
        s += -math.log(max(1.0 - e, 1e-300))
    return s


def compact_fidelity(qc, out):
    """|<ref|out>|^2 on the touched physical qubits only (as in test_c3_prune.py)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    n = qc.num_qubits
    fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
    used = sorted({out.find_bit(q).index for ins in out.data for q in ins.qubits} | set(fin))
    pos = {p: i for i, p in enumerate(used)}
    comp = QuantumCircuit(len(used))
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure"):
            continue
        comp.append(ins.operation, [pos[out.find_bit(q).index] for q in ins.qubits])
    ref = QuantumCircuit(len(used))
    ref.compose(qc, qubits=[pos[p] for p in fin], inplace=True)
    return abs(Statevector(ref).inner(Statevector(comp))) ** 2




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


def test_version(mods):
    assert mods[0].VERSION == "2026-10-03.c8"
    assert mods[1].VERSION == "2026-10-06.2"  # current release (2026-10-02.2 when this candidate was evaluated)


def test_default_identical_to_release(mods):
    c8, rel = mods
    for name in ("FakeTorino", "FakeAuckland"):
        tgt = backend(name).target
        for qc in (ring(4, seed=4), ring(6, seed=6), chain(5, seed=5)):
            kw = kw_for(tgt)
            for extra in ({}, {"target": tgt}, {"target": tgt, "placement_refine": True}):
                assert sig(c8.compile_for_hardware(qc, **kw, **extra)) == sig(rel.compile_for_hardware(qc, **kw, **extra)), \
                    (name, extra.keys())


def test_needs_target(mods):
    tgt = backend("FakeTorino").target
    kw = kw_for(tgt)
    with pytest.raises(ValueError):
        mods[0].compile_for_hardware(ring(4), final_resynthesis=True, **kw)


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeHanoiV2", "FakeGeneva", "FakeTorino", "FakeKingston"])
def test_resynthesis_exact_on_target_same_layout(mods, name):
    """Exact, every instruction on the target, no failed qubit or failed direction (direction-aware, Addendum 319
    s. 4), the same final layout and no more two-qubit gates than the release."""
    c8, rel = mods
    tgt = backend(name).target
    edges, qubits = c8._failed_elements(tgt, 0.5)
    for qc in (ring(4, seed=1), ring(6, seed=2), chain(5, seed=3), xxz_chain(seed=4)):
        kw = kw_for(tgt)
        a = rel.compile_for_hardware(qc, target=tgt, placement_refine=True, **kw)
        b = c8.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis=True, **kw)
        assert compact_fidelity(qc, b) > 1 - 1e-6, name
        n = qc.num_qubits
        assert list(a.layout.final_index_layout(filter_ancillas=True)[:n]) == \
            list(b.layout.final_index_layout(filter_ancillas=True)[:n])
        assert sum(len(i.qubits) == 2 for i in b.data) <= sum(len(i.qubits) == 2 for i in a.data)
        for ins in b.data:
            q = tuple(b.find_bit(x).index for x in ins.qubits)
            assert ins.operation.name in tgt.operation_names and q in tgt[ins.operation.name], (name, ins.operation.name, q)
            assert not any(i in qubits for i in q)
            if len(q) == 2:
                assert q not in edges, (name, ins.operation.name, q)


def test_resynthesis_reduces_x_on_xxz_chain(mods):
    """Addendum 322's mechanism on one cx device: fewer x gates than the release on open XXZ chains, and exact."""
    c8, rel = mods
    tgt = backend("FakeAuckland").target
    kw = kw_for(tgt)
    xa = xb = 0
    for s in range(3):
        qc = xxz_chain(seed=10 + s)
        a = rel.compile_for_hardware(qc, target=tgt, placement_refine=True, **kw)
        b = c8.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis=True, **kw)
        assert compact_fidelity(qc, b) > 1 - 1e-6
        xa += sum(i.operation.name == "x" for i in a.data)
        xb += sum(i.operation.name == "x" for i in b.data)
    assert xb < xa, (xa, xb)


def test_kept_original_when_result_uses_failed_direction(mods):
    """The backstop: if re-synthesis yields an off-target or failed-direction gate, the original is returned."""
    c8 = mods[0]
    tgt = backend("FakeAuckland").target
    kw = kw_for(tgt)
    qc = chain(4, seed=7)
    a = c8.compile_for_hardware(qc, target=tgt, placement_refine=True, **kw)
    # an impossible bound: every gate counts as failed, so re-synthesis must be refused
    out = c8._final_resynthesis(a, tgt, -1.0)
    assert out is a


def test_bad_mode_raises(mods):
    tgt = backend("FakeTorino").target
    with pytest.raises(ValueError):
        mods[0].compile_for_hardware(ring(4), target=tgt, final_resynthesis="sometimes", **kw_for(tgt))


def test_excitation_cost_matches_statevector(mods):
    """The numpy state in excitation_cost against Qiskit's Statevector, on a compiled circuit of a cx device."""
    from qiskit.quantum_info import Statevector
    c8 = mods[0]
    tgt = backend("FakeAuckland").target
    out = c8.compile_for_hardware(xxz_chain(seed=3), target=tgt, placement_refine=True, **kw_for(tgt))
    ops = [(i.operation, tuple(out.find_bit(b).index for b in i.qubits)) for i in out.data
           if i.operation.name not in ("barrier", "measure", "delay")]
    active = sorted({i for _, q in ops for i in q})
    pos = {p: j for j, p in enumerate(active)}
    sv = Statevector.from_label("0" * len(active))
    ref = 0.0
    for op, q in ops:
        props = tgt[op.name][q]
        if props.error is not None:
            ref += -math.log(1.0 - props.error)
        if props.duration:
            for i in q:
                ref += props.duration / tgt.qubit_properties[i].t1 * float(sv.probabilities([pos[i]])[1])
        sv = sv.evolve(op, qargs=[pos[i] for i in q])
    assert abs(c8.excitation_cost(out, tgt) - ref) <= 1e-9 * max(1.0, ref)


@pytest.mark.parametrize("name", ["FakeAuckland", "FakeHanoiV2", "FakeTorino"])
def test_select_exact_and_minimal(mods, name):
    """"select" returns the release's circuit or the re-synthesised one, whichever has the lower estimate; exact and
    on the target either way."""
    c8 = mods[0]
    tgt = backend(name).target
    kw = kw_for(tgt)
    for qc in (chain(5, seed=3), xxz_chain(seed=5), ring(6, seed=6)):
        a = c8.compile_for_hardware(qc, target=tgt, placement_refine=True, **kw)
        b = c8.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis=True, **kw)
        s = c8.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select", **kw)
        assert compact_fidelity(qc, s) > 1 - 1e-6
        ca, cb, cs = (c8.excitation_cost(x, tgt) for x in (a, b, s))
        assert cs == min(ca, cb), (name, ca, cb, cs)
        assert sig(s) in (sig(a), sig(b))
