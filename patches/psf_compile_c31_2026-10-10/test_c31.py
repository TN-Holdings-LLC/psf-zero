"""Tests for candidate psf_compile 2026-10-10.c31 (changelog items 58 and 59a) against release 2026-10-10.2, on which
it is based.

Item 58: with a device (target= or the new backend=), a circuit that uses a failed coupler or qubit is never returned
silently; a qubit with a failed measurement counts as failed; without a Target the first call warns. Item 59a records
the infidelities of item 39's checks and changes no decision.

Run from the repository root:  python -m pytest patches/psf_compile_c31_2026-10-10/test_c31.py -q
"""
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "patches", "psf_compile_release_2026-10-10.2", "psf_compile.py"), "psf_compile_rel_for_c31_test")
    c31 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c31_test")
    rel_warn, c31_warn = getattr(rel, "WARN_WITHOUT_TARGET", None), c31.WARN_WITHOUT_TARGET
    c31.WARN_WITHOUT_TARGET = False
    yield dict(rel=rel, c31=c31)
    c31.WARN_WITHOUT_TARGET = c31_warn
    assert rel_warn is None  # the release has no such switch


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _used(circ):
    return {circ.find_bit(q).index for ins in circ.data if ins.operation.name not in ("barrier", "delay")
            for q in ins.qubits}


def _line_backend(n=5, seed=58):
    from qiskit.providers.fake_provider import GenericBackendV2
    return GenericBackendV2(num_qubits=n, basis_gates=["cz", "rz", "sx", "x", "id"],
                            coupling_map=[[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)],
                            seed=seed)


def _kw(backend):
    t = backend.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    return dict(coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx", layout_search=True,
                seed_transpiler=0)


def test_version(mods):
    assert mods["c31"].VERSION == "2026-10-10.c31"
    assert mods["rel"].VERSION == "2026-10-10.2"


def test_same_output_as_the_release(mods, torino):
    """Same arguments: the release's output by value, wherever the release's output uses no element that c31 counts
    as failed; otherwise c31's output uses none."""
    from qiskit.circuit.random import random_circuit
    rel, c31 = mods["rel"], mods["c31"]
    target = torino.target
    edges, qubits = c31._failed_elements(target, 0.5)
    seen = {"same": 0, "safer": 0}
    for k in range(10):
        qc = random_circuit(3 + k % 6, 6 + k, max_operands=2, measure=k % 2 == 0, seed=58_100 + k)
        for extra in ({}, dict(target=target), dict(target=target, **RECOMMENDED)):
            kw = dict(_kw(torino), **extra)
            a = rel.compile_for_hardware(qc, **kw)
            b = c31.compile_for_hardware(qc, **kw)
            if not extra or not c31._uses_failed(a, edges, qubits):
                assert _sig(a) == _sig(b), (k, sorted(extra))
                seen["same"] += 1
            else:
                assert not c31._uses_failed(b, edges, qubits), (k, sorted(extra))
                seen["safer"] += 1
    assert seen["same"] >= 20


def test_backend_argument(mods, torino):
    """backend= gives what target=, coupling_map= and basis_gates= taken from it give."""
    from qiskit.circuit.random import random_circuit
    c31 = mods["c31"]
    t = torino.target
    basis = [g for g in t.operation_names if g in c31._BACKEND_BASIS]
    for k in range(4):
        qc = random_circuit(4, 8, max_operands=2, measure=True, seed=58_200 + k)
        a = c31.compile_for_hardware(qc, backend=torino, entangling_basis="cx", layout_search=True, seed_transpiler=0)
        b = c31.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis, target=t,
                                     entangling_basis="cx", layout_search=True, seed_transpiler=0)
        assert _sig(a) == _sig(b)
    with pytest.raises(ValueError):
        c31.compile_for_hardware(qc)


def test_warning_once_without_target(mods):
    from qiskit import QuantumCircuit
    c31 = mods["c31"]
    be = _line_backend()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    c31.WARN_WITHOUT_TARGET, c31._WARNED["no_target"] = True, False
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c31.compile_for_hardware(qc, target=be.target, **_kw(be))
            assert not [x for x in w if "without the device's Target" in str(x.message)]
            c31.compile_for_hardware(qc, **_kw(be))
            c31.compile_for_hardware(qc, **_kw(be))
        assert len([x for x in w if "without the device's Target" in str(x.message)]) == 1
    finally:
        c31.WARN_WITHOUT_TARGET = False


def test_raise_or_keep_when_failed_elements_cannot_be_avoided(mods):
    """A line of five with its coupler (1, 2) failed: a four-qubit chain fits on neither side."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import InstructionProperties
    c31 = mods["c31"]
    be = _line_backend()
    t = be.target
    for qargs in ((1, 2), (2, 1)):
        t.update_instruction_properties("cz", qargs, InstructionProperties(error=1.0, duration=t["cz"][qargs].duration))
    qc = QuantumCircuit(4)
    qc.h(0)
    for k in range(3):
        qc.cx(k, k + 1)
    with pytest.raises(c31.FailedElementsError):
        c31.compile_for_hardware(qc, target=t, **_kw(be))
    with pytest.warns(RuntimeWarning):
        out = c31.compile_for_hardware(qc, target=t, on_failed_elements="keep", **_kw(be))
    edges, qubits = c31._failed_elements(t, 0.5)
    assert c31._uses_failed(out, edges, qubits)  # "keep" returns it, as item 43 did
    two = QuantumCircuit(2)
    two.h(0)
    two.cx(0, 1)
    assert not c31._uses_failed(c31.compile_for_hardware(two, target=t, **_kw(be)), edges, qubits)


def test_failed_measurement_counts_as_failed_qubit(mods):
    from qiskit import QuantumCircuit
    from qiskit.transpiler import InstructionProperties
    c31 = mods["c31"]
    be = _line_backend(seed=59)
    t = be.target
    t.update_instruction_properties("measure", (2,), InstructionProperties(error=0.9, duration=t["measure"][(2,)].duration))
    edges, qubits = c31._failed_elements(t, 0.5)
    assert 2 in qubits
    assert not c31.prune_coupling_map(t.build_coupling_map(), t, 0.5).neighbors(2)
    for k in range(3):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.1 * (k + 1), 1)
        qc.measure([0, 1], [0, 1])
        out = c31.compile_for_hardware(qc, target=t, **_kw(be))
        assert 2 not in _used(out)


def test_exact_seen_records_without_changing_decisions(mods, torino):
    from qiskit import transpile
    from qiskit.circuit.random import random_circuit
    rel, c31 = mods["rel"], mods["c31"]
    del c31.EXACT_SEEN[:]
    for k in range(5):
        qc = random_circuit(4, 10, max_operands=2, seed=59_100 + k)
        out = transpile(qc, backend=torino, optimization_level=2, seed_transpiler=k)
        assert c31._implements(qc, out) == rel._implements(qc, out) is True
    assert len(c31.EXACT_SEEN) == 5
    assert all(where == "implements" and -1e-12 < v < 1e-10 for where, v in c31.EXACT_SEEN)
