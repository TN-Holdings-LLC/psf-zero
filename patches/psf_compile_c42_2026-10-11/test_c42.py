"""Tests for prototype psf_compile 2026-10-11.c42 (changelog item 74 on c41): an output that uses a near-failed element
(error in [near_failed_error, prune_max_error): a coupler, a qubit's gates, a measured qubit's readout) is compared with
the input compiled with those elements avoided too, and the one with the higher ESP is returned.

Run from the repository root:  python -m pytest patches/psf_compile_c42_2026-10-11/test_c42.py -q
"""
import contextlib
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    c41 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c41_2026-10-11", "psf_compile.py"),
                        "psf_compile_c41_for_c42_test")
    c42 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c42_test")
    c41.WARN_WITHOUT_TARGET = c42.WARN_WITHOUT_TARGET = False
    return dict(c41=c41, c42=c42)


def _backend(near_edge=None, near_readout=None, seed=74):
    """A 7-qubit device (a line with a branch) with differing errors; optionally one coupler and one readout made
    near-failed (0.2 and 0.25)."""
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap, InstructionProperties
    cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 6], [6, 5]])
    cm.make_symmetric()
    b = GenericBackendV2(7, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cm, seed=seed)
    rng = random.Random(seed)
    for q in range(7):
        b.target.update_instruction_properties("measure", (q,), InstructionProperties(error=rng.uniform(0.005, 0.03)))
        for g in ("sx", "x"):
            b.target.update_instruction_properties(g, (q,), InstructionProperties(error=rng.uniform(1e-4, 1e-3)))
    for qa in b.target["cz"]:
        b.target.update_instruction_properties("cz", qa, InstructionProperties(error=rng.uniform(2e-3, 1e-2)))
    if near_edge is not None:
        for qa in (near_edge, near_edge[::-1]):
            b.target.update_instruction_properties("cz", qa, InstructionProperties(error=0.2))
    if near_readout is not None:
        b.target.update_instruction_properties("measure", (near_readout,), InstructionProperties(error=0.25))
    return b


def _call(mod, qc, **kw):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return mod.compile_for_hardware(qc, entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits():
    from qiskit import QuantumCircuit
    from qiskit.circuit.random import random_circuit
    out = []
    for seed in range(4):
        qc = random_circuit(6, 8, max_operands=2, seed=740 + seed)
        qc.measure_all()
        out.append(qc)
    ghz = QuantumCircuit(7)
    ghz.h(0)
    for q in range(6):
        ghz.cx(q, q + 1)
    ghz.measure_all()
    out.append(ghz)
    return out


def _exact_without_measure(out, qc):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    body = out.copy_empty_like()
    for i in out.data:
        if i.operation.name != "measure":
            body.append(i.operation, i.qubits, i.clbits)
    body._layout = out._layout
    ref = qc.remove_final_measurements(inplace=False)
    ext = QuantumCircuit(out.num_qubits)
    ext.compose(ref, range(qc.num_qubits), inplace=True)
    return Operator.from_circuit(body).equiv(Operator(ext))


def test_version(mods):
    assert mods["c42"].VERSION == "2026-10-11.c42"


def test_off_and_unaffected_devices_give_c41_by_value(mods):
    """near_failed_error=None, and a device without near-failed elements: c41's outputs by value."""
    c41, c42 = mods["c41"], mods["c42"]
    clean = _backend()
    near = _backend(near_edge=(2, 3), near_readout=4)
    for qc in _circuits():
        assert _sig(_call(c42, qc, backend=clean)) == _sig(_call(c41, qc, backend=clean))
        assert _sig(_call(c42, qc, backend=near, near_failed_error=None)) == _sig(_call(c41, qc, backend=near))


def test_never_worse_and_avoids_when_better(mods):
    c41, c42 = mods["c41"], mods["c42"]
    b = _backend(near_edge=(2, 3), near_readout=4)
    t = b.target
    tried = 0
    for qc in _circuits():
        before = dict(c42.NEAR_STATS)
        out = _call(c42, qc, backend=b)
        first = _call(c41, qc, backend=b)
        assert c42._st_cost(out, t) <= c42._st_cost(first, t) + 1e-12
        tried += c42.NEAR_STATS["tried"] - before["tried"]
        if c42._near_used(first, t, 0.1, 0.5):
            assert c42.NEAR_STATS["tried"] > before["tried"]
        else:
            assert _sig(out) == _sig(first)
    assert tried >= 1


def test_output_is_exact(mods):
    """The kept alternative implements the input (measurements aside, as Qiskit's level 3 may drop diagonal gates
    before them; the inputs here are compared without them)."""
    c42 = mods["c42"]
    b = _backend(near_edge=(2, 3))
    for qc in _circuits():
        body = qc.remove_final_measurements(inplace=False)
        out = _call(c42, body, backend=b)
        assert _exact_without_measure(out, body)


def test_near_used(mods):
    """Counts a two-qubit gate on a near-failed coupler, any gate on a near-failed qubit, a measurement of a qubit
    with near-failed readout; not a gate on a qubit whose readout alone is near-failed."""
    from qiskit import QuantumCircuit
    c42 = mods["c42"]
    t = _backend(near_edge=(2, 3), near_readout=4).target
    a = QuantumCircuit(7)
    a.cz(3, 2)
    assert c42._near_used(a, t, 0.1, 0.5)
    b = QuantumCircuit(7, 1)
    b.sx(4)
    assert not c42._near_used(b, t, 0.1, 0.5)
    b.measure(4, 0)
    assert c42._near_used(b, t, 0.1, 0.5)
    c = QuantumCircuit(7)
    c.cz(0, 1)
    assert not c42._near_used(c, t, 0.1, 0.5)


def test_no_failed_element_on_a_device(mods):
    from qiskit.circuit.random import random_circuit
    from qiskit_ibm_runtime.fake_provider import FakeBoston
    c42 = mods["c42"]
    b = FakeBoston()
    edges, bad = c42._failed_elements(b.target, 0.5)
    qc = random_circuit(20, 6, max_operands=2, seed=741)
    qc.measure_all()
    out = _call(c42, qc, backend=b)
    assert not c42._uses_failed(out, edges, bad)
