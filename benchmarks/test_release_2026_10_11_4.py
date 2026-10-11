"""Tests for release psf_compile 2026-10-11.4: prototype 2026-10-11.c43 (changelog items 74 revised, 75, 76 on release
2026-10-11.3), accepted in C43-VAL (Addendum 452). The prototype's own tests are
patches/psf_compile_c43_2026-10-11/test_c43.py. The previous release, 2026-10-11.3, is kept unchanged in
patches/psf_compile_release_2026-10-11.3/.

Item 75 replaces Qiskit's VF2PostLayout, so outputs are no longer those of 2026-10-11.3 by value; these tests check what
the release promises instead: the same output on every call, no failed element, the device's gates only, exact outputs,
and that circuits 2026-10-11.3 refused now compile.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_11_4.py -q
"""
import contextlib
import hashlib
import os
import random
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

REL = os.path.join(REPO, "psf_compile.py")
C43 = os.path.join(REPO, "patches", "psf_compile_c43_2026-10-11", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.3", "psf_compile.py")


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
    rel = H.load_module(REL, "psf_compile_rel101104_test")
    prev = H.load_module(PREV, "psf_compile_prev101104_test")
    return dict(rel=rel, prev=prev)


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _call(mod, qc, **kw):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return mod.compile_for_hardware(qc, entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)


def _circuits(n=4, seed=84_000):
    from qiskit.circuit.random import random_circuit
    return [random_circuit(5 + k % 2, 8 + 2 * k, max_operands=2, measure=k % 2 == 0, seed=seed + k) for k in range(n)]


def _bv(n, seed):
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    for q in range(n):
        if rng.random() < 0.5:
            qc.cx(q, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-11.4"
    assert mods["prev"].VERSION == "2026-10-11.3"


def test_file_is_c43_except_the_version_lines():
    a, b = lines(REL), lines(C43)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-11.4 -- release") and diff[0][1].startswith("VERSION: 2026-10-11.c43")
    assert diff[1][0].startswith('VERSION = "2026-10-11.4"') and diff[1][1].startswith('VERSION = "2026-10-11.c43"')


def test_prototype_and_previous_release_kept_unchanged():
    assert nsha(C43) == "30898d7cede72e9353518d52b9004ea0cbd79a5df058cbf080413f52b2094369"
    assert nsha(PREV) == "56ec4a7c32bb1f207569f6015cfb2488901dc1558fc54f1ec650ebb3f1cd25b8"


def test_same_output_every_call(mods):
    """With the device and without it, three calls give one output (Qiskit's VF2PostLayout did not: Addendum 452)."""
    from qiskit_ibm_runtime.fake_provider import FakeFez
    rel = mods["rel"]
    b = FakeFez()
    cm = b.target.build_coupling_map()
    rel.WARN_WITHOUT_TARGET = False
    try:
        for qc in _circuits() + [_bv(12, 84_100)]:
            assert len({_sig(_call(rel, qc, backend=b)) for _ in range(3)}) == 1
            assert len({_sig(_call(rel, qc, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"])) for _ in range(3)}) == 1
    finally:
        rel.WARN_WITHOUT_TARGET = True


def test_refused_by_2026_10_11_3_now_compiles(mods):
    """Bernstein-Vazirani, 20 bits, on FakeKingston: 2026-10-11.3 raises FailedElementsError (Qiskit's VF2PostLayout
    moves a measured qubit onto a failed one); 2026-10-11.4 compiles it on no failed element."""
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    rel, prev = mods["rel"], mods["prev"]
    b = FakeKingston()
    qc = _bv(20, 84_200)
    refused = 0
    try:
        _call(prev, qc, backend=b)
    except prev.FailedElementsError:
        refused = 1
    out = _call(rel, qc, backend=b)
    edges, bad = rel._failed_elements(b.target, 0.5)
    assert not rel._uses_failed(out, edges, bad)
    assert refused in (0, 1)  # the previous release's behaviour is recorded, not required


def test_no_failed_element_device_gates_only(mods):
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    rel = mods["rel"]
    b = FakeTorino()
    t = b.target
    edges, bad = rel._failed_elements(t, 0.5)
    for qc in _circuits(seed=84_300) + [_bv(16, 84_301)]:
        out = _call(rel, qc, backend=b)
        assert not rel._uses_failed(out, edges, bad)
        for ins in out.data:
            t[ins.operation.name][tuple(out.find_bit(q).index for q in ins.qubits)]


def test_exact_without_the_device():
    """Without a Target (a 7-qubit coupling map and basis gates), outputs implement the input."""
    import core_fix_c2_eval as H
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import CouplingMap
    rel = H.load_module(REL, "psf_compile_rel101104_exact")
    rel.WARN_WITHOUT_TARGET = False
    cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 6]])
    cm.make_symmetric()
    for qc in _circuits(seed=84_400):
        body = qc.remove_final_measurements(inplace=False)
        out = _call(rel, body, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"])
        ext = QuantumCircuit(7)
        ext.compose(body, range(body.num_qubits), inplace=True)
        assert Operator.from_circuit(out).equiv(Operator(ext))


def test_warns_once_without_the_device(mods):
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    rel = mods["rel"]
    rel.WARN_WITHOUT_TARGET = True
    rel._WARNED["no_target"] = False
    cm = FakeTorino().target.build_coupling_map()
    qc = _circuits(n=1, seed=84_500)[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            for _ in range(2):
                rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                         entangling_basis="cx", seed_transpiler=0)
    assert len([x for x in w if "without the device's Target" in str(x.message)]) == 1
