"""Tests for release psf_compile 2026-10-11.3: prototype 2026-10-11.c41 (changelog items 69-73 on release 2026-10-11.2),
accepted in C39-VAL to C41-VAL (Addenda 447-449). The prototypes' own tests are
patches/psf_compile_c39_2026-10-11/test_c39.py to patches/psf_compile_c41_2026-10-11/test_c41.py. The previous release,
2026-10-11.2, is kept unchanged in patches/psf_compile_release_2026-10-11.2/.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_11_3.py -q
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
C41 = os.path.join(REPO, "patches", "psf_compile_c41_2026-10-11", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.2", "psf_compile.py")


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
    rel = H.load_module(REL, "psf_compile_rel101103_test")
    prev = H.load_module(PREV, "psf_compile_prev101103_test")
    return dict(rel=rel, prev=prev)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _device(seed=83):
    """A 7-qubit device with differing errors."""
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap
    cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [2, 6], [6, 5]])
    cm.make_symmetric()
    b = GenericBackendV2(7, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cm, seed=seed)
    rng = random.Random(seed)
    for q in range(7):
        b.target.update_instruction_properties("measure", (q,), _props(rng.uniform(0.005, 0.03)))
        for g in ("sx", "x"):
            b.target.update_instruction_properties(g, (q,), _props(rng.uniform(1e-4, 1e-3)))
    for qa in b.target["cz"]:
        b.target.update_instruction_properties("cz", qa, _props(rng.uniform(2e-3, 1e-2)))
    return b


def _props(e):
    from qiskit.transpiler import InstructionProperties
    return InstructionProperties(error=e)


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits(n=6, seed=83_000, width=4):
    from qiskit.circuit.random import random_circuit
    return [random_circuit(width + k % 3, 8 + 2 * k, max_operands=2, measure=k % 2 == 0, seed=seed + k)
            for k in range(n)]


def _call(mod, qc, **kw):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return mod.compile_for_hardware(qc, entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)


def _same_as_prev(mods, qc, prev_kw=None, **kw):
    """True when both releases give one output each over three calls and they are the same; None when either varies
    from call to call (Qiskit's own transpile, with a fixed seed, can return different outputs from call to call on
    some circuits: Addendum 451); an AssertionError when both are reproducible and differ."""
    rs = {_sig(_call(mods["rel"], qc, **kw)) for _ in range(3)}
    ps = {_sig(_call(mods["prev"], qc, **(kw if prev_kw is None else prev_kw))) for _ in range(3)}
    if len(rs) > 1 or len(ps) > 1:
        return None
    assert rs == ps
    return True


def _mostly_same(mods, circuits, **kw):
    got = [_same_as_prev(mods, qc, **kw) for qc in circuits]
    assert sum(1 for g in got if g) >= (len(got) + 1) // 2, got


def _qaoa_gates(n, seed):
    """QAOA written as gates (cx-rz-cx), as QASM files have it: item 73 reads it."""
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    edges = sorted({tuple(sorted(rng.sample(range(n), 2))) for _ in range(2 * n)})
    for _ in range(2):
        g, beta = rng.uniform(0, 3), rng.uniform(0, 3)
        for a, b in edges:
            qc.cx(a, b)
            qc.rz(g, b)
            qc.cx(a, b)
        qc.rx(beta, range(n))
    qc.measure_all()
    return qc


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-11.3"
    assert mods["prev"].VERSION == "2026-10-11.2"


def test_file_is_c41_except_the_version_lines():
    a, b = lines(REL), lines(C41)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-11.3 -- release") and diff[0][1].startswith("VERSION: 2026-10-11.c41")
    assert diff[1][0].startswith('VERSION = "2026-10-11.3"') and diff[1][1].startswith('VERSION = "2026-10-11.c41"')


def test_prototype_and_previous_release_kept_unchanged():
    assert nsha(C41) == "9a8d19818c08c2054c15cf82cc4786188bb5053b130ddc1bf0e73766b0c8bcd6"
    assert nsha(PREV) == "56be71fb65fcecfc857536e83fd3f871df6f29168d4484d205df04994bf52b9b"


def test_unread_inputs_are_2026_10_11_2(mods):
    """Ordinary circuits that items 72-73 do not read: 2026-10-11.2's outputs, by value (where both are reproducible)."""
    b = _device()
    circuits = _circuits()
    for qc in circuits:
        assert mods["rel"]._st_parse(qc) is None and mods["rel"]._st_parse_gates(qc) is None
    _mostly_same(mods, circuits, backend=b)


def test_steiner_off_gives_2026_10_11_2(mods, torino):
    """steiner=False: 2026-10-11.2's steiner=False outputs, by value (where reproducible), on FakeTorino."""
    _mostly_same(mods, _circuits(seed=83_100) + [_qaoa_gates(6, 83_101)], backend=torino, steiner=False)


def test_never_worse_than_2026_10_11_2_and_no_failed_element(mods, torino):
    """Gate-level QAOA (item 73) and ordinary circuits on FakeTorino: ESP never below 2026-10-11.2's, no failed
    element, only the device's gates."""
    rel, prev = mods["rel"], mods["prev"]
    t = torino.target
    edges, bad = rel._failed_elements(t, 0.5)
    for qc in [_qaoa_gates(8, 83_200), _qaoa_gates(12, 83_201)] + _circuits(n=3, seed=83_202):
        out = _call(rel, qc, backend=torino)
        assert rel._st_cost(out, t) <= rel._st_cost(_call(prev, qc, backend=torino), t) + 1e-12
        assert not rel._uses_failed(out, edges, bad)
        for ins in out.data:
            t[ins.operation.name][tuple(out.find_bit(q).index for q in ins.qubits)]


def test_without_the_device_unchanged(mods, torino):
    """Without a Target, 2026-10-11.2's default call, by value (where reproducible)."""
    cm = torino.target.build_coupling_map()
    basis = [g for g in torino.target.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    mods["rel"].WARN_WITHOUT_TARGET = mods["prev"].WARN_WITHOUT_TARGET = False
    try:
        _mostly_same(mods, _circuits(seed=83_400) + [_qaoa_gates(6, 83_401)], coupling_map=cm, basis_gates=basis)
    finally:
        mods["rel"].WARN_WITHOUT_TARGET = mods["prev"].WARN_WITHOUT_TARGET = True


def test_warns_once_without_the_device(mods, torino):
    rel = mods["rel"]
    rel._WARNED["no_target"] = False
    cm = torino.target.build_coupling_map()
    qc = _circuits(n=1, seed=83_500)[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            for _ in range(2):
                rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                         entangling_basis="cx", seed_transpiler=0)
    assert len([x for x in w if "without the device's Target" in str(x.message)]) == 1
