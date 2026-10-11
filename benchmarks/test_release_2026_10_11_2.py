"""Tests for release psf_compile 2026-10-11.2: candidate 2026-10-11.c38 (changelog item 68, a Pauli evolution whose
terms all commute compiled by STEINER6's construction), accepted in C38-VAL (Addendum 445). The candidate's own tests
are patches/psf_compile_c38_2026-10-11/test_c38.py. The previous release, 2026-10-11.1, is kept unchanged in
patches/psf_compile_release_2026-10-11.1/.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_11_2.py -q
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

REL = os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.2", "psf_compile.py")
C38 = os.path.join(REPO, "patches", "psf_compile_c38_2026-10-11", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.1", "psf_compile.py")


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
    rel = H.load_module(REL, "psf_compile_rel101102_test")
    prev = H.load_module(PREV, "psf_compile_prev101102_test")
    return dict(rel=rel, prev=prev)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _circuits(n=6, seed=82_000):
    from qiskit.circuit.random import random_circuit
    return [random_circuit(4 + k, 8 + 2 * k, max_operands=2, measure=k % 2 == 0, seed=seed + k) for k in range(n)]


def _call(mod, qc, **kw):
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return mod.compile_for_hardware(qc, entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)


def _evolution(n, terms, seed):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list([(t, rng.uniform(-1, 1)) for t in terms]), time=0.8),
              range(n))
    return qc


def _zz(n, pairs):
    out = []
    for a, b in pairs:
        lb = ["I"] * n
        lb[n - 1 - a] = lb[n - 1 - b] = "Z"
        out.append("".join(lb))
    return out


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-11.2"
    assert mods["prev"].VERSION == "2026-10-11.1"


def test_file_is_c38_except_the_version_lines():
    a, b = lines(REL), lines(C38)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-11.2 -- release") and diff[0][1].startswith("VERSION: 2026-10-11.c38")
    assert diff[1][0].startswith('VERSION = "2026-10-11.2"') and diff[1][1].startswith('VERSION = "2026-10-11.c38"')


def test_candidate_and_previous_release_kept_unchanged():
    assert nsha(C38) == "31a2c38662c2432c1c8c3b625f76c8af734cf0c871100ee662f27211840fecb8"
    assert nsha(PREV) == "2ab863edcaf867c55db565093cbe19ae4ad872403e788e9b5ed76d23a6c1c210"


def test_other_inputs_are_2026_10_11_1(mods, torino):
    """Ordinary circuits and a non-commuting evolution: 2026-10-11.1's outputs, by value."""
    n = 6
    tfim = _evolution(n, _zz(n, [(q, q + 1) for q in range(n - 1)]) +
                      ["".join("X" if j == n - 1 - q else "I" for j in range(n)) for q in range(n)], 82_100)
    for qc in [tfim] + _circuits():
        assert _sig(_call(mods["rel"], qc, backend=torino)) == _sig(_call(mods["prev"], qc, backend=torino))


def test_commuting_evolution_by_steiner(mods, torino):
    """A commuting evolution takes the new path, uses no failed element and only the device's gates."""
    rel = mods["rel"]
    n = 10
    rng = random.Random(82_200)
    pairs = sorted({tuple(sorted(rng.sample(range(n), 2))) for _ in range(18)})
    qc = _evolution(n, _zz(n, pairs), 82_201)
    used = rel.STEINER_STATS["used"]
    out = _call(rel, qc, backend=torino)
    assert rel.STEINER_STATS["used"] == used + 1
    edges, bad = rel._failed_elements(torino.target, 0.5)
    assert not rel._uses_failed(out, edges, bad)
    for ins in out.data:
        torino.target[ins.operation.name][tuple(out.find_bit(q).index for q in ins.qubits)]
    assert out.layout is not None and len(out.layout.final_index_layout()) == n


def test_without_the_device_unchanged(mods, torino):
    """Without a Target, 2026-10-11.1's default call, by value."""
    cm = torino.target.build_coupling_map()
    basis = [g for g in torino.target.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    mods["rel"].WARN_WITHOUT_TARGET = mods["prev"].WARN_WITHOUT_TARGET = False
    try:
        for qc in _circuits(seed=82_300) + [_evolution(6, _zz(6, [(0, 1), (2, 5), (1, 4)]), 82_301)]:
            assert _sig(_call(mods["rel"], qc, coupling_map=cm, basis_gates=basis)) == \
                _sig(_call(mods["prev"], qc, coupling_map=cm, basis_gates=basis))
    finally:
        mods["rel"].WARN_WITHOUT_TARGET = mods["prev"].WARN_WITHOUT_TARGET = True


def test_warns_once_without_the_device(mods, torino):
    rel = mods["rel"]
    rel._WARNED["no_target"] = False
    cm = torino.target.build_coupling_map()
    qc = _circuits(n=1, seed=82_400)[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            for _ in range(2):
                rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                         entangling_basis="cx", seed_transpiler=0)
    assert len([x for x in w if "without the device's Target" in str(x.message)]) == 1
