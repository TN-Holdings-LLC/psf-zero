"""Tests for release psf_compile 2026-10-10.3: candidate 2026-10-10.c35 (changelog items 58-63; Addenda 431-432), accepted
by the pre-registered C35-VAL (Addenda 436-437). The candidate's own tests are
patches/psf_compile_c31_2026-10-10/test_c31.py to patches/psf_compile_c35_2026-10-10/test_c35.py. The previous release,
2026-10-10.2, and its psf_smart_layout.py are kept unchanged in patches/psf_compile_release_2026-10-10.2/.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_10_3.py -q
"""
import contextlib
import hashlib
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

REL = os.path.join(REPO, "psf_compile.py")
LAYOUT = os.path.join(REPO, "benchmarks", "psf_smart_layout.py")
C35 = os.path.join(REPO, "patches", "psf_compile_c35_2026-10-10")
PREV_DIR = os.path.join(REPO, "patches", "psf_compile_release_2026-10-10.2")
PREV = os.path.join(PREV_DIR, "psf_compile.py")


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
    lay = H.load_module(LAYOUT, "psf_smart_layout")
    rel = H.load_module(REL, "psf_compile_rel101003_test")
    prev = H.load_module(PREV, "psf_compile_prev101003_test")
    return dict(rel=rel, prev=prev, layout=lay)


@pytest.fixture(scope="module")
def torino():
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    return FakeTorino()


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-10.3"
    assert mods["prev"].VERSION == "2026-10-10.2"
    assert mods["rel"].ABSORB_SYNTH == "qiskit" and mods["rel"].PRUNE_FIRST is True
    assert mods["layout"].LAYOUT_VERSION == "2026-10-01.1"


def test_file_is_c35_except_the_version_lines():
    a, b = lines(REL), lines(os.path.join(C35, "psf_compile.py"))
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-10.3 -- release") and diff[0][1].startswith("VERSION: 2026-10-10.c35")
    assert diff[1][0].startswith('VERSION = "2026-10-10.3"') and diff[1][1].startswith('VERSION = "2026-10-10.c35"')


def test_layout_is_c35s_and_the_previous_files_are_kept():
    assert nsha(LAYOUT) == nsha(os.path.join(C35, "psf_smart_layout.py"))
    assert nsha(os.path.join(C35, "psf_compile.py")) == "3bea06a94d859cdb5bc41a59cfb3f4bebc949629d81a59979be60f71640a9f92"
    assert nsha(LAYOUT) == "8472cd86d48c798a5991486924969c73ad24f15b4fc57753e2109e2d88c06a41"
    assert nsha(PREV) == "3b027b856e1b997906f1ad19cfa123d4f9503473c737eeb7ba40bac266c0ad8b"
    assert nsha(os.path.join(PREV_DIR, "psf_smart_layout.py")) == \
        "624e8f8a00e1635a1ee3bc77b5b0f41bd69a94022e214d679b86cc66cc1cf241"


def _circuits(n=6, seed=71_000):
    from qiskit.circuit.random import random_circuit
    return [random_circuit(4 + k, 6 + 2 * k, max_operands=2, measure=k % 2 == 0, seed=seed + k) for k in range(n)]


def _q2(c):
    return sum(1 for i in c.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "delay"))


def _on_failed(out, target):
    n = 0
    for ins in out.data:
        if ins.operation.name in ("barrier", "delay"):
            continue
        qa = tuple(out.find_bit(q).index for q in ins.qubits)
        try:
            p = target[ins.operation.name][qa]
        except KeyError:
            continue
        n += p is not None and p.error is not None and p.error >= 0.5
    return n


def test_with_the_device_no_failed_element(mods, torino):
    """C35-VAL's V1 in small: the plain call given backend= never puts an operation on a failed element."""
    rel = mods["rel"]
    for qc in _circuits():
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            out = rel.compile_for_hardware(qc, backend=torino, entangling_basis="cx", layout_search=True,
                                           seed_transpiler=0)
        assert _on_failed(out, torino.target) == 0


def test_without_the_device_the_previous_two_qubit_counts(mods, torino):
    """C35-VAL's V5 in small: without the Target, the two-qubit counts of 2026-10-10.2's default call."""
    cm = torino.target.build_coupling_map()
    basis = [g for g in torino.target.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    mods["rel"].WARN_WITHOUT_TARGET = False
    try:
        for qc in _circuits(seed=71_100):
            kw = dict(coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True,
                      seed_transpiler=0)
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                a = mods["rel"].compile_for_hardware(qc, **kw)
                b = mods["prev"].compile_for_hardware(qc, **kw)
            assert _q2(a) == _q2(b)
    finally:
        mods["rel"].WARN_WITHOUT_TARGET = True


def test_warns_once_without_the_device(mods, torino):
    rel = mods["rel"]
    rel._WARNED["no_target"] = False
    cm = torino.target.build_coupling_map()
    qc = _circuits(n=1, seed=71_200)[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            for _ in range(2):
                rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"],
                                         entangling_basis="cx", seed_transpiler=0)
    msgs = [str(x.message) for x in w if "without the device's Target" in str(x.message)]
    assert len(msgs) == 1


def test_full_choice_by_hybrid_cost_with_its_known_exception(mods):
    """test_release_2026_10_04_1's full-choice property on 2026-10-10.3, with the exception REL3-DIAG found (Addendum
    438): on its 20 cells (5 devices x 4 circuits) the full-option call is exact, uses no failed element and has no
    higher hybrid_cost than c10's placement_refine + final_resynthesis="select" call, except on at most one cell and
    there by at most 3% (FakeHanoiV2, ring(6): 0.24515 against 0.23931, from item 63's single-qubit gates)."""
    import core_fix_c2_eval as H
    T = H.load_module(os.path.join(REPO, "benchmarks", "test_release_2026_10_04_1.py"), "t10041_for_rel101003_test")
    c10 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c10_2026-10-03", "psf_compile.py"),
                        "psf_compile_c10_for_rel101003_test")
    rel = mods["rel"]
    worse = []
    for name in ("FakeAuckland", "FakeHanoiV2", "FakeGeneva", "FakeTorino", "FakeKingston"):
        tgt = T.backend(name).target
        edges, qubits = rel._failed_elements(tgt, 0.5)
        kw = T.kw_for(tgt)
        for cname, qc in (("ghz", T.ghz(seed=1)), ("ring6", T.ring(6, seed=2)), ("chain5", T.chain(5, seed=3)),
                          ("xxz", T.xxz_chain(seed=4))):
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                c = rel.compile_for_hardware(qc, target=tgt, **T.FULL, **kw)
                a = c10.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select", **kw)
            assert T.compact_fidelity(qc, c) > 1 - 1e-6, (name, cname)
            assert not rel._uses_failed(c, edges, qubits), (name, cname)
            hc, ha = rel.hybrid_cost(c, tgt), rel.hybrid_cost(a, tgt)
            if hc > ha + 1e-12:
                worse.append((name, cname, hc / ha))
    assert len(worse) <= 1 and all(r <= 1.03 for _, _, r in worse), worse
