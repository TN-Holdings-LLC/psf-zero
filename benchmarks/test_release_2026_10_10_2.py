"""Tests for release psf_compile 2026-10-10.2: candidate 2026-10-10.c30 of Addendum 426, which adds changelog item
57b (the estimates' and checks' per-gate loops in psf_zero_core57, an optional module) to release 2026-10-10.1
(C30-ID, Addenda 426-427). The candidate's own tests are patches/psf_compile_c30_2026-10-10/test_c30.py. The
previous release, 2026-10-10.1, is kept unchanged in patches/psf_compile_release_2026-10-10.1/psf_compile.py.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_10_2.py -q -s
"""
import contextlib
import hashlib
import os
import sys
import warnings

import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

REL = os.path.join(REPO, "patches", "psf_compile_release_2026-10-10.2", "psf_compile.py")  # kept copy
C30 = os.path.join(REPO, "patches", "psf_compile_c30_2026-10-10", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-10.1", "psf_compile.py")


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
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    return dict(rel=H.load_module(REL, "psf_compile_rel101002_test"),
                prev=H.load_module(PREV, "psf_compile_prev101002_test"), layout=lay)


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-10.2"
    assert mods["prev"].VERSION == "2026-10-10.1"
    assert set(mods["rel"].CORE57_STATS) == {"rust", "python", "fallback"}
    assert not hasattr(mods["prev"], "CORE57_STATS")


def test_file_is_c30_except_the_version_lines():
    a, b = lines(REL), lines(C30)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-10.2 -- release")
    assert diff[0][1].startswith("VERSION: 2026-10-07.1 -- release")  # c30 kept its base's header line
    assert diff[1][0].startswith('VERSION = "2026-10-10.2"') and diff[1][1].startswith('VERSION = "2026-10-10.c30"')


def test_candidate_and_previous_release_kept_unchanged():
    assert nsha(C30) == "e753f72a918854b2e9cee786557cb137d7ace56d0de4038d18364008b180ff25"
    assert nsha(PREV) == "0999bb063393e4fb174cd4018d9b9a0ffabf3146494a3fed5a55e0658968da09"


def _run(mods, name, qc, recommended):
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    import c25_identity2 as C2
    b = FakeTorino()
    basis = [g for g in b.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    kw = dict(coupling_map=b.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
              seed_transpiler=0)
    if recommended:
        kw.update(target=b.target, **C2.RECOMMENDED)
    mods["layout"].time = C2._VirtualTime()
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return mods[name].compile_for_hardware(qc, **kw)


@pytest.mark.parametrize("core", [True, False])
@pytest.mark.parametrize("recommended", [False, True])
def test_same_output_as_the_previous_release(mods, recommended, core):
    """2026-10-10.1's output by value (as C30-ID compares) wherever 2026-10-10.1 run twice gives one output, with
    psf_zero_core57 (if installed) and without it."""
    from qiskit.circuit.random import random_circuit
    from c29_identity import value_sig
    rel = mods["rel"]
    if core and rel._CORE57 is None:
        pytest.skip("psf_zero_core57 is not installed")
    saved = rel._CORE57
    if not core:
        rel._CORE57 = None
    try:
        before = dict(rel.CORE57_STATS)
        scored = 0
        for k in range(4):
            qc = random_circuit(3 + k, 4 + 2 * k, max_operands=2, measure=False, seed=10_200 + k)
            x, x2 = _run(mods, "prev", qc, recommended), _run(mods, "prev", qc, recommended)
            y = _run(mods, "rel", qc, recommended)
            if value_sig(x) == value_sig(x2):
                assert value_sig(y) == value_sig(x), k
                scored += 1
        assert scored >= 3
        used = rel.CORE57_STATS["rust"] - before["rust"]
        assert (used > 0) == (core and recommended)  # the estimates are made only with a target
    finally:
        rel._CORE57 = saved
