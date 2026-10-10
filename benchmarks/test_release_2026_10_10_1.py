"""Tests for release psf_compile 2026-10-10.1: candidate 2026-10-10.c29 of Addendum 422, which carries changelog
items 53 (candidate c26; C26-ID, Addenda 413-414) and 56 (C29-ID, Addenda 422-423). The candidates' own tests are
patches/psf_compile_c26_2026-10-09/test_c26.py and patches/psf_compile_c29_2026-10-10/test_c29.py.
The previous release, 2026-10-07.1, is kept unchanged in patches/psf_compile_release_2026-10-07.1/psf_compile.py.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_10_1.py -q -s
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

REL = os.path.join(REPO, "patches", "psf_compile_release_2026-10-10.1", "psf_compile.py")  # kept copy
C29 = os.path.join(REPO, "patches", "psf_compile_c29_2026-10-10", "psf_compile.py")
PREV = os.path.join(REPO, "patches", "psf_compile_release_2026-10-07.1", "psf_compile.py")


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
    return dict(rel=H.load_module(REL, "psf_compile_rel101001_test"),
                prev=H.load_module(PREV, "psf_compile_prev101001_test"), layout=lay)


@contextlib.contextmanager
def limit(mods, n):
    """EXACT_MAX_OPS set to n in both releases, so that item 56 acts on small circuits."""
    old = {k: mods[k].EXACT_MAX_OPS for k in ("rel", "prev")}
    for k in old:
        mods[k].EXACT_MAX_OPS = n
    try:
        yield
    finally:
        for k, v in old.items():
            mods[k].EXACT_MAX_OPS = v


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-10.1"
    assert mods["prev"].VERSION == "2026-10-07.1"
    assert set(mods["rel"].FEASIBILITY_STATS) == {"candidate", "level3", "input", "resynthesis"}
    assert not hasattr(mods["prev"], "FEASIBILITY_STATS")
    assert mods["rel"].ESTIMATE_TIE_TOL == 1e-12 and mods["rel"].EXACT_MAX_OPS == 200_000


def test_file_is_c29_except_the_version_lines():
    a, b = lines(REL), lines(C29)
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2
    assert diff[0][0].startswith("VERSION: 2026-10-10.1 -- release")
    assert diff[0][1].startswith("VERSION: 2026-10-07.1 -- release")  # c29 kept its base's header line
    assert diff[1][0].startswith('VERSION = "2026-10-10.1"') and diff[1][1].startswith('VERSION = "2026-10-10.c29"')


def test_candidate_and_previous_release_kept_unchanged():
    assert nsha(C29) == "36fdd78d854bb8f3ce170e76cded8f313b6f0429e13ea6af9e0d6b57cd4936a0"
    assert nsha(PREV) == "73fb2cb0b1acc5870339c23599829b326fbf57aa198945231e8551e55c1884dc"


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


@pytest.mark.parametrize("recommended", [False, True])
def test_same_output_as_the_previous_release(mods, recommended):
    """The previous release's output, by value (as C29-ID compares), wherever the previous release run twice gives
    one output; with EXACT_MAX_OPS lowered as well, so that item 56 skips estimates in the recommended call."""
    from qiskit.circuit.random import random_circuit
    from c29_identity import value_sig
    scored = 0
    before = sum(mods["rel"].FEASIBILITY_STATS.values())
    for k in range(4):
        qc = random_circuit(3 + k, 4 + 2 * k, max_operands=2, measure=False, seed=10_100 + k)
        for n in (12, 200_000):
            with limit(mods, n):
                x, x2 = _run(mods, "prev", qc, recommended), _run(mods, "prev", qc, recommended)
                y = _run(mods, "rel", qc, recommended)
            if value_sig(x) == value_sig(x2):
                assert value_sig(y) == value_sig(x), (k, n)
                scored += 1
    assert scored >= 5
    skipped = sum(mods["rel"].FEASIBILITY_STATS.values()) - before
    assert (skipped > 0) == recommended  # item 56 acts only in the recommended call (with a target)
