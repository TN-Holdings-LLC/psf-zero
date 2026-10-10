"""Tests for candidate psf_compile 2026-10-07.c24 (changelog item 51: the recommended call switches to an alternative
only for an estimated gain above SWITCH_MARGIN) against release 2026-10-07.1 (psf_compile.py), on which it is based.

Run from the repository root:  python -m pytest patches/psf_compile_c24_2026-10-07/test_c24.py -q -s
"""
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)

RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
DEVICES = ("FakeAuckland", "FakeTorino", "FakeHanoiV2")
STALE_BASE = 96_000_000  # the tests' own stale draws (STALE 80,000,000; CALSPLIT 91-93,000,000; MARGIN 97-99,000,000)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "patches", "psf_compile_release_2026-10-07.1", "psf_compile.py"),
                        "psf_compile_rel_c24_test")  # 2026-10-07.1, kept since release 2026-10-10.1
    c24 = H.load_module(os.path.join(HERE, "psf_compile.py"), "psf_compile_c24_test")
    return dict(rel=rel, c24=c24)


def backend(name):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def stale(dev):
    import stale_eval as S
    old = S.BASE
    S.BASE = STALE_BASE
    try:
        return S.stale_target(backend(dev).target, dev)[0]
    finally:
        S.BASE = old


def call(mod, qc, dev, target=None, recommended=False):
    b = backend(dev)
    t = target if target is not None else b.target
    basis = [g for g in ("cx", "cz", "ecr", "rz", "sx", "x") if g in t.operation_names]
    kw = dict(target=t, **RECOMMENDED) if recommended else {}
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = mod.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis,
                                       entangling_basis="cx", layout_search=True, seed_transpiler=0, **kw)
    return out, time.perf_counter() - t0


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def family(name, n, seed):
    import skip_eval
    return skip_eval.family_circuit(name, n, np.random.default_rng(seed))


def classifier(n, L, seed):
    """DEPTH's classifier circuit (depth_eval.circuit, imported unchanged) with random data and parameters."""
    import depth_r_eval as R
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 0.6, R.E.n_params(n, L))
    return R.E.circuit(rng.uniform(-1, 1, n), th, n, L)


def inputs():
    out = [("ring6", family("ring", 6, 51_000_001)), ("brick8", family("brick", 8, 51_000_002)),
           ("pauli8", family("pauli", 8, 51_000_003)), ("qft6", family("qft", 6, 51_000_004))]
    for k, (n, L) in enumerate(((4, 2), (4, 8), (6, 4), (6, 12))):
        out.append((f"cls_n{n}_L{L}", classifier(n, L, 51_000_100 + k)))
    return out


def test_versions(mods):
    assert mods["c24"].VERSION == "2026-10-07.c24"
    assert mods["rel"].VERSION == "2026-10-07.1"
    assert mods["c24"].SWITCH_MARGIN == 0.05
    assert mods["c24"].ESTIMATE_TIE_TOL == mods["rel"].ESTIMATE_TIE_TOL == 1e-12
    assert "margin_kept" in mods["c24"].COMPARE_STATS and "margin_kept" in mods["c24"].RESYNTH_STATS


def test_better_unit(mods):
    c = mods["c24"]
    assert c._better(0.94, 1.0) and not c._better(0.96, 1.0) and not c._better(1.0, 1.0)
    assert not c._better(1.0 - 5e-13, 1.0) and not c._better(1.1, 1.0)
    old = c.SWITCH_MARGIN
    try:
        rng = np.random.default_rng(51_000_200)
        pairs = list(zip(rng.uniform(0, 1, 20000), rng.uniform(0, 1, 20000)))
        pairs += [(a * (1 - d), a) for a, d in zip(rng.uniform(0.01, 1, 2000), rng.uniform(0, 3e-12, 2000))]
        for m in (0.0, c.ESTIMATE_TIE_TOL):
            c.SWITCH_MARGIN = m
            assert all(c._better(b, a) == c._lower(b, a) for b, a in pairs), m
    finally:
        c.SWITCH_MARGIN = old


@pytest.mark.parametrize("dev", DEVICES)
def test_default_call_unchanged(mods, dev):
    """Item 51 touches only the recommended call's choices: the default call returns the release's circuit."""
    for name, qc in inputs()[:4]:
        assert sig(call(mods["c24"], qc, dev)[0]) == sig(call(mods["rel"], qc, dev)[0]), name


@pytest.mark.parametrize("dev", DEVICES)
def test_margin_at_tie_band_is_release(mods, dev):
    """SWITCH_MARGIN set to the tie band: the recommended call returns the release's circuit, with the true and with a
    stale Target."""
    c = mods["c24"]
    old = c.SWITCH_MARGIN
    c.SWITCH_MARGIN = c.ESTIMATE_TIE_TOL
    try:
        for tname, t in (("true", None), ("stale", stale(dev))):
            for name, qc in inputs():
                a = call(c, qc, dev, t, True)[0]
                b = call(mods["rel"], qc, dev, t, True)[0]
                assert sig(a) == sig(b), (tname, name)
    finally:
        c.SWITCH_MARGIN = old


@pytest.mark.parametrize("dev", DEVICES)
def test_margin_exact_and_accounted(mods, dev):
    """SWITCH_MARGIN 0.05: every output implements its input and uses no failed element; wherever it differs from the
    release's, the margin turned an alternative away in that call (counted, and printed)."""
    c, rel = mods["c24"], mods["rel"]
    differ = kept = 0
    tc = tr = 0.0
    for tname, t in (("true", None), ("stale", stale(dev))):
        tt = t if t is not None else backend(dev).target
        for name, qc in inputs():
            k0 = c.COMPARE_STATS["margin_kept"] + c.RESYNTH_STATS["margin_kept"]
            a, ta = call(c, qc, dev, t, True)
            dk = c.COMPARE_STATS["margin_kept"] + c.RESYNTH_STATS["margin_kept"] - k0
            b, tb = call(rel, qc, dev, t, True)
            tc, tr = tc + ta, tr + tb
            assert c._implements(qc, a), (tname, name)
            assert c._acceptable(a, backend(dev).target, 0.5), (tname, name)  # no failed element of the TRUE device
            kept += dk > 0
            if sig(a) != sig(b):
                differ += 1
                assert dk > 0, (tname, name)
    print(f"\n{dev}: {differ} of {2 * len(inputs())} outputs differ from the release's; the margin turned an "
          f"alternative away in {kept} calls; time c24 {tc:.1f} s, release {tr:.1f} s")
