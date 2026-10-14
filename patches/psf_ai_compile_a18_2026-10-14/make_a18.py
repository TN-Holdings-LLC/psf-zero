"""make_a18.py -- builds the AI front end 2026-10-14.a18 from the workplace's a17 by exact substitutions (home,
after the owner's scope declaration of 2026-10-14, Addendum 488): item 32, a14's seed search is no longer a
candidate; item 33, an input the front end does not build by formula is handed on unchanged, to psf_compile's default
call with the device's Target, or to Qiskit's level 3 without one, checked exact where it can be.
    python make_a18.py <a17 psf_ai_compile.py> <output psf_ai_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub('"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-14.a17\n'
    '(workplace proposal: a16 + items 30-31).',
    '"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-14.a18\n'
    '(a17, the workplace\'s, + items 32-33 at home: no search; what is not built by formula is handed on).')
sub('''Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
"""''', '''  32. (a18) a14's searched routing is no longer a candidate (SEARCH_FALLBACK = False). Why: the owner's scope
      declaration of 2026-10-14 (Addendum 488) -- PSF-Zero competes only where it builds by formula, and does not
      search. On W3's 64 cells a17 chose the search once per Target (FakeTorino, n = 4, L = 1).
  33. (a18) Hand-off. An input that item 28's reader does not read as a ring classifier, and every input a17 sent to
      a14 (no Target, more than SMALL_MAX_QUBITS qubits, no basis, a psf_compile without `_better`), is handed on
      unchanged: with the device's Target to psf_compile's default call (`compile_for_hardware`, which builds chains
      and embeddable lattices by formula itself and applies its own scope), without one to Qiskit's level 3 with the
      coupling map and basis given (seed `seeds[0]`). The front end's own candidates (a15's starts and polish, the
      rings, the rx merge) are made only for a ring classifier. The handed-on output is checked exact as item 30 does
      where it can be; an inexact one is returned unchanged with a warning and counted (A18_STATS), since the
      hand-off promises the receiver's own output. Why: on real devices PSF-Zero's own choices outside the circuits
      it builds by formula lost on average (F 0.78 x over 25 pairs, Addendum 488). HAND_OFF = False restores a17.
Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
"""''')
sub('AI_COMPILE_VERSION = "2026-10-14.a17"  # workplace proposal: a15 + items 28-31 (ring formula candidates; rx one pulse; every candidate checked exact)',
    'AI_COMPILE_VERSION = "2026-10-14.a18"  # prototype: a17 (workplace: items 28-31) + items 32-33 (no search; hand-off outside the ring formula)')
sub('SEARCH_FALLBACK = True  # item 27\n', 'SEARCH_FALLBACK = False  # item 27; False since a18 (item 32)\nHAND_OFF = True  # item 33\n')
sub('A17_STATS = {"checked": 0, "inexact_dropped": 0, "unchecked": 0, "largest_tvd_dropped": 0.0, "all_inexact": 0}\n',
    'A17_STATS = {"checked": 0, "inexact_dropped": 0, "unchecked": 0, "largest_tvd_dropped": 0.0, "all_inexact": 0}\n'
    'A18_STATS = {"handed_compiler": 0, "handed_qiskit": 0, "handoff_exact": 0, "handoff_unchecked": 0, "handoff_inexact": 0}\n')
sub('''    """a16: a15 (items 23-27) with N2's ring formula as one more candidate (item 28) and, with rx in the Target, every
    candidate's single-qubit runs in one pulse (item 29); otherwise a14."""
    if (not CONSTRUCT_BY_DURATION or target is None or basis_gates is None or qc.num_qubits > SMALL_MAX_QUBITS
            or not hasattr(pc, "_better")):
        return a14.compile_for_model_circuit(''', '''    """a18: for a ring classifier (item 28's reader), a16's candidates -- a15's (items 23-26), N2's ring formula and
    the second ring schedule (items 28, 31), single-qubit runs in one pulse with rx (item 29), every candidate checked
    exact (item 30), no search (item 32) -- the shortest returned; every other input handed on unchanged (item 33)."""
    if HAND_OFF and (not CONSTRUCT_BY_DURATION or target is None or basis_gates is None
                     or qc.num_qubits > SMALL_MAX_QUBITS or not hasattr(pc, "_better") or _ring_read(qc) is None):
        return _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info,
                         kwargs)
    if (not CONSTRUCT_BY_DURATION or target is None or basis_gates is None or qc.num_qubits > SMALL_MAX_QUBITS
            or not hasattr(pc, "_better")):
        return a14.compile_for_model_circuit(''')
sub('''def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,''',
    '''def _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info, kwargs):
    """Item 33: `qc` handed on unchanged -- with a Target to psf_compile's default call, without one to Qiskit's level
    3 (coupling map and basis given, seed seeds[0]) -- and checked exact as item 30 does where it can be."""
    seed = seeds[0] if seeds else 0
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if target is not None:
            out = pc.compile_for_hardware(qc, target=target, basis_gates=basis_gates, entangling_basis=entangling_basis,
                                          layout_search=layout_search, seed_transpiler=seed, **kwargs)
            label = "compiler"
            A18_STATS["handed_compiler"] += 1
        else:
            from qiskit import transpile
            cmap = coupling_map if coupling_map is None or isinstance(coupling_map, CouplingMap) else CouplingMap(coupling_map)
            out = transpile(qc, coupling_map=cmap, basis_gates=basis_gates, optimization_level=3, seed_transpiler=seed)
            label = "QK3"
            A18_STATS["handed_qiskit"] += 1
    verdict = _exact(qc, out)
    if verdict is None:
        A18_STATS["handoff_unchecked"] += 1
    elif verdict:
        A18_STATS["handoff_exact"] += 1
    else:
        A18_STATS["handoff_inexact"] += 1
        warnings.warn(f"psf_ai_compile item 33: the handed-on output ({label}) is not exact within EXACT_TVD; returned "
                      "unchanged", RuntimeWarning, stacklevel=3)
    info = {"path": "hand-off", "chosen": label, "candidates": [label], "version": AI_COMPILE_VERSION,
            "exact": verdict}
    return (out, info) if return_info else out


def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
