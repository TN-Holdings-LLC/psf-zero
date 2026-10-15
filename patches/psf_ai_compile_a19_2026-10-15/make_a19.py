"""make_a19.py -- the AI front end 2026-10-15.a19 from the released a18 (benchmarks/psf_ai_compile.py) by exact
substitutions: item 34, formula only for a ring classifier -- the two ring schedules (items 28, 31), the shorter
returned; a15's starts, polish and the per-candidate simulation check (item 30) are not made.
    python make_a19.py <a18 psf_ai_compile.py> <output psf_ai_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub('"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-14.a18\n',
    '"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-15.a19\n'
    '(a18 + item 34: formula only for a ring classifier). a18\'s description:\n')
sub('''Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
"""''', '''  34. (a19) Formula only. For a ring classifier only the two ring schedules are built (items 28 and 31), each with
      item 29's rx merge and rzz fold, and the shorter (`_better`) of those on no failed element is returned; a15's
      starts and polish and item 30's per-candidate simulation check are not made (FORMULA_CHECK = True adds item 30's
      check). If neither schedule builds, the input is handed on as item 33 does. Why: on 2026-10-15 a18 spent 0.17-1.2
      s per classifier, about 70% of it on item 30's checks and most of the rest on candidates that lost (the ring won
      63 of 64 W3 cells), against 28-36 ms for the ring formula alone (Addendum 493). FORMULA_ONLY = False restores a18.
Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
"""''')
sub('AI_COMPILE_VERSION = "2026-10-14.a18"  # adopted 2026-10-14 with psf_compile 2026-10-14.1 (Addendum 492); a12 is psf_ai_compile_a12.py; prototype:',
    'AI_COMPILE_VERSION = "2026-10-15.a19"  # prototype: item 34 (formula only for ring classifiers) on a18, which was:')
sub('HAND_OFF = True  # item 33\n', 'HAND_OFF = True  # item 33\nFORMULA_ONLY = True  # item 34\nFORMULA_CHECK = False  # item 34\n'
    'A19_STATS = {"formula": 0, "not_built": 0, "checked": 0, "check_failed": 0}\n')
sub('''        return _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info,
                         kwargs)
''', '''        return _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info,
                         kwargs)
    if FORMULA_ONLY and target is not None and basis_gates is not None:  # item 34
        label, best = _formula_only(qc, target, basis_gates, seeds[0] if seeds else 0)
        if best is not None and FORMULA_CHECK:
            A19_STATS["checked"] += 1
            if _exact(qc, best) is False:
                A19_STATS["check_failed"] += 1
                best = None
        if best is not None:
            A19_STATS["formula"] += 1
            info = {"path": "formula", "chosen": label, "candidates": ["ring", "ring2"], "version": AI_COMPILE_VERSION}
            return (best, info) if return_info else best
        A19_STATS["not_built"] += 1
        return _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info,
                         kwargs)
''')
sub('''def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,''',
    '''def _formula_only(qc, target, basis_gates, seed):
    """Item 34: (label, circuit) -- the shorter of the two ring schedules (items 28, 31) after item 29's rx merge and
    rzz fold, on no failed element -- or (None, None)."""
    cands = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ring = _ring_candidate(qc, target, basis_gates, seed)
        if ring is not None:
            cands.append(("ring", ring))
            ring2 = _ring_candidate(qc, target, basis_gates, seed, ring_rounds_q4)
            if ring2 is not None:
                cands.append(("ring2", ring2))
    cands = [(lb, _executable(_rx_merged(c, target), target)) for lb, c in cands]
    edges, bad = pc._failed_elements(target, a14.FAILED_MAX_ERROR)
    cands = [(lb, c) for lb, c in cands if c is not None and not pc._uses_failed(c, edges, bad)]
    if not cands:
        return None, None
    label, best = cands[0]
    for lb, c in cands[1:]:
        if pc._better(c, best, target):
            label, best = lb, c
    return label, best


def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
