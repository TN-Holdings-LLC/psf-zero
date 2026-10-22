"""Builds psf_ai_compile a24 from a23 (run from the repository root):
    python patches/psf_ai_compile_a24_2026-10-22/make_a24.py

a24 = a23 + three items (owner, 2026-10-22):
  46. The ring template built in Rust (rust/psfbind, `ring.rs`): the default path of `ring_template` /
      `model_template`. Without the library, or with one older than ring r2, the call stops (no silent fallback).
      PSF_RUST_TEMPLATE=check also builds a23's Python template beside it and stops on any difference.
  47. E5, formula only: the ring is placed on the longest piece of the device's snake (on the couplers of the most
      common gate length, else on every working coupler), or refused with OutOfScopeError; never by VF2's search
      (both the template and the default call).
  48. The zipper schedule (the workplace's prototype P1), only with PSF_RING_ZIP=1: one more candidate after item
      44's rule's schedule, chosen by item 84's rule (`_better`). Without the switch, the output is a23's.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "psf_ai_compile_a23_2026-10-16", "psf_ai_compile.py")
DST = os.path.join(HERE, "psf_ai_compile.py")
TAIL = os.path.join(HERE, "a24_tail.py")


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


s = open(SRC, encoding="utf-8").read()

s = rep(s, '''"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-16.a23
(a22 + items 44-45:''', '''"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, 2026-10-22.a24
(a23 + items 46-48: the ring template built in Rust, the ring placed by formula only, the zipper schedule as an option
-- see patches/psf_ai_compile_a24_2026-10-22/make_a24.py). a23's description:
(a22 + items 44-45:''')

s = rep(s, '''Placement is psf_compile's (first VF2 match, no score); no seed, error value or T1/T2 decides; failed elements are
never used (item 26).
"""''', '''  46. (a24) The ring template (item 35, `ring_template` and `model_template` for a ring that does not embed) is built
      by PSF-Zero's rust/psfbind (`ring.rs`, ported line by line from a22 and psf_compile c61, CPython-exact complex
      arithmetic), called through ctypes; the result is the same RingTemplate class, so binding and every caller are
      unchanged. The library is found by PSF_BIND_LIB or at rust/psfbind/target/release; without it, or with a library
      older than "psfbind ring r2", the call raises RuntimeError (build it: `cargo build --release --manifest-path
      rust/psfbind/Cargo.toml`). A ring the Rust does not port (item 41's other diagonal gates) is built by the
      Python, with items 47-48 (counted, A24_STATS["rust_refused"]). PSF_RUST_TEMPLATE=check also builds the Python
      template and raises on any difference (gate list, qubits, fixed angles or factors above 1e-12, runs, layout).
      Evidence before the release (TOQB, 2026-10-21/22): identical templates on 146 of 150 cases over five devices
      with random failed elements (the other 4 are item 47's), 295 of 295 at the workplace, 1,180 bindings; used on
      the device day of 2026-10-22 (job on ibm_marrakesh) in check mode.
  47. (a24) E5, formula only (owner, 2026-10-21, "B"; scope settled 2026-10-22): the ring's line is the longest
      piece of the device's snake on the couplers of the most common gate length, else on every working coupler
      (psf_compile item 87's first rule, both by formula). a23 tried, in order, the first, the first path
      rustworkx's VF2 finds on those couplers, the second, and VF2 on every working coupler; a24 never searches
      and raises OutOfScopeError where neither snake piece holds the ring: in the template (Rust and Python alike)
      and in the default call. Embeddable rings (item 42) are psf_compile's and unchanged.
  48. (a24) The zipper schedule, an option: with PSF_RING_ZIP=1, for an even ring of n >= 4, the workplace's zipper
      (prototype P1 of 2026-10-22: start 0, 1, 2, n-1, n-2, 3, 4, ...; rounds A, B (CZ then SWAP at both ends), C
      (interior SWAPs), D; 5n/2 - 4 CZ per layer once each CZ + SWAP is merged, against item 31's 28 at n = 10) is
      built after item 44's rule's schedule and item 84's rule chooses (`_better`: an operation not offered or at
      error >= 1 loses, then the shorter duration, then fewer two-qubit gates; a full tie keeps item 44's). In the
      template and in the default call. Without the switch nothing changes. Evidence (TOQB, report only, on TOQB's
      twin of ibm_marrakesh): 25 % fewer CZ and 31 % shorter at n = 10; on the twin its noise acts as one uniform
      shrink, which at 16 layers is stronger than item 31's (results/lam-ablate-2026-10-22); not run on a device.
Placement is psf_compile's (first VF2 match, no score) for what psf_compile builds; the ring formula's line is the
snake piece (item 47). No seed, error value or T1/T2 decides; failed elements are never used (item 26).
"""''')

s = rep(s, '''AI_COMPILE_VERSION = "2026-10-16.a23"  # prototype: items 44-45''',
        '''AI_COMPILE_VERSION = "2026-10-22.a24"  # items 46-48 (Rust template, formula-only placement, zipper option) + 2026-10-16.a23: items 44-45''')

# item 47 in the default call's ring candidate and in the Python template builder
old_place_c = '''    K, kadj, fadj, snake = lat
    line, _ = pc._chain_place(n, K, fadj, snake)
    if line is None:
        line, _ = pc._chain_place(n, K, kadj, snake)
    if line is None:
        A16_STATS["ring_no_path"] += 1
        return None
'''
s = rep(s, old_place_c, '''    K, kadj, fadj, snake = lat
    line = _snake_line(n, K, fadj, kadj, snake)  # item 47: the snake piece, or OutOfScopeError
''')
old_place_t = '''    K, kadj, fadj, snake = lat
    line, _ = pc._chain_place(n, K, fadj, snake)
    if line is None:
        line, _ = pc._chain_place(n, K, kadj, snake)
    if line is None:
        return None
'''
s = rep(s, old_place_t, '''    K, kadj, fadj, snake = lat
    line = _snake_line(n, K, fadj, kadj, snake)  # item 47: the snake piece, or OutOfScopeError
''')

# item 48 in the default call
s = rep(s, '''def _formula_only(qc, target, basis_gates, seed):
    """Items 34 and 44: (label, circuit) -- the ring schedule item 44's rule names, after item 29's rx merge and rzz
    fold, on no failed element -- or (None, None)."""
    rounds_fn, label = _ring_rounds_for(qc.num_qubits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ring = _ring_candidate(qc, target, basis_gates, seed, rounds_fn)
    if ring is None:
        return None, None
    ring = _executable(_rx_merged(ring, target), target)
    edges, bad = pc._failed_elements(target, FAILED_MAX_ERROR)
    if ring is None or pc._uses_failed(ring, edges, bad):
        return None, None
    return label, ring
''', '''def _formula_only(qc, target, basis_gates, seed):
    """Items 34, 44 and 48: (label, circuit) -- the ring schedule item 44's rule names (and, with PSF_RING_ZIP=1 on an
    even ring, the zipper, item 84's rule choosing), after item 29's rx merge and rzz fold, on no failed element -- or
    (None, None). Item 47: OutOfScopeError where the snake holds no line for the ring."""
    rounds_fn, label = _ring_rounds_for(qc.num_qubits)
    cands = [(label, rounds_fn)]
    if _zip_on(qc.num_qubits):
        cands.append(("zip", ring_rounds_zip))
    edges, bad = pc._failed_elements(target, FAILED_MAX_ERROR)
    best, best_label = None, None
    for lab, fn in cands:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ring = _ring_candidate(qc, target, basis_gates, seed, fn)
        if ring is None:
            continue
        ring = _executable(_rx_merged(ring, target), target)
        if ring is None or pc._uses_failed(ring, edges, bad):
            continue
        if best is None or pc._better(ring, best, target):
            best, best_label = ring, lab
    if best is None:
        return None, None
    return best_label, best
''')

# item 46 / 48: ring_template
s = rep(s, '''def ring_template(qc, target, basis_gates=None, seed=0, rounds_fn=None):
    """Items 35 and 44: the RingTemplate of the ring schedule `rounds_fn` names, or else item 44's rule names; None
    when it does not build."""
    if rounds_fn is None:
        rounds_fn, _ = _ring_rounds_for(qc.num_qubits)
    try:
        return _ring_template_one(qc, target, basis_gates, seed, rounds_fn)
    except RuntimeError:
        return None
''', '''def ring_template(qc, target, basis_gates=None, seed=0, rounds_fn=None):
    """Items 35, 44, 46-48: the RingTemplate for `qc` on `target`, built in Rust (item 46) with the schedule item 44's
    rule names (and the zipper with PSF_RING_ZIP=1, item 48); None when it does not build; OutOfScopeError where the
    snake holds no line (item 47). With `rounds_fn` (one schedule named by the caller) a23's Python builder, with
    item 47."""
    if rounds_fn is not None:
        try:
            return _ring_template_one(qc, target, basis_gates, seed, rounds_fn)
        except RuntimeError:
            return None
    return _ring_template_rust(qc, target, basis_gates, seed)
''')

s = s.rstrip("\n") + "\n\n\n" + open(TAIL, encoding="utf-8").read()
open(DST, "w", encoding="utf-8").write(s)
print("written", DST)
