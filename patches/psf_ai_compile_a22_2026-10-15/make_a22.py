"""make_a22.py -- the AI front end 2026-10-15.a22 from a21 by exact substitutions and one appended block:
  42. a ring whose interaction graph embeds in the device is handed to psf_compile, whose item 99 (c60 on) builds it
      as written on the embedded ring, without SWAPs; a ring that does not embed keeps the ring formula
  43. `model_template(qc, target)`: the template the same rule picks -- psf_compile's `compile_template` (item 100,
      c61 on) for an embeddable ring and for every other in-scope input, the ring template (item 35) for a ring that
      does not embed
With a psf_compile without item 99 (or item 100), a22 behaves as a21.
    python make_a22.py <a21 psf_ai_compile.py> <output psf_ai_compile.py>"""
import sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub('"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-15.a21\n'
    '(a20 + items 37-41: the workplace\'s review of a18 taken up). a18\'s description:\n',
    '"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-15.a22\n'
    '(a21 + items 42-43: embeddable rings and templates through psf_compile\'s items 99-100). a18\'s description:\n')
sub('''      written with cx does not commute and is still not read). DIAG_RINGS = False restores a20.
''', '''      written with cx does not commute and is still not read). DIAG_RINGS = False restores a20.
  42. (a22) A ring whose two-qubit interaction graph embeds in the device's working couplers (psf_compile's item 96
      test: first VF2 match, no score) is handed to psf_compile, whose item 99 (c60 on) builds it as written on the
      embedded ring, commuting gates layered by edge colouring, with no SWAP. A ring that does not embed (on
      heavy-hex, every ring shorter than 12) keeps the ring formula, which folds it onto a line with SWAPs. A rule
      (embeds or not), not a comparison. Measured on FakeFez (Addendum 497): a 12-qubit CZ ring classifier, 4 layers,
      2.47 us by item 99 against 5.40 us by the ring formula. Only with a psf_compile that has item 99;
      EMBED_RINGS = False restores a21.
  43. (a22) `model_template(qc, target)`: the template for a training loop, by the same rule. An embeddable ring, and
      every other in-scope input, gets psf_compile's `compile_template` (item 100, c61 on); a ring that does not
      embed gets the ring template (item 35). The result binds a circuit of the same structure (`bind(qc)`) and
      carries `layout` (the device qubit of each input qubit at the start). None where neither applies.
''')
sub('AI_COMPILE_VERSION = "2026-10-15.a21"  # prototype: items 37-41 (workplace review of a18)',
    'AI_COMPILE_VERSION = "2026-10-15.a22"  # prototype: items 42-43 (embeddable rings and templates through psf_compile items 99-100) + items 37-41 (workplace review of a18)')
sub('DIAG_RINGS = True  # item 41\n',
    'DIAG_RINGS = True  # item 41\n'
    'EMBED_RINGS = True  # item 42\n'
    'A22_STATS = {"ring_embedded": 0, "templates_compiler": 0, "templates_ring": 0, "templates_none": 0}\n')
sub('''    if FORMULA_ONLY and target is not None and basis_gates is not None:  # item 34
        label, best = _formula_only(qc, target, basis_gates, seeds[0] if seeds else 0)''',
    '''    if target is not None and _ring_embeds(qc, target, kwargs.get("prune_max_error", 0.5)):  # item 42
        A22_STATS["ring_embedded"] += 1
        return _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info,
                         kwargs)
    if FORMULA_ONLY and target is not None and basis_gates is not None:  # item 34
        label, best = _formula_only(qc, target, basis_gates, seeds[0] if seeds else 0)''')

s = s.rstrip("\n") + '''


# ---------------------------------------------------------------------------------------------------------------------
# Prototype a22 (home, 2026-10-15): item 42, embeddable rings through psf_compile's item 99; item 43, model_template.
# ---------------------------------------------------------------------------------------------------------------------

def _ring_embeds(qc, target, max_error=0.5):
    """Item 42: True when `qc` is read as a ring (items 28, 41), psf_compile has item 99, and the ring's interaction
    graph embeds in the device's working couplers by item 96's test."""
    if not (EMBED_RINGS and getattr(pc, "NATIVE_BRICKWORK", False) and hasattr(pc, "_native_brickwork")
            and hasattr(pc, "_scope96_edges") and hasattr(pc, "_scope96_embeds")):
        return False
    if _ring_read_path(qc) is None:
        return False
    edges, reason = pc._scope96_edges(qc)
    return bool(reason is None and edges and pc._scope96_embeds(edges, qc.num_qubits, target, max_error))


class ModelTemplate:
    """Item 43: psf_compile's CompileTemplate (item 100) behind the RingTemplate's interface: `bind(qc)` checks the
    circuit's structure (as RingTemplate does) and binds; `layout` is the device qubit of each input qubit at the
    start; `inner` is the CompileTemplate."""

    def __init__(self, inner, qc):
        self.inner, self.sig = inner, _structure_sig(qc)
        lay = getattr(inner.circuit, "_layout", None)
        n = qc.num_qubits
        self.layout = (list(lay.initial_index_layout(filter_ancillas=True))[:n] if lay is not None else
                       list(range(n)))

    def bind(self, qc):
        if _structure_sig(qc) != self.sig:
            TEMPLATE_STATS["structure_mismatch"] += 1
            raise ValueError("item 43: the circuit's structure differs from the template's")
        TEMPLATE_STATS["bound"] += 1
        return self.inner.bind(qc)


def model_template(qc, target, basis_gates=None, seed=0, max_error=0.5):
    """Item 43: the template for `qc` on `target` by item 42's rule -- psf_compile's compile_template for an
    embeddable ring or any other in-scope input, item 35's ring template for a ring that does not embed -- or None."""
    if _ring_embeds(qc, target, max_error) or _ring_read_path(qc) is None:
        if not hasattr(pc, "compile_template"):
            A22_STATS["templates_none"] += 1
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inner = pc.compile_template(qc, target, seed_transpiler=seed, prune_max_error=max_error)
        except pc.TranspilerError:
            A22_STATS["templates_none"] += 1
            return None
        A22_STATS["templates_compiler"] += 1
        return ModelTemplate(inner, qc)
    tpl = ring_template(qc, target, basis_gates, seed)
    A22_STATS["templates_ring" if tpl is not None else "templates_none"] += 1
    return tpl
'''
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
