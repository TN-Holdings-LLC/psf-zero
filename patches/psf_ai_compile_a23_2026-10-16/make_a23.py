"""Builds psf_ai_compile a23 from a22 (run from the repository root):
    python patches/psf_ai_compile_a23_2026-10-16/make_a23.py

a23 = a22 with what costs time and is not the formula removed, and no switches left (owner, 2026-10-16):
  44. Compare-and-choose removed. The ring schedule is chosen by the ring size: N2 (item 28) for 4 and 6 qubits, item
      31's otherwise. On four devices, 4-20 qubits and 1-8 layers, this is the schedule a22's comparison kept every
      time (N2 won at 6, tied at 4 where a22 keeps N2, item 31's won elsewhere). One ring is built, not two.
  45. Dead paths and switches removed. Kept as a22's defaults, with their switch gone:
      - item 34's formula-only path (FORMULA_ONLY), and the hand-off of everything else (HAND_OFF);
      - item 36's refusal without a Target (GATE_CLOSED);
      - items 41-42 (DIAG_RINGS, EMBED_RINGS) and items 28-29 (RING_FORMULA, RX_ONE_PULSE).
      Removed:
      - a15/a14 (their candidates, polish and searched routing; SEARCH_FALLBACK, CONSTRUCT_BY_DURATION);
      - the exactness checks (items 30, 37, 38: EXACT_CHECK, ONE_CHECK, HANDOFF_CHECK, FORMULA_CHECK);
      - the Qiskit level-3 hand-off without a Target, which item 36 already refused.
      a14's two constants are written in (SMALL_MAX_QUBITS 8 is no longer used; FAILED_MAX_ERROR 0.5; DEFAULT_SEEDS).
"""
import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "psf_ai_compile_a22_2026-10-15", "psf_ai_compile.py")
DST = os.path.join(HERE, "psf_ai_compile.py")


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


def drop_def(s, name):
    """Remove a top-level def/class `name` (with its trailing blank lines)."""
    tree = ast.parse(s)
    lines = s.split("\n")
    for nd in tree.body:
        if isinstance(nd, (ast.FunctionDef, ast.ClassDef)) and nd.name == name:
            a, b = nd.lineno - 1, nd.end_lineno
            while b < len(lines) and lines[b].strip() == "":
                b += 1
            return "\n".join(lines[:a] + lines[b:])
    raise AssertionError(name)


s = open(SRC, encoding="utf-8").read()

# --- header and version
s = rep(s, '"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-15.a22\n',
        '"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-16.a23\n'
        '(a22 + items 44-45: compare-and-choose replaced by a rule on the ring size; a15/a14, the exactness checks and\n'
        'every switch removed -- see patches/psf_ai_compile_a23_2026-10-16/make_a23.py). a22\'s description:\n')
s = re.sub(r'AI_COMPILE_VERSION = "2026-10-15\.a22"  # prototype: ',
           'AI_COMPILE_VERSION = "2026-10-16.a23"  # prototype: items 44-45 (no compare-and-choose, no checks, no switches) + ', s)
assert '"2026-10-16.a23"' in s

# --- a15 / a14 no longer loaded; their constants written in
s = rep(s, s[s.index("_A15_FILE = os.path.join("):s.index("AI_COMPILE_VERSION")],
        "FAILED_MAX_ERROR = 0.5  # a14 (item 22): as the release's prune_max_error\n"
        "DEFAULT_SEEDS = (0, 1, 2, 3)  # a14's; only the first is used (item 39)\n\n")
s = s.replace("a14.FAILED_MAX_ERROR", "FAILED_MAX_ERROR")
for line in ("SMALL_MAX_QUBITS = a15.SMALL_MAX_QUBITS\n", "DEFAULT_SEEDS = a15.DEFAULT_SEEDS\n",
             "ONE_CHECK = True  # item 37\n", "HANDOFF_CHECK = False  # item 38\n", "DIAG_RINGS = True  # item 41\n",
             "EMBED_RINGS = True  # item 42\n", "CONSTRUCT_BY_DURATION = True  # items 23-29; False gives a14\n",
             "SEARCH_FALLBACK = False  # item 27; False since a18 (item 32)\n", "HAND_OFF = True  # item 33\n",
             "FORMULA_ONLY = True  # item 34\n", "FORMULA_CHECK = False  # item 34\n", "RING_FORMULA = True  # item 28\n",
             "RX_ONE_PULSE = True  # item 29\n", "EXACT_CHECK = True  # item 30\n", "GATE_CLOSED = True  # item 36\n"):
    s = rep(s, line, "")
s = rep(s, s[s.index("EXACT_TVD = 1e-9"):s.index("A18_STATS = ")], "")

s = rep(s, """Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
\"\"\"""", """  44. (a23) No compare-and-choose. The ring formula (items 34, 35) builds one schedule, by a rule on the ring size:
      N2's (item 28) for n = 4 and 6, item 31's otherwise. Measured before the rule was fixed (2026-10-16): over
      FakeFez, FakeTorino, FakeMarrakesh and FakeKingston, n = 4-20, L = 1, 2, 4, 8 (272 cases), a22's comparison
      chose N2's at n = 6, tied at n = 4 (a22 kept N2's) and chose item 31's at every other size (12 and 20 embed
      and are handed on, item 42). Inputs that are not read as rings, are wider than RING_MAX_QUBITS, or come
      without a Target or basis are handed to psf_compile's `compile_for_hardware` (a Target is required).
  45. (a23) Removed, because the default calls never ran them or they cost time: a15 and a14 (their two constants
      written in), the exactness checks of items 30 and 37 (`_outcomes`, `_exact`, `_exact_filter`), and every
      switch (ONE_CHECK, HANDOFF_CHECK, DIAG_RINGS, EMBED_RINGS, CONSTRUCT_BY_DURATION, SEARCH_FALLBACK, HAND_OFF,
      FORMULA_ONLY, FORMULA_CHECK, RING_FORMULA, RX_ONE_PULSE, EXACT_CHECK, GATE_CLOSED, ...), each fixed at a22's
      default. The items above that name a switch describe its history. Checked: a22 and a23 return the same
      circuit, bit for bit (every operation, angle as a float's hex, global phase, initial and final layout), on
      the 272 cases (recompile and template), on refusals without a Target, and on 40 handed-on inputs
      (`check_same_a22.py`).
Placement is psf_compile's (first VF2 match, no score); no seed, error value or T1/T2 decides; failed elements are
never used (item 26).
\"\"\"""")
s = rep(s, "import importlib.util\n", "")
s = rep(s, "import os\nimport sys\n", "")

# --- switches hard-wired to a22's defaults
s = rep(s, "    return _ring_read2(qc) if DIAG_RINGS else _ring_read(qc)\n", "    return _ring_read2(qc)\n")
s = rep(s, "    if not RING_FORMULA:\n        return None\n", "")
s = rep(s, "    if not RX_ONE_PULSE or target is None or", "    if target is None or")
s = rep(s, "    if not (EMBED_RINGS and getattr(pc,", "    if not (getattr(pc,")

# --- item 45: the hand-off: a Target is required (item 36); no Qiskit branch, no check
i, j = s.index("def _hand_off("), s.index("def _formula_only(")
s = s[:i] + '''def _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info, kwargs):
    """Item 33: `qc` handed on unchanged to psf_compile's `compile_for_hardware` (with this front end's own
    `entangling_basis` and `layout_search`, "cx" and True by default). Without a Target it is refused (item 36)."""
    seed = seeds[0] if seeds else 0
    if target is None:  # item 36
        raise OutOfScopeError("PSF-Zero compiles only for a device's Target (changelog item 36); without one, use "
                              "Qiskit yourself")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pc.compile_for_hardware(qc, target=target, basis_gates=basis_gates, entangling_basis=entangling_basis,
                                      layout_search=layout_search, seed_transpiler=seed, **kwargs)
    A18_STATS["handed_compiler"] += 1
    info = {"path": "hand-off", "chosen": "compiler", "candidates": ["compiler"], "version": AI_COMPILE_VERSION,
            "exact": None}
    return (out, info) if return_info else out


def _ring_rounds_for(n):
    """Item 44: the ring schedule for an n-qubit ring -- N2's (item 28) for 4 and 6 qubits, item 31's otherwise --
    and its label. No candidate is built to be compared."""
    return (None, "ring") if n in (4, 6) else (ring_rounds_q4, "ring2")


''' + s[j:]

# --- item 44: one ring schedule, by rule
i, j = s.index("def _formula_only("), s.index("def compile_for_model_circuit(")
s = s[:i] + '''def _formula_only(qc, target, basis_gates, seed):
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


''' + s[j:]

# --- the entry: formula-only path only
i = s.index("def compile_for_model_circuit(")
j = s.index("# ---", i)
s = s[:i] + '''def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,
                              commutation_start=True, do_polish=True, layout_search=True, return_info=False,
                              target=None, **kwargs):
    """a23: a ring classifier (items 28, 41) that does not embed is built by the ring formula with the schedule item
    44's rule names; an embeddable ring (item 42) and every other input are handed on to psf_compile (item 33);
    without a Target, refused (item 36). `commutation_start` and `do_polish` are accepted and ignored (a15's)."""
    if "seed_transpiler" in kwargs:  # item 39
        given = kwargs.pop("seed_transpiler")
        if given is not None:
            seeds = (given,)
    if (target is None or basis_gates is None or qc.num_qubits > RING_MAX_QUBITS
            or _ring_read_path(qc) is None):  # item 40
        return _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info,
                         kwargs)
    if _ring_embeds(qc, target, kwargs.get("prune_max_error", 0.5)):  # item 42
        A22_STATS["ring_embedded"] += 1
        return _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info,
                         kwargs)
    label, best = _formula_only(qc, target, basis_gates, seeds[0] if seeds else 0)  # items 34, 44
    if best is not None:
        A19_STATS["formula"] += 1
        info = {"path": "formula", "chosen": label, "candidates": [label], "version": AI_COMPILE_VERSION}
        return (best, info) if return_info else best
    A19_STATS["not_built"] += 1
    return _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info,
                     kwargs)


''' + s[j:]

# --- item 44 in the template
i, j = s.index("def ring_template("), s.index("# ---", s.index("def ring_template("))
s = s[:i] + '''def ring_template(qc, target, basis_gates=None, seed=0, rounds_fn=None):
    """Items 35 and 44: the RingTemplate of the ring schedule `rounds_fn` names, or else item 44's rule names; None
    when it does not build."""
    if rounds_fn is None:
        rounds_fn, _ = _ring_rounds_for(qc.num_qubits)
    try:
        return _ring_template_one(qc, target, basis_gates, seed, rounds_fn)
    except RuntimeError:
        return None


''' + s[j:]

# --- the exactness checks (items 30, 37, 38) and the private helpers nothing calls any more (public names are kept)
for name in ("_outcomes", "_exact", "_exact_filter"):
    s = drop_def(s, name)
s = s.replace("GATE_CLOSED and ", "")
changed = True
while changed:
    changed = False
    tree = ast.parse(s)
    for nd in tree.body:
        if isinstance(nd, (ast.FunctionDef, ast.ClassDef)) and nd.name.startswith("_") and not nd.name.startswith("__"):
            if len(re.findall(r"\b" + re.escape(nd.name) + r"\b", s)) == 1:  # only its own definition
                s = drop_def(s, nd.name)
                changed = True
                break
for stat in ("A17_STATS",):
    if len(re.findall(r"\b" + stat + r"\b", s)) == 1:
        s = re.sub(r"^" + stat + r" = .*\n", "", s, flags=re.M)
open(DST, "w", encoding="utf-8", newline="\n").write(s)
print("a23 written:", len(s.splitlines()), "lines")
