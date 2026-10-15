"""make_a21.py -- the AI front end 2026-10-15.a21 from a20 by exact substitutions. The workplace's review of a18
(delivery 2026-10-15a, W1; Addendum 495) taken up:
  37. one exactness check: psf_compile's item 93 (the input unrolled to its own product formula first; one tolerance)
  38. the hand-off's output is not simulated again (HANDOFF_CHECK = False)
  39. `seed_transpiler` given as a keyword is the seed, not a TypeError
  40. the ring formula up to RING_MAX_QUBITS qubits, not SMALL_MAX_QUBITS
  41. rings of commuting diagonal two-qubit gates (cz, rzz, cp, crz, cs, csdg), not only CZ
    python make_a21.py <a20 psf_ai_compile.py> <output psf_ai_compile.py>"""
import sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


# --- header, changelog, version
sub('"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-15.a20\n'
    '(a19 + item 35: the ring template; item 36: the gate closed). a18\'s description:\n',
    '"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-15.a21\n'
    '(a20 + items 37-41: the workplace\'s review of a18 taken up). a18\'s description:\n')
sub('''Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
"""''', '''  37. (a21) One exactness check. `_exact` is psf_compile's item 93 (`_exact93`): the input is first unrolled to its own
      synthesis -- a PauliEvolutionGate to its product formula -- and the largest outcome difference accepted is
      psf_compile's EXACT_CHECK_TOL (1e-6). Before, item 30 built the state vector from the input as written: a
      PauliEvolutionGate was exponentiated exactly (960 s at 12 qubits, the process killed for memory at 20) and
      compared with a product formula, so a correct output was called inexact; and its tolerance (1e-9) was not item
      93's. Workplace review W1, findings A and B (Addendum 495). ONE_CHECK = False restores item 30's own check.
  38. (a21) The hand-off's output is not simulated again (HANDOFF_CHECK = False). psf_compile answers for its own
      output: the release checks it by item 93, and the formula-only prototypes return their formula unsimulated by
      the owner's choice (psf_compile item 97), as this front end does for its ring (item 34). HANDOFF_CHECK = True
      checks it by item 37.
  39. (a21) `seed_transpiler=` given as a keyword is the seed (it replaces `seeds`); before, it was passed twice and
      raised a TypeError. Workplace review W1, finding E.
  40. (a21) The ring path (items 28, 31, 34, 35) up to RING_MAX_QUBITS qubits (20) instead of SMALL_MAX_QUBITS (8), so
      that rings of 9-20 qubits are built by the formula rather than handed on. Workplace review W1, finding C. Only
      with FORMULA_ONLY; a15's candidate path keeps SMALL_MAX_QUBITS.
  41. (a21) Rings of commuting diagonal two-qubit gates. Item 28's schedule needs only that a layer's ring gates
      commute; every diagonal gate does. A layer may now be any of cz, rzz, cp, crz, cs, csdg on the ring's edges
      (`_ring_read2`), each emitted with its own angle (a crz in its written orientation, the symmetric ones as a20
      orients its CZ, so a CZ ring compiles exactly as in a20); on a Target with rzz the formula is
      compiled with rzz in the basis. The template (item 35) stays CZ-only. Workplace review W1, finding F (a ring
      written with cx does not commute and is still not read). DIAG_RINGS = False restores a20.
Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
"""''')
sub('AI_COMPILE_VERSION = "2026-10-15.a20"  # prototype: items 35-36 (ring template; gate closed) + item 34 (formula only) on a18, which was:',
    'AI_COMPILE_VERSION = "2026-10-15.a21"  # prototype: items 37-41 (workplace review of a18) + items 35-36 (ring template; gate closed) + item 34 (formula only) on a18, which was:')
sub('SMALL_MAX_QUBITS = a15.SMALL_MAX_QUBITS\n',
    'SMALL_MAX_QUBITS = a15.SMALL_MAX_QUBITS\n'
    'ONE_CHECK = True  # item 37\n'
    'HANDOFF_CHECK = False  # item 38\n'
    'RING_MAX_QUBITS = 20  # item 40\n'
    'DIAG_RINGS = True  # item 41\n'
    'RING_DIAGONAL = ("cz", "rzz", "cp", "crz", "cs", "csdg")  # item 41: commuting diagonal two-qubit gates\n')

# --- item 41: the reader for diagonal rings (a20's _ring_read stays, CZ-only, for the template)
sub('''def ring_rounds(n):''', '''def _ring_read2(qc):
    """Item 41: as `_ring_read`, with every layer's ring gates any commuting diagonal two-qubit gates (RING_DIAGONAL):
    (cycle, segments, tail, meas, ent) with ent[layer] = {(a, b) with a < b: (operation, written qubits)}; else None."""
    n = qc.num_qubits
    if n < 4:
        return None
    segments, cur, block, blocks, tail_meas, ents, ent = [], [], set(), [], [], [], {}
    touched = set()
    measured = set()
    for ins in qc.data:
        nm = ins.operation.name
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if nm == "barrier":
            continue
        if getattr(ins.operation, "condition", None) is not None or ins.operation.num_clbits > (1 if nm == "measure" else 0):
            return None
        if nm == "measure":
            if qs[0] in measured:
                return None
            measured.add(qs[0])
            tail_meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))
            continue
        if measured & set(qs):
            return None
        if len(qs) == 1:
            if qs[0] in touched:
                if len(block) != n:
                    return None
                blocks.append(frozenset(block))
                segments.append(cur)
                ents.append(ent)
                cur, block, touched, ent = [], set(), set(), {}
            cur.append((ins.operation, qs[0]))
            continue
        if len(qs) != 2 or nm not in RING_DIAGONAL or getattr(ins.operation, "is_parameterized", lambda: False)():
            return None
        e = (min(qs), max(qs))
        if e in block:
            return None
        block.add(e)
        ent[e] = (ins.operation, tuple(qs))
        touched |= set(qs)
    if block:
        if len(block) != n:
            return None
        blocks.append(frozenset(block))
        segments.append(cur)
        ents.append(ent)
        cur = []
    if not blocks or len(set(blocks)) != 1:
        return None
    if measured and len(measured) != n:
        return None
    adj = {q: [] for q in range(n)}
    for a, b in blocks[0]:
        adj[a].append(b)
        adj[b].append(a)
    if any(len(v) != 2 for v in adj.values()):
        return None
    cycle = [0]
    nxt = min(adj[0])
    while nxt != 0:
        cycle.append(nxt)
        nxt = [v for v in adj[nxt] if v != cycle[-2]][0]
        if len(cycle) > n:
            return None
    if len(cycle) != n:
        return None
    return cycle, segments, cur, tail_meas, ents


def _ring_read_path(qc):
    """Items 28 and 41: the reader the ring path uses."""
    return _ring_read2(qc) if DIAG_RINGS else _ring_read(qc)


def ring_rounds(n):''')
sub('''    cycle, segments, tail, meas = read
''', '''    cycle, segments, tail, meas = read[:4]
    ent = read[4] if len(read) > 4 else None  # item 41
''')
sub('''    def do_czs(czs):
        for a, b in czs:
            out.cz(wire_at[pos[a]], wire_at[pos[b]])''', '''    def do_czs(czs, layer):
        for a, b in czs:
            if ent is None:
                out.cz(wire_at[pos[a]], wire_at[pos[b]])
                continue
            op, (qa, qb) = ent[layer][(min(cycle[a], cycle[b]), max(cycle[a], cycle[b]))]  # item 41
            if op.name == "crz":  # not symmetric: its written orientation
                out.append(op, [here(qa), here(qb)])
            else:  # symmetric: oriented as a20 orients its CZ, so that a CZ ring compiles exactly as in a20
                out.append(op, [wire_at[pos[a]], wire_at[pos[b]]])''')
sub('''                do_czs(czs)
                do_swaps(swaps)''', '''                do_czs(czs, layer)
                do_swaps(swaps)''')
sub('''                do_swaps(list(reversed(swaps)))
                do_czs(czs)''', '''                do_swaps(list(reversed(swaps)))
                do_czs(czs, layer)''')
sub('''    if not RING_FORMULA:
        return None
    read = _ring_read(qc)''', '''    if not RING_FORMULA:
        return None
    read = _ring_read_path(qc)  # item 41''')
sub('''    two = [g for g in basis_gates if g in ("cz", "cx", "ecr")] or ["cz"]
    one = [g for g in basis_gates if g in ("rz", "sx", "x")]
    try:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=two[:1] + one,''', '''    two = [g for g in basis_gates if g in ("cz", "cx", "ecr")] or ["cz"]
    one = [g for g in basis_gates if g in ("rz", "sx", "x")]
    if len(read) > 4 and "rzz" in target.operation_names and any(  # item 41: the device's own rzz
            op.name != "cz" for layer in read[4] for op, _ in layer.values()):
        one = one + ["rzz"]
    try:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=two[:1] + one,''')

# --- item 37: one exactness check
sub('''    if not EXACT_CHECK:
        return None
    if any(i.operation.name == "measure" for i in qc.data):
        p0 = _outcomes(qc) if ref is None else ref''', '''    if not EXACT_CHECK:
        return None
    if ONE_CHECK and hasattr(pc, "_exact93"):  # item 37: psf_compile's item 93, the input unrolled first
        try:
            return pc._exact93(qc, c)
        except Exception:  # noqa: BLE001
            return None
    if any(i.operation.name == "measure" for i in qc.data):
        p0 = _outcomes(qc) if ref is None else ref''')
sub('''    ref = _outcomes(qc) if any(i.operation.name == "measure" for i in qc.data) else None''',
    '''    ref = None if ONE_CHECK and hasattr(pc, "_exact93") else (  # item 37: no state vector of the input as written
        _outcomes(qc) if any(i.operation.name == "measure" for i in qc.data) else None)''')

# --- items 38 (no second simulation) and D (docstring)
sub('''    """Item 33: `qc` handed on unchanged -- with a Target to psf_compile's default call, without one to Qiskit's level
    3 (coupling map and basis given, seed seeds[0]) -- and checked exact as item 30 does where it can be."""''',
    '''    """Item 33: `qc` handed on unchanged -- with a Target to psf_compile's `compile_for_hardware` (with this front
    end's own `entangling_basis` and `layout_search`, "cx" and True by default, as C57-VAL calls it; the formula-only
    prototypes do not use them), without one to Qiskit's level 3 (coupling map and basis given, seed seeds[0]) -- and
    checked exact by item 37 only with HANDOFF_CHECK (item 38)."""''')
sub('''    verdict = _exact(qc, out)
    if verdict is None:
        A18_STATS["handoff_unchecked"] += 1''', '''    verdict = _exact(qc, out) if HANDOFF_CHECK else None  # item 38
    if verdict is None:
        A18_STATS["handoff_unchecked"] += 1''')

# --- items 39 (seed keyword) and 40 (ring limit), 41 (reader)
sub('''    exact (item 30), no search (item 32) -- the shortest returned; every other input handed on unchanged (item 33)."""
    if HAND_OFF and (not CONSTRUCT_BY_DURATION or target is None or basis_gates is None
                     or qc.num_qubits > SMALL_MAX_QUBITS or not hasattr(pc, "_better") or _ring_read(qc) is None):''',
    '''    exact (item 30), no search (item 32) -- the shortest returned; every other input handed on unchanged (item 33)."""
    if "seed_transpiler" in kwargs:  # item 39
        given = kwargs.pop("seed_transpiler")
        if given is not None:
            seeds = (given,)
    ring_max = RING_MAX_QUBITS if FORMULA_ONLY else SMALL_MAX_QUBITS  # item 40
    if HAND_OFF and (not CONSTRUCT_BY_DURATION or target is None or basis_gates is None
                     or qc.num_qubits > ring_max or not hasattr(pc, "_better") or _ring_read_path(qc) is None):''')

open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
