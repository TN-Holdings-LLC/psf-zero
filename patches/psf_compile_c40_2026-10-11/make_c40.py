"""Builds candidate 2026-10-11.c40 from candidate 2026-10-11.c39 by exact substitutions (changelog items 71-72: STEINER6
for every Pauli-rotation circuit it can read, commuting or not, under item 69's choice; product formulas of any order
and repetitions, sequences of evolutions and rotation gates, a leading layer of single-qubit gates, and exact merging
of repeated terms).
    python make_c40.py <c39 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

PARSE = '''

_ST_ROT = {"rz": "Z", "rx": "X", "ry": "Y", "rzz": "ZZ", "rxx": "XX", "ryy": "YY"}  # item 72: exp(-i theta/2 P)
_ST_FORMULAS = ("LieTrotter", "SuzukiTrotter")  # item 72: deterministic product formulas (not QDrift)


def _st_flatten(qc, depth=0):
    """Item 72: ([(operation, qubit indices, clbit indices)], global phase) of `qc`, with composite instructions that
    are not read directly (a PauliEvolutionGate, a rotation, a single-qubit gate, measure, barrier) replaced by their
    definitions, up to three levels."""
    out, phase = [], float(qc.global_phase) if not hasattr(qc.global_phase, "parameters") else None
    if phase is None:
        return None, None
    for ins in qc.data:
        op = ins.operation
        qs = [qc.find_bit(q).index for q in ins.qubits]
        cs = [qc.find_bit(c).index for c in ins.clbits]
        leaf = (op.name in ("barrier", "measure", "PauliEvolution") or op.name in _ST_ROT
                or (len(qs) == 1 and not cs and getattr(op, "blocks", None) is None))
        if not leaf and depth < 3 and not cs and getattr(op, "definition", None) is not None:
            inner, ph = _st_flatten(op.definition, depth + 1)
            if inner is None:
                return None, None
            phase += ph
            out += [(o, [qs[i] for i in q2], c2) for o, q2, c2 in inner]
        else:
            out.append((op, qs, cs))
    return out, phase


def _st_pauli_terms(gate, qs, n):
    """Item 72: the terms a PauliEvolutionGate on qubits `qs` applies, in order, as [(label on n qubits, angle, 1.0)]
    with each term exp(-i angle P), from its product formula's own expansion (LieTrotter or SuzukiTrotter, any order and
    repetitions); and the global phase of identity terms. None if not readable (a list of operators, an unbound time,
    complex coefficients, another formula)."""
    from qiskit.synthesis import LieTrotter
    if isinstance(gate.operator, list):
        return None
    try:
        float(gate.time)
    except (TypeError, ValueError):
        return None
    coeffs = np.asarray(gate.operator.coeffs)
    if coeffs.dtype == object or np.any(np.abs(np.imag(coeffs)) > 1e-12):
        return None
    syn = gate.synthesis if gate.synthesis is not None else LieTrotter()
    if type(syn).__name__ not in _ST_FORMULAS:
        return None
    try:
        seq = syn.expand(gate)
    except Exception:  # noqa: BLE001 - not expandable: the usual path
        return None
    out, phase = [], 0.0
    for item in seq:
        pl, idx, coeff = item[0], list(item[1]), float(np.real(item[2]))
        lab = ["I"] * n
        for ch, i in zip(pl, idx):  # character k acts on the gate's qubit idx[k]
            if ch != "I":
                lab[n - 1 - qs[i]] = ch
        lab = "".join(lab)
        if set(lab) <= {"I"}:
            phase -= coeff / 2
            continue
        out.append((lab, coeff / 2, 1.0))
    return out, phase


def _st_parse(qc):
    """Item 72: (leading single-qubit gates [(operation, qubit)], final measurements [(qubit, clbit)], terms
    [(label, angle, 1.0)] in order, global phase) when `qc` is: single-qubit gates first, then only Pauli evolutions and
    rotation gates (rz, rx, ry, rzz, rxx, ryy), then measurements (barriers anywhere); else None."""
    flat, phase = _st_flatten(qc)
    if flat is None:
        return None
    n = qc.num_qubits
    prefix, meas, terms = [], [], []
    started = False
    for op, qs, cs in flat:
        name = op.name
        if name == "barrier":
            continue
        if name == "measure":
            meas.append((qs[0], cs[0]))
            continue
        if meas:
            return None  # an operation after a measurement
        if name == "PauliEvolution":
            got = _st_pauli_terms(op, qs, n)
            if got is None:
                return None
            terms += got[0]
            phase += got[1]
            started = True
            continue
        if name in _ST_ROT and len(op.params) == 1:
            try:
                angle = float(op.params[0])
            except (TypeError, ValueError):
                return None
            lab = ["I"] * n
            for ch, q in zip(_ST_ROT[name], qs):
                lab[n - 1 - q] = ch
            terms.append(("".join(lab), angle / 2, 1.0))
            started = True
            continue
        if len(qs) == 1 and not cs and not started:
            try:
                op.to_matrix()
            except Exception:  # noqa: BLE001 - not a unitary gate
                return None
            prefix.append((op, qs[0]))
            continue
        return None
    if not terms:
        return None
    return prefix, meas, terms, phase


def _st_merge(terms):
    """Item 72: repeated terms joined where exact: a term is added to an earlier one with the same Pauli if no term in
    between anticommutes with it (it could then be moved next to it). [(label, angle, 1.0)] in order."""
    out, open_ = [], {}
    for lb, c, t in terms:
        x, z, _ = _st_term_masks(lb)
        if lb in open_:
            out[open_[lb]][1] += c * t
            continue
        for other in [k for k, i in open_.items() if bin((out[i][2] & z) ^ (out[i][3] & x)).count("1") & 1]:
            del open_[other]
        out.append([lb, c * t, x, z])
        open_[lb] = len(out) - 1
    return [(lb, a, 1.0) for lb, a, _, _ in out]


'''

for old, new in [
    ("VERSION: 2026-10-11.c39 -- candidate (from release 2026-10-11.2): items 69-70",
     "VERSION: 2026-10-11.c40 -- candidate (from c39, from release 2026-10-11.2): items 69-72"),
    ('VERSION = "2026-10-11.c39"  # candidate: release 2026-10-11.2 + items 69-70 (STEINER6 kept only where its ESP is '
     'higher; a size limit)',
     'VERSION = "2026-10-11.c40"  # candidate: release 2026-10-11.2 + items 69-72 (STEINER6 kept where its ESP is higher; '
     'a size limit; non-commuting, multi-step and QAOA-type inputs)'),
    ('''    10 s). Counts in `STEINER_STATS` ("too_large").
"""''', '''    10 s). Counts in `STEINER_STATS` ("too_large").
71. **ACCURACY: STEINER6 also for terms that do not all commute (prototype c40).** Item 68 required all terms to
    commute; with item 69 choosing by ESP, the construction is tried on every input it can read and kept only where it
    is better. Terms never pass an anticommuting earlier term, so the result is exact.
72. **ACCURACY: more inputs read as Pauli rotations (prototype c40).** Product formulas of any order and repetitions
    (LieTrotter, SuzukiTrotter, from their own `expand()`), sequences of PauliEvolutionGates and rotation gates (rz, rx,
    ry, rzz, rxx, ryy), composite instructions such as QAOAAnsatz (read through their definitions), a leading layer of
    single-qubit gates (as QAOA's Hadamards) and final measurements. Repeated terms are merged where exact (no
    anticommuting term in between): a commuting Hamiltonian over r Trotter steps becomes one step. The size limit of
    item 70 applies after merging.
"""'''),
    ("def _st_compile(qc, target, max_error, seed, call_limit, max_trials):", PARSE.lstrip("\n") +
     "def _st_compile(qc, target, max_error, seed, call_limit, max_trials):"),
    ('''    ev = _st_evolution(qc)
    if ev is None:
        STEINER_STATS["not_applicable"] += 1
        return None
    evo, meas, triples = ev
    if len(triples) > _ST_MAX_TERMS:  # item 70
        STEINER_STATS["too_large"] += 1
        return None
    if not triples or not _st_all_commute(triples):
        STEINER_STATS["not_commuting"] += 1
        return None
''', '''    ev = _st_parse(qc)  # item 72
    if ev is None:
        STEINER_STATS["not_applicable"] += 1
        return None
    prefix, meas, triples, extra_phase = ev
    triples = _st_merge(triples)
    if len(triples) > _ST_MAX_TERMS:  # item 70
        STEINER_STATS["too_large"] += 1
        return None
    if not _st_all_commute(triples):  # item 71: tried all the same; counted
        STEINER_STATS["not_commuting"] += 1
'''),
    ('''    phys = QuantumCircuit(N, qc.num_clbits)
    at = list(range(N))  # at[p]: the physical qubit whose state is at p (SWAPs move idle qubits' states too)
    for name, q, prm in m.ops:''', '''    phys = QuantumCircuit(N, qc.num_clbits)
    for op, q in prefix:  # item 72: the leading single-qubit gates, at the initial placement
        phys.append(op, [pi0[q]])
    at = list(range(N))  # at[p]: the physical qubit whose state is at p (SWAPs move idle qubits' states too)
    for name, q, prm in m.ops:'''),
    ('''    phys.global_phase = qc.global_phase + m.gphase''', '''    phys.global_phase = extra_phase + m.gphase'''),
]:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
