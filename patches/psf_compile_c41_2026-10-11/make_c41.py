"""Builds prototype 2026-10-11.c41 from prototype 2026-10-11.c40 by exact substitutions (changelog item 73: circuits
written as gates -- Clifford gates such as cx, cz, h, s, sx and rotations such as rz, rx, ry, p, t, any single-qubit
gate, and composite gates through their definitions -- are read as Pauli rotations by carrying the Clifford gates to
the end of the circuit; STEINER6 is then tried on them under item 69's choice).
    python make_c41.py <c40 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

GATES = '''

_ST_GATE_LIMIT = 200_000  # item 73: circuits with more instructions are not read
_ST_PAULI_1Q = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_ST_CONJ_CACHE = {}
_ST_CLIFF1Q = None


def _st_pmul(a, b):
    """Item 73: the product a.b of two Paulis written (x mask, z mask, k) for i^k X^x Z^z."""
    return a[0] ^ b[0], a[1] ^ b[1], (a[2] + b[2] + 2 * bin(a[1] & b[0]).count("1")) % 4


def _st_conj(op):
    """Item 73: for a gate on k <= 2 qubits whose conjugation maps Paulis to Paulis (a Clifford gate), the images
    G^dagger P G of X and Z on each of its qubits, as [(label on its k qubits, +1 or -1)] in the order X0, Z0, X1, Z1;
    None if it is not Clifford or has no matrix. From the gate's own matrix, cached by name and parameters."""
    try:
        key = (op.name, tuple(float(p) for p in op.params))
    except (TypeError, ValueError):
        return None
    if key in _ST_CONJ_CACHE:
        return _ST_CONJ_CACHE[key]
    from itertools import product
    from qiskit.quantum_info import Pauli
    k = op.num_qubits
    try:
        m = np.asarray(op.to_matrix(), dtype=complex)
    except Exception:  # noqa: BLE001 - no matrix
        _ST_CONJ_CACHE[key] = None
        return None
    basis = ["".join(t) for t in product("IXYZ", repeat=k)]
    mats = {lb: Pauli(lb).to_matrix() for lb in basis}
    out = []
    for j in range(k):
        for ch in "XZ":
            lb = ["I"] * k
            lb[k - 1 - j] = ch
            img = m.conj().T @ mats["".join(lb)] @ m
            hit = None
            for r in basis:
                c = np.trace(mats[r].conj().T @ img) / 2 ** k
                if abs(c) > 1e-6:
                    if hit is not None or abs(abs(c) - 1) > 1e-9 or abs(c.imag) > 1e-9:
                        hit = None
                        break
                    hit = (r, 1 if c.real > 0 else -1)
            if hit is None:
                _ST_CONJ_CACHE[key] = None
                return None
            out.append(hit)
    _ST_CONJ_CACHE[key] = out
    return out


def _st_cliff1q():
    """Item 73: {(image of X, image of Z) as (character, sign): [gate, ...]} for the 24 single-qubit Cliffords, from
    the shortest words in H and S."""
    global _ST_CLIFF1Q
    if _ST_CLIFF1Q is None:
        from qiskit.circuit.library import HGate, SGate
        table, frontier = {}, [[]]
        while frontier and len(table) < 24:
            nxt = []
            for word in frontier:
                m = np.eye(2, dtype=complex)
                for g in word:
                    m = g.to_matrix() @ m
                c = _st_conj(type("W", (), {"name": "w" + "".join(g.name for g in word), "params": [],
                                            "num_qubits": 1, "to_matrix": lambda self, m=m: m})())
                key = (c[0], c[1])
                if key not in table:
                    table[key] = word
                    nxt += [word + [HGate()], word + [SGate()]]
            frontier = nxt
        _ST_CLIFF1Q = table
    return _ST_CLIFF1Q


def _st_parse_gates(qc):
    """Item 73: (leading single-qubit gates [(operation, qubit)], final measurements [(qubit, clbit)], terms
    [(label, angle, 1.0)] in order, global phase, final single-qubit Cliffords [(operation, qubit)]) such that the
    circuit equals, up to a global phase, the leading gates, then the terms exp(-i angle P), then the final gates, then
    the measurements; None if it cannot be written so.

    Single-qubit gates on a qubit before its first multi-qubit gate are kept as the leading gates. After that, Clifford
    gates (any gate on one or two qubits that maps Paulis to Paulis, from its matrix, including rotations by multiples
    of pi/2) are carried to the end: a later rotation exp(-i a P) becomes exp(-i a C^dagger P C), where C is the product
    of the Clifford gates so far. Rotations rz, rx, ry, rzz, rxx, ryy, p, u1, t, tdg and Pauli evolutions give terms; any
    other single-qubit gate is written as rz.ry.rz; other gates are read through their definitions (three levels). The
    Clifford gates left at the end must act on each qubit alone (a product of single-qubit Cliffords)."""
    from qiskit.synthesis import OneQubitEulerDecomposer
    n = qc.num_qubits
    if n == 0 or len(qc.data) > _ST_GATE_LIMIT:
        return None
    try:
        phase = float(qc.global_phase)
    except (TypeError, ValueError):
        return None
    frame = [[(1 << q, 0, 0), (0, 1 << q, 0)] for q in range(n)]  # C^dagger X_q C, C^dagger Z_q C
    touched = [False] * n
    prefix, meas, terms = [], [], []
    count = [0]
    euler = OneQubitEulerDecomposer("ZYZ")

    def image(chars):  # chars: {qubit: "X" | "Y" | "Z"} -> C^dagger P C as (x, z, k)
        acc = (0, 0, 0)
        for q, ch in chars.items():
            if ch == "X":
                acc = _st_pmul(acc, frame[q][0])
            elif ch == "Z":
                acc = _st_pmul(acc, frame[q][1])
            elif ch == "Y":
                acc = _st_pmul(acc, _st_pmul((0, 0, 1), _st_pmul(frame[q][0], frame[q][1])))
        return acc

    def term(chars, angle):
        x, z, k = image(chars)
        if not x and not z:
            return None
        sign = (k - bin(x & z).count("1")) % 4
        if sign not in (0, 2):
            return None
        lab = ["I"] * n
        both = x | z
        while both:
            low = both & -both
            j = low.bit_length() - 1
            lab[n - 1 - j] = "Y" if x & low and z & low else ("X" if x & low else "Z")
            both ^= low
        terms.append(("".join(lab), angle if sign == 0 else -angle, 1.0))
        return True

    def clifford(op, qs):
        c = _st_conj(op)
        if c is None:
            return False
        new = []
        for j in range(len(qs)):
            row = []
            for r, sg in c[2 * j: 2 * j + 2]:
                chars = {qs[len(qs) - 1 - i]: ch for i, ch in enumerate(r) if ch != "I"}
                acc = image(chars)
                if sg < 0:
                    acc = (acc[0], acc[1], (acc[2] + 2) % 4)
                row.append(acc)
            new.append(row)
        for j, q in enumerate(qs):
            frame[q] = new[j]
        return True

    def one(op, qs, depth):
        nonlocal phase
        count[0] += 1
        if count[0] > _ST_GATE_LIMIT:
            return False
        name = op.name
        if len(qs) == 1 and not touched[qs[0]] and name != "PauliEvolution":
            try:
                op.to_matrix()
            except Exception:  # noqa: BLE001 - not a unitary gate
                return False
            prefix.append((op, qs[0]))
            return True
        if name == "PauliEvolution":
            got = _st_pauli_terms(op, qs, n)
            if got is None:
                return False
            for q in qs:
                touched[q] = True
            phase += got[1]
            for lb, a, _ in got[0]:
                chars = {n - 1 - i: ch for i, ch in enumerate(lb) if ch != "I"}
                if term(chars, a) is None:
                    return False
            return True
        for q in qs:
            touched[q] = True
        if name in ("p", "u1", "t", "tdg") and len(qs) == 1:
            try:
                lam = {"t": math.pi / 4, "tdg": -math.pi / 4}.get(name) or float(op.params[0])
            except (TypeError, ValueError):
                return False
            phase += lam / 2
            from qiskit.circuit.library import RZGate
            return one(RZGate(lam), qs, depth)
        if name in _ST_ROT and len(op.params) == 1:
            try:
                angle = float(op.params[0])
            except (TypeError, ValueError):
                return False
            if len(qs) == 1 and abs(math.remainder(angle, math.pi / 2)) < 1e-12:
                return clifford(op, qs)
            return term({q: ch for q, ch in zip(qs, _ST_ROT[name])}, angle / 2) is not None
        if len(qs) <= 2 and getattr(op, "blocks", None) is None and _st_conj(op) is not None:
            return clifford(op, qs)
        if len(qs) == 1:
            try:
                sub = euler(op.to_matrix())
            except Exception:  # noqa: BLE001 - not a unitary gate
                return False
            phase += float(sub.global_phase)
            return all(one(i.operation, qs, depth) for i in sub.data)
        definition = getattr(op, "definition", None)
        if depth >= 3 or definition is None or getattr(op, "blocks", None) is not None:
            return False
        try:
            phase += float(definition.global_phase)
        except (TypeError, ValueError):
            return False
        for i in definition.data:
            if i.clbits:
                return False
            if not one(i.operation, [qs[definition.find_bit(b).index] for b in i.qubits], depth + 1):
                return False
        return True

    for ins in qc.data:
        op = ins.operation
        name = op.name
        if name == "barrier":
            continue
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if name == "measure":
            meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))
            continue
        if meas or ins.clbits:
            return None
        if not one(op, qs, 0):
            return None
    if not terms:
        return None
    suffix = []
    table = None
    for q in range(n):
        if frame[q] == [(1 << q, 0, 0), (0, 1 << q, 0)]:
            continue
        key = []
        for x, z, k in frame[q]:
            if (x | z) & ~(1 << q) or not (x | z):
                return None  # the Clifford left at the end acts on more than this qubit
            ch = "Y" if x and z else ("X" if x else "Z")
            sg = (k - (1 if ch == "Y" else 0)) % 4
            if sg not in (0, 2):
                return None
            key.append((ch, 1 if sg == 0 else -1))
        table = table or _st_cliff1q()
        word = table.get(tuple(key))
        if word is None:
            return None
        suffix += [(g, q) for g in word]
    STEINER_STATS["gate_level"] += 1
    return prefix, meas, terms, phase, suffix


'''

for old, new in [
    ("VERSION: 2026-10-11.c40 -- candidate (from c39, from release 2026-10-11.2): items 69-72",
     "VERSION: 2026-10-11.c41 -- prototype (from c40, c39, release 2026-10-11.2): items 69-73"),
    ('VERSION = "2026-10-11.c40"  # candidate: release 2026-10-11.2 + items 69-72 (STEINER6 kept where its ESP is higher; '
     'a size limit; non-commuting, multi-step and QAOA-type inputs)',
     'VERSION = "2026-10-11.c41"  # prototype: release 2026-10-11.2 + items 69-73 (STEINER6 kept where its ESP is higher; '
     'a size limit; non-commuting, multi-step, QAOA-type and gate-level inputs)'),
    ('''    item 70 applies after merging.
"""''', '''    item 70 applies after merging.
73. **ACCURACY: circuits written as gates are read too (prototype c41).** A circuit that item 72 cannot read is read
    gate by gate: single-qubit gates before a qubit's first multi-qubit gate stay in front; Clifford gates after that
    (cx, cz, swap, ecr, h, s, sx, rotations by multiples of pi/2: any gate on one or two qubits that maps Paulis to
    Paulis, from its matrix) are carried to the end, turning each later rotation exp(-i a P) into exp(-i a C^dagger P C);
    rotations, Pauli evolutions, p, t and any other single-qubit gate (as rz.ry.rz) give the terms; other gates are read
    through their definitions. It applies when the Clifford gates left at the end act on each qubit alone; they are
    then placed after the terms at the qubits' final places. The output equals the input up to a global phase (the
    Clifford gates' phases are not followed). QASM files of QAOA and Trotter circuits (cx-rz-cx) are of this kind.
"""'''),
    ('''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0}  # changelog items 68-70''',
     '''STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0}  # changelog items 68-73'''),
    ("def _st_compile(qc, target, max_error, seed, call_limit, max_trials):", GATES.lstrip("\n") +
     "def _st_compile(qc, target, max_error, seed, call_limit, max_trials):"),
    ('''    ev = _st_parse(qc)  # item 72
    if ev is None:
        STEINER_STATS["not_applicable"] += 1
        return None
    prefix, meas, triples, extra_phase = ev
''', '''    ev = _st_parse(qc)  # item 72
    ev = ev + ([],) if ev is not None else _st_parse_gates(qc)  # item 73
    if ev is None:
        STEINER_STATS["not_applicable"] += 1
        return None
    prefix, meas, triples, extra_phase, suffix = ev
'''),
    ('''    phys.global_phase = extra_phase + m.gphase
    for q, cb in meas:''', '''    phys.global_phase = extra_phase + m.gphase
    for op, q in suffix:  # item 73: the Clifford gates left at the end, at the final places
        phys.append(op, [m.pi[q]])
    for q, cb in meas:'''),
]:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
