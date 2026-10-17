"""PSF-Zero compiler -- VERSION: 2026-10-16.2 -- release, adopted on 2026-10-17 from prototype 2026-10-16.c66
(previous release: 2026-10-16.1, kept in patches/psf_compile_release_2026-10-16.1/).

Two entry points: `compile_for_hardware(qc, target=...)` and `compile_template(qc, target)`. A circuit whose
two-qubit interaction graph embeds in the device's working couplers is built by formula (items 87-101), placed by the
first VF2 match with no score; every other circuit, and every call without the device's Target, is refused with
`OutOfScopeError`: nothing is handed to a search (item 98). Item 96's embedding test decides by formula before VF2
(item 102). In a template on a fractional-gate target, item 88's frame is applied by segments, solved over GF(2)
(item 103): the template's shape never changes with the values, without option A's extra pulses. 2026-10-16.2 is
2026-10-16.1 plus item 103; validated in C66-VAL (Addendum 513): its default calls are 2026-10-16.1's bit for bit.

The changelog of items 1-103 is docs/psf_compile_CHANGELOG.md. The functions of earlier releases that the default
calls never used (`compile()`, `prune_coupling_map`, `edge_errors_from_target`, the cost models, ...) are not in this
release; they remain in patches/psf_compile_release_2026-10-14.1/psf_compile.py."""

import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager, generate_preset_pass_manager
from qiskit.transpiler.exceptions import TranspilerError


VERSION = "2026-10-16.2"  # release, adopted on 2026-10-17 from prototype 2026-10-16.c66 (item 103 on release 2026-10-16.1)


def qubit_errors_from_target(target, gate_name: str = "sx") -> dict:
    """`{q: error}` of the single-qubit `gate_name` (default `sx`, the only
    non-virtual single-qubit gate on IBM devices). Qubits without an error
    value are left out. For `layout_qubit_errors` (changelog item 19)."""
    out = {}
    if gate_name not in target.operation_names:
        return out
    for qargs, props in target[gate_name].items():
        if qargs is None or props is None or props.error is None:
            continue
        out[qargs[0]] = props.error
    return out


WARN_WITHOUT_TARGET = True  # item 58: warn once per process when compile_for_hardware has no device Target


def readout_errors_from_target(target) -> dict:
    """Item 58: `{q: error}` of `measure`, for the qubits that have an error value."""
    out = {}
    if "measure" not in target.operation_names:
        return out
    for qargs, props in target["measure"].items():
        if qargs is None or props is None or props.error is None:
            continue
        out[qargs[0]] = props.error
    return out


def _count_2q(qc):
    """Item 50: the number of instructions on two qubits (barriers aside)."""
    return sum(1 for ins in qc.data if len(ins.qubits) == 2 and ins.operation.name != "barrier")


import math  # noqa: E402


def _st_term_masks(label):
    """(x mask, z mask, support) of a label; character j from the right acts on logical qubit j."""
    x = z = 0
    sup = []
    for j, ch in enumerate(reversed(label)):
        if ch == "I":
            continue
        sup.append(j)
        if ch in "XY":
            x |= 1 << j
        if ch in "ZY":
            z |= 1 << j
    return x, z, tuple(sup)


# ------------------------------------------------------------------ item 83: time-aware order of STEINER6's blocks

_ST_DUR = {"cx": 1.0, "swap": 3.0, "rz": 0.0}  # item 83: durations in CX units for the ordering (rz is virtual)
_ST_DUR_1Q = 0.3


STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0, "too_large": 0, "chosen": 0,
                 "kept_usual": 0, "gate_level": 0, "permuted": 0, "mirror_tried": 0,
                 "mirror_kept": 0, "reordered": 0, "reorder_kept_order": 0,
                 "diag_dropped": 0}  # changelog items 68-76, 82, 83, 89


def _st_cost(circ, target):
    """Item 69: the summed -ln(1 - error) of every operation of `circ` on the device (-ln ESP); inf if one is not
    offered there or has error >= 1. Barriers and delays count 0, and so does control flow (no error reported)."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    total = 0.0
    for ins in circ.data:
        name = ins.operation.name
        if name in ("barrier", "delay"):
            continue
        if name not in target.operation_names:
            return math.inf
        entries = target[name]
        qargs = tuple(pos[q] for q in ins.qubits)
        if qargs in entries:
            props = entries[qargs]
        elif None in entries:
            continue
        else:
            return math.inf
        e = props.error if props is not None and props.error is not None else 0.0
        if e >= 1.0:
            return math.inf
        total += -math.log1p(-max(e, 0.0))
    return total


# ---------------------------------------------------------------------------------------------------------------------
# Prototype c49: item 84 (choices by duration, not by the device's error table) and item 85 (failed qubits cut first)
# ---------------------------------------------------------------------------------------------------------------------

SELECT_STATS = {"duration": 0, "cx_units": 0, "tie_2q": 0, "tie_kept": 0}


def _duration_s(circ, target):
    """Item 84: the as-soon-as-possible makespan of `circ` in seconds on the durations `target` reports (each
    operation starts when all its qubits are free; barriers take no time), or None when an operation has no reported
    duration on its qubits (then both circuits are compared in CX units)."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    clock = {}
    names = set(target.operation_names)
    for ins in circ.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        if name not in names:
            return None
        qargs = tuple(pos[q] for q in ins.qubits)
        entries = target[name]
        props = entries.get(qargs) if qargs in entries else entries.get(None)
        d = getattr(props, "duration", None) if props is not None else None
        if d is None:
            if name == "rz":  # virtual on every IBM device
                d = 0.0
            else:
                return None
        s = max((clock.get(x, 0.0) for x in qargs), default=0.0)
        for x in qargs:
            clock[x] = s + d
    return max(clock.values()) if clock else 0.0


def _duration_cx(circ):
    """Item 84: the same makespan in item 83's CX units (CX 1, SWAP 3, rz 0, any other single-qubit gate 0.3, any
    other two-qubit gate 1), for a Target without durations."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    clock = {}
    for ins in circ.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        qargs = tuple(pos[q] for q in ins.qubits)
        if name in _ST_DUR:
            d = _ST_DUR[name]
        else:
            d = 1.0 if len(qargs) >= 2 else _ST_DUR_1Q
        s = max((clock.get(x, 0.0) for x in qargs), default=0.0)
        for x in qargs:
            clock[x] = s + d
    return max(clock.values()) if clock else 0.0


def _better(a, b, target):
    """Item 84: True if circuit `a` is to be returned instead of `b` (both compiled for `target`).

    "duration" (default): a circuit with an operation the device does not offer, or one reported at error >= 1, loses
    (item 69's safety, unchanged). Otherwise the shorter as-soon-as-possible schedule on the device's reported gate
    durations wins; on equal durations (within 1e-12 s), fewer two-qubit gates; on a full tie, `b` (the caller's
    default) is kept. No error value of the device is used beyond that. "esp" restores c48's choice (lower summed
    -ln(1 - error))."""
    ca, cb = _st_cost(a, target), _st_cost(b, target)
    fa, fb = math.isinf(ca), math.isinf(cb)
    if fa != fb:
        return fb
    da, db = _duration_s(a, target), _duration_s(b, target)
    if da is None or db is None:
        da, db, tol = _duration_cx(a), _duration_cx(b), 1e-9
        SELECT_STATS["cx_units"] += 1
    else:
        tol = 1e-12
        SELECT_STATS["duration"] += 1
    if abs(da - db) > tol:
        return da < db
    na, nb = _count_2q(a), _count_2q(b)
    if na != nb:
        SELECT_STATS["tie_2q"] += 1
        return na < nb
    SELECT_STATS["tie_kept"] += 1
    return False


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


_ST_GATE_LIMIT = 200_000  # item 73: circuits with more instructions are not read
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
    if meas and {q for q, _ in meas} == set(range(n)):  # item 89
        dropped = False
        for q in range(n):
            xi, zi = frame[q]
            bz, bx = zi[0] | zi[1], xi[0] | xi[1]
            if not bz or bz & (bz - 1):
                continue  # Z's image is not a single-qubit Pauli: no single-qubit L exists
            w = bz.bit_length() - 1
            if bx and not bx & (bx - 1) and bx == bz:
                continue  # already a single-qubit Clifford on that wire
            # an X image on wire w that anticommutes with Z's image: Z_w if Z's image has an X part, else X_w
            frame[q] = [(0, 1 << w, 0) if zi[0] & bz else (1 << w, 0, 0), zi]
            dropped = True
        if dropped:
            STEINER_STATS["diag_dropped"] += 1
    suffix = []
    table = None
    perm = list(range(n))  # item 76: perm[q] is the wire whose state logical qubit q holds at the end
    seen = set()
    for q in range(n):
        if frame[q] == [(1 << q, 0, 0), (0, 1 << q, 0)]:
            seen.add(q)
            continue
        key = []
        wire = None
        for x, z, k in frame[q]:
            both = x | z
            if not both or both & (both - 1):
                return None  # the Clifford left at the end acts on more than one qubit here
            if wire is None:
                wire = both.bit_length() - 1
            elif both != 1 << wire:
                return None
            ch = "Y" if x and z else ("X" if x else "Z")
            sg = (k - (1 if ch == "Y" else 0)) % 4
            if sg not in (0, 2):
                return None
            key.append((ch, 1 if sg == 0 else -1))
        if wire in seen:
            return None
        seen.add(wire)
        perm[q] = wire
        table = table or _st_cliff1q()
        word = table.get(tuple(key))
        if word is None:
            return None
        suffix += [(g, wire) for g in word]
    if len(seen) != n:
        return None
    STEINER_STATS["gate_level"] += 1
    if perm != list(range(n)):
        STEINER_STATS["permuted"] += 1
    return prefix, meas, terms, phase, suffix, perm


def _failed_elements(target, max_error: float, gate_names=("cz", "ecr", "cx"), qubit_gate: str = "sx"):
    """(set of directed failed edges, set of failed qubits) as reported by `target` (changelog item 31)."""
    name = next((g for g in gate_names if g in target.operation_names), None)
    edges = set()
    if name is not None:
        for qargs, props in target[name].items():
            if qargs is not None and props is not None and props.error is not None and props.error >= max_error:
                edges.add(tuple(qargs))
    qubits = {q for q, e in qubit_errors_from_target(target, qubit_gate).items() if e >= max_error}
    qubits |= {q for q, e in readout_errors_from_target(target).items() if e >= max_error}  # item 58
    return edges, qubits


_SYMMETRIC_2Q = frozenset(("cz", "swap", "iswap", "rzz", "rxx", "ryy", "cp", "xx_plus_yy"))


def _uses_failed(circ, edges, qubits) -> bool:
    """Item 31 (direction-aware from item 41): a gate on a failed qubit, a two-qubit gate in a direction reported
    failed, or a symmetric two-qubit gate on a coupler failed in either direction."""
    pos = {q: i for i, q in enumerate(circ.qubits)}  # item 62: the same indices as find_bit, without its cost
    for inst in circ.data:
        idx = tuple(pos[q] for q in inst.qubits)
        if any(i in qubits for i in idx):
            return True
        if len(idx) == 2 and (idx in edges or (inst.operation.name in _SYMMETRIC_2Q and idx[::-1] in edges)):
            return True
    return False


# ---------------------------------------------------------------------------------------------------------------------
# Prototype c50: item 86 (the device's native fractional gates: rzz and rx); c51: item 86 revised (every rzz folded)
# ---------------------------------------------------------------------------------------------------------------------

FRAC_STATS = {"tried": 0, "kept": 0, "longer": 0, "not_isa": 0, "not_exact": 0, "unverified": 0, "error": 0,
              "folded": 0, "fold_failed": 0}


def _errorless(target):
    """Item 86: a copy of `target` with every reported error removed (durations kept), so that Qiskit's synthesis
    passes cannot choose by the device's error table."""
    import copy
    from qiskit.transpiler import InstructionProperties
    t = copy.deepcopy(target)
    for name in t.operation_names:
        for qargs, p in list(t[name].items()):
            if qargs is not None and p is not None and p.error is not None:
                t.update_instruction_properties(name, qargs, InstructionProperties(duration=p.duration, error=None))
    return t


def _strip_global_phase_gates(new):
    phase, kept = 0.0, []
    for ins in new.data:
        if ins.operation.name == "global_phase":
            phase += float(ins.operation.params[0])
        else:
            kept.append(ins)
    if len(kept) != len(new.data):
        new.data = kept
        new.global_phase += phase
    return new


def _rzz_in_range(circ):
    return all(0 < float(i.operation.params[0]) <= math.pi / 2 + 1e-12 for i in circ.data if i.operation.name == "rzz")


def _fold(out, target):
    """Item 86 revised (c51): `out` with every rzz angle folded into IBM's executable range 0 < theta <= pi/2
    (qiskit-ibm-runtime's FoldRzzAngle, then single-qubit runs merged on an error-free copy of the Target). Qiskit's
    own stages inside the compiler emit rzz when the Target offers it, with any angle; only a backend object of
    qiskit-ibm-runtime folds them by itself. Raises TranspilerError if an angle stays outside the range."""
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition
    from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
    new = PassManager([FoldRzzAngle(), Optimize1qGatesDecomposition(target=_errorless(target))]).run(out)
    new = _strip_global_phase_gates(new)
    new._layout = getattr(out, "_layout", None)
    if not _rzz_in_range(new):
        FRAC_STATS["fold_failed"] += 1
        raise TranspilerError("an rzz angle is outside 0 < theta <= pi/2 after folding (changelog item 86)")
    FRAC_STATS["folded"] += 1
    return new


# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c51 (2026-10-14): item 87, a chain Hamiltonian built by formula as a brickwork (N1);
# item 88, on a Target with rzz and rx, its terms emitted as rzz / rx with the signs carried in a Pauli X frame
# ---------------------------------------------------------------------------------------------------------------------


CHAIN_STATS = {"tried": 0, "not_chain": 0, "not_cz": 0, "no_path": 0, "snake": 0, "vf2_path": 0,
               "common_length": 0, "used": 0, "failed_check": 0, "kept": 0, "kept_default": 0,
               "frame": 0, "frame_x_end": 0, "frame_folded": 0}
_CHAIN_VF2_CALLS = 5_000_000


def _chain_support(lb):
    n = len(lb)
    return [(n - 1 - i, ch) for i, ch in enumerate(lb) if ch != "I"]


def _chain_commute(a, b):
    sa, sb = dict(_chain_support(a)), dict(_chain_support(b))
    return sum(1 for q in set(sa) & set(sb) if sa[q] != sb[q]) % 2 == 0


def _chain_read(qc):
    """Item 87: (prefix, meas, terms [(label, angle)], phase, suffix, perm, path) when every term (read by items
    72-73, merged where exact by `_st_merge`) acts on one or two qubits and the two-qubit supports form one path
    through all the circuit's qubits; else None."""
    ev = _st_parse(qc)
    ev = ev + ([], None) if ev is not None else _st_parse_gates(qc)
    if ev is None:
        return None
    prefix, meas, triples, phase, suffix, perm = ev
    terms = [(lb, a * t) for lb, a, t in _st_merge(triples)]
    n = qc.num_qubits
    adj = {q: set() for q in range(n)}
    for lb, _ in terms:
        sup = [q for q, _ in _chain_support(lb)]
        if len(sup) > 2:
            return None
        if len(sup) == 2:
            adj[sup[0]].add(sup[1])
            adj[sup[1]].add(sup[0])
    if any(len(v) > 2 for v in adj.values()):
        return None
    ends = [q for q in range(n) if len(adj[q]) == 1]
    if n < 2 or len(ends) != 2:
        return None
    path, prev, cur = [min(ends)], None, min(ends)
    while True:
        nxt = [v for v in adj[cur] if v != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)
    if len(path) != n:
        return None
    return prefix, meas, terms, phase, suffix, perm or list(range(n)), path


def _chain_lattice(target, max_error):
    """Item 87: (K, its adjacency, the adjacency restricted to couplers of the device's most common two-qubit gate
    length, the snake path) -- W5's cut: qubits with sx or readout error >= max_error and couplers with error >=
    max_error removed, the largest connected component kept (ties: the smallest qubit)."""
    N = target.num_qubits
    err, dur = {}, {}
    for qa, pr in target["cz"].items():
        if qa is None:
            continue
        k = (min(qa), max(qa))
        err[k] = max(err.get(k, 0.0), pr.error if pr is not None and pr.error is not None else 0.0)
        dur[k] = max(dur.get(k, 0.0), pr.duration if pr is not None and pr.duration is not None else 0.0)
    adj = {q: set() for q in range(N)}
    for a, b in err:
        adj[a].add(b)
        adj[b].add(a)
    bad = set()
    for name in ("sx", "measure"):
        if name in target.operation_names:
            for qa, pr in target[name].items():
                if qa is not None and pr is not None and pr.error is not None and pr.error >= max_error:
                    bad.add(qa[0])
    gadj = {q: set() for q in range(N) if q not in bad}
    for (a, b), e in err.items():
        if e < max_error and a not in bad and b not in bad:
            gadj[a].add(b)
            gadj[b].add(a)
    seen, comps = set(), []
    for q in sorted(gadj):
        if q in seen:
            continue
        comp, fr = {q}, [q]
        while fr:
            u = fr.pop()
            for v in gadj[u]:
                if v not in comp:
                    comp.add(v)
                    fr.append(v)
        seen |= comp
        comps.append(comp)
    if not comps:
        return None
    K = max(comps, key=lambda c: (len(c), -min(c)))
    kadj = {q: {v for v in gadj[q] if v in K} for q in K}
    lens = {}
    for a in K:
        for b in kadj[a]:
            if a < b:
                k = round(dur[(a, b)] * 1e12)
                lens[k] = lens.get(k, 0) + 1
    modal = max(lens, key=lambda x: (lens[x], -x)) if lens else None
    fadj = {q: {v for v in kadj[q] if round(dur[(min(q, v), max(q, v))] * 1e12) == modal} for q in K}
    return K, kadj, fadj, _chain_snake(N, adj)


def _chain_snake(N, adj):
    """Item 87: Heron's snake (W5): rows of consecutive qubits joined by couplers, row 0 left to right, the bridge
    at the far end down, the next row back, and so on. A lattice that is not read as rows gives a shorter path."""
    rows, cur = [], [0]
    for q in range(1, N):
        if q in adj[q - 1]:
            cur.append(q)
        else:
            rows.append(cur)
            cur = [q]
    rows.append(cur)
    rows = [r for r in rows if len(r) > 2]
    inrow = {q: (i, j) for i, r in enumerate(rows) for j, q in enumerate(r)}
    path, col = [], 0
    for i, r in enumerate(rows):
        leftward = i % 2 == 1
        if i + 1 < len(rows):
            br = [(inrow[a][1], b) for b in range(N) if b not in inrow for a in adj[b]
                  if a in inrow and inrow[a][0] == i and any(c in inrow and inrow[c][0] == i + 1 for c in adj[b])]
            if not br:
                return path + list(r)
            exit_col, bq = (min(br) if leftward else max(br))
        else:
            exit_col, bq = (0 if leftward else len(r) - 1), None
        step = -1 if leftward else 1
        start = col if i else 0
        path += [r[j] for j in range(start, exit_col + step, step)]
        if bq is not None:
            path.append(bq)
            below = next(c for c in adj[bq] if c in inrow and inrow[c][0] == i + 1)
            col = inrow[below][1]
    return path


def _chain_place(n, K, kadj, snake):
    """Item 87: n physical qubits forming a path in K: the longest surviving snake piece, centred; else the first
    path rustworkx's VF2 finds with nodes in index order (no score). (path, how) or (None, None)."""
    pieces, cur = [], []
    for q in snake:
        if q in K and (not cur or q in kadj[cur[-1]]):
            cur.append(q)
        else:
            if cur:
                pieces.append(cur)
            cur = [q] if q in K else []
    if cur:
        pieces.append(cur)
    if pieces:
        best = max(pieces, key=lambda p: (len(p), -snake.index(p[0])))
        if len(best) >= n:
            s = (len(best) - n) // 2
            return best[s:s + n], "snake"
    import rustworkx as rx
    kn = sorted(K)
    ki = {q: i for i, q in enumerate(kn)}
    G = rx.PyGraph()
    G.add_nodes_from(kn)
    G.add_edges_from_no_data([(ki[a], ki[b]) for a in kn for b in kadj[a] if a < b])
    P = rx.PyGraph()
    P.add_nodes_from(range(n))
    P.add_edges_from_no_data([(i, i + 1) for i in range(n - 1)])
    m = next(rx.vf2_mapping(G, P, subgraph=True, induced=False, id_order=True, call_limit=_CHAIN_VF2_CALLS), None)
    if m is None:
        return None, None
    inv = {h: kn[g] for g, h in m.items()}
    return [inv[i] for i in range(n)], "vf2_path"


def _chain_rotation(qc, lb, a, frame=False):
    """Item 87: exp(-i a P) on the circuit's own qubits; XX, YY, ZZ as Qiskit's rxx, ryy, rzz. Item 88 (`frame`):
    every two-qubit term as rzz between single-qubit basis changes (X: h; Y: rx(pi/2) before, rx(-pi/2) after), so
    that the device's one-pulse rzz carries it; single-qubit terms as rx / ry / rz."""
    sup = _chain_support(lb)
    if len(sup) == 1:
        q, ch = sup[0]
        getattr(qc, {"X": "rx", "Y": "ry", "Z": "rz"}[ch])(2 * a, q)
        return
    (q1, c1), (q2, c2) = sup
    if c1 == c2 and not frame:
        getattr(qc, {"X": "rxx", "Y": "ryy", "Z": "rzz"}[c1])(2 * a, q1, q2)
        return
    for q, ch in ((q1, c1), (q2, c2)):
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            qc.rx(math.pi / 2, q)
    if frame:
        qc.rzz(2 * a, q1, q2)
    else:
        qc.cx(q1, q2)
        qc.rz(2 * a, q2)
        qc.cx(q1, q2)
    for q, ch in ((q1, c1), (q2, c2)):
        if ch == "X":
            qc.h(q)
        elif ch == "Y":
            qc.rx(-math.pi / 2, q)


def _chain_frame(log):
    """Item 88: `log` (single-qubit gates, rzz with any angle, measurements) rewritten so that every rzz angle lies in
    IBM's executable range 0 < theta <= pi/2, with the same action (unitary, and outcome distribution when measured).

    Each angle is reduced to (-pi, pi]; an angle of size above pi/2 is moved by pi with Z x Z (rz(pi) on both qubits,
    the phase kept). A negative angle (after the frame's sign, below) is made positive by one x on the rzz's first
    qubit before it, merged into the single-qubit run before it; the x that would follow it is not placed but kept in
    a Pauli X frame: a single-qubit gate G on a qubit in the frame is placed as X G X (merged into one pulse with its
    run), an rzz with exactly one of its qubits in the frame has its sign flipped, and the frame is undone inside the
    circuit -- one x before the qubit's measurement, or at the end of an unmeasured qubit, merged into the last
    single-qubit gate where there is one. No measured bit is flipped afterwards: the circuit itself outputs the
    input's distribution, and acts as the input's unitary when it is not measured."""
    n = log.num_qubits
    out = QuantumCircuit(*log.qregs, *log.cregs, global_phase=log.global_phase)
    f = [0] * n
    for ins in log.data:
        nm = ins.operation.name
        qs = [log.find_bit(q).index for q in ins.qubits]
        if nm == "barrier":
            out.append(ins.operation, qs)
            continue
        if nm == "measure":
            q = qs[0]
            if f[q]:
                out.x(q)
                f[q] = 0
            out.measure(q, log.find_bit(ins.clbits[0]).index)
            continue
        if nm == "rzz":
            a, b = qs
            th = float(ins.operation.params[0])
            th = math.pi - (math.pi - th) % (2 * math.pi)  # (-pi, pi]
            if abs(th) > math.pi / 2:  # rzz(th) = rzz(th - sign(th) pi) (rz(pi) x rz(pi)) e^{i pi/2} for th > 0
                out.rz(math.pi, a)
                out.rz(math.pi, b)
                out.global_phase += math.copysign(math.pi / 2, th) + math.pi * (f[a] + f[b])  # X rz(pi) X = -rz(pi)
                th -= math.copysign(math.pi, th)
                CHAIN_STATS["frame_folded"] += 1
            if abs(th) < 1e-12:
                continue
            eff = -th if f[a] != f[b] else th
            if eff < 0:
                out.x(a)
                f[a] ^= 1
                eff = -eff
            out.rzz(eff, a, b)
            continue
        if len(qs) != 1:
            raise TranspilerError(f"item 88: unexpected {nm} on {len(qs)} qubits")
        q = qs[0]
        if f[q]:
            out.x(q)
        out.append(ins.operation, [q])
        if f[q]:
            out.x(q)
    for q in range(n):
        if f[q]:
            out.x(q)
            CHAIN_STATS["frame_x_end"] += 1
    return out


def _chain_brickwork(qc, target, max_error, seed):
    """Item 87 (workplace proposal, N1): a Pauli evolution whose two-qubit terms form one chain, compiled by formula.
    The chain goes on a path of the device (`_chain_place`: first on couplers of the device's most common two-qubit
    gate length, then on all of the cut device); every term gets a level, 1 + the largest level of an earlier term
    that shares a qubit and does not commute with it; terms are emitted by level, two-qubit terms on even edges of the
    chain, then odd edges, then single-qubit terms (every non-commuting pair keeps its order: exact). Without rzz in
    the Target, Qiskit's level-3 preset then translates on that fixed layout without routing, given the coupling map
    and basis only (no error value is read). Item 88: with rzz and rx in the Target (`CHAIN_FRAME`), the terms are
    emitted as rzz between basis changes, diagonal gates directly before measurements are dropped (Qiskit's
    RemoveDiagonalGatesBeforeMeasure), the signs are carried in a Pauli X frame (`_chain_frame`), Qiskit's level-0
    preset places it on the fixed layout, and every single-qubit run is merged into rz rx rz (one pulse); an rzz angle
    outside 0 < theta <= pi/2 would go through item 86's `_fold`. None when the input is not such a chain, the
    device's two-qubit gate is not cz, no path of n qubits exists, or the output would use a failed element."""
    CHAIN_STATS["tried"] += 1
    if "cz" not in target.operation_names:
        CHAIN_STATS["not_cz"] += 1
        return None
    got = _chain_read(qc)
    if got is None:
        CHAIN_STATS["not_chain"] += 1
        return None
    prefix, meas, terms, phase, suffix, perm, path = got
    n, N = qc.num_qubits, target.num_qubits
    lat = _chain_lattice(target, max_error)
    if lat is None:
        CHAIN_STATS["no_path"] += 1
        return None
    K, kadj, fadj, snake = lat
    phys, how = _chain_place(n, K, fadj, snake)
    if phys is not None:
        CHAIN_STATS["common_length"] += 1
    else:
        phys, how = _chain_place(n, K, kadj, snake)
    if phys is None:
        CHAIN_STATS["no_path"] += 1
        return None
    CHAIN_STATS[how] += 1
    rank = {path[i]: i for i in range(n)}
    level, last = [], {}
    for k, (lb, _) in enumerate(terms):
        lv = 0
        for q, _ in _chain_support(lb):
            for j, lb2 in last.get(q, ()):
                if not _chain_commute(lb, lb2):
                    lv = max(lv, level[j] + 1)
        level.append(lv)
        for q, _ in _chain_support(lb):
            last.setdefault(q, []).append((k, lb))

    def key(k):
        sup = [q for q, _ in _chain_support(terms[k][0])]
        if len(sup) == 2:
            lo = min(rank[q] for q in sup)
            return (level[k], 0, lo % 2, lo, k)
        return (level[k], 1, 0, rank[sup[0]], k)
    frame = "rzz" in target.operation_names and "rx" in target.operation_names
    log = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=phase)
    for op, q in prefix:
        log.append(op, [q])
    for k in sorted(range(len(terms)), key=key):
        _chain_rotation(log, *terms[k], frame=frame)
    for op, q in suffix:
        log.append(op, [q])
    for q, cb in meas:
        log.measure(perm[q], cb)  # item 76: logical q's state is on wire perm[q]
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for a in sorted(K):
        for b in sorted(kadj[a]):
            cm.add_edge(a, b)
    layout = [phys[rank[q]] for q in range(n)]
    if _T100_HOOK[0] is not None:  # item 100: compile_template reads the structure here (nothing is compiled)
        _T100_HOOK[0](dict(prefix=prefix, terms=terms, order=sorted(range(len(terms)), key=key), suffix=suffix,
                           meas=meas, perm=perm, layout=layout, frame=frame, phase=phase, n=n,
                           extra=locals().get("path", locals().get("edges"))))
    if frame:
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveDiagonalGatesBeforeMeasure
        log = PassManager([RemoveDiagonalGatesBeforeMeasure()]).run(log)
        low = _chain_frame(log)
        pm = generate_preset_pass_manager(0, coupling_map=cm, basis_gates=["rzz", "rx", "rz", "sx", "x", "cz"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        placed = pm.run(low)
        out = PassManager([Optimize1qGatesDecomposition(basis=["rz", "rx"])]).run(placed)
        out._layout = placed._layout
        if not _rzz_in_range(out):
            out = _fold(out, target)
        CHAIN_STATS["frame"] += 1
    else:  # item 101: PSF-Zero emits the CZ path itself
        out = _cz101_emit(qc, target, dict(prefix=prefix, terms=terms, order=sorted(range(len(terms)), key=key),
                                           suffix=suffix, meas=meas, perm=perm, layout=layout, n=n))
    edges, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges, bad):
        CHAIN_STATS["failed_check"] += 1
        return None
    CHAIN_STATS["used"] += 1
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on c53 (2026-10-14): item 91, a Pauli evolution whose interaction graph embeds in the device,
# built by formula: first VF2 embedding (no score), edges coloured into at most 3 matchings (Koenig), one layer per
# colour
# ---------------------------------------------------------------------------------------------------------------------

GRAPH_STATS = {"tried": 0, "not_read": 0, "degree": 0, "not_cz": 0, "no_embedding": 0, "common_length": 0,
               "used": 0, "colours": {}, "failed_check": 0, "frame": 0}
_GRAPH_VF2_CALLS = 2_000_000


def _graph_read(qc):
    """Item 91: (prefix, meas, terms [(label, angle)], phase, suffix, perm, edges) when every term (items 72-73's
    reader, merged where exact) acts on one or two qubits; edges: the sorted two-qubit supports. Else None."""
    ev = _st_parse(qc)
    ev = ev + ([], None) if ev is not None else _st_parse_gates(qc)
    if ev is None:
        return None
    prefix, meas, triples, phase, suffix, perm = ev
    terms = [(lb, a * t) for lb, a, t in _st_merge(triples)]
    edges = set()
    for lb, _ in terms:
        sup = sorted(q for q, _ in _chain_support(lb))
        if len(sup) > 2:
            return None
        if len(sup) == 2:
            edges.add((sup[0], sup[1]))
    if not edges:
        return None
    return prefix, meas, terms, phase, suffix, perm or list(range(qc.num_qubits)), sorted(edges)


def _graph_colours(edges):
    """Item 91: a proper edge colouring of a bipartite graph with max-degree colours (Koenig), deterministic: edges in
    sorted order; an edge (u, v) takes the smallest colour a free at u; if a is used at v, the a/b path from v (b the
    smallest colour free at v) is swapped first. {edge: colour} or None when the graph is not bipartite."""
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    D = max(deg.values())
    at = {x: {} for x in deg}  # node -> {colour: neighbour}
    for u, v in edges:
        a = min(c for c in range(D) if c not in at[u])
        if a in at[v]:
            b = min(c for c in range(D) if c not in at[v])
            path, x, c, seen = [], v, a, {v}
            while c in at[x]:
                y = at[x][c]
                path.append((x, y, c))
                if y in seen or y == u:
                    return None  # an odd cycle: not bipartite
                seen.add(y)
                x, c = y, (b if c == a else a)
            for x, y, c in path:
                del at[x][c]
                del at[y][c]
            for x, y, c in path:
                d = b if c == a else a
                at[x][d] = y
                at[y][d] = x
        at[u][a] = v
        at[v][a] = u
    col = {}
    for x, m in at.items():
        for c, y in m.items():
            col[(min(x, y), max(x, y))] = c
    return col


def _graph_place(n, edges, K, kadj):
    """Item 91: the first VF2 embedding (rustworkx, nodes in index order, no score) of the interaction graph into the
    cut device K: {logical qubit: physical qubit} or None."""
    import rustworkx as rx
    kn = sorted(K)
    ki = {q: i for i, q in enumerate(kn)}
    G = rx.PyGraph()
    G.add_nodes_from(kn)
    G.add_edges_from_no_data([(ki[a], ki[b]) for a in kn for b in kadj[a] if a < b])
    P = rx.PyGraph()
    P.add_nodes_from(range(n))
    P.add_edges_from_no_data(edges)
    if n > len(kn):
        return None
    m = next(rx.vf2_mapping(G, P, subgraph=True, induced=False, id_order=True, call_limit=_GRAPH_VF2_CALLS), None)
    if m is None:
        return None
    return {h: kn[g] for g, h in m.items()}


def _graph_brickwork(qc, target, max_error, seed):
    """Item 91 (workplace proposal on c53): a Pauli evolution whose terms act on one or two qubits and whose
    interaction graph embeds in the cut device (max degree <= 3, bipartite), compiled by formula. Placement: the
    first VF2 embedding, tried first on the couplers of the device's most common two-qubit gate length, then on all
    of the cut device (item 87's lattice). Order: every term at level 1 + the largest level of an earlier term that
    shares a qubit and does not commute with it (exact); within a level the two-qubit terms by colour of a proper
    edge colouring with max-degree colours (each colour a matching: one layer), then the single-qubit terms.
    Emission as item 87 (Qiskit level 3 on the fixed layout, CZ) or, with rzz and rx in the Target, item 88 (rzz,
    rx, Pauli frame undone inside the circuit). None when not applicable."""
    GRAPH_STATS["tried"] += 1
    if "cz" not in target.operation_names:
        GRAPH_STATS["not_cz"] += 1
        return None
    got = _graph_read(qc)
    if got is None:
        GRAPH_STATS["not_read"] += 1
        return None
    prefix, meas, terms, phase, suffix, perm, edges = got
    n, N = qc.num_qubits, target.num_qubits
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    if max(deg.values()) > 3:
        GRAPH_STATS["degree"] += 1
        return None
    col = _graph_colours(edges)
    if col is None:
        GRAPH_STATS["no_embedding"] += 1
        return None
    lat = _chain_lattice(target, max_error)
    if lat is None:
        GRAPH_STATS["no_embedding"] += 1
        return None
    K, kadj, fadj, _ = lat
    place = _graph_place(n, edges, K, fadj)
    if place is not None:
        GRAPH_STATS["common_length"] += 1
    else:
        place = _graph_place(n, edges, K, kadj)
    if place is None:
        GRAPH_STATS["no_embedding"] += 1
        return None
    ncol = max(col.values()) + 1
    GRAPH_STATS["colours"][ncol] = GRAPH_STATS["colours"].get(ncol, 0) + 1
    level, last = [], {}
    for k, (lb, _) in enumerate(terms):
        lv = 0
        for q, _ in _chain_support(lb):
            for j, lb2 in last.get(q, ()):
                if not _chain_commute(lb, lb2):
                    lv = max(lv, level[j] + 1)
        level.append(lv)
        for q, _ in _chain_support(lb):
            last.setdefault(q, []).append((k, lb))

    def key(k):
        sup = sorted(q for q, _ in _chain_support(terms[k][0]))
        if len(sup) == 2:
            return (level[k], 0, col[(sup[0], sup[1])], k)
        return (level[k], 1, 0, k)
    frame = "rzz" in target.operation_names and "rx" in target.operation_names
    log = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=phase)
    for op, q in prefix:
        log.append(op, [q])
    for k in sorted(range(len(terms)), key=key):
        _chain_rotation(log, *terms[k], frame=frame)
    for op, q in suffix:
        log.append(op, [q])
    for q, cb in meas:
        log.measure(perm[q], cb)  # item 76
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for a in sorted(K):
        for b in sorted(kadj[a]):
            cm.add_edge(a, b)
    used = set(place.values())
    free = [q for q in sorted(K) if q not in used]
    layout = [place[q] if q in place else free.pop(0) for q in range(n)]  # qubits without a term: unused places
    if _T100_HOOK[0] is not None:  # item 100: compile_template reads the structure here (nothing is compiled)
        _T100_HOOK[0](dict(prefix=prefix, terms=terms, order=sorted(range(len(terms)), key=key), suffix=suffix,
                           meas=meas, perm=perm, layout=layout, frame=frame, phase=phase, n=n,
                           extra=locals().get("path", locals().get("edges"))))
    if frame:
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveDiagonalGatesBeforeMeasure
        log = PassManager([RemoveDiagonalGatesBeforeMeasure()]).run(log)
        low = _chain_frame(log)
        pm = generate_preset_pass_manager(0, coupling_map=cm, basis_gates=["rzz", "rx", "rz", "sx", "x", "cz"],
                                          initial_layout=layout, routing_method="none", seed_transpiler=seed)
        placed = pm.run(low)
        out = PassManager([Optimize1qGatesDecomposition(basis=["rz", "rx"])]).run(placed)
        out._layout = placed._layout
        if not _rzz_in_range(out):
            out = _fold(out, target)
        GRAPH_STATS["frame"] += 1
    else:  # item 101: PSF-Zero emits the CZ path itself
        out = _cz101_emit(qc, target, dict(prefix=prefix, terms=terms, order=sorted(range(len(terms)), key=key),
                                           suffix=suffix, meas=meas, perm=perm, layout=layout, n=n))
    edges_f, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges_f, bad):
        GRAPH_STATS["failed_check"] += 1
        return None
    GRAPH_STATS["used"] += 1
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Prototype c54: item 93, every output checked exact at a tight tolerance where it can be (home, after the workplace's
# Q3: Qiskit's two-qubit synthesis can snap a nearly degenerate block, 1.4e-5 in distribution, inside a level-3
# transpile)
# ---------------------------------------------------------------------------------------------------------------------

_X93_BASIS = ["cx", "cz", "rz", "sx", "x", "h", "rzz", "rx", "ry", "u", "measure"]  # standard names only (barriers kept)


# ---------------------------------------------------------------------------------------------------------------------
# Prototype c56: item 96, the scope gate (home, after the owner's scope declaration of 2026-10-14, Addendum 488):
# PSF-Zero compiles a circuit only when its two-qubit interaction graph embeds in the device's working couplers;
# every other circuit is handed to Qiskit's level 3 unchanged
# ---------------------------------------------------------------------------------------------------------------------

_S96_VF2_CALLS = 2_000_000  # item 96: VF2 states tried before an embedding counts as not found (a count, not a time)


def _scope96_edges(qc):
    """Item 96: (sorted two-qubit interaction edges, None), or (None, reason) when the circuit cannot be in scope.
    A circuit items 72-73's reader reads as Pauli terms on one or two qubits gives the terms' two-qubit supports
    (item 91's `_graph_read`); otherwise every two-qubit operation as written gives an edge, and an operation on more
    than two qubits puts the circuit out of scope."""
    got = _graph_read(qc)
    if got is not None:
        return sorted(got[-1]), None
    edges = set()
    for ins in qc.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if len(qs) > 2:
            return None, "multi_qubit_operation"
        if len(qs) == 2:
            edges.add((min(qs), max(qs)))
    return sorted(edges), None

# ---------------------------------------------------------------------------------------------------------------------
# Item 102 (c65, home, 2026-10-16): item 96's embedding test decides by formula before VF2.
# ---------------------------------------------------------------------------------------------------------------------
_S96_GRAPH_CACHE = {}  # item 102: per working graph, (bipartite, shortest cycle, longest working piece of the snake)
_S96_GRAPH_CACHE_MAX = 64
S96_FORMULA_STATS = {"not_bipartite": 0, "short_cycle": 0, "path_on_snake": 0, "vf2": 0}


def _s96_bipartite(adj):
    """Item 102: True when the graph {node: neighbours} has no odd cycle (two-colouring by breadth-first search)."""
    import collections
    col = {}
    for r in sorted(adj):
        if r in col:
            continue
        col[r] = 0
        dq = collections.deque([r])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in col:
                    col[v] = 1 - col[u]
                    dq.append(v)
                elif col[v] == col[u]:
                    return False
    return True


def _s96_girth(adj):
    """Item 102: the length of the graph's shortest cycle (breadth-first search from every node), None for a forest."""
    import collections
    best = None
    for r in sorted(adj):
        dist, par = {r: 0}, {r: None}
        dq = collections.deque([r])
        while dq:
            u = dq.popleft()
            if best is not None and 2 * dist[u] + 1 >= best:
                break
            for v in adj[u]:
                if v not in dist:
                    dist[v], par[v] = dist[u] + 1, u
                    dq.append(v)
                elif par[u] != v:
                    c = dist[u] + dist[v] + 1
                    if best is None or c < best:
                        best = c
    return best


def _s96_is_path(adj, n_edges):
    """Item 102: True when the graph {node: neighbours} with n_edges edges is one simple path."""
    if n_edges != len(adj) - 1 or any(len(v) > 2 for v in adj.values()):
        return False
    start = next((q for q in sorted(adj) if len(adj[q]) == 1), None)
    if start is None:
        return False
    seen, prev, cur = {start}, None, start
    while True:
        nxt = [v for v in adj[cur] if v != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur in seen:
            return False
        seen.add(cur)
    return len(seen) == len(adj)


def _scope96_embeds(edges, n, target, max_error):
    """Item 96: True when the interaction graph `edges` (of an n-qubit circuit) embeds in the device's working
    coupling graph -- couplers and qubits the Target reports failed (error >= max_error, either direction) removed --
    by the first match of rustworkx's VF2 (subgraph monomorphism, nodes in index order, no score). Not finding one
    within _S96_VF2_CALLS states counts as no embedding.

    Item 102: decided by formula first, VF2 only when the formula does not decide.
      - The working graph is bipartite (heavy-hex is) and the interaction graph is not: no embedding (an embedding
        maps an odd cycle to an odd cycle).
      - The interaction graph has a cycle shorter than the working graph's shortest cycle (12 on heavy-hex): no
        embedding (an embedding maps a cycle to a cycle of the same length).
      - The interaction graph is a path, and the device's snake (item 87) has a working piece at least as long:
        it embeds.
    The first two are necessary conditions, so their answer is the one VF2 gives. Why: VF2 proving that a ring of 4-11
    qubits, or of odd length, does not embed took up to 70% of a compile of TOQB's ring circuits (prototype B,
    2026-10-16)."""
    import rustworkx as rx
    cm = target.build_coupling_map()
    if cm is None:  # all-to-all: every graph embeds
        return True
    fedges, fq = _failed_elements(target, max_error)
    und = set()
    for a, b in cm.get_edges():
        if a in fq or b in fq or (a, b) in fedges or (b, a) in fedges:
            continue
        und.add((min(a, b), max(a, b)))
    working = target.num_qubits - len(fq)
    if n > working or not und:
        return False
    key = (target.num_qubits, frozenset(und), frozenset(tuple(e) for e in cm.get_edges()))
    facts = _S96_GRAPH_CACHE.get(key)
    if facts is None:
        gadj = {}
        for a, b in und:
            gadj.setdefault(a, set()).add(b)
            gadj.setdefault(b, set()).add(a)
        full = {q: set() for q in range(target.num_qubits)}
        for a, b in cm.get_edges():
            full[a].add(b)
            full[b].add(a)
        longest, cur = 0, []
        for q in _chain_snake(target.num_qubits, full):
            if q in gadj and (not cur or q in gadj[cur[-1]]):
                cur.append(q)
            else:
                cur = [q] if q in gadj else []
            longest = max(longest, len(cur))
        facts = (_s96_bipartite(gadj), _s96_girth(gadj), longest)
        if len(_S96_GRAPH_CACHE) >= _S96_GRAPH_CACHE_MAX:
            _S96_GRAPH_CACHE.clear()
        _S96_GRAPH_CACHE[key] = facts
    g_bipartite, g_girth, longest = facts
    padj = {}
    for a, b in edges:
        padj.setdefault(a, set()).add(b)
        padj.setdefault(b, set()).add(a)
    if g_bipartite and not _s96_bipartite(padj):
        S96_FORMULA_STATS["not_bipartite"] += 1
        return False
    p_girth = _s96_girth(padj)
    if p_girth is not None and (g_girth is None or p_girth < g_girth):
        S96_FORMULA_STATS["short_cycle"] += 1
        return False
    if _s96_is_path(padj, len(set(edges))) and longest >= len(padj):
        S96_FORMULA_STATS["path_on_snake"] += 1
        return True
    S96_FORMULA_STATS["vf2"] += 1
    gn = sorted({q for e in und for q in e})
    gi = {q: i for i, q in enumerate(gn)}
    G = rx.PyGraph()
    G.add_nodes_from(gn)
    G.add_edges_from_no_data([(gi[a], gi[b]) for a, b in sorted(und)])
    pn = sorted({q for e in edges for q in e})
    pi = {q: i for i, q in enumerate(pn)}
    P = rx.PyGraph()
    P.add_nodes_from(pn)
    P.add_edges_from_no_data([(pi[a], pi[b]) for a, b in edges])
    if len(pn) > len(gn) or max(P.degree(i) for i in P.node_indices()) > max(G.degree(i) for i in G.node_indices()):
        return False
    m = next(rx.vf2_mapping(G, P, subgraph=True, induced=False, id_order=True, call_limit=_S96_VF2_CALLS), None)
    return m is not None


# ---------------------------------------------------------------------------------------------------------------------
# Prototype c58 (home, 2026-10-15): item 97, formula only inside the scope -- an in-scope circuit is built by item 87's
# chain brickwork or item 91's edge colouring and returned as it is; nothing else is built or compared, and no output
# is simulated to check it. A circuit the formulas do not build goes to Qiskit's level 3 as item 96 hands it on.
# ---------------------------------------------------------------------------------------------------------------------

FORMULA97_STATS = {"calls": 0, "chain": 0, "graph": 0, "native": 0, "not_built": 0, "checked": 0, "check_failed": 0}
GATE98_STATS = {"no_target": 0, "out_of_scope": 0, "not_built": 0}


class OutOfScopeError(TranspilerError):
    """Item 98: the call is outside PSF-Zero's scope; compile it with Qiskit (or another compiler) yourself."""


def compile_for_hardware(qc, *args, **kwargs):
    """Item 97 (prototype c58), items 98-99, c64: with the device's Target (`target=` or `backend=`) and an input
    inside item 96's scope, the output of item 87's `_chain_brickwork`, else item 91's `_graph_brickwork`, else item
    99's `_native_brickwork`, returned without building or comparing anything else and without simulating it (an rzz
    outside IBM's range goes through item 86's `_fold`). Without a Target, with positional arguments, out of scope or
    with no formula reading the input: OutOfScopeError (item 98). Counts in `FORMULA97_STATS` and `GATE98_STATS`."""
    target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
    if target is None or args:  # item 98
        GATE98_STATS["no_target"] += 1
        raise OutOfScopeError("PSF-Zero compiles only with the device's Target given as target= or backend= "
                              "(changelog item 98); compile this call with Qiskit yourself")
    max_error = kwargs.get("prune_max_error", 0.5)
    edges, reason = _scope96_edges(qc)
    in_scope = reason is None and (not edges or _scope96_embeds(edges, qc.num_qubits, target, max_error))
    if not in_scope:  # item 98
        GATE98_STATS["out_of_scope"] += 1
        why = ("an operation acts on more than two qubits" if reason == "multi_qubit_operation" else
               "its two-qubit interaction graph does not embed in the device's working couplers")
        raise OutOfScopeError(f"outside PSF-Zero's scope: {why} (changelog item 98); compile it with Qiskit yourself")
    FORMULA97_STATS["calls"] += 1
    seed = kwargs.get("seed_transpiler")
    seed = 0 if seed is None else seed
    out = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "cz" in target.operation_names:
            out = _chain_brickwork(qc, target, max_error, seed)
            kind = "chain"
            if out is None:
                out = _graph_brickwork(qc, target, max_error, seed)
                kind = "graph"
            if out is None:  # item 99
                out = _native_brickwork(qc, target, max_error, seed)
                kind = "native"
    if out is not None and "rzz" in target.operation_names and not _rzz_in_range(out):
        try:
            out = _fold(out, target)
        except TranspilerError:
            out = None
    if out is not None:
        FORMULA97_STATS[kind] += 1
        return out
    FORMULA97_STATS["not_built"] += 1
    GATE98_STATS["not_built"] += 1
    raise OutOfScopeError("inside the scope, but no formula (item 87's chain, item 91's graph, item 99's native "
                          "gates) reads this circuit (changelog items 98-99); compile it with Qiskit yourself")


# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c59 (2026-10-15): item 99, native two-qubit gate circuits on an embeddable graph built
# by formula as written -- placement by item 91's first VF2 embedding, each gate in a fixed native form, commuting
# diagonal gates reordered by item 91's edge colouring, single-qubit runs merged in closed form (ZYZ) by PSF-Zero
# itself. One fixed pass: nothing is searched, compared or simulated.
# ---------------------------------------------------------------------------------------------------------------------

import cmath  # noqa: E402  (item 99)
from qiskit.circuit import Gate, QuantumRegister  # noqa: E402  (item 99)

NATIVE_BRICKWORK = True  # item 99
NATIVE99_STATS = {"tried": 0, "not_read": 0, "degree": 0, "no_colouring": 0, "no_embedding": 0, "not_cz": 0,
                  "block_not_exact": 0, "failed_check": 0, "built": 0, "gates": {}}
_N99_TOL = 1e-12  # item 99: angles and matrix entries within this of a special value are taken as that value
_N99_BLOCK_TOL = 1e-9  # item 99: every two-qubit gate's native form is multiplied out and checked to this (up to phase)
# item 99: gates diagonal for every angle (the diagonal class is decided by the name for gates with a parameter, so
# that it never depends on the angle's value); a gate without parameters is diagonal when its matrix is
_N99_DIAG = frozenset(("id", "rz", "p", "u1", "z", "s", "sdg", "t", "tdg", "cz", "cp", "cu1", "crz", "rzz",
                       "ccz"))
_N99_PI = math.pi


def _n99_t(op):
    """Item 99: a single-qubit operation as a tuple matrix (a, b, c, d) = [[a, b], [c, d]] (closed forms for rz, ry,
    rx, p; else the operation's matrix)."""
    nm = op.name
    if nm in ("rz", "ry", "rx", "p", "u1"):
        t = float(op.params[0])
        if nm == "rz":
            return (cmath.exp(-0.5j * t), 0j, 0j, cmath.exp(0.5j * t))
        if nm in ("p", "u1"):
            return (1 + 0j, 0j, 0j, cmath.exp(1j * t))
        c, s = math.cos(t / 2), math.sin(t / 2)
        if nm == "ry":
            return (complex(c), complex(-s), complex(s), complex(c))
        return (complex(c), -1j * s, -1j * s, complex(c))
    m = op.to_matrix()
    return (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))


def _n99_mul(A, B):
    """Item 99: the tuple-matrix product A @ B."""
    a, b, c, d = A
    e, f, g, h = B
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def _n99_rz(t):
    return (cmath.exp(-0.5j * t), 0j, 0j, cmath.exp(0.5j * t))


_N99_H = (complex(1 / math.sqrt(2)), complex(1 / math.sqrt(2)), complex(1 / math.sqrt(2)), complex(-1 / math.sqrt(2)))
_N99_X = (0j, 1 + 0j, 1 + 0j, 0j)


def _n99_zyz(U):
    """Item 99: (theta, phi, lam) with U = e^{i alpha} Rz(phi) Ry(theta) Rz(lam) (as a20's `_t_zyz`)."""
    a, b, c, d = U
    sq = cmath.sqrt(a * d - b * c)
    a, c, d = a / sq, c / sq, d / sq
    if abs(c) <= 1e-15:  # diagonal: only phi + lam is defined
        return 0.0, 2 * cmath.phase(d), 0.0
    if abs(a) <= 1e-15:  # anti-diagonal: only phi - lam is defined
        return _N99_PI, 2 * cmath.phase(c), 0.0
    p11, p10 = cmath.phase(d), cmath.phase(c)
    return 2 * math.atan2(abs(c), abs(a)), p11 + p10, p11 - p10


def _n99_wrap(t):
    """Item 99: t reduced to (-pi, pi]."""
    t = math.fmod(t, 2 * _N99_PI)
    if t <= -_N99_PI:
        t += 2 * _N99_PI
    elif t > _N99_PI:
        t -= 2 * _N99_PI
    return t


def _n99_run_class(U, use_rx):
    """Item 99: how a merged single-qubit run is emitted -- (class, angles): "none" (identity up to phase), "z"
    (one rz), "half" (rz sx rz, theta = pi/2, without rx), "full" (rz sx rz sx rz, or rz rx rz with rx)."""
    theta, phi, lam = _n99_zyz(U)
    if abs(theta) < _N99_TOL:
        z = _n99_wrap(phi + lam)
        return ("none", ()) if abs(z) < _N99_TOL else ("z", (z,))
    if use_rx:
        return "full", (lam - _N99_PI / 2, theta, phi + _N99_PI / 2)
    if abs(theta - _N99_PI / 2) < _N99_TOL:
        return "half", (lam - _N99_PI / 2, phi + _N99_PI / 2)
    return "full", (lam, theta + _N99_PI, phi + _N99_PI)


def _n99_emit_run(out, p, cls, ang, use_rx):
    """Item 99: append a run's gates (`_n99_run_class`) on physical qubit p."""
    if cls == "z":
        out.rz(ang[0], p)
    elif cls == "half":
        out.rz(ang[0], p)
        out.sx(p)
        out.rz(ang[1], p)
    elif cls == "full":
        out.rz(ang[0], p)
        if use_rx:
            out.rx(ang[1], p)
        else:
            out.sx(p)
            out.rz(ang[1], p)
            out.sx(p)
        out.rz(ang[2], p)


def _n99_diag(op, nq):
    """Item 99: True if `op` is diagonal (by name for the gates of _N99_DIAG; by its matrix for a gate without
    parameters; never for any other gate with a parameter)."""
    if op.name in _N99_DIAG:
        return True
    if op.params:
        return False
    try:
        m = np.asarray(op.to_matrix())
    except Exception:  # noqa: BLE001
        return False
    return bool(np.all(np.abs(m - np.diag(np.diag(m))) < _N99_TOL))


def _n99_read(qc):
    """Item 99: ([(name, qubits, op, diagonal)], [(qubit, clbit)]) when `qc` is gates on one or two qubits (each with
    a matrix and bound parameters), barriers and final measurements; else None."""
    ops, meas, measured = [], [], set()
    for ins in qc.data:
        op = ins.operation
        nm = op.name
        if nm == "barrier":
            continue
        if getattr(op, "condition", None) is not None or getattr(op, "blocks", None) is not None:
            return None
        qs = tuple(qc.find_bit(q).index for q in ins.qubits)
        if nm == "measure":
            if qs[0] in measured:
                return None
            measured.add(qs[0])
            meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))
            continue
        if ins.clbits or measured & set(qs) or len(qs) > 2 or not isinstance(op, Gate):
            return None
        if getattr(op, "is_parameterized", lambda: False)():
            return None
        if not qs:  # a global phase: no effect on outcomes
            continue
        ops.append((nm, qs, op, _n99_diag(op, len(qs))))
    return ops, meas


def _n99_zz(q0, q1, th, frac, native_rzz):
    """Item 99: steps for RZZ(th) = exp(-i th/2 Z Z) on (q0, q1), up to a global phase. RZZ(pi) is local (Rz(pi) on
    both), so th is first reduced to t in (-pi/2, pi/2]. Then: t = 0, no two-qubit gate; |t| = pi/2, one cz (unless
    `native_rzz`); otherwise, with rzz in the Target (`frac`), one rzz(|t|) in IBM's range, an X on q0 on both sides
    when t < 0; without it, cz, rz(t), cz with H on q1 (CX = H cz H)."""
    k = round(th / _N99_PI)
    t = th - k * _N99_PI
    if t <= -_N99_PI / 2 + _N99_TOL:
        t += _N99_PI
        k -= 1
    steps = []
    if k % 2:
        steps += [("1", q0, _n99_rz(_N99_PI)), ("1", q1, _n99_rz(_N99_PI))]
    if abs(t) < _N99_TOL:
        return steps
    if frac and (native_rzz or abs(abs(t) - _N99_PI / 2) >= _N99_TOL):
        if t > 0:
            return steps + [("rzz", q0, q1, t)]
        return steps + [("1", q0, _N99_X), ("rzz", q0, q1, -t), ("1", q0, _N99_X)]
    if abs(abs(t) - _N99_PI / 2) < _N99_TOL:
        s = _N99_PI / 2 if t > 0 else -_N99_PI / 2
        return steps + [("1", q0, _n99_rz(s)), ("1", q1, _n99_rz(s)), ("cz", q0, q1)]
    return steps + [("1", q1, _N99_H), ("cz", q0, q1), ("1", q1, _N99_H), ("1", q1, _n99_rz(t)), ("1", q1, _N99_H),
                    ("cz", q0, q1), ("1", q1, _N99_H)]


def _n99_same(U, W):
    k = int(np.argmax(np.abs(U)))
    ph = U.flat[k] / W.flat[k]
    return abs(abs(ph) - 1) < _N99_BLOCK_TOL and float(np.max(np.abs(U - ph * W))) < _N99_BLOCK_TOL


def _n99_steps(nm, qs, op, diag, frac):
    """Item 99: the native form of one two-qubit gate on logical qubits qs = (q0, q1): [("1", q, 2 x 2 tuple) |
    ("cz", q0, q1) | ("rzz", q0, q1, t)] in time order, or None if no form passes the check. cz stays cz. A diagonal
    gate is Rz (x) Rz times RZZ(theta), its angles read off its four phases (closed form), RZZ by `_n99_zz`. Any other
    gate by its Weyl decomposition (Qiskit's, no specialisation): a single interaction (b = c = 0: cx, rxx, ryy, crx,
    ...) is local gates around RZZ(-2a) (XX = (H x H) ZZ (H x H)), by `_n99_zz`; otherwise Qiskit's closed-form CZ
    decomposition (KAK, two or three cz)."""
    q0, q1 = qs
    if nm == "cz":
        return [("cz", q0, q1)]
    U = np.asarray(op.to_matrix(), dtype=complex)
    if diag:
        ph = np.angle(np.diag(U))  # index x0 + 2 x1
        z0, z1 = np.array([1, -1, 1, -1]), np.array([1, 1, -1, -1])
        b0, b1, th = -0.5 * float(ph @ z0), -0.5 * float(ph @ z1), -0.5 * float(ph @ (z0 * z1))
        steps = [("1", q0, _n99_rz(b0)), ("1", q1, _n99_rz(b1))] + _n99_zz(q0, q1, th, frac, nm == "rzz")
        return steps if _n99_same(U, _n99_kron_steps(steps, q0, q1)) else None
    from qiskit.synthesis import TwoQubitWeylDecomposition
    try:
        d = TwoQubitWeylDecomposition(U, fidelity=None)
    except Exception:  # noqa: BLE001
        d = None
    if d is not None and abs(d.b) < _N99_TOL and abs(d.c) < _N99_TOL:
        t = lambda m: (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))  # noqa: E731
        steps = ([("1", q0, t(d.K2r)), ("1", q1, t(d.K2l)), ("1", q0, _N99_H), ("1", q1, _N99_H)]
                 + _n99_zz(q0, q1, -2 * d.a, frac, False)
                 + [("1", q0, _N99_H), ("1", q1, _N99_H), ("1", q0, t(d.K1r)), ("1", q1, t(d.K1l))])
        if _n99_same(U, _n99_kron_steps(steps, q0, q1)):
            return steps
    from qiskit.circuit.library import CZGate
    from qiskit.synthesis import TwoQubitBasisDecomposer
    for approximate in (True, False):
        dec = TwoQubitBasisDecomposer(CZGate(), euler_basis="ZSX")(U, approximate=approximate)
        steps = []
        for ins in dec.data:
            w = [dec.find_bit(q).index for q in ins.qubits]
            if len(w) == 2:
                steps.append(("cz", q0, q1))
            else:
                steps.append(("1", q0 if w[0] == 0 else q1, _n99_t(ins.operation)))
        if _n99_same(U, _n99_kron_steps(steps, q0, q1)):
            return steps
    return None


def _n99_layout(qc, out, phys):
    """Item 99: the output's TranspileLayout (input qubit q on physical qubit phys[q]; no routing)."""
    from qiskit.transpiler import Layout, TranspileLayout
    N = out.num_qubits
    used = set(phys)
    anc = QuantumRegister(N - qc.num_qubits, "ancilla") if N > qc.num_qubits else None
    mapping = {qc.qubits[q]: phys[q] for q in range(qc.num_qubits)}
    if anc is not None:
        rest = [p for p in range(N) if p not in used]
        mapping.update({anc[i]: p for i, p in enumerate(rest)})
    lay = Layout(mapping)
    for r in qc.qregs:
        lay.add_register(r)
    if anc is not None:
        lay.add_register(anc)
    iq = {qc.qubits[q]: q for q in range(qc.num_qubits)}
    if anc is not None:
        iq.update({anc[i]: qc.num_qubits + i for i in range(len(anc))})
    return TranspileLayout(initial_layout=lay, input_qubit_mapping=iq, final_layout=None,
                           _input_qubit_count=qc.num_qubits, _output_qubit_list=list(out.qubits))


def _native_brickwork(qc, target, max_error, seed):
    """Item 99 (workplace proposal on c59): a circuit of gates on one or two qubits whose interaction graph embeds in
    the cut device (max degree <= 3, bipartite), built by formula as written, in one fixed pass:
      1. placement: item 91's -- the first VF2 embedding (`_graph_place`), on the couplers of the device's most common
         two-qubit gate length first, then on all of item 87's cut device; qubits without a two-qubit gate take the
         cut device's free qubits in order;
      2. order: every gate at level 1 + the largest level of an earlier gate on a shared qubit that does not commute
         with it (diagonal gates commute with each other; nothing else is assumed to commute) -- exact; within a level
         the two-qubit gates by the colour of item 91's edge colouring (`_graph_colours`), then the single-qubit ones;
      3. each two-qubit gate in its native form (`_n99_steps`: cz; rzz native and folded into IBM's range when the
         Target has rzz, else cz rz cz; a diagonal or single-interaction gate through RZZ; any other by the closed-form
         CZ decomposition), every form multiplied out and checked to 1e-9;
      4. every single-qubit run merged and emitted in closed form (ZYZ): rz sx rz sx rz (rz sx rz at theta = pi/2,
         one rz when diagonal), or rz rx rz with rx in the Target;
      5. measurements as written (item 89 not applied).
    No Qiskit pass manager runs and nothing is simulated. None when not applicable (counted in NATIVE99_STATS)."""
    NATIVE99_STATS["tried"] += 1
    if "cz" not in target.operation_names:
        NATIVE99_STATS["not_cz"] += 1
        return None
    got = _n99_read(qc)
    if got is None:
        NATIVE99_STATS["not_read"] += 1
        return None
    ops, meas = got
    n, N = qc.num_qubits, target.num_qubits
    edges = sorted({(min(qs), max(qs)) for _, qs, _, _ in ops if len(qs) == 2})
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    if deg and max(deg.values()) > 3:
        NATIVE99_STATS["degree"] += 1
        return None
    col = _graph_colours(edges) if edges else {}
    if col is None:
        NATIVE99_STATS["no_colouring"] += 1
        return None
    lat = _chain_lattice(target, max_error)
    if lat is None:
        NATIVE99_STATS["no_embedding"] += 1
        return None
    K, kadj, fadj, _ = lat
    place = {}
    if edges:
        place = _graph_place(n, edges, K, fadj) or _graph_place(n, edges, K, kadj)
        if place is None:
            NATIVE99_STATS["no_embedding"] += 1
            return None
    free = [q for q in sorted(K) if q not in set(place.values())]
    if n - len(place) > len(free):
        NATIVE99_STATS["no_embedding"] += 1
        return None
    phys = [place[q] if q in place else free.pop(0) for q in range(n)]
    # 2. levels and order
    maxall, maxnd, level = {}, {}, []
    for _, qs, _, dg in ops:
        if dg:
            lv = max([maxnd.get(q, -1) for q in qs]) + 1
        else:
            lv = max([maxall.get(q, -1) for q in qs]) + 1
        level.append(lv)
        for q in qs:
            maxall[q] = max(maxall.get(q, -1), lv)
            if not dg:
                maxnd[q] = max(maxnd.get(q, -1), lv)

    def key(k):
        qs = ops[k][1]
        if len(qs) == 2:
            return (level[k], 0, col[(min(qs), max(qs))], k)
        return (level[k], 1, 0, k)
    order = sorted(range(len(ops)), key=key)
    # 3-4. emission
    frac = "rzz" in target.operation_names
    use_rx = "rx" in target.operation_names
    out = QuantumCircuit(N, name=qc.name)
    for cr in qc.cregs:
        out.add_register(cr)
    one = (1 + 0j, 0j, 0j, 1 + 0j)
    pending = {}

    def flush(q):
        U = pending.pop(q, None)
        if U is not None:
            cls, ang = _n99_run_class(U, use_rx)
            _n99_emit_run(out, phys[q], cls, ang, use_rx)

    def two(name, a, b):
        pa, pb = phys[a], phys[b]
        if (pa, pb) not in target[name]:
            pa, pb = pb, pa
        return pa, pb
    for k in order:
        nm, qs, op, dg = ops[k]
        if len(qs) == 1:
            pending[qs[0]] = _n99_mul(_n99_t(op), pending.get(qs[0], one))
            continue
        steps = _n99_steps(nm, qs, op, dg, frac)
        if steps is None:
            NATIVE99_STATS["block_not_exact"] += 1
            return None
        NATIVE99_STATS["gates"][nm] = NATIVE99_STATS["gates"].get(nm, 0) + 1
        for st in steps:
            if st[0] == "1":
                pending[st[1]] = _n99_mul(st[2], pending.get(st[1], one))
                continue
            flush(st[1])
            flush(st[2])
            if st[0] == "cz":
                out.cz(*two("cz", st[1], st[2]))
            else:
                out.rzz(st[3], *two("rzz", st[1], st[2]))
    for q in sorted(pending):
        flush(q)
    for q, cb in meas:
        out.measure(phys[q], cb)
    out._layout = _n99_layout(qc, out, phys)
    edges_f, bad = _failed_elements(target, max_error)
    if _uses_failed(out, edges_f, bad):
        NATIVE99_STATS["failed_check"] += 1
        return None
    NATIVE99_STATS["built"] += 1
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c60 (2026-10-15): item 101, the CZ path of items 87 and 91 emitted by PSF-Zero
# itself -- the last Qiskit level-3 stage inside the scope removed. The rule is by block kind: a single interaction
# (one two-qubit term: a local Clifford around RZZ) in closed form; any other block (several terms on one pair, with
# the single-qubit operations between them) by its Weyl decomposition and closed-form circuits. Nothing is compared.
# ---------------------------------------------------------------------------------------------------------------------

CZ101_STATS = {"emitted": 0, "single_blocks": 0, "weyl_blocks": 0, "removed_diagonal": 0}
_T100_ONE = (1 + 0j, 0j, 0j, 1 + 0j)


def _t100_emit(items, phys, target, use_rx, use_x=False):
    """Item 100: the output's shape [(gate, physical qubits...)] and its angle slots, from emitter items
    ("1", q, 2 x 2 tuple) | ("cz", a, b) | ("rzz", a, b, t) | ("end",) | ("meas", q, clbit) -- single-qubit runs merged
    and emitted exactly as item 99's `_native_brickwork` does (`_n99_run_class`)."""
    shape, slots, pending = [], [], {}
    cz_q, rzz_q = target["cz"], (target["rzz"] if "rzz" in target.operation_names else None)

    def flush(q):
        U = pending.pop(q, None)
        if U is None:
            return
        cls, ang = _n99_run_class(U, use_rx)
        p = phys[q]
        if use_x and cls == "full" and abs(ang[1] - 2 * math.pi) < _N99_TOL:  # theta = pi: one x pulse
            shape.extend((("rz", p), ("x", p), ("rz", p)))
            slots.extend((ang[0] - math.pi / 2, ang[2] - math.pi / 2))
            return
        if cls == "z":
            shape.append(("rz", p))
            slots.append(ang[0])
        elif cls == "half":
            shape.extend((("rz", p), ("sx", p), ("rz", p)))
            slots.extend(ang)
        elif cls == "full":
            if use_rx:
                shape.extend((("rz", p), ("rx", p), ("rz", p)))
            else:
                shape.extend((("rz", p), ("sx", p), ("rz", p), ("sx", p), ("rz", p)))
            slots.extend(ang)
    for it in items:
        k = it[0]
        if k == "1":
            q = it[1]
            pending[q] = _n99_mul(it[2], pending.get(q, _T100_ONE))
        elif k == "cz" or k == "rzz":
            a, b = it[1], it[2]
            flush(a)
            flush(b)
            pa, pb = phys[a], phys[b]
            if (pa, pb) not in (cz_q if k == "cz" else rzz_q):
                pa, pb = pb, pa
            shape.append((k, pa, pb))
            if k == "rzz":
                slots.append(it[3])
        elif k == "end":
            for q in sorted(pending):
                flush(q)
        else:  # ("meas", q, clbit)
            flush(it[1])
            shape.append(("measure", phys[it[1]], it[2]))
    return tuple(shape), slots


_T100_PAULI = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
               "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]).astype(complex)}


def _t100_rot1(ch, a):
    """Item 100: exp(-i a P) for a single-qubit Pauli P as a tuple matrix (rx / ry / rz(2a))."""
    if ch == "Z":
        return _n99_rz(2 * a)
    c, s = math.cos(a), math.sin(a)
    if ch == "X":
        return (complex(c), -1j * s, -1j * s, complex(c))
    return (complex(c), complex(-s), complex(s), complex(c))


def _t100_k1(a, b, c, d, low):
    """Item 100: a 2 x 2 matrix [[a, b], [c, d]] on one qubit of a pair as 4 x 4 (low: the less significant qubit)."""
    if low:
        return np.array([[a, b, 0, 0], [c, d, 0, 0], [0, 0, a, b], [0, 0, c, d]], dtype=complex)
    return np.array([[a, 0, b, 0], [0, a, 0, b], [c, 0, d, 0], [0, c, 0, d]], dtype=complex)


_N99_CZ4 = np.diag([1, 1, 1, -1]).astype(complex)


def _n99_kron_steps(steps, q0, q1):  # noqa: F811 - item 100: item 99's check, the same matrices without np.kron
    """Item 99 (faster form, c61): the 4 x 4 matrix (qubit q0 least significant) of a list of steps on (q0, q1)."""
    U = np.eye(4, dtype=complex)
    for st in steps:
        if st[0] == "1":
            U = _t100_k1(*st[2], st[1] == q0) @ U
        elif st[0] == "cz":
            U = _N99_CZ4 @ U
        else:
            t = st[3]
            e, f = cmath.exp(-0.5j * t), cmath.exp(0.5j * t)
            U = np.array([e, f, f, e])[:, None] * U
    return U


_T100_MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / math.sqrt(2)
_T100_XYZ = np.array([np.real(np.diag(_T100_MAGIC.conj().T @ np.kron(_T100_PAULI[p], _T100_PAULI[p]) @ _T100_MAGIC))
                      for p in "XYZ"])  # eigenvalues of XX, YY, ZZ on the magic basis (all three diagonal there)


def _t100_canonical(q0, q1, a, b, c):
    """Item 100: steps for exp(i(a XX + b YY + c ZZ)) on (q0, q1), any real a, b, c, up to a global phase: Rz(pi/2)
    on q1; CX(q1 -> q0); Rz(-pi/2 - 2c) on q0, Ry(-pi/2 - 2a) on q1; CX(q0 -> q1); Ry(pi/2 + 2b) on q1;
    CX(q1 -> q0); Rz(-pi/2) on q0 -- CX = H cz H (three cz). An identity, checked in test_c61."""
    hp, H = math.pi / 2, _N99_H
    return [("1", q1, _n99_rz(hp)),
            ("1", q0, H), ("cz", q0, q1), ("1", q0, H),
            ("1", q0, _n99_rz(-hp - 2 * c)), ("1", q1, _T100_RY(-hp - 2 * a)),
            ("1", q1, H), ("cz", q0, q1), ("1", q1, H),
            ("1", q1, _T100_RY(hp + 2 * b)),
            ("1", q0, H), ("cz", q0, q1), ("1", q0, H),
            ("1", q0, _n99_rz(-hp))]


def _t100_weyl_ok(U, d):
    """Item 100: True if Qiskit's Weyl decomposition `d` reproduces U to 1e-9 (the closed-form circuits built from it
    are identities, so this checks the step list without multiplying it out)."""
    a, b = d.K1l, d.K1r
    k1 = (a[:, None, :, None] * b[None, :, None, :]).reshape(4, 4)
    a, b = d.K2l, d.K2r
    k2 = (a[:, None, :, None] * b[None, :, None, :]).reshape(4, 4)
    Nd = np.exp(1j * (d.a * _T100_XYZ[0] + d.b * _T100_XYZ[1] + d.c * _T100_XYZ[2]))
    W = k1 @ ((_T100_MAGIC * Nd) @ _T100_MAGIC.conj().T) @ k2
    return _n99_same(U, W)


_T100_RY = lambda t: (complex(math.cos(t / 2)), complex(-math.sin(t / 2)), complex(math.sin(t / 2)),  # noqa: E731
                      complex(math.cos(t / 2)))


def _t100_block_steps(U, q0, q1):
    """Item 100: steps for a two-qubit block U (q0 least significant), from its Weyl decomposition (Qiskit's, without
    specialisation): U = e^{i phi} (K1l x K1r) exp(i(a XX + b YY + c ZZ)) (K2l x K2r). A single interaction
    (b = c = 0) through RZZ (`_n99_zz`); c = 0 by Qiskit's closed-form CZ decomposition (two cz); otherwise the
    closed-form three-CNOT circuit of exp(i(a XX + b YY + c ZZ)) -- Rz(pi/2) on q1; CX(q1 -> q0); Rz(-pi/2 - 2c) on q0,
    Ry(-pi/2 - 2a) on q1; CX(q0 -> q1); Ry(pi/2 + 2b) on q1; CX(q1 -> q0); Rz(-pi/2) on q0 -- with CX = H cz H. Every
    form is multiplied out and checked to 1e-9."""
    from qiskit.synthesis import TwoQubitWeylDecomposition
    d = TwoQubitWeylDecomposition(U, fidelity=None)
    t = lambda m: (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))  # noqa: E731
    pre = [("1", q0, t(d.K2r)), ("1", q1, t(d.K2l))]
    post = [("1", q0, t(d.K1r)), ("1", q1, t(d.K1l))]
    H = _N99_H
    exact_form = True  # closed forms of the Weyl decomposition: the decomposition itself is checked
    if abs(d.b) < _N99_TOL and abs(d.c) < _N99_TOL:
        steps = pre + [("1", q0, H), ("1", q1, H)] + _n99_zz(q0, q1, -2 * d.a, False, False) + \
            [("1", q0, H), ("1", q1, H)] + post
    elif abs(d.c) < _N99_TOL:
        exact_form = False
        steps = _n99_steps("unitary", (q0, q1), _T100Mat(U), False, False)
        if steps is not None:
            return steps
        steps = []
    else:
        steps = pre + _t100_canonical(q0, q1, d.a, d.b, d.c) + post
    if steps and (_t100_weyl_ok(U, d) if exact_form else _n99_same(U, _n99_kron_steps(steps, q0, q1))):
        return steps
    TEMPLATE100_STATS["block_not_exact"] += 1
    raise TranspilerError("item 100: a block's native form failed its check")


class _T100Op:
    """An operation of the input with new parameters (name and params as the original's; matrix from a new instance
    of its class), for `_n99_t`."""
    __slots__ = ("name", "params", "_cls")

    def __init__(self, op, params):
        self.name, self.params, self._cls = op.name, params, op.__class__

    def to_matrix(self):
        return self._cls(*self.params).to_matrix()


class _T100Mat:
    """Item 100: a fixed 4 x 4 block for `_n99_steps`."""
    __slots__ = ("U",)
    name = "unitary"
    params = ()

    def __init__(self, U):
        self.U = U

    def to_matrix(self):
        return self.U


class _CZ101Program:
    """Item 101: the program items 87/91 build (prefix gates, terms in their order, suffix gates, measurements, on
    their fixed layout) as PSF-Zero emits it on the CZ path: every two-qubit term a local Clifford around RZZ
    (`_n99_zz`); consecutive two-qubit terms on one pair one block (single-qubit operations between two of its terms
    absorbed); a block of one term in closed form, any other by its Weyl decomposition (`_t100_block_steps`);
    diagonal operations directly before the measurements left out (repeatedly; exact for the outcome distribution).
    Values: the prefix gates' parameters, the terms' angles (reader order), the suffix gates' parameters."""

    def __init__(self, info):
        self.n, self.phys = info["n"], list(info["layout"])
        self.prefix, self.terms, self.order = info["prefix"], info["terms"], info["order"]
        self.suffix, self.meas, self.perm = info["suffix"], info["meas"], info["perm"]
        self.pre_sl, pos = [], 0
        for op, _ in self.prefix:
            m = len(op.params)
            self.pre_sl.append((pos, pos + m) if m else None)
            pos += m
        self.term0 = pos
        pos += len(self.terms)
        self.suf_sl = []
        for op, _ in self.suffix:
            m = len(op.params)
            self.suf_sl.append((pos, pos + m) if m else None)
            pos += m
        self.nvalues = pos
        self.values0 = ([float(p) for op, _ in self.prefix for p in op.params] + [float(a) for _, a in self.terms]
                        + [float(p) for op, _ in self.suffix for p in op.params])
        self._build_blocks()

    def _gate1(self, op, sl, values):
        return _n99_t(op if sl is None else _T100Op(op, values[sl[0]:sl[1]]))

    def _build_blocks(self):
        """The logical program: prefix gates, terms in items 87/91's order, suffix gates, measurements; diagonal
        operations directly before the measurements left out (repeatedly); consecutive two-qubit terms on one pair
        grouped into a block."""
        prog = []  # (kind, qubits, ref, diagonal)
        for j, (op, q) in enumerate(self.prefix):
            prog.append(("pre", (q,), j, _n99_diag(op, 1)))
        for k in self.order:
            lb, _ = self.terms[k]
            sup = _chain_support(lb)
            prog.append(("term", tuple(q for q, _ in sup), k, all(ch == "Z" for _, ch in sup)))
        for j, (op, q) in enumerate(self.suffix):
            prog.append(("suf", (q,), j, _n99_diag(op, 1)))
        measured = {self.perm[q] for q, _ in self.meas}
        if measured:
            while True:
                last = {}
                for i, (_, qs, _, _) in enumerate(prog):
                    for q in qs:
                        last[q] = i
                drop = [i for i in sorted(set(last.values()), reverse=True)
                        if prog[i][3] and all(q in measured and last[q] == i for q in prog[i][1])]
                if not drop:
                    break
                for i in drop:
                    del prog[i]
                CZ101_STATS["removed_diagonal"] += len(drop)
        # blocks: consecutive two-qubit terms on one pair; single-qubit operations on the pair's qubits between two
        # of its terms are absorbed into the block, those after its last term stay outside (decided here, once)
        seq, open_ = [], {}

        def close(blk):
            for x in blk[1]:
                if open_.get(x) is blk:
                    del open_[x]
            seq.extend(blk[3])
            blk[3] = []
        for kind, qs, ref, _ in prog:
            if kind == "term" and len(qs) == 2:
                pair = tuple(sorted(qs))
                blk = open_.get(qs[0])
                if blk is not None and blk is open_.get(qs[1]) and blk[1] == pair:
                    blk[2].extend(("g",) + h for h in blk[3])
                    blk[3] = []
                    blk[2].append(("t", ref))
                    continue
                for q in qs:
                    if open_.get(q) is not None:
                        close(open_[q])
                blk = ["block", pair, [("t", ref)], []]
                open_[qs[0]] = open_[qs[1]] = blk
                seq.append(blk)
                continue
            q = qs[0] if len(qs) == 1 else None
            if q is not None and open_.get(q) is not None:
                open_[q][3].append((kind, qs, ref))
                continue
            for x in qs:
                if open_.get(x) is not None:
                    close(open_[x])
            seq.append((kind, qs, ref))
        for blk in list({id(v): v for v in open_.values()}.values()):
            close(blk)
        self.prog = seq
        self.paulis = {}
        for blk in seq:
            if blk[0] == "block":
                qa, qb = blk[1]
                for m in blk[2]:
                    if m[0] == "t":
                        ch = dict(_chain_support(self.terms[m[1]][0]))
                        self.paulis[m[1]] = np.kron(_T100_PAULI[ch.get(qb, "I")], _T100_PAULI[ch.get(qa, "I")])

    def _one(self, kind, ref, values):
        """A single-qubit operation of the program: (qubit, 2 x 2 tuple)."""
        if kind == "pre" or kind == "suf":
            op, q = (self.prefix if kind == "pre" else self.suffix)[ref]
            return q, self._gate1(op, (self.pre_sl if kind == "pre" else self.suf_sl)[ref], values)
        (q, ch), = _chain_support(self.terms[ref][0])
        return q, _t100_rot1(ch, values[self.term0 + ref])

    def _block_items(self, values):
        t0 = self.term0
        for b in self.prog:
            if b[0] != "block":
                q, M = self._one(b[0], b[2], values)
                yield ("1", q, M)
            elif len(b[2]) == 1:  # one two-qubit term: a local Clifford around RZZ(2a), exact in closed form
                CZ101_STATS["single_blocks"] += 1
                k = b[2][0][1]
                (q1, c1), (q2, c2) = _chain_support(self.terms[k][0])
                for q, ch in ((q1, c1), (q2, c2)):
                    if ch == "X":
                        yield ("1", q, _N99_H)
                    elif ch == "Y":
                        yield ("1", q, _t100_rot1("X", math.pi / 4))
                yield from _n99_zz(q1, q2, 2 * values[t0 + k], False, False)
                for q, ch in ((q1, c1), (q2, c2)):
                    if ch == "X":
                        yield ("1", q, _N99_H)
                    elif ch == "Y":
                        yield ("1", q, _t100_rot1("X", -math.pi / 4))
            else:  # several terms (and the single-qubit operations between them) on one pair: the Weyl form (item 101)
                CZ101_STATS["weyl_blocks"] += 1
                qa, qb = b[1]
                U = np.eye(4, dtype=complex)
                for m in b[2]:
                    if m[0] == "t":
                        a = values[t0 + m[1]]
                        U = (math.cos(a) * np.eye(4) - 1j * math.sin(a) * self.paulis[m[1]]) @ U
                    else:
                        q, (x, y, z, w) = self._one(m[1], m[3], values)
                        U = _t100_k1(x, y, z, w, q == qa) @ U
                yield from _t100_block_steps(U, qa, qb)
        yield ("end",)
        for q, cb in self.meas:
            yield ("meas", self.perm[q], cb)

    def items(self, values):
        return self._block_items(values)


def _cz101_circuit(qc, target, shape, slots, phys):
    """Item 101: the concrete circuit of an emitted shape and its angles (the same gates, in the same order, as the
    compile template's parametrised circuit bound to them)."""
    out = QuantumCircuit(target.num_qubits, name=qc.name)
    for cr in qc.cregs:
        out.add_register(cr)
    k = 0
    for g in shape:
        nm = g[0]
        if nm == "rz":
            out.rz(slots[k], g[1])
            k += 1
        elif nm == "sx":
            out.sx(g[1])
        elif nm == "x":
            out.x(g[1])
        elif nm == "rx":
            out.rx(slots[k], g[1])
            k += 1
        elif nm == "cz":
            out.cz(g[1], g[2])
        elif nm == "rzz":
            out.rzz(slots[k], g[1], g[2])
            k += 1
        else:
            out.measure(g[1], g[2])
    out._layout = _n99_layout(qc, out, phys)
    return out


def _cz101_emit(qc, target, info):
    """Item 101: items 87/91's CZ path emitted by PSF-Zero (`_CZ101Program`, runs merged as item 99's, a run with
    theta = pi as one x pulse when the Target has x). No pass manager, no comparison, no simulation."""
    p = _CZ101Program(info)
    use_x = "x" in target.operation_names
    shape, slots = _t100_emit(p.items(p.values0), p.phys, target, "rx" in target.operation_names, use_x)
    CZ101_STATS["emitted"] += 1
    return _cz101_circuit(qc, target, shape, slots, p.phys)


# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on prototype c62 (2026-10-15): item 100 as in c61 with a planned binding (c63) -- a circuit inside
# the scope is built once (the formula path: item 87's chain, item 91's graph or item 99's native gates); new angles are
# bound in closed form only (no pass manager, no simulation); a shape signature guards every value-dependent branch;
# each shape keeps its emission plan, so a bind multiplies each run's factors and merges it without re-emitting.
# ---------------------------------------------------------------------------------------------------------------------

TEMPLATE100_STATS = {"built": 0, "bound": 0, "rebuilt": 0, "switched": 0, "structure_mismatch": 0,
                     "block_not_exact": 0}
TEMPLATE_CACHE = 256  # item 100: shapes kept per template (a shape seen before is not built again)
_T100_HOOK = [None]  # item 100: set while compile_template reads items 87/91's structure (they stop before Qiskit)


_T100_RZPI = _n99_rz(math.pi)  # c63: a kept object, so that runs of constant factors are recognised


class _T100Captured(Exception):
    """Item 100: raised by the hook inside items 87/91 with the structure they built (nothing is compiled)."""

    def __init__(self, info):
        super().__init__("item 100: structure captured")
        self.info = info


def _t100_hook(info):
    raise _T100Captured(info)


def _t100_params(op):
    return [float(p) for p in op.params] if op.params else []


class _T100Native:
    """Item 100: item 99's construction (`_native_brickwork`) split into its structure (placement, order, fixed gates'
    native forms -- decided once) and its values (the input's gate parameters, in the input's order)."""
    kind = "native"

    def __init__(self, qc, target, max_error):
        got = _n99_read(qc)
        if got is None:
            raise _T100NotHere("native: not read")
        ops, meas = got
        n = qc.num_qubits
        edges = sorted({(min(qs), max(qs)) for _, qs, _, _ in ops if len(qs) == 2})
        deg = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        if deg and max(deg.values()) > 3:
            raise _T100NotHere("native: degree")
        col = _graph_colours(edges) if edges else {}
        lat = _chain_lattice(target, max_error)
        if col is None or lat is None:
            raise _T100NotHere("native: no colouring or no lattice")
        K, kadj, fadj, _ = lat
        place = {}
        if edges:
            place = _graph_place(n, edges, K, fadj) or _graph_place(n, edges, K, kadj)
            if place is None:
                raise _T100NotHere("native: no embedding")
        free = [q for q in sorted(K) if q not in set(place.values())]
        if n - len(place) > len(free):
            raise _T100NotHere("native: no embedding")
        self.phys = [place[q] if q in place else free.pop(0) for q in range(n)]
        maxall, maxnd, level = {}, {}, []
        for _, qs, _, dg in ops:
            lv = max([(maxnd if dg else maxall).get(q, -1) for q in qs]) + 1
            level.append(lv)
            for q in qs:
                maxall[q] = max(maxall.get(q, -1), lv)
                if not dg:
                    maxnd[q] = max(maxnd.get(q, -1), lv)

        def key(k):
            qs = ops[k][1]
            if len(qs) == 2:
                return (level[k], 0, col[(min(qs), max(qs))], k)
            return (level[k], 1, 0, k)
        self.order = sorted(range(len(ops)), key=key)
        self.ops, self.meas, self.qc = ops, meas, qc
        self.frac = "rzz" in target.operation_names
        self.slices, pos = [], 0
        for _, _, op, _ in ops:
            m = len(op.params)
            self.slices.append((pos, pos + m) if m else None)
            pos += m
        self.nvalues = pos
        self.fixed = {}
        self.fixed1 = {k: _n99_t(op) for k, (_, qs, op, _) in enumerate(ops) if len(qs) == 1 and not op.params}
        for k, (nm, qs, op, dg) in enumerate(ops):
            if len(qs) == 2 and not op.params:
                st = _n99_steps(nm, qs, op, dg, self.frac)
                if st is None:
                    raise _T100NotHere("native: block not exact")
                self.fixed[k] = st
        self.input_sig = ("native",) + tuple((nm, qs, dg) for nm, qs, _, dg in ops) + (tuple(meas),)

    def values_of(self, qc):
        got = _n99_read(qc)
        if got is None or ("native",) + tuple((nm, qs, dg) for nm, qs, _, dg in got[0]) + (tuple(got[1]),) != \
                self.input_sig:
            TEMPLATE100_STATS["structure_mismatch"] += 1
            raise ValueError("item 100: the circuit's structure differs from the template's")
        return [v for _, _, op, _ in got[0] for v in _t100_params(op)]

    def items(self, values):
        ops, sl, frac = self.ops, self.slices, self.frac
        for k in self.order:
            nm, qs, op, dg = ops[k]
            s = sl[k]
            if len(qs) == 1:
                yield ("1", qs[0], self.fixed1[k] if s is None else _n99_t(_T100Op(op, values[s[0]:s[1]])))
                continue
            st = self.fixed.get(k)
            if st is None:
                st = _n99_steps(nm, qs, _T100Op(op, values[s[0]:s[1]]), dg, frac)
                if st is None:
                    TEMPLATE100_STATS["block_not_exact"] += 1
                    raise TranspilerError("item 100: a gate's native form failed its check")
            yield from st
        yield ("end",)
        for q, cb in self.meas:
            yield ("meas", q, cb)


class _T100NotHere(Exception):
    """Item 100: this formula path does not build the input (the next one is tried)."""


class _T100Formula:
    """Item 100: items 87/91's structure (as they read and place the input; captured before their Qiskit stage) and a
    PSF-Zero emission of it that needs no pass manager.

    With rzz and rx in the Target (item 88's frame path) the logical circuit, Qiskit's RemoveDiagonalGatesBeforeMeasure
    (applied once at build; the same positions are left out at every bind) and the Pauli X frame are exactly c60's;
    the single-qubit runs are merged by PSF-Zero (`_n99_run_class`) instead of Qiskit's Optimize1qGatesDecomposition.
    Without them (c60: Qiskit's level 3 on the fixed layout), every two-qubit term is a local Clifford around RZZ (by
    `_n99_zz`), consecutive two-qubit terms on one pair are one block (closed form when it is a single term; Qiskit's
    Weyl decomposition or closed-form CZ decomposition otherwise, as in item 99), and diagonal terms directly before
    the measurements are left out (repeatedly; exact for the outcome distribution)."""

    def __init__(self, kind, qc, target, info):
        self.kind, self.qc = kind, qc
        self.n = info["n"]
        self.frame = info["frame"]
        self.phys = list(info["layout"])
        self.prefix, self.terms, self.order = info["prefix"], info["terms"], info["order"]
        self.suffix, self.meas, self.perm = info["suffix"], info["meas"], info["perm"]
        self.use_rx = "rx" in target.operation_names
        # values: prefix gates' parameters, the terms' angles (reader order), suffix gates' parameters
        self.pre_sl, pos = [], 0
        for op, _ in self.prefix:
            m = len(op.params)
            self.pre_sl.append((pos, pos + m) if m else None)
            pos += m
        self.term0 = pos
        pos += len(self.terms)
        self.suf_sl = []
        for op, _ in self.suffix:
            m = len(op.params)
            self.suf_sl.append((pos, pos + m) if m else None)
            pos += m
        self.nvalues = pos
        self.input_sig = self._read_sig(self.prefix, self.terms, self.suffix, self.meas, self.perm, info.get("extra"))
        self.values0 = self._flat(self.prefix, self.terms, self.suffix)
        if self.frame:
            self._build_frame()
        else:
            self._build_blocks()

    @staticmethod
    def _read_sig(prefix, terms, suffix, meas, perm, extra):
        return (tuple((op.name, q, len(op.params)) for op, q in prefix), tuple(lb for lb, _ in terms),
                tuple((op.name, q, len(op.params)) for op, q in suffix), tuple(meas), tuple(perm),
                tuple(extra) if extra is not None else None)

    @staticmethod
    def _flat(prefix, terms, suffix):
        return ([v for op, _ in prefix for v in _t100_params(op)] + [float(a) for _, a in terms]
                + [v for op, _ in suffix for v in _t100_params(op)])

    def values_of(self, qc):
        got = (_chain_read if self.kind == "chain" else _graph_read)(qc)
        if got is not None:
            prefix, meas, terms, _, suffix, perm, extra = got
            if self._read_sig(prefix, terms, suffix, meas, perm, extra) == self.input_sig:
                return self._flat(prefix, terms, suffix)
        TEMPLATE100_STATS["structure_mismatch"] += 1
        raise ValueError("item 100: the circuit's structure differs from the template's")

    def _gate1(self, op, sl, values):
        return _n99_t(op if sl is None else _T100Op(op, values[sl[0]:sl[1]]))

    # -- item 88's frame path ----------------------------------------------------------------------------------------
    def _build_frame(self):
        """The logical circuit exactly as items 87/91 write it (`_chain_rotation`, frame=True), each instruction's
        parameter traced to its source, and the positions Qiskit's RemoveDiagonalGatesBeforeMeasure removes."""
        from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
        log = QuantumCircuit(*self.qc.qregs, *self.qc.cregs)
        src = []
        for j, (op, q) in enumerate(self.prefix):
            log.append(op, [q])
            src.append(("pre", j))
        for k in self.order:
            lb, a = self.terms[k]
            before = len(log.data)
            _chain_rotation(log, lb, a, frame=True)
            new = log.data[before:]
            rot = [i for i, ins in enumerate(new) if ins.operation.name in ("rx", "ry", "rz", "rzz")
                   and len(new) == 1 or ins.operation.name == "rzz"]
            for i, ins in enumerate(new):
                src.append(("term", k) if i in rot else ("fixed",))
        for j, (op, q) in enumerate(self.suffix):
            log.append(op, [q])
            src.append(("suf", j))
        for q, cb in self.meas:
            log.measure(self.perm[q], cb)
            src.append(("meas",))
        # Qiskit's rule, applied here by position: a measurement's predecessor on its qubit is removed when it is a
        # diagonal single-qubit gate, or a diagonal two-qubit gate whose other qubit's successor is a measurement too
        d1 = {"rz", "z", "t", "s", "tdg", "sdg", "u1", "p"}
        d2 = {"cz", "crz", "cu1", "rzz", "cp"}
        qs_of = [tuple(log.find_bit(x).index for x in ins.qubits) for ins in log.data]
        nxt, prev, last = {}, {}, {}
        for i, qs in enumerate(qs_of):
            for q in qs:
                if q in last:
                    nxt[(last[q], q)] = i
                    prev[(i, q)] = last[q]
                last[q] = i
        names = [ins.operation.name for ins in log.data]
        drop = set()
        for i, nm in enumerate(names):
            if nm != "measure":
                continue
            p = prev.get((i, qs_of[i][0]))
            if p is None:
                continue
            if names[p] in d1:
                drop.add(p)
            elif names[p] in d2 and all(names[nxt.get((p, q), -1)] == "measure" if (p, q) in nxt else False
                                        for q in qs_of[p]):
                drop.add(p)
        keep = [i for i in range(len(names)) if i not in drop]
        kept = PassManager([RemoveDiagonalGatesBeforeMeasure()]).run(log)  # checked against Qiskit's own result

        def perq(c, idx=None):
            out = {}
            for k, ins in enumerate(c.data):
                if idx is not None and k not in idx:
                    continue
                for x in ins.qubits:
                    out.setdefault(c.find_bit(x).index, []).append(ins.operation.name)
            return out
        if perq(log, set(keep)) != perq(kept):
            raise _T100NotHere("frame: removal differs from Qiskit's")
        a = qs_of
        self.fixed_m, self.fixed_mc = {}, {}
        for i in keep:
            fixed = src[i][0] == "fixed" or (src[i][0] in ("pre", "suf") and
                                             (self.pre_sl if src[i][0] == "pre" else self.suf_sl)[src[i][1]] is None)
            if fixed and len(a[i]) == 1 and log.data[i].operation.name != "measure":
                m = _n99_t(log.data[i].operation)
                self.fixed_m[i], self.fixed_mc[i] = m, (m[3], m[2], m[1], m[0])  # c63: X m X kept too
        self.log = [(log.data[i].operation, a[i], log.find_bit(log.data[i].clbits[0]).index
                     if log.data[i].clbits else None, src[i], self.fixed_m.get(i), self.fixed_mc.get(i))
                    for i in keep]

    def _frame_items(self, values):
        """W3 option A (c63a): item 88's frame fixed at build. A negative rzz angle becomes X RZZ(|theta|) X on its first
        qubit, locally; a fold (|theta| crossing pi/2) is Rz(pi) on both qubits; the items on both sides of every rzz are
        pinned (`_t63_emit`), so a sign change or a fold alters angles only, never the shape."""
        X = _N99_X
        t0 = self.term0
        for op, qs, cb, s, fm, fmc in self.log:
            nm = op.name
            if nm == "measure":
                yield ("meas", qs[0], cb)
                continue
            if s[0] == "term":
                prm = 2 * values[t0 + s[1]]
            elif s[0] == "pre" or s[0] == "suf":
                sl = (self.pre_sl if s[0] == "pre" else self.suf_sl)[s[1]]
                prm = None if sl is None else values[sl[0]:sl[1]]
            else:
                prm = None
            if nm == "rzz":
                a, b = qs
                th = math.pi - (math.pi - prm) % (2 * math.pi)
                fold = abs(th) > math.pi / 2
                if fold:
                    th -= math.copysign(math.pi, th)
                if abs(th) < 1e-12 and not fold:
                    continue
                # new tuple objects (never constant): the sign and the fold may change from bind to bind. Time order on
                # qubit a: the fold's Rz(pi), then X if theta < 0; on qubit b: the fold's Rz(pi). All pinned.
                Z = _T100_RZPI if fold else _T63A_I
                Pa = _n99_mul((X[0], X[1], X[2], X[3]), Z) if th < 0 else (Z[0], Z[1], Z[2], Z[3])
                yield ("1", a, Pa, 1)
                yield ("1", b, (Z[0], Z[1], Z[2], Z[3]), 1)
                yield ("rzz", a, b, abs(th))
                Q = X if th < 0 else _T63A_I
                yield ("1", a, (Q[0], Q[1], Q[2], Q[3]), 1)  # built anew: a literal tuple would be one kept object
                continue
            q = qs[0]
            if fm is not None:
                yield ("1", q, fm)
                continue
            if s[0] == "term":
                yield ("1", q, _t100_rot1({"rx": "X", "ry": "Y", "rz": "Z"}[nm], prm / 2))
            elif prm is None:
                yield ("1", q, _n99_t(op))
            else:
                yield ("1", q, _n99_t(_T100Op(op, prm)))
        yield ("end",)
    # -- the CZ path: PSF-Zero's own blocks ---------------------------------------------------------------------------
    def _build_blocks(self):
        """The logical program: prefix gates, terms in items 87/91's order, suffix gates, measurements; diagonal
        operations directly before the measurements left out (repeatedly); consecutive two-qubit terms on one pair
        grouped into a block."""
        prog = []  # (kind, qubits, ref, diagonal)
        for j, (op, q) in enumerate(self.prefix):
            prog.append(("pre", (q,), j, _n99_diag(op, 1)))
        for k in self.order:
            lb, _ = self.terms[k]
            sup = _chain_support(lb)
            prog.append(("term", tuple(q for q, _ in sup), k, all(ch == "Z" for _, ch in sup)))
        for j, (op, q) in enumerate(self.suffix):
            prog.append(("suf", (q,), j, _n99_diag(op, 1)))
        measured = {self.perm[q] for q, _ in self.meas}
        if measured:
            while True:
                last = {}
                for i, (_, qs, _, _) in enumerate(prog):
                    for q in qs:
                        last[q] = i
                drop = [i for i in sorted(set(last.values()), reverse=True)
                        if prog[i][3] and all(q in measured and last[q] == i for q in prog[i][1])]
                if not drop:
                    break
                for i in drop:
                    del prog[i]
        # blocks: consecutive two-qubit terms on one pair; single-qubit operations on the pair's qubits between two
        # of its terms are absorbed into the block, those after its last term stay outside (decided here, once)
        seq, open_ = [], {}

        def close(blk):
            for x in blk[1]:
                if open_.get(x) is blk:
                    del open_[x]
            seq.extend(blk[3])
            blk[3] = []
        for kind, qs, ref, _ in prog:
            if kind == "term" and len(qs) == 2:
                pair = tuple(sorted(qs))
                blk = open_.get(qs[0])
                if blk is not None and blk is open_.get(qs[1]) and blk[1] == pair:
                    blk[2].extend(("g",) + h for h in blk[3])
                    blk[3] = []
                    blk[2].append(("t", ref))
                    continue
                for q in qs:
                    if open_.get(q) is not None:
                        close(open_[q])
                blk = ["block", pair, [("t", ref)], []]
                open_[qs[0]] = open_[qs[1]] = blk
                seq.append(blk)
                continue
            q = qs[0] if len(qs) == 1 else None
            if q is not None and open_.get(q) is not None:
                open_[q][3].append((kind, qs, ref))
                continue
            for x in qs:
                if open_.get(x) is not None:
                    close(open_[x])
            seq.append((kind, qs, ref))
        for blk in list({id(v): v for v in open_.values()}.values()):
            close(blk)
        for blk in seq:  # item 100: a block of exactly XX, YY and ZZ (no single-qubit operation) is the canonical gate
            if blk[0] == "block":
                byp = {}
                if all(m[0] == "t" for m in blk[2]) and len(blk[2]) == 3:
                    for m in blk[2]:
                        sup = _chain_support(self.terms[m[1]][0])
                        if len(sup) == 2 and sup[0][1] == sup[1][1]:
                            byp[sup[0][1]] = m[1]
                blk.append((byp["X"], byp["Y"], byp["Z"]) if sorted(byp) == ["X", "Y", "Z"] else None)
        self.prog = seq
        self.sup = [tuple(_chain_support(lb)) for lb, _ in self.terms]  # c63: each term's support, read once
        self.paulis = {}
        for blk in seq:
            if blk[0] == "block":
                qa, qb = blk[1]
                for m in blk[2]:
                    if m[0] == "t":
                        ch = dict(_chain_support(self.terms[m[1]][0]))
                        self.paulis[m[1]] = np.kron(_T100_PAULI[ch.get(qb, "I")], _T100_PAULI[ch.get(qa, "I")])

    def _one(self, kind, ref, values):
        """A single-qubit operation of the program: (qubit, 2 x 2 tuple)."""
        if kind == "pre" or kind == "suf":
            op, q = (self.prefix if kind == "pre" else self.suffix)[ref]
            return q, self._gate1(op, (self.pre_sl if kind == "pre" else self.suf_sl)[ref], values)
        (q, ch), = self.sup[ref]
        return q, _t100_rot1(ch, values[self.term0 + ref])

    def _block_items(self, values):
        t0 = self.term0
        # c63: every general block's 4 x 4 matrix first, then their Weyl decompositions with one batched check
        gen = []
        for b in self.prog:
            if b[0] == "block" and len(b[2]) > 1 and b[4] is None:
                qa, qb = b[1]
                U = _T63_I4
                for m in b[2]:
                    if m[0] == "t":
                        a = values[t0 + m[1]]
                        U = (math.cos(a) * _T63_I4 - 1j * math.sin(a) * self.paulis[m[1]]) @ U
                    else:
                        q, (x, y, z, w) = self._one(m[1], m[3], values)
                        U = _t100_k1(x, y, z, w, q == qa) @ U
                gen.append((U, qa, qb))
        gen = iter(_t63_blocks(gen))
        for b in self.prog:
            if b[0] != "block":
                q, M = self._one(b[0], b[2], values)
                yield ("1", q, M)
            elif len(b[2]) == 1:  # one two-qubit term: a local Clifford around RZZ(2a), exact in closed form
                k = b[2][0][1]
                (q1, c1), (q2, c2) = self.sup[k]
                for q, ch in ((q1, c1), (q2, c2)):
                    if ch == "X":
                        yield ("1", q, _N99_H)
                    elif ch == "Y":
                        yield ("1", q, _t100_rot1("X", math.pi / 4))
                yield from _n99_zz(q1, q2, 2 * values[t0 + k], False, False)
                for q, ch in ((q1, c1), (q2, c2)):
                    if ch == "X":
                        yield ("1", q, _N99_H)
                    elif ch == "Y":
                        yield ("1", q, _t100_rot1("X", -math.pi / 4))
            elif b[4] is not None:  # exactly XX, YY and ZZ on one pair (they commute): the canonical gate itself
                al, be, ga = (values[t0 + k] for k in b[4])
                yield from _t100_canonical(b[1][0], b[1][1], -al, -be, -ga)
            else:  # several terms (and the single-qubit operations between them) on one pair: item 99's general rule
                for st in next(gen):  # c63: a Weyl block's single-qubit factors pin their run
                    yield st + (1,) if st[0] == "1" else st
        yield ("end",)
        for q, cb in self.meas:
            yield ("meas", self.perm[q], cb)

    def items(self, values):
        return self._frame_items(values) if self.frame else self._block_items(values)


import itertools as _itertools

_T63_I4 = np.eye(4, dtype=complex)


def _t63_blocks(blocks):
    """Item 100 (c63): `_t100_block_steps` for several blocks [(U, q0, q1)], with the check of every closed-form
    three-CNOT block (the Weyl decomposition reproduces U to item 99's block tolerance) done in one numpy batch.
    A block with c = 0 (one interaction, or two cz) takes `_t100_block_steps` itself, with its own check."""
    from qiskit.synthesis import TwoQubitWeylDecomposition
    out, chk = [], []
    t = lambda m: (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))  # noqa: E731
    for U, q0, q1 in blocks:
        d = TwoQubitWeylDecomposition(U, fidelity=None)
        if abs(d.c) < _N99_TOL:
            out.append(_t100_block_steps(U, q0, q1))
            continue
        out.append([("1", q0, t(d.K2r)), ("1", q1, t(d.K2l))] + _t100_canonical(q0, q1, d.a, d.b, d.c) +
                   [("1", q0, t(d.K1r)), ("1", q1, t(d.K1l))])
        chk.append((U, d))
    if chk:
        n = len(chk)
        st = lambda f: np.array([f(d) for _, d in chk])  # noqa: E731
        k1l, k1r, k2l, k2r = st(lambda d: d.K1l), st(lambda d: d.K1r), st(lambda d: d.K2l), st(lambda d: d.K2r)
        K1 = (k1l[:, :, None, :, None] * k1r[:, None, :, None, :]).reshape(n, 4, 4)
        K2 = (k2l[:, :, None, :, None] * k2r[:, None, :, None, :]).reshape(n, 4, 4)
        abc = np.array([(d.a, d.b, d.c) for _, d in chk])
        Nd = np.exp(1j * (abc @ _T100_XYZ))
        W = K1 @ ((_T100_MAGIC[None] * Nd[:, None, :]) @ _T100_MAGIC.conj().T) @ K2
        Us = np.array([U for U, _ in chk])
        fu, fw = Us.reshape(n, 16), W.reshape(n, 16)
        k = np.argmax(np.abs(fu), axis=1)
        r = np.arange(n)
        ph = fu[r, k] / fw[r, k]
        ok = (np.abs(np.abs(ph) - 1) < _N99_BLOCK_TOL) & \
            (np.max(np.abs(fu - ph[:, None] * fw), axis=1) < _N99_BLOCK_TOL)
        if not ok.all():
            TEMPLATE100_STATS["block_not_exact"] += 1
            raise TranspilerError("item 100: a block's native form failed its check")
    return out


def _t63_full(U, use_rx):
    """Item 100 (c63): a pinned run -- one with a factor of a Weyl block -- in the full form whatever its values
    (rz sx rz sx rz, or rz rx rz with rx; exact for every U, theta = 0 included), so that a Weyl decomposition's
    discrete branches do not change the template's shape."""
    theta, phi, lam = _n99_zyz(U)
    if use_rx:
        return "full", (lam - _N99_PI / 2, theta, phi + _N99_PI / 2)
    return "full", (lam, theta + _N99_PI, phi + _N99_PI)


def _t63_emit(items, phys, target, use_rx, use_x=False):
    """Item 100 (c63): `_t100_emit` for templates. An item ("1", q, M, 1) pins its run to the full form
    (`_t63_full`); every other run is emitted exactly as `_t100_emit` emits it."""
    shape, slots, pending, pinned = [], [], {}, set()
    cz_q, rzz_q = target["cz"], (target["rzz"] if "rzz" in target.operation_names else None)

    def flush(q):
        U = pending.pop(q, None)
        if U is None:
            return
        p = phys[q]
        if q in pinned:
            pinned.discard(q)
            _, ang = _t63_full(U, use_rx)
            if use_rx:
                shape.extend((("rz", p), ("rx", p), ("rz", p)))
            else:
                shape.extend((("rz", p), ("sx", p), ("rz", p), ("sx", p), ("rz", p)))
            slots.extend(ang)
            return
        cls, ang = _n99_run_class(U, use_rx)
        if use_x and cls == "full" and abs(ang[1] - 2 * math.pi) < _N99_TOL:  # theta = pi: one x pulse
            shape.extend((("rz", p), ("x", p), ("rz", p)))
            slots.extend((ang[0] - math.pi / 2, ang[2] - math.pi / 2))
            return
        if cls == "z":
            shape.append(("rz", p))
            slots.append(ang[0])
        elif cls == "half":
            shape.extend((("rz", p), ("sx", p), ("rz", p)))
            slots.extend(ang)
        elif cls == "full":
            if use_rx:
                shape.extend((("rz", p), ("rx", p), ("rz", p)))
            else:
                shape.extend((("rz", p), ("sx", p), ("rz", p), ("sx", p), ("rz", p)))
            slots.extend(ang)
    for it in items:
        k = it[0]
        if k == "1":
            q = it[1]
            pending[q] = _n99_mul(it[2], pending.get(q, _T100_ONE))
            if len(it) > 3:
                pinned.add(q)
        elif k == "cz" or k == "rzz":
            a, b = it[1], it[2]
            flush(a)
            flush(b)
            pa, pb = phys[a], phys[b]
            if (pa, pb) not in (cz_q if k == "cz" else rzz_q):
                pa, pb = pb, pa
            shape.append((k, pa, pb))
            if k == "rzz":
                slots.append(it[3])
        elif k == "end":
            for q in sorted(pending):
                flush(q)
        else:  # ("meas", q, clbit)
            flush(it[1])
            shape.append(("measure", phys[it[1]], it[2]))
    return tuple(shape), slots


class CompileTemplate:
    """Item 100: a circuit inside PSF-Zero's scope compiled once into a parametrised circuit on the device.

    `bind_values(values)` returns the compiled circuit for new values, computed in closed form (single-qubit runs by
    ZYZ, two-qubit forms as at build) and assigned with Qiskit's assign_parameters: no pass manager, no simulation.
    `values_of(qc)` reads the values from a circuit of the same structure (item 99: every gate parameter, in the
    input's order; items 87/91: the reader's prefix-gate parameters, term angles and suffix-gate parameters).
    `bind(qc)` = `bind_values(values_of(qc))`. `signature` is the input's structure and the output's shape; if new
    values change the shape (a value-dependent branch: a single-qubit run's class, an rzz folded or dropped, item 88's
    frame, a block's CZ count), the circuit of the new shape is built and counted (TEMPLATE100_STATS["rebuilt"]), or
    taken from the template's cache of shapes already built (["switched"]); a wrong output is never returned for the
    old shape."""

    def __init__(self, gen, target, seed=0):
        self._gen, self.target, self.kind, self.seed = gen, target, gen.kind, seed
        self.use_rx = "rx" in target.operation_names
        # item 100: on the CZ path of items 87/91 a run with theta = pi is one x pulse (as Qiskit's single-qubit
        # synthesis there gives); item 99's own runs stay exactly as c60 emits them
        self.use_x = gen.kind != "native" and not gen.frame and "x" in target.operation_names
        self._cache = {}  # shape -> parametrised circuit (TEMPLATE_CACHE shapes at most, oldest dropped first)
        # c63: items 87/91's runs are merged in one numpy batch per bind (their outputs are checked for exactness, not
        # compared value for value); item 99's runs keep CPython's arithmetic, bit for bit c60's
        self.batch = gen.kind != "native"
        self._build(gen.values0 if hasattr(gen, "values0") else None)
        TEMPLATE100_STATS["built"] += 1

    def _circuit(self, shape, plan):
        """The parametrised circuit of an output shape: a run of constant factors (`plan`) with its angles as numbers,
        every other angle a parameter (in slot order)."""
        from qiskit.circuit import ParameterVector
        pv = ParameterVector("psf_t100", plan["nslots"])
        fixed = plan["fixed"]  # shape slot position -> constant angle
        remap = plan["remap"]  # shape slot position -> parameter index
        qc = self._gen.qc
        out = QuantumCircuit(self.target.num_qubits, name=qc.name)
        for cr in qc.cregs:
            out.add_register(cr)
        k = 0

        def ang():
            nonlocal k
            v = fixed[k] if k in fixed else pv[remap[k]]
            k += 1
            return v
        for g in shape:
            nm = g[0]
            if nm == "rz":
                out.rz(ang(), g[1])
            elif nm == "sx":
                out.sx(g[1])
            elif nm == "x":
                out.x(g[1])
            elif nm == "rx":
                out.rx(ang(), g[1])
            elif nm == "cz":
                out.cz(g[1], g[2])
            elif nm == "rzz":
                out.rzz(ang(), g[1], g[2])
            else:
                out.measure(g[1], g[2])
        out._layout = _n99_layout(qc, out, self._gen.phys)
        return out

    def _build(self, values, items=None, shape=None, slots=None):
        if values is None:
            values = self._gen.values_of(self._gen.qc)
        if items is None:
            items = list(self._gen.items(values))
        if shape is None:
            shape, slots = _t63_emit(iter(items), self._gen.phys, self.target, self.use_rx, self.use_x)
        kinds = tuple(it[0] for it in items)
        key = (kinds, shape)  # c63: constant runs depend on the items' kinds as well as on the shape
        hit = self._cache.get(key)
        if hit is None:
            again = list(self._gen.items(values))  # a kept object is the same object in both: a constant factor
            const = [it[0] == "1" and it[2] is ag[2] for it, ag in zip(items, again)]
            plan = self._plan(items, const, slots)
            hit = (self._circuit(shape, plan), plan)
            if len(self._cache) >= TEMPLATE_CACHE:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = hit
            new = True
        else:
            new = False
        self.circuit, self._planned = hit
        self._shape, self._values0 = shape, list(values)
        self._slots0 = [slots[i] for i in sorted(self._planned["remap"], key=self._planned["remap"].get)]
        self.signature = (self.kind, self._gen.input_sig, shape)
        return new

    def _run(self, U, pin=False):
        """Item 100 (c63): a merged run's class and angles exactly as `_t63_emit` emits them."""
        if pin:
            return _t63_full(U, self.use_rx)
        cls, ang = _n99_run_class(U, self.use_rx)
        if self.use_x and cls == "full" and abs(ang[1] - 2 * math.pi) < _N99_TOL:
            return "x", (ang[0] - math.pi / 2, ang[2] - math.pi / 2)
        return cls, ang

    def _plan(self, items, const, slots):
        """Item 100 (c63): the emission plan of a shape -- every run's factors (item positions, in the order
        `_t100_emit` multiplies them), class and slots; a run of constant factors only is fixed now (its angles become
        numbers in the circuit); every rzz's item and slot; the items' kinds."""
        pending, runs, rzz, pos = {}, [], [], 0
        fixed, remap, nvar = {}, {}, 0

        def flush(q):
            nonlocal pos, nvar
            idx = pending.pop(q, None)
            if idx is None:
                return
            U = _T100_ONE
            for i in idx:
                U = _n99_mul(items[i][2], U)
            pin = any(len(items[i]) > 3 for i in idx)
            cls, ang = self._run(U, pin)
            if all(const[i] for i in idx):
                for j, x in enumerate(ang):
                    fixed[pos + j] = x
            else:
                runs.append((tuple(idx), cls, nvar, len(ang), pin))
                for j in range(len(ang)):
                    remap[pos + j] = nvar + j
                nvar += len(ang)
            pos += len(ang)
        for i, it in enumerate(items):
            k = it[0]
            if k == "1":
                pending.setdefault(it[1], []).append(i)
            elif k == "cz" or k == "rzz":
                flush(it[1])
                flush(it[2])
                if k == "rzz":
                    rzz.append((i, nvar))
                    remap[pos] = nvar
                    nvar += 1
                    pos += 1
            elif k == "end":
                for q in sorted(pending):
                    flush(q)
            else:
                flush(it[1])
        if pos != len(slots):
            raise TranspilerError("item 100: the emission plan does not match the shape")
        return dict(kinds=[it[0] for it in items], runs=runs, rzz=rzz, nslots=nvar, fixed=fixed, remap=remap)

    def _batch_plan(self, p):
        """Item 100 (c63): a plan's runs as index arrays for `_batch` (made once per shape)."""
        runs = p["runs"]
        need, rows = [], []
        for idx, *_ in runs:
            rows.append(list(range(len(need), len(need) + len(idx))))
            need.extend(idx)
        lmax = max((len(r) for r in rows), default=0)
        IDX = np.full((len(runs), lmax), len(need), dtype=np.intp)  # padded with the identity (the last row)
        for k, r in enumerate(rows):
            IDX[k, :len(r)] = r
        code = {"none": 0, "z": 1, "half": 2, "full": 3, "x": 4}
        want = np.array([5 if r[4] else code[r[1]] for r in runs], dtype=np.int8)
        groups = {}
        for k, r in enumerate(runs):
            groups.setdefault(int(want[k]), []).append(k)
        groups = {c: (np.array(ks, dtype=np.intp), np.array([runs[k][2] for k in ks], dtype=np.intp))
                  for c, ks in groups.items()}
        b = dict(need=need, IDX=IDX.T.copy(), want=want, groups=groups,
                 rzz_i=[i for i, _ in p["rzz"]], rzz_pos=np.array([q for _, q in p["rzz"]], dtype=np.intp))
        p["batch"] = b
        return b

    def _batch(self, items, p):
        """Item 100 (c63): every planned run's product, ZYZ angles and class in one numpy batch (as `_n99_zyz` and
        `_n99_run_class`, to floating-point rounding); None if a run's class differs from the plan's (the shape
        changed: the caller re-emits)."""
        b = p.get("batch") or self._batch_plan(p)
        slots = np.zeros(p["nslots"])
        if b["rzz_i"]:
            slots[b["rzz_pos"]] = [items[i][3] for i in b["rzz_i"]]
        if not b["need"]:
            return slots.tolist()
        need = b["need"]
        A = np.fromiter(_itertools.chain.from_iterable([items[i][2] for i in need] + [_T100_ONE]), complex,
                        4 * len(need) + 4).reshape(-1, 4)
        IDX = b["IDX"]
        M = A[IDX[0]]
        a, bb, c, d = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
        for row in IDX[1:]:
            M = A[row]
            m0, m1, m2, m3 = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
            a, bb, c, d = m0 * a + m1 * c, m0 * bb + m1 * d, m2 * a + m3 * c, m2 * bb + m3 * d
        sq = np.sqrt(a * d - bb * c)
        a, c, d = a / sq, c / sq, d / sq
        ac, aa = np.abs(c), np.abs(a)
        pd, pc = np.angle(d), np.angle(c)
        diag, anti = ac <= 1e-15, (aa <= 1e-15) & (ac > 1e-15)
        theta = np.where(diag, 0.0, np.where(anti, _N99_PI, 2 * np.arctan2(ac, aa)))
        phi = np.where(diag, 2 * pd, np.where(anti, 2 * pc, pd + pc))
        lam = np.where(diag | anti, 0.0, pd - pc)
        z = np.fmod(phi + lam, 2 * _N99_PI)
        z = np.where(z <= -_N99_PI, z + 2 * _N99_PI, np.where(z > _N99_PI, z - 2 * _N99_PI, z))
        small = np.abs(theta) < _N99_TOL
        if self.use_rx:
            got = np.where(small, np.where(np.abs(z) < _N99_TOL, 0, 1), 3)
        else:
            got = np.where(small, np.where(np.abs(z) < _N99_TOL, 0, 1),
                           np.where(np.abs(theta - _N99_PI / 2) < _N99_TOL, 2, 3))
            if self.use_x:
                got = np.where((got == 3) & (np.abs(theta + _N99_PI - 2 * _N99_PI) < _N99_TOL), 4, got)
        want = b["want"]
        got = np.where(want == 5, 5, got)
        if not np.array_equal(got, want):
            return None
        for cl, (ks, pos) in b["groups"].items():
            if cl == 1:
                slots[pos] = z[ks]
            elif cl == 2:
                slots[pos] = lam[ks] - _N99_PI / 2
                slots[pos + 1] = phi[ks] + _N99_PI / 2
            elif cl == 4:
                slots[pos] = lam[ks] - _N99_PI / 2
                slots[pos + 1] = phi[ks] + _N99_PI / 2
            elif cl == 3 or cl == 5:
                if self.use_rx:
                    slots[pos] = lam[ks] - _N99_PI / 2
                    slots[pos + 1] = theta[ks]
                    slots[pos + 2] = phi[ks] + _N99_PI / 2
                else:
                    slots[pos] = lam[ks]
                    slots[pos + 1] = theta[ks] + _N99_PI
                    slots[pos + 2] = phi[ks] + _N99_PI
        return slots.tolist()

    def values_of(self, qc):
        return self._gen.values_of(qc)

    def bind_values(self, values):
        values = [float(v) for v in values]
        if len(values) != self._gen.nvalues:
            raise ValueError(f"item 100: {self._gen.nvalues} values expected, {len(values)} given")
        items = list(self._gen.items(values))
        p = self._planned
        slots = None
        if self.batch and [it[0] for it in items] == p["kinds"]:
            slots = self._batch(items, p)
        elif [it[0] for it in items] == p["kinds"]:  # c63: the planned emission, the same arithmetic as _t100_emit
            slots = [0.0] * p["nslots"]  # constant runs are already numbers in the circuit
            mul, one, run = _n99_mul, _T100_ONE, self._run
            for idx, cls, pos, m, pin in p["runs"]:
                U = one
                for i in idx:
                    U = mul(items[i][2], U)
                c, ang = run(U, pin)
                if c != cls:
                    slots = None
                    break
                slots[pos:pos + m] = ang
            if slots is not None:
                for i, pos in p["rzz"]:
                    slots[pos] = items[i][3]
        if slots is None:  # a value-dependent branch changed the shape: built anew, or taken from the cache
            shape, full = _t63_emit(iter(items), self._gen.phys, self.target, self.use_rx, self.use_x)
            if self._build(values, items, shape, full):
                TEMPLATE100_STATS["rebuilt"] += 1
            else:
                TEMPLATE100_STATS["switched"] += 1
            slots = self._slots0
        out = self.circuit.assign_parameters(slots)
        TEMPLATE100_STATS["bound"] += 1
        return out

    def bind(self, qc):
        return self.bind_values(self.values_of(qc))


def compile_template(qc, target, seed_transpiler=0, prune_max_error=0.5):
    """Item 100: the CompileTemplate of `qc` on the device `target`, by the formula path c60's default call takes
    (item 96's scope, then item 87's chain, item 91's graph, item 99's native gates). OutOfScopeError where c60
    refuses."""
    if target is None:
        raise OutOfScopeError("item 100: a template needs the device's Target (changelog item 98)")
    max_error = prune_max_error
    edges, reason = _scope96_edges(qc)
    if not (reason is None and (not edges or _scope96_embeds(edges, qc.num_qubits, target, max_error))):
        raise OutOfScopeError("item 100: outside PSF-Zero's scope (changelog item 98)")
    if "cz" not in target.operation_names:
        raise OutOfScopeError("item 100: the device's two-qubit gate is not cz")
    fe, fq = _failed_elements(target, max_error)
    for kind, fn in (("chain", _chain_brickwork), ("graph", _graph_brickwork)):
        info = None
        _T100_HOOK[0] = _t100_hook
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn(qc, target, max_error, seed_transpiler)
        except _T100Captured as e:
            info = e.info
        finally:
            _T100_HOOK[0] = None
        if info is None:
            continue
        try:
            tpl = CompileTemplate(_T100Formula(kind, qc, target, info), target, seed_transpiler)
        except _T100NotHere:
            continue
        if not _uses_failed(tpl.circuit, fe, fq):
            return tpl
    try:
        tpl = CompileTemplate(_T100Native(qc, target, max_error), target, seed_transpiler)
    except _T100NotHere:
        tpl = None
    if tpl is not None and not _uses_failed(tpl.circuit, fe, fq):
        return tpl
    raise OutOfScopeError("item 100: inside the scope, but no formula reads this circuit (changelog items 98-99)")

_T63A_I = (1 + 0j, 0j, 0j, 1 + 0j)  # W3 option A (c63a): the identity, copied into new tuples by _frame_items

# ======================================================================================================================
# Item 103 (prototype c66, home, 2026-10-16; release 2026-10-16.2): item 88's frame in a template by segments, solved over GF(2).
#
# Option A (c63a, kept by the owner) makes every rzz's frame local: a negative angle becomes X RZZ(|theta|) X on its
# first qubit, and the runs on both sides are pinned to the full form, so a sign change never changes the shape. Its
# cost is an extra pulse in every pinned run that would otherwise be z-only or empty: on fractional targets, same-sign
# inputs are up to 1.21-1.35 x c62's duration (C63A-VAL, Addendum 507).
#
# Item 103 keeps the shape fixed without those pulses. On every qubit, the runs that are not diagonal ("anchors": a
# term's rx or ry, a non-diagonal prefix or suffix gate, a non-diagonal fixed gate) cut its timeline into segments. A
# segment between two anchors may be conjugated by X as a whole: the two X's merge into the anchors' runs (their class
# does not change: X U has the same |U_00| as U up to the order of the factors) and every rzz inside it on that qubit
# changes sign. Which segments to conjugate, so that every rzz angle is positive, is a linear system over GF(2): one
# variable per inner segment, one equation per rzz (c_a + c_b = [theta < 0]). Its spanning forest is found once, at
# build; a bind solves it by one pass over the forest's edges.
#
# - An rzz whose qubits are each before their first anchor (a basis state: the rzz is a phase) or, in a measured
#   circuit, after their last anchor (only diagonal operations follow until the measurement) needs no positive angle:
#   it is emitted with |theta|.
# - An rzz that closes a cycle of the forest, or joins two fixed segments, is emitted as in option A (pinned, local X).
#   On a chain there are none.
# - A fold (|theta| > pi/2) is Rz(pi) on both qubits, a Z frame carried to the next anchor (or dropped on a basis state
#   or before a measurement); in the pinned case, as in option A.
#
# The items' kinds and every run's class are the same for every value (up to values of measure zero), so a template
# never changes its shape. Exactness is checked by the prototype's tests against the input, not assumed.
# ----------------------------------------------------------------------------------------------------------------------
_T103_I = (1 + 0j, 0j, 0j, 1 + 0j)
_T103_Z = (1 + 0j, 0j, 0j, -1 + 0j)
_T103_TOL = 1e-12
T103_STATS = {"plans": 0, "tree": 0, "free": 0, "pinned": 0, "anchors": 0}


def _t103_new(M):
    """A new tuple object with M's entries (never a kept object: `CompileTemplate._build` treats an item that is the
    same object in two generations as a constant factor)."""
    return (M[0] + 0j, M[1] + 0j, M[2] + 0j, M[3] + 0j)


def _t103_xc(M):
    """X M X, as a new tuple."""
    return (M[3] + 0j, M[2] + 0j, M[1] + 0j, M[0] + 0j)


def _t103_bound(c, z):
    """X^c Z^z as a new tuple (the order only changes a global phase)."""
    if c and z:
        return (0j, -1 + 0j, 1 + 0j, 0j)
    if c:
        return (0j, 1 + 0j, 1 + 0j, 0j)
    if z:
        return (1 + 0j, 0j, 0j, -1 + 0j)
    return (1 + 0j, 0j, 0j, 1 + 0j)


def _t103_matrix(self, op, s, fm, values):
    """A single-qubit log entry's matrix at `values` (as `_frame_items` emits it, without a frame)."""
    nm = op.name
    if fm is not None:
        return fm
    if s[0] == "term":
        return _t100_rot1({"rx": "X", "ry": "Y", "rz": "Z"}[nm], values[self.term0 + s[1]])
    if s[0] == "pre" or s[0] == "suf":
        sl = (self.pre_sl if s[0] == "pre" else self.suf_sl)[s[1]]
        return _n99_t(op) if sl is None else _n99_t(_T100Op(op, values[sl[0]:sl[1]]))
    return _n99_t(op)


def _t103_plan(self):
    """Item 103: the segments, the anchors and the rzz equations of this template's log, and the spanning forest that
    solves them (made once, from the build values)."""
    values = self.values0
    n = self.n
    seg = [0] * n
    anchor = set()
    edges = []  # (log index, a, seg_a, b, seg_b)
    measured = False
    # the single-qubit runs (consecutive single-qubit entries on a qubit between its two-qubit entries); a run is an
    # anchor when its product at the build values is neither diagonal nor anti-diagonal (X U, U X and X U X then have
    # its class), and its first non-diagonal entry is where the segments meet
    runs, cur = [], {}
    for i, (op, qs, cb, s, fm, fmc) in enumerate(self.log):
        if op.name == "measure" or len(qs) == 2:
            for q in qs:
                if cur.get(q):
                    runs.append(cur.pop(q))
            continue
        cur.setdefault(qs[0], []).append(i)
    runs.extend(r for r in cur.values() if r)
    for r in runs:
        U, first = _T103_I, None
        for i in r:
            op, qs, cb, s, fm, fmc = self.log[i]
            M = _t103_matrix(self, op, s, fm, values)
            U = _n99_mul(M, U)
            if first is None and (abs(M[1]) > _T103_TOL or abs(M[2]) > _T103_TOL):
                first = i
        if first is not None and min(abs(U[0]), abs(U[1])) > 1e-6:
            anchor.add(first)
    for i, (op, qs, cb, s, fm, fmc) in enumerate(self.log):
        nm = op.name
        if nm == "measure":
            measured = True
            continue
        if nm == "rzz":
            a, b = qs
            edges.append((i, a, seg[a], b, seg[b]))
            continue
        if i in anchor:
            seg[qs[0]] += 1
    last = seg  # number of anchors per qubit: segments 0 .. last[q]; 0 is "start", last[q] (> 0) is "end"

    def kind(q, k):
        if k == 0:
            return "S"
        if k == last[q]:
            return "E"
        return "I"
    node = {}
    for q in range(n):
        for k in range(1, last[q]):
            node[(q, k)] = len(node) + 1  # 0 is the ground (fixed segments, c = 0)
    parent = list(range(len(node) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ekind, enodes, tree_adj = {}, {}, {}
    for i, a, ka, b, kb in edges:
        ta, tb = kind(a, ka), kind(b, kb)
        free_a = ta == "S" or (ta == "E" and measured)
        free_b = tb == "S" or (tb == "E" and measured)
        if free_a and free_b:
            ekind[i] = "free"
            continue
        u = node.get((a, ka), 0)
        v = node.get((b, kb), 0)
        enodes[i] = (u, v)
        ru, rv = find(u), find(v)
        if ru == rv:
            ekind[i] = "pin"
            continue
        parent[ru] = rv
        ekind[i] = "tree"
        tree_adj.setdefault(u, []).append((v, i))
        tree_adj.setdefault(v, []).append((u, i))
    order, seen = [], set()
    for root in [0] + list(range(1, len(node) + 1)):
        if root in seen:
            continue
        seen.add(root)
        stack = [root]
        while stack:
            x = stack.pop()
            for y, i in tree_adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    order.append((y, x, i))
                    stack.append(y)
    # Z frame at the end of a qubit whose last segment is not dropped (an unmeasured circuit): emitted pinned
    end_pin = set()
    if not measured:
        for i, a, ka, b, kb in edges:
            if ekind[i] == "tree":
                for q, k in ((a, ka), (b, kb)):
                    if kind(q, k) == "E":
                        end_pin.add(q)
    T103_STATS["plans"] += 1
    T103_STATS["anchors"] += len(anchor)
    for k in ("tree", "free", "pin"):
        T103_STATS["pinned" if k == "pin" else k] += sum(1 for v in ekind.values() if v == k)
    return dict(anchor=anchor, last=last, node=node, ekind=ekind, enodes=enodes, order=order, measured=measured,
                end_pin=end_pin, nnodes=len(node) + 1)


def _t103_frame_items(self, values):
    """Item 103: item 88's frame by segments (see the comment above); the same output distribution (or state) as
    c63a's, with no pinned run but where a cycle or a fixed segment needs one."""
    plan = getattr(self, "_t103", None)
    if plan is None:
        plan = self._t103 = _t103_plan(self)
    X = _N99_X
    t0 = self.term0
    log = self.log
    ekind, last, node, anchor = plan["ekind"], plan["last"], plan["node"], plan["anchor"]
    # each rzz's angle after the fold, then the segments' conjugations
    ang = {}
    neg = {}
    for i, k in ekind.items():
        th = 2 * values[t0 + log[i][3][1]] if log[i][3][0] == "term" else None
        if th is None:  # an rzz not traced to a term: as written
            th = float(log[i][0].params[0])
        th = math.pi - (math.pi - th) % (2 * math.pi)
        fold = abs(th) > math.pi / 2
        if fold:
            th -= math.copysign(math.pi, th)
        ang[i] = (th, fold)
        neg[i] = th < 0
    c = [0] * plan["nnodes"]
    for y, x, i in plan["order"]:
        c[y] = c[x] ^ neg[i]
    n = self.n
    seg = [0] * n
    zf = [0] * n

    def cseg(q, k):
        v = node.get((q, k))
        return c[v] if v is not None else 0

    def is_inner(q, k):
        return 0 < k < last[q]
    for i, (op, qs, cb, s, fm, fmc) in enumerate(log):
        nm = op.name
        if nm == "measure":
            yield ("meas", qs[0], cb)
            continue
        if nm == "rzz":
            a, b = qs
            th, fold = ang[i]
            k = ekind[i]
            if abs(th) < 1e-12 and not fold:
                continue
            if k == "free":
                yield ("rzz", a, b, abs(th))
                continue
            ca, cb_ = cseg(a, seg[a]), cseg(b, seg[b])
            eff = -th if ca != cb_ else th
            if k == "tree":
                if fold:
                    for q in (a, b):
                        if is_inner(q, seg[q]) or q in plan["end_pin"]:
                            zf[q] ^= 1
                yield ("rzz", a, b, abs(eff))
                continue
            # "pin": option A's local form, with this bind's effective sign
            Z = _T100_RZPI if fold else _T103_I
            Pa = _n99_mul((X[0], X[1], X[2], X[3]), Z) if eff < 0 else _t103_new(Z)
            yield ("1", a, Pa, 1)
            yield ("1", b, _t103_new(Z), 1)
            yield ("rzz", a, b, abs(eff))
            Q = X if eff < 0 else _T103_I
            yield ("1", a, _t103_new(Q), 1)
            continue
        q = qs[0]
        k = seg[q]
        if i in anchor:
            if is_inner(q, k):  # closing a conjugated segment: X^c and the segment's Z frame, into the anchor's run
                yield ("1", q, _t103_bound(cseg(q, k), zf[q]))
            zf[q] = 0
            M = _t103_matrix(self, op, s, fm, values)
            yield ("1", q, M if fm is not None and not is_inner(q, k) and not is_inner(q, k + 1) else _t103_new(M))
            seg[q] = k + 1
            if is_inner(q, k + 1):  # opening the next segment
                yield ("1", q, _t103_bound(cseg(q, k + 1), 0))
            continue
        M = _t103_matrix(self, op, s, fm, values)  # any other single-qubit operation inside segment k
        if is_inner(q, k):
            yield ("1", q, _t103_xc(M) if cseg(q, k) else _t103_new(M))
        else:
            yield ("1", q, M)
    for q in sorted(plan["end_pin"]):
        yield ("1", q, _t103_bound(0, zf[q]), 1)
    yield ("end",)


_T100Formula._frame_items = _t103_frame_items  # item 103 (release 2026-10-16.2)
