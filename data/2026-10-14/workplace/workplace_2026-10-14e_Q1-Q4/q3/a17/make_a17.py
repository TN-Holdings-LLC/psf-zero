"""make_a17.py -- builds the workplace's AI front end 2026-10-14.a17 from a16 by exact substitutions: item 30, every
candidate is checked against the input before item 25 compares them (outcome distribution within 1e-9 in total
variation, or the unitary within a state infidelity of 1e-12), and an inexact candidate is never returned; item 31, a
second ring schedule (constant depth for any n) as one more candidate.
    python make_a17.py <a16 psf_ai_compile.py> <output psf_ai_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub('"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-14.a16\n'
    '(workplace proposal).',
    '"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-14.a17\n'
    '(workplace proposal: a16 + items 30-31).')
sub('''      output is not folded by itself (W3 cells on a fractional FakeTorino, a15: 2 of 64 returned such a circuit).
''', '''      output is not folded by itself (W3 cells on a fractional FakeTorino, a15: 2 of 64 returned such a circuit).
  30. (a17) Every candidate is checked against the input before item 25 compares them, and an inexact one is never
      returned. Measured inputs: the outcome distribution over the input's classical bits, computed exactly
      (statevector of the active qubits), within 1e-9 in total variation. Unmeasured inputs: psf_compile's item 39
      check (`_implements`) at a state infidelity of 1e-12 instead of its EXACT_TOL = 1e-6. Why: a15 returned one
      circuit 1.4e-5 away from the input in total variation (W3, FakeFez fractional, n = 6, L = 12; workplace Q3).
      The cause is Qiskit's two-qubit synthesis, not a tolerance of PSF-Zero's: TwoQubitWeylDecomposition snaps a
      block to a specialised form when the snap costs at most 1e-9 in average gate fidelity, and Qiskit's level 2/3
      (TwoQubitPeepholeOptimization, also UnitarySynthesis) uses it even at approximation_degree = 1.0. A block with
      Weyl coordinates (0.1489, 0.1489, 0) came back with infidelity 8.5e-10, an amplitude error of about 3e-5, and
      item 39's state-infidelity bound (1e-6) cannot see that. A candidate that cannot be checked (more than
      EXACT_MAX_ACTIVE = 20 active qubits, mid-circuit measurement, classical control) stands unchecked (counted).
  31. (a17) A second ring schedule, as one more candidate after item 28's ("ring2"): the same folded order, then
      three rounds with two SWAP layers -- the ring edges already adjacent, SWAPs SA = (1,2), (5,6), ...; the edges
      made adjacent, then SA undone together with SB = (3,4), (7,8), ... (disjoint, one layer); the rest -- and the
      next layer mirrored. Every ring of n >= 4 is covered (checked to n = 199) at CZ-depth 9 and about 3n CZ per
      layer; N2's odd-even transposition rounds grow with n (55 CZ per layer at n = 7, 203 at n = 11). At n = 4 both
      are the same; at n = 6 N2's is shorter (same CZ count), at n = 5, 7, 8 this one (n = 7, L = 8: 7.90 against
      22.75 us). Both are built and item 25 takes the shorter.
''')
sub('AI_COMPILE_VERSION = "2026-10-14.a16"  # workplace proposal: a15 + items 28-29 (ring formula candidate; rx one pulse)',
    'AI_COMPILE_VERSION = "2026-10-14.a17"  # workplace proposal: a15 + items 28-31 (ring formula candidates; rx one pulse; every candidate checked exact)')
sub('''RX_ONE_PULSE = True  # item 29
''', '''RX_ONE_PULSE = True  # item 29
EXACT_CHECK = True  # item 30
EXACT_TVD = 1e-9  # item 30: largest total variation distance accepted between the candidate's and the input's outcomes
EXACT_STATE = 1e-12  # item 30: largest state infidelity accepted for an unmeasured input (item 39's check)
EXACT_MAX_ACTIVE = 20  # item 30: largest number of active qubits simulated for the check
A17_STATS = {"checked": 0, "inexact_dropped": 0, "unchecked": 0, "largest_tvd_dropped": 0.0, "all_inexact": 0}
''')
sub('''def _executable(c, target):''', '''def _outcomes(circ):
    """Item 30: the exact outcome distribution of `circ` over its classical bits (statevector of its active qubits;
    measurements only at the end), or None when it cannot be computed."""
    import numpy as np
    from qiskit.quantum_info import Statevector
    meas, body, done = {}, [], set()
    for ins in circ.data:
        nm = ins.operation.name
        qs = [circ.find_bit(q).index for q in ins.qubits]
        if nm in ("barrier", "delay"):
            continue
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = qs[0]
            done.add(qs[0])
            continue
        if done & set(qs) or ins.operation.num_clbits or not hasattr(ins.operation, "to_matrix") and \\
                getattr(ins.operation, "definition", None) is None:
            return None
        body.append((ins.operation, qs))
    if len(meas) != circ.num_clbits or not meas:
        return None
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    if len(active) > EXACT_MAX_ACTIVE:
        return None
    pos = {q: i for i, q in enumerate(active)}
    red = QuantumCircuit(len(active))
    for op, qs in body:
        red.append(op, [pos[q] for q in qs])
    try:
        return np.asarray(Statevector(red).probabilities([pos[meas[k]] for k in range(circ.num_clbits)]))
    except Exception:  # noqa: BLE001
        return None


def _exact(qc, c, ref=None):
    """Item 30: True if candidate `c` reproduces input `qc` (outcome distribution within EXACT_TVD, or for an
    unmeasured input item 39's check at EXACT_STATE), False if it does not, None if the check cannot be made."""
    if not EXACT_CHECK:
        return None
    if any(i.operation.name == "measure" for i in qc.data):
        p0 = _outcomes(qc) if ref is None else ref
        p1 = _outcomes(c)
        if p0 is None or p1 is None or len(p0) != len(p1):
            return None
        tvd = 0.5 * float(abs(p0 - p1).sum())
        if tvd > EXACT_TVD:
            A17_STATS["largest_tvd_dropped"] = max(A17_STATS["largest_tvd_dropped"], tvd)
            return False
        return True
    if qc.num_qubits > SMALL_MAX_QUBITS:
        return None
    try:
        return bool(pc._implements(qc, c, tol=EXACT_STATE))
    except Exception:  # noqa: BLE001
        return None


def _exact_filter(qc, cands):
    """Item 30: the candidates that are exact or cannot be checked (counted); inexact ones dropped."""
    if not EXACT_CHECK:
        return cands
    ref = _outcomes(qc) if any(i.operation.name == "measure" for i in qc.data) else None
    kept = []
    for lb, c in cands:
        ok = _exact(qc, c, ref)
        if ok is None:
            A17_STATS["unchecked"] += 1
            kept.append((lb, c))
        elif ok:
            A17_STATS["checked"] += 1
            kept.append((lb, c))
        else:
            A17_STATS["checked"] += 1
            A17_STATS["inexact_dropped"] += 1
    return kept


def _executable(c, target):''')
sub('''    cands = [(lb, c) for lb, c in cands if c is not None]
''', '''    cands = [(lb, c) for lb, c in cands if c is not None]
    cands = _exact_filter(qc, cands)  # item 30
''')
sub('''        best, label = _executable(_rx_merged(device_call(qc), target), target), "REL"
''', '''        best, label = _executable(_rx_merged(device_call(qc), target), target), "REL"
        if best is not None and _exact(qc, best) is False:  # item 30: nothing exact was found; counted
            A17_STATS["all_inexact"] += 1
''')
sub('"version": AI_COMPILE_VERSION,', '"version": AI_COMPILE_VERSION, "exact_check": dict(A17_STATS),')

sub("""def _ring_logical(qc, read):""", """def ring_rounds_q4(n):
    \"\"\"Item 31: (folded order of ring indices, rounds [(CZ pairs, SWAPs (position p, p+1))]) in three rounds with two
    SWAP layers: SA = (1,2), (5,6), ...; then SA undone together with SB = (3,4), (7,8), ... .\"\"\"
    order, lo, hi = [], 0, n - 1
    while lo <= hi:
        order.append(lo)
        if hi != lo:
            order.append(hi)
        lo, hi = lo + 1, hi - 1
    edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    occ, rounds = list(order), []

    def adjacent():
        czs = []
        for p in range(n - 1):
            e = (min(occ[p], occ[p + 1]), max(occ[p], occ[p + 1]))
            if e in edges:
                czs.append((occ[p], occ[p + 1]))
                edges.discard(e)
        return czs
    SA = [(p, p + 1) for p in range(1, n - 1, 4)]
    SB = [(p, p + 1) for p in range(3, n - 1, 4)]
    czs = adjacent()
    for sw in (SA, SA + SB):
        if not edges:
            break
        rounds.append((czs, sw))
        for p, p1 in sw:
            occ[p], occ[p1] = occ[p1], occ[p]
        czs = adjacent()
    rounds.append((czs, []))
    if edges:
        raise RuntimeError("item 31: no schedule")
    return order, rounds


def _ring_logical(qc, read, rounds_fn=None):""")
sub("""    order, rounds = ring_rounds(n)
""", """    order, rounds = (rounds_fn or ring_rounds)(n)
""")
sub("""def _ring_candidate(qc, target, basis_gates, seed):""", """def _ring_candidate(qc, target, basis_gates, seed, rounds_fn=None):""")
sub("""    built = _ring_logical(qc, read)
""", """    built = _ring_logical(qc, read, rounds_fn)
""")
sub("""    if ring is not None:
        cands.append(("ring", ring))
""", """    if ring is not None:
        cands.append(("ring", ring))
        with warnings.catch_warnings():  # item 31
            warnings.simplefilter("ignore")
            ring2 = _ring_candidate(qc, target, basis_gates, seed, ring_rounds_q4)
        if ring2 is not None:
            cands.append(("ring2", ring2))
""")
sub("""    if label == "ring":
        A16_STATS["chosen_ring"] += 1""", """    if label in ("ring", "ring2"):
        A16_STATS["chosen_ring"] += 1""")
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
