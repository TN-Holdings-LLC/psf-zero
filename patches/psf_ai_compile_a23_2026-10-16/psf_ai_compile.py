"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-16.a23
(a22 + items 44-45: compare-and-choose replaced by a rule on the ring size; a15/a14, the exactness checks and
every switch removed -- see patches/psf_ai_compile_a23_2026-10-16/make_a23.py). a22's description:
(a21 + items 42-43: embeddable rings and templates through psf_compile's items 99-100). a18's description:
(a17, the workplace's, + items 32-33 at home: no search; what is not built by formula is handed on).

a16 is a15 (patches/psf_ai_compile_a15_2026-10-14) with two additions at or below SMALL_MAX_QUBITS when the device's
Target is given. Without a Target, and above SMALL_MAX_QUBITS, it is a15's own fallback (a14).

Why. W3 and N2 (2026-10-14): heavy-hex has no cycle shorter than 12, so any 4 or 6 qubits induce a forest and every
layer of a CZ ring needs at least one SWAP or a CZ fused with one, whatever the placement. a15's constructions then
lost to a14's searched routing on the 4-qubit classifier, and a15 returned the search. N2's formula -- the ring on a
line in folded order, rounds of CZ and interior odd-even SWAPs, the next layer mirrored -- was shorter than every
other method on 61 of W3's 64 cells.

a16 (with a Target):
  28. A circuit that reads as layers of single-qubit gates, each followed by CZ gates forming one cycle through all
      its n >= 4 qubits (the same cycle every time; the CZs of a layer may come in any order, since they commute),
      then single-qubit gates and measurements at the end, is also built by N2's formula: the cycle's qubits in folded
      order (c0, c[n-1], c1, c[n-2], ...) on a line of the device placed by psf_compile's item 87 rule (W5's cut, the
      longest piece of Heron's snake, centred, else the first VF2 path; couplers of the device's most common gate
      length first), each layer's CZs done round by round with one odd-even transposition round of SWAPs on the
      interior positions between rounds, the next layer running the rounds backwards. Qiskit's level-3 preset
      translates it on that fixed layout without routing (coupling map and basis only). The result is one more
      candidate of item 25, after a15's constructions and before a14's search (item 27, still last). An unmeasured
      circuit with an odd number of ring layers (its qubits end permuted) is not built. Needs psf_compile with item
      87 (`_chain_lattice`, `_chain_place`); without it the candidate is skipped (counted).
  29. With rx in the Target (IBM's fractional gates: rx(theta) in one pulse), every candidate's single-qubit runs are
      merged into rz rx rz (one pulse each; Qiskit's Optimize1qGatesDecomposition with basis rz, rx; exact) before
      item 25 compares them. With rzz in the Target, a candidate with an rzz angle outside IBM's executable range
      0 < theta <= pi/2 is folded by psf_compile's item 86 (`_fold`) and dropped if that fails: a14's searched
      output is not folded by itself (W3 cells on a fractional FakeTorino, a15: 2 of 64 returned such a circuit).
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
  32. (a18) a14's searched routing is no longer a candidate (SEARCH_FALLBACK = False). Why: the owner's scope
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
  34. (a19) Formula only. For a ring classifier only the two ring schedules are built (items 28 and 31), each with
      item 29's rx merge and rzz fold, and the shorter (`_better`) of those on no failed element is returned; a15's
      starts and polish and item 30's per-candidate simulation check are not made (FORMULA_CHECK = True adds item 30's
      check). If neither schedule builds, the input is handed on as item 33 does. Why: on 2026-10-15 a18 spent 0.17-1.2
      s per classifier, about 70% of it on item 30's checks and most of the rest on candidates that lost (the ring won
      63 of 64 W3 cells), against 28-36 ms for the ring formula alone (Addendum 493). FORMULA_ONLY = False restores a18.
  35. (a20) The ring template. `ring_template(qc, target)` builds the ring formula once by PSF-Zero itself, with no
      Qiskit pass manager: the fixed CZ / SWAP blocks are decomposed once and checked to 1e-12, and every
      single-qubit run between CZs is kept as a product of the input's own gates and fixed matrices.
      `RingTemplate.bind(qc)` compiles a circuit of the same structure with other angles by re-synthesising only those
      runs in closed form (ZYZ: rz sx rz sx rz, or rz rx rz with rx in the Target). For a training loop, where only the
      angles change. The owner's request of 2026-10-15 (Addendum 494).
  36. (a20) The gate closed: no input is handed to Qiskit's level 3. Without a Target, OutOfScopeError; an input that
      is not a ring classifier goes to psf_compile's own call, which builds it by formula or refuses it
      (GATE_CLOSED = False restores a19's hand-off).
  37. (a21) One exactness check. `_exact` is psf_compile's item 93 (`_exact93`): the input is first unrolled to its own
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
  44. (a23) No compare-and-choose. The ring formula (items 34, 35) builds one schedule, by a rule on the ring size:
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
"""
import contextlib
import io
import warnings

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

import psf_compile as pc

FAILED_MAX_ERROR = 0.5  # a14 (item 22): as the release's prune_max_error
DEFAULT_SEEDS = (0, 1, 2, 3)  # a14's; only the first is used (item 39)

AI_COMPILE_VERSION = "2026-10-16.a23"  # prototype: items 44-45 (no compare-and-choose, no checks, no switches) + items 42-43 (embeddable rings and templates through psf_compile items 99-100) + items 37-41 (workplace review of a18) + items 35-36 (ring template; gate closed) + item 34 (formula only) on a18, which was: a17 (workplace: items 28-31) + items 32-33 (no search; hand-off outside the ring formula)
RING_MAX_QUBITS = 20  # item 40
A22_STATS = {"ring_embedded": 0, "templates_compiler": 0, "templates_ring": 0, "templates_none": 0}
RING_DIAGONAL = ("cz", "rzz", "cp", "crz", "cs", "csdg")  # item 41: commuting diagonal two-qubit gates
A19_STATS = {"formula": 0, "not_built": 0, "checked": 0, "check_failed": 0}
A18_STATS = {"handed_compiler": 0, "handed_qiskit": 0, "handoff_exact": 0, "handoff_unchecked": 0, "handoff_inexact": 0}
A16_STATS = {"calls": 0, "candidates": 0, "chosen_start": {}, "chosen_search": 0, "chosen_ring": 0,
             "ring_built": 0, "ring_not_read": 0, "ring_no_path": 0, "ring_no_item87": 0, "ring_odd_unmeasured": 0,
             "ring_failed": 0, "rx_merged": 0, "rzz_folded": 0, "rzz_dropped": 0, "failed_skipped": 0, "release_fallback": 0, "compile_failed": 0,
             "search_failed": 0}


# --------------------------------------------------------------------------------------------- item 28: reading

def _ring_read(qc):
    """Item 28: (cycle, segments, tail, meas) when `qc` is layers of single-qubit gates each followed by CZ gates
    forming one cycle through all its n >= 4 qubits (the same cycle each time), then single-qubit gates and final
    measurements; else None. cycle: the qubits in cycle order from the smallest, toward its smaller neighbour;
    segments: per layer, the single-qubit (operation, qubit) before its CZs; tail: those after the last layer;
    meas: [(qubit, clbit)]."""
    n = qc.num_qubits
    if n < 4:
        return None
    segments, cur, block, blocks, tail_meas = [], [], set(), [], []
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
            if qs[0] in touched:  # the layer's CZs on this qubit are done: the layer must be complete
                if len(block) != n:
                    return None
                blocks.append(frozenset(block))
                segments.append(cur)
                cur, block, touched = [], set(), set()
            cur.append((ins.operation, qs[0]))
            continue
        if len(qs) != 2 or nm != "cz":
            return None
        e = (min(qs), max(qs))
        if e in block:
            return None
        block.add(e)
        touched |= set(qs)
    if block:
        if len(block) != n:
            return None
        blocks.append(frozenset(block))
        segments.append(cur)
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
    return cycle, segments, cur, tail_meas


def _ring_read2(qc):
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
    return _ring_read2(qc)


def ring_rounds(n):
    """N2: (folded order of ring indices, rounds [(CZ pairs of ring indices, SWAPs (position p, p+1))])."""
    order, lo, hi = [], 0, n - 1
    while lo <= hi:
        order.append(lo)
        if hi != lo:
            order.append(hi)
        lo, hi = lo + 1, hi - 1
    edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    occ, rounds, parity = list(order), [], 1
    while True:
        czs = []
        for p in range(n - 1):
            e = (min(occ[p], occ[p + 1]), max(occ[p], occ[p + 1]))
            if e in edges:
                czs.append((occ[p], occ[p + 1]))
                edges.discard(e)
        swaps = []
        if edges:
            for p in range(1 if parity else 2, n - 2, 2):
                swaps.append((p, p + 1))
                occ[p], occ[p + 1] = occ[p + 1], occ[p]
            parity ^= 1
        rounds.append((czs, swaps))
        if not edges:
            return order, rounds
        if len(rounds) > 4 * n:
            raise RuntimeError("N2: no schedule")


# --------------------------------------------------------------------------------------------- item 28: building

def ring_rounds_q4(n):
    """Item 31: (folded order of ring indices, rounds [(CZ pairs, SWAPs (position p, p+1))]) in three rounds with two
    SWAP layers: SA = (1,2), (5,6), ...; then SA undone together with SB = (3,4), (7,8), ... ."""
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


def _ring_logical(qc, read, rounds_fn=None):
    """Item 28: the formula as a circuit on qc's own wires, wire w bound to line position p_of_wire[w] for the whole
    circuit (a SWAP moves states, not wires); (circuit, p_of_wire) or None (unmeasured, odd number of layers)."""
    cycle, segments, tail, meas = read[:4]
    ent = read[4] if len(read) > 4 else None  # item 41
    n = qc.num_qubits
    order, rounds = (rounds_fn or ring_rounds)(n)
    L = len(segments)
    if not meas and L % 2 == 1:
        A16_STATS["ring_odd_unmeasured"] += 1
        return None
    ridx = {q: i for i, q in enumerate(cycle)}  # input qubit -> ring index
    wire_at = [cycle[order[p]] for p in range(n)]  # position -> wire (fixed)
    p_of_wire = {w: p for p, w in enumerate(wire_at)}
    pos = {order[p]: p for p in range(n)}  # ring index -> current position
    occ = list(order)
    out = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=qc.global_phase)

    def here(q):  # the wire currently holding input qubit q's state
        return wire_at[pos[ridx[q]]]

    def do_swaps(swaps):
        for p, p1 in swaps:
            out.swap(wire_at[p], wire_at[p1])
            a, b = occ[p], occ[p1]
            occ[p], occ[p1] = b, a
            pos[a], pos[b] = p1, p

    def do_czs(czs, layer):
        for a, b in czs:
            if ent is None:
                out.cz(wire_at[pos[a]], wire_at[pos[b]])
                continue
            op, (qa, qb) = ent[layer][(min(cycle[a], cycle[b]), max(cycle[a], cycle[b]))]  # item 41
            if op.name == "crz":  # not symmetric: its written orientation
                out.append(op, [here(qa), here(qb)])
            else:  # symmetric: oriented as a20 orients its CZ, so that a CZ ring compiles exactly as in a20
                out.append(op, [wire_at[pos[a]], wire_at[pos[b]]])
    for layer, seg in enumerate(segments):
        for op, q in seg:
            out.append(op, [here(q)])
        if layer % 2 == 0:
            for czs, swaps in rounds:
                do_czs(czs, layer)
                do_swaps(swaps)
        else:
            for czs, swaps in reversed(rounds):
                do_swaps(list(reversed(swaps)))
                do_czs(czs, layer)
    for op, q in tail:
        out.append(op, [here(q)])
    for q, cb in meas:
        out.measure(here(q), cb)
    return out, p_of_wire


def _ring_candidate(qc, target, basis_gates, seed, rounds_fn=None):
    """Item 28: N2's formula on a line of the device, or None."""
    read = _ring_read_path(qc)  # item 41
    if read is None:
        A16_STATS["ring_not_read"] += 1
        return None
    if not (hasattr(pc, "_chain_lattice") and hasattr(pc, "_chain_place")):
        A16_STATS["ring_no_item87"] += 1
        return None
    n, N = qc.num_qubits, target.num_qubits
    lat = pc._chain_lattice(target, FAILED_MAX_ERROR)
    if lat is None:
        A16_STATS["ring_no_path"] += 1
        return None
    K, kadj, fadj, snake = lat
    line, _ = pc._chain_place(n, K, fadj, snake)
    if line is None:
        line, _ = pc._chain_place(n, K, kadj, snake)
    if line is None:
        A16_STATS["ring_no_path"] += 1
        return None
    built = _ring_logical(qc, read, rounds_fn)
    if built is None:
        return None
    logical, p_of_wire = built
    cm = CouplingMap()
    for q in range(N):
        cm.add_physical_qubit(q)
    for i in range(n - 1):
        cm.add_edge(line[i], line[i + 1])
        cm.add_edge(line[i + 1], line[i])
    two = [g for g in basis_gates if g in ("cz", "cx", "ecr")] or ["cz"]
    one = [g for g in basis_gates if g in ("rz", "sx", "x")]
    if len(read) > 4 and "rzz" in target.operation_names and any(  # item 41: the device's own rzz
            op.name != "cz" for layer in read[4] for op, _ in layer.values()):
        one = one + ["rzz"]
    try:
        pm = generate_preset_pass_manager(3, coupling_map=cm, basis_gates=two[:1] + one,
                                          initial_layout=[line[p_of_wire[w]] for w in range(n)],
                                          routing_method="none", seed_transpiler=seed)
        out = pm.run(logical)
    except Exception:  # noqa: BLE001 - counted; the other candidates stand
        A16_STATS["ring_failed"] += 1
        return None
    A16_STATS["ring_built"] += 1
    return out


def _rx_merged(c, target):
    """Item 29: single-qubit runs merged into rz rx rz when the Target offers rx (exact; layout kept)."""
    if target is None or "rx" not in target.operation_names:
        return c
    new = PassManager([Optimize1qGatesDecomposition(basis=["rz", "rx"])]).run(c)
    new._layout = getattr(c, "_layout", None)
    A16_STATS["rx_merged"] += 1
    return new


def _executable(c, target):
    """Item 29: `c` with every rzz angle in 0 < theta <= pi/2 (psf_compile's item 86 `_fold` where needed), or None."""
    if target is None or "rzz" not in target.operation_names:
        return c
    angles = [float(i.operation.params[0]) for i in c.data if i.operation.name == "rzz"]
    if all(0 < a <= 1.5707963267948966 + 1e-12 for a in angles):
        return c
    try:
        new = pc._fold(c, target)
    except Exception:  # noqa: BLE001 - no item 86, or the fold failed: the candidate is not returned
        A16_STATS["rzz_dropped"] += 1
        return None
    A16_STATS["rzz_folded"] += 1
    return _rx_merged(new, target)


def _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info, kwargs):
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


def _formula_only(qc, target, basis_gates, seed):
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


def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,
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


# ---------------------------------------------------------------------------------------------------------------------
# Prototype a20 (home, 2026-10-15): item 35, the ring template -- the ring formula built once by PSF-Zero itself (no
# Qiskit pass manager), then bound to new angles by closed-form single-qubit synthesis; item 36, the gate closed --
# no input is handed to Qiskit's level 3.
# ---------------------------------------------------------------------------------------------------------------------

TEMPLATE_STATS = {"built": 0, "bound": 0, "structure_mismatch": 0}


class OutOfScopeError(pc.TranspilerError):
    """Item 36: the input is not built by PSF-Zero's formulas; compile it with Qiskit (or another compiler) yourself."""


def _m_rz(t):
    import numpy as np
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]])


def _m_ry(t):
    import numpy as np
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _m_rx(t):
    import numpy as np
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


_FAST_1Q = {"rz": _m_rz, "ry": _m_ry, "rx": _m_rx}


def _mul(A, B):
    """Item 35: the 2 x 2 product A @ B on tuples (a, b, c, d) = [[a, b], [c, d]] (plain Python: faster than numpy at
    this size)."""
    a, b, c, d = A
    e, f, g, h = B
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def _t_op(op):
    """Item 35: a single-qubit operation as a tuple matrix (closed forms for rz, ry, rx; else Qiskit's to_matrix)."""
    import cmath
    import math
    nm = op.name
    if nm == "rz":
        t = float(op.params[0])
        return (cmath.exp(-0.5j * t), 0j, 0j, cmath.exp(0.5j * t))
    if nm == "ry":
        t = float(op.params[0])
        c, s = math.cos(t / 2), math.sin(t / 2)
        return (complex(c), complex(-s), complex(s), complex(c))
    if nm == "rx":
        t = float(op.params[0])
        c, s = math.cos(t / 2), math.sin(t / 2)
        return (complex(c), -1j * s, -1j * s, complex(c))
    m = op.to_matrix()
    return (complex(m[0][0]), complex(m[0][1]), complex(m[1][0]), complex(m[1][1]))


def _t_zyz(U):
    """Item 35: (theta, phi, lam) with U = e^{i alpha} Rz(phi) Ry(theta) Rz(lam), U a tuple matrix."""
    import cmath
    import math
    a, b, c, d = U
    sq = cmath.sqrt(a * d - b * c)
    a, c, d = a / sq, c / sq, d / sq
    theta = 2 * math.atan2(abs(c), abs(a))
    p11, p10 = cmath.phase(d), cmath.phase(c) if abs(c) > 1e-15 else 0.0
    if abs(c) <= 1e-15:  # diagonal: only phi + lam is defined
        return 0.0, 2 * p11, 0.0
    if abs(a) <= 1e-15:  # anti-diagonal: only phi - lam is defined
        return math.pi, 2 * p10, 0.0
    return theta, p11 + p10, p11 - p10


def _rot(kind, t):
    """Item 35: rz / ry / rx(t) as a tuple matrix."""
    import cmath
    import math
    if kind == "rz":
        return (cmath.exp(-0.5j * t), 0j, 0j, cmath.exp(0.5j * t))
    c, s = math.cos(t / 2), math.sin(t / 2)
    if kind == "ry":
        return (complex(c), complex(-s), complex(s), complex(c))
    return (complex(c), -1j * s, -1j * s, complex(c))


class RingTemplate:
    """Item 35: a ring classifier compiled once by the ring formula into a parametrised circuit.

    `bind_values(values)` compiles the same classifier with new angles: `values` are the angles of the input's rz /
    ry / rx gates in the input's order (`values_of(qc)` reads them from a circuit; any other single-qubit gate of the
    input is fixed as it was at build time). The angles of every single-qubit run that holds such a gate are computed
    in closed form (ZYZ) and assigned with Qiskit's assign_parameters; runs of fixed gates only were synthesised once,
    with the fewest pulses. `bind(qc)` does the same from a circuit, after checking its structure."""

    def __init__(self, sig, circuit, var_runs, params, use_rx, layout, var_kinds):
        self.sig, self.circuit, self.var_runs, self.params = sig, circuit, var_runs, params
        self.use_rx, self.layout, self.var_kinds = use_rx, layout, var_kinds

    def values_of(self, qc):
        ops = qc.data
        return [float(ops[k].operation.params[0]) for k, _ in self.var_kinds]

    def angles(self, values):
        import math
        out = []
        half = math.pi / 2
        for factors in self.var_runs:
            U = (1 + 0j, 0j, 0j, 1 + 0j)
            for f in factors:
                U = _mul(_rot(f[2], values[f[1]]) if len(f) == 3 else f, U)
            theta, phi, lam = _t_zyz(U)
            if self.use_rx:
                out += (lam - half, theta, phi + half)
            else:
                out += (lam, theta + math.pi, phi + math.pi)
        return out

    def bind_values(self, values):
        if len(values) != len(self.var_kinds):
            raise ValueError(f"item 35: {len(self.var_kinds)} angles expected, {len(values)} given")
        out = self.circuit.assign_parameters(self.angles(values))
        TEMPLATE_STATS["bound"] += 1
        return out

    def bind(self, qc):
        if _structure_sig(qc) != self.sig:
            TEMPLATE_STATS["structure_mismatch"] += 1
            raise ValueError("item 35: the circuit's structure differs from the template's")
        return self.bind_values(self.values_of(qc))


def _structure_sig(qc):
    """Item 35: the circuit with its angles left out (gate names, qubits, clbits) -- what a template must match."""
    return tuple((i.operation.name, tuple(qc.find_bit(q).index for q in i.qubits),
                  tuple(qc.find_bit(c).index for c in i.clbits)) for i in qc.data)


_BLOCK_CACHE = {}


def _block_ops(kinds):
    """Item 35: a fixed two-qubit block (a sequence of "cz" and "swap" on wires (0, 1)) as time-ordered
    [("1q", wire, 2 x 2 matrix) | ("cz",)] with the fewest CZ, decomposed once and checked to 1e-12 up to phase."""
    if kinds in _BLOCK_CACHE:
        return _BLOCK_CACHE[kinds]
    import numpy as np
    from qiskit.circuit.library import CZGate
    from qiskit.quantum_info import Operator
    from qiskit.synthesis import TwoQubitBasisDecomposer
    blk = QuantumCircuit(2)
    for k in kinds:
        blk.cz(0, 1) if k == "cz" else blk.swap(0, 1)
    U = Operator(blk).data
    dec = TwoQubitBasisDecomposer(CZGate(), euler_basis="ZSX")(U)
    seq = []
    for ins in dec.data:
        qs = [dec.find_bit(q).index for q in ins.qubits]
        if len(qs) == 2:
            seq.append(("cz",))
        else:
            seq.append(("1q", qs[0], np.asarray(ins.operation.to_matrix(), dtype=complex)))
    W = Operator(dec).data
    k = np.argmax(abs(U))
    ph = U.flat[k] / W.flat[k]
    if np.max(abs(U - ph * W)) > 1e-12:
        raise RuntimeError(f"item 35: block {kinds} not exact")
    _BLOCK_CACHE[kinds] = seq
    return seq


def _ring_template_one(qc, target, basis_gates=None, seed=0, rounds_fn=None):
    """Item 35: a RingTemplate for the ring classifier `qc` on the device `target` (the ring formula of item 28, or
    item 31's schedule with `rounds_fn=ring_rounds_q4`, placed by item 87's rule as in item 28), or None when `qc` is
    not read as a ring classifier or no line is found. No Qiskit pass manager runs: the fixed CZ / SWAP blocks are
    decomposed once (`_block_ops`), and every single-qubit run between CZs is re-synthesised at `bind` time."""
    read = _ring_read(qc)
    if read is None:
        return None
    n, N = qc.num_qubits, target.num_qubits
    lat = pc._chain_lattice(target, FAILED_MAX_ERROR)
    if lat is None:
        return None
    K, kadj, fadj, snake = lat
    line, _ = pc._chain_place(n, K, fadj, snake)
    if line is None:
        line, _ = pc._chain_place(n, K, kadj, snake)
    if line is None:
        return None
    built = _ring_logical(qc, read, rounds_fn)
    if built is None:
        return None
    logical, p_of_wire = built
    phys = {w: line[p_of_wire[w]] for w in range(n)}
    # the logical circuit's single-qubit operations are qc's, in qc's order: map each to its index in qc.data
    src = [k for k, ins in enumerate(qc.data) if ins.operation.name not in ("barrier", "measure", "cz")]
    pending = {p: [] for p in phys.values()}
    items = []
    lg = [(ins.operation.name, [logical.find_bit(q).index for q in ins.qubits],
           [logical.find_bit(c).index for c in ins.clbits]) for ins in logical.data if ins.operation.name != "barrier"]
    used = [False] * len(lg)
    si = 0
    var_kinds = []  # (index in qc.data, gate name) of every rz / ry / rx of the input, in order: bind_values' order

    def flush(p):
        if pending[p]:
            items.append(("run", p, list(pending[p])))
            pending[p] = []
    for i, (nm, ws, cs) in enumerate(lg):
        if used[i]:
            continue
        used[i] = True
        if len(ws) == 1 and nm != "measure":
            op = qc.data[src[si]].operation
            if op.name in ("rz", "ry", "rx") and not getattr(op, "is_parameterized", lambda: False)():
                pending[phys[ws[0]]].append(("v", len(var_kinds), op.name))
                var_kinds.append((src[si], op.name))
            else:
                pending[phys[ws[0]]].append(_t_op(op))
            si += 1
            continue
        if nm == "measure":
            p = phys[ws[0]]
            flush(p)
            items.append(("measure", p, cs[0]))
            continue
        # a block: this cz / swap and the next ones on the same pair, up to any other operation on either wire
        a, b = sorted(ws)
        kinds = [nm]
        for j in range(i + 1, len(lg)):
            if used[j]:
                continue
            nj, wj, _ = lg[j]
            if not ({a, b} & set(wj)):
                continue
            if nj in ("cz", "swap") and sorted(wj) == [a, b]:
                kinds.append(nj)
                used[j] = True
                continue
            break
        pa, pb = phys[a], phys[b]
        for op in _block_ops(tuple(kinds)):
            if op[0] == "cz":
                flush(pa)
                flush(pb)
                items.append(("cz", pa, pb))
            else:
                m = op[2]
                pending[pa if op[1] == 0 else pb].append((complex(m[0, 0]), complex(m[0, 1]), complex(m[1, 0]),
                                                          complex(m[1, 1])))
    for p in list(pending):
        flush(p)
    if si != len(src):
        return None
    use_rx = "rx" in target.operation_names
    from qiskit.circuit import ParameterVector
    isvar = lambda f: len(f) == 3  # noqa: E731 - ("v", position, kind); a fixed matrix is a 4-tuple
    nvar = sum(1 for it in items if it[0] == "run" and any(isvar(f) for f in it[2]))
    pv = ParameterVector("psf_t", 3 * nvar)
    out = QuantumCircuit(N, name=qc.name)
    for cr in qc.cregs:
        out.add_register(cr)
    var_runs, k = [], 0
    half = 3.141592653589793 / 2
    for it in items:
        if it[0] == "cz":
            out.cz(it[1], it[2])
        elif it[0] == "measure":
            out.measure(it[1], it[2])
        else:
            _, p, factors = it
            merged, acc = [], None  # consecutive fixed matrices pre-multiplied
            for f in factors:
                if isvar(f):
                    if acc is not None:
                        merged.append(acc)
                        acc = None
                    merged.append(f)
                else:
                    acc = f if acc is None else _mul(f, acc)
            if acc is not None:
                merged.append(acc)
            if any(isvar(f) for f in merged):
                var_runs.append(merged)
                a, t, c = pv[k], pv[k + 1], pv[k + 2]
                k += 3
                if use_rx:
                    out.rz(a, p)
                    out.rx(t, p)
                    out.rz(c, p)
                else:
                    out.rz(a, p)
                    out.sx(p)
                    out.rz(t, p)
                    out.sx(p)
                    out.rz(c, p)
            else:  # fixed only: the fewest pulses, decided now
                theta, phi, lam = _t_zyz(merged[0])
                if abs(theta) < 1e-12:
                    if abs(((phi + lam + 3.141592653589793) % (2 * 3.141592653589793)) - 3.141592653589793) > 1e-12:
                        out.rz(phi + lam, p)
                elif use_rx:
                    out.rz(lam - half, p)
                    out.rx(theta, p)
                    out.rz(phi + half, p)
                elif abs(theta - half) < 1e-12:
                    out.rz(lam - half, p)
                    out.sx(p)
                    out.rz(phi + half, p)
                else:
                    out.rz(lam, p)
                    out.sx(p)
                    out.rz(theta + 3.141592653589793, p)
                    out.sx(p)
                    out.rz(phi + 3.141592653589793, p)
    TEMPLATE_STATS["built"] += 1
    return RingTemplate(_structure_sig(qc), out, var_runs, list(pv), use_rx, [phys[w] for w in range(n)], var_kinds)


def ring_template(qc, target, basis_gates=None, seed=0, rounds_fn=None):
    """Items 35 and 44: the RingTemplate of the ring schedule `rounds_fn` names, or else item 44's rule names; None
    when it does not build."""
    if rounds_fn is None:
        rounds_fn, _ = _ring_rounds_for(qc.num_qubits)
    try:
        return _ring_template_one(qc, target, basis_gates, seed, rounds_fn)
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Prototype a22 (home, 2026-10-15): item 42, embeddable rings through psf_compile's item 99; item 43, model_template.
# ---------------------------------------------------------------------------------------------------------------------

def _ring_embeds(qc, target, max_error=0.5):
    """Item 42: True when `qc` is read as a ring (items 28, 41), psf_compile has item 99, and the ring's interaction
    graph embeds in the device's working couplers by item 96's test."""
    if not (getattr(pc, "NATIVE_BRICKWORK", False) and hasattr(pc, "_native_brickwork")
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
