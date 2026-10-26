"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, 2026-10-26.a26
(a25 + item 50: the ring folded onto a device cycle, the default -- see patches/psf_ai_compile_a26_2026-10-26/
make_a26.py). a25's description:
(a24 + item 49: the zipper schedule also for odd n -- see patches/psf_ai_compile_a25_2026-10-25/make_a25.py). a24's
description:
(a23 + items 46-48: the ring template built in Rust, the ring placed by formula only, the zipper schedule as an option
-- see patches/psf_ai_compile_a24_2026-10-22/make_a24.py). a23's description:
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
  46. (a24) The ring template (item 35, `ring_template` and `model_template` for a ring that does not embed) is built
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
  49. (a25) The zipper also for odd n (owner, 2026-10-25): with PSF_RING_ZIP=1, a ring of odd n >= 5 also gets the
      zipper, by formula: the same start order (0, 1, 2, n-1, n-2, 3, 4, ...); round A CZ on positions (1,2), ...,
      (n-2,n-1); round B CZ then SWAP on (0,1) only; round C SWAP on (2,3), ..., (n-3,n-2); round D as round A;
      (5n - 7)/2 CZ per layer once the CZ + SWAP is merged (9, 14, 19 at n = 5, 7, 9; N2's and item 31's counts are
      higher). Chosen as in item 48. Evidence (TOQB results/zipper-odd-2026-10-25, analysis): one layer equals the
      ring's CZ layer and a permutation, and forward then backward is the identity, for n = 4 to 41; (5n - 7)/2 is
      the least found by exhaustive search (every state for n = 5, 7, 9; every minimum-length start for n = 5-11);
      not run on a device. Even n: the rounds are a24's, unchanged. Needs psfbind ring r3 or later (item 46's check).
  50. (a26) The ring folded onto one of the device's own cycles, the default (owner, 2026-10-26; PSF_RING_FOLD=0
      gives a25's output). Heavy-hex has cycles of 12, 20, 24, 28, ... qubits and none between (TOQB
      results/fold-holes-2026-10-25). A ring classifier (item 28's reading, any one cycle through the qubits) that
      does not embed (item 42) is laid on the outline of a fixed shape of cells, read off the coupling graph by
      formula (bridges, rows, cells; the first working shape in a fixed order; no search, no error value decides;
      a coupler or qubit at error >= FAILED_MAX_ERROR is never used, every listed direction of a coupler must work):
      - pendants: the longest such cycle of m <= n qubits, the n - m extra ring nodes hung off cycle nodes that have
        a working outside neighbour (spread in cycle order); each layer every pendant node trades places with its
        cycle node by one CZ + SWAP block: n + (n - m) two-qubit gates per layer, on exactly n qubits;
      - holes: the shortest such cycle of m >= n qubits, its m - n spare qubits starting in |0> and moved across by
        two CX each: n + 2 (m - n) per layer;
      - the rule: of pendants n - m, holes 2 (m - n) and the zipper's extra gates per layer, the fewest decides
        (ties: pendants, then holes); where the zipper's are fewest (small n), or no cycle fits, a25's path.
      Built once into a FoldTemplate (Qiskit's level-1 preset on the fixed layout, no routing, as item 28's
      translation), angles bound per call; the default call binds the same template. Evidence (TOQB, models of a
      2026-10-13 calibration of ibm_marrakesh, not a device): flow E (pendants against holes and the zipper, n = 13-
      18, L = 8, 16) and flow F, a declared match (the rule against the zipper, n = 13-18, L = 16, four random models:
      WIN at every n, 24 of 24 models; the rule's lam 0.51-0.76, the zipper's 0.09-0.46). Not run on a device.
Placement is psf_compile's (first VF2 match, no score) for what psf_compile builds; the ring formula's line is the
snake piece (item 47). No seed, error value or T1/T2 decides; failed elements are never used (item 26).
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

AI_COMPILE_VERSION = "2026-10-26.a26"  # item 50 (the ring folded onto a device cycle, default) + 2026-10-25.a25: item 49 (the zipper for odd n) + 2026-10-22.a24: items 46-48 (Rust template, formula-only placement, zipper option) + 2026-10-16.a23: items 44-45 (no compare-and-choose, no checks, no switches) + items 42-43 (embeddable rings and templates through psf_compile items 99-100) + items 37-41 (workplace review of a18) + items 35-36 (ring template; gate closed) + item 34 (formula only) on a18, which was: a17 (workplace: items 28-31) + items 32-33 (no search; hand-off outside the ring formula)
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
    line = _snake_line(n, K, fadj, kadj, snake)  # item 47: the snake piece, or OutOfScopeError
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
    line = _snake_line(n, K, fadj, kadj, snake)  # item 47: the snake piece, or OutOfScopeError
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


# ---------------------------------------------------------------------------------------------------------------------
# a24 (home, 2026-10-22): item 46, the ring template built in Rust; item 47, formula-only placement (E5); item 48, the
# zipper schedule as an option.
# ---------------------------------------------------------------------------------------------------------------------

import ctypes  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

A24_STATS = {"rust_built": 0, "rust_none": 0, "rust_refused": 0, "checked": 0, "out_of_scope": 0, "zip_chosen": 0}
PSFBIND_RING_MIN = 3  # item 46: "psfbind ring r3" or later (r2 has the zipper, r3 also for odd n: item 49)


def _zip_on(n):
    """Items 48-49: the zipper is a candidate only with PSF_RING_ZIP=1, for a ring of n >= 4 (odd n: item 49)."""
    return os.environ.get("PSF_RING_ZIP", "") == "1" and n >= 4


def _snake_line(n, K, fadj, kadj, snake):
    """Item 47: the ring's line, the longest piece of the snake on the couplers of the most common gate length, else
    on every working coupler (psf_compile item 87's first rule, both by formula), or OutOfScopeError where only VF2's
    search would find one."""
    for adj in (fadj, kadj):
        line, how = pc._chain_place(n, K, adj, snake)
        if line is not None and how == "snake":
            return line
    A24_STATS["out_of_scope"] += 1
    raise OutOfScopeError("item 47 (formula only): no piece of the device's snake holds the "
                          f"{n}-qubit ring; a placement would need VF2's search, which is outside the scope")


def ring_rounds_zip(n):
    """Item 48: the zipper schedule (the workplace's prototype P1, 2026-10-22; TOQB's ring_zip.py, unchanged):
    (start order: ring index at each line position, rounds [(CZ pairs of ring indices, SWAPs (p, p + 1))])."""
    if n < 4:
        raise RuntimeError("zipper: n must be >= 4")
    top = list(range(1, n))
    bot = list(range(n - 1, 0, -1))
    order, used, t, b, from_top = [0], {0}, 0, 0, True
    while len(order) < n:
        src = top if from_top else bot
        k = 0
        while k < 2 and len(order) < n:
            i = t if from_top else b
            while src[i] in used:
                i += 1
            order.append(src[i])
            used.add(src[i])
            if from_top:
                t = i + 1
            else:
                b = i + 1
            k += 1
        from_top = not from_top
    occ = list(order)
    edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}

    def take(p):
        a, b_ = occ[p], occ[p + 1]
        e = (min(a, b_), max(a, b_))
        if e not in edges:
            raise RuntimeError(f"zipper: positions ({p}, {p + 1}) hold {a}, {b_}, not a ring edge")
        edges.discard(e)
        return (a, b_)

    rounds = []
    odd = list(range(1, n - 1, 2))  # item 49: even n up to (n-3, n-2) as a24, odd n up to (n-2, n-1)
    rounds.append(([take(p) for p in odd], []))
    ends = [0, n - 2] if n % 2 == 0 else [0]  # item 49: odd n, one end block
    czb = [take(p) for p in ends]
    rounds.append((czb, [(p, p + 1) for p in ends]))
    for p in ends:
        occ[p], occ[p + 1] = occ[p + 1], occ[p]
    mid = [(p, p + 1) for p in range(2, n - 3 + n % 2, 2)]  # item 49: odd n up to (n-3, n-2)
    if mid:
        rounds.append(([], mid))
        for p, p1 in mid:
            occ[p], occ[p1] = occ[p1], occ[p]
    rounds.append(([take(p) for p in odd], []))
    if edges:
        raise RuntimeError(f"zipper: edges left {sorted(edges)}")
    return order, rounds


_SCHED_FN = {"n2": None, "q4": ring_rounds_q4, "zip": ring_rounds_zip}


def _schedules(n):
    """Items 44 and 48: the schedules to build, in order (item 44's rule's, then the zipper when asked)."""
    first = "n2" if _ring_rounds_for(n)[0] is None else "q4"
    return [first, "zip"] if _zip_on(n) else [first]


# ----------------------------------------------------------------------------------------- item 46: the Rust bridge

def _f(x):
    return None if x is None else float(x)


def _device_json(target):
    names = sorted(target.operation_names)
    entries = []
    for name in names:
        for qargs, props in target[name].items():
            err = getattr(props, "error", None) if props is not None else None
            dur = getattr(props, "duration", None) if props is not None else None
            entries.append([name, None if qargs is None else [int(q) for q in qargs], _f(err), _f(dur)])
    return {"num_qubits": int(target.num_qubits), "names": names, "entries": entries}


def _input_json(qc):
    ops = []
    for ins in qc.data:
        op = ins.operation
        qs = [qc.find_bit(q).index for q in ins.qubits]
        cs = [qc.find_bit(c).index for c in ins.clbits]
        angle = mat = None
        if len(qs) == 1 and op.name not in ("measure", "barrier"):
            parameterised = getattr(op, "is_parameterized", lambda: False)()
            if op.name in ("rz", "ry", "rx"):
                if not parameterised:
                    angle = float(op.params[0])
            elif not parameterised:
                m = op.to_matrix()
                mat = [float(v) for z in (m[0][0], m[0][1], m[1][0], m[1][1]) for v in (complex(z).real,
                                                                                      complex(z).imag)]
        ops.append([op.name, qs, cs, angle, mat])
    return {"num_qubits": qc.num_qubits, "ops": ops}


_BLOCKS_JSON = []


def _blocks_json(max_len=4):
    """Every sequence of cz / swap up to `max_len`, decomposed by this module's `_block_ops` (Qiskit's
    TwoQubitBasisDecomposer, checked there): constants handed to the builder, not chosen by it."""
    if not _BLOCKS_JSON:
        out = []
        for m in range(1, max_len + 1):
            for kinds in itertools.product(("cz", "swap"), repeat=m):
                try:
                    seq = _block_ops(tuple(kinds))
                except RuntimeError:
                    continue
                ops = []
                for op in seq:
                    if op[0] == "cz":
                        ops.append(["cz"])
                    else:
                        M = op[2]
                        ops.append(["1q", int(op[1]), [float(v) for z in (M[0, 0], M[0, 1], M[1, 0], M[1, 1])
                                                       for v in (complex(z).real, complex(z).imag)]])
                out.append([list(kinds), ops])
        _BLOCKS_JSON.append(out)
    return _BLOCKS_JSON[0]


_PSFBIND = []


def _psfbind_path():
    env = os.environ.get("PSF_BIND_LIB")
    if env:
        return env
    roots = (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), os.getcwd())
    for root in roots:
        for name in ("libpsfbind.so", "libpsfbind.dylib", "psfbind.dll"):
            p = os.path.join(root, "rust", "psfbind", "target", "release", name)
            if os.path.exists(p):
                return p
    return None


def _psfbind():
    """Item 46: the library, checked to be "psfbind ring r3" or later (a25); RuntimeError otherwise (no fallback)."""
    if _PSFBIND:
        return _PSFBIND[0]
    path = _psfbind_path()
    if path is None or not os.path.exists(path) or not path.endswith((".so", ".dylib", ".dll")):
        raise RuntimeError("item 46: the ring template is built by PSF-Zero's rust/psfbind, which was not found; "
                           "build it with `cargo build --release --manifest-path rust/psfbind/Cargo.toml` or set "
                           "PSF_BIND_LIB to the library")
    lib = ctypes.CDLL(path)
    try:
        lib.psfbind_version.restype = ctypes.c_char_p
        lib.psfbind_version.argtypes = []
        version = lib.psfbind_version().decode()
    except AttributeError:
        version = ""
    m = re.match(r"psfbind ring r(\d+)", version)
    if m is None or int(m.group(1)) < PSFBIND_RING_MIN:
        raise RuntimeError(f"item 46: {path} is an older psfbind ({version or 'no version'}); rebuild it with "
                           "`cargo build --release --manifest-path rust/psfbind/Cargo.toml`")
    lib.psfbind_ring_template.restype = ctypes.c_void_p
    lib.psfbind_ring_template.argtypes = [ctypes.c_char_p]
    lib.psfbind_free.argtypes = [ctypes.c_void_p]
    lib.psfbind_free.restype = None
    lib.version = version
    _PSFBIND.append(lib)
    return lib


def _rust_build(qc, target, schedules, max_error=FAILED_MAX_ERROR):
    req = {"device": _device_json(target), "input": _input_json(qc), "blocks": _blocks_json(),
           "max_error": float(max_error), "schedules": list(schedules)}
    lib = _psfbind()
    p = lib.psfbind_ring_template(json.dumps(req).encode())
    try:
        return json.loads(ctypes.string_at(p).decode())
    finally:
        lib.psfbind_free(p)


def _from_rust(t, qc):
    from qiskit.circuit import ParameterVector
    N = int(t["num_qubits"])
    pv = ParameterVector("psf_t", 3 * int(t["nvar"]))
    out = QuantumCircuit(N, name=qc.name)
    for cr in qc.cregs:
        out.add_register(cr)
    for name, qs, cs, p in t["ops"]:
        qs = [int(q) for q in qs]
        if name == "cz":
            out.cz(qs[0], qs[1])
        elif name == "measure":
            out.measure(qs[0], int(cs[0]))
        elif name == "sx":
            out.sx(qs[0])
        elif name in ("rz", "rx"):
            getattr(out, name)(pv[int(p["p"])] if isinstance(p, dict) else float(p), qs[0])
        else:
            raise RuntimeError(f"item 46: unknown operation {name!r} in the Rust template")
    var_runs = []
    for run in t["var_runs"]:
        factors = []
        for f in run:
            if len(f) == 8:
                factors.append(tuple(complex(f[2 * i], f[2 * i + 1]) for i in range(4)))
            else:
                factors.append(("v", int(f[0]), f[1]))
        var_runs.append(factors)
    var_kinds = [(int(k), nm) for k, nm in t["var_kinds"]]
    tpl = RingTemplate(_structure_sig(qc), out, var_runs, list(pv), bool(t["use_rx"]),
                       [int(q) for q in t["layout"]], var_kinds)
    tpl.schedule = t["schedule"]
    return tpl


def _ring_template_py(qc, target, basis_gates, seed, schedules):
    """a23's Python builder (item 47's placement) over `schedules`, item 84's rule choosing (a full tie keeps the
    earlier): the reference of PSF_RUST_TEMPLATE=check, and the builder of a ring the Rust does not port."""
    best, bc = None, None
    for name in schedules:
        try:
            t = _ring_template_one(qc, target, basis_gates, seed, _SCHED_FN[name])
        except RuntimeError:
            t = None
        if t is None:
            continue
        t.schedule = name
        c = t.bind(qc)
        if best is None or pc._better(c, bc, target):
            best, bc = t, c
    return best


def _ring_template_rust(qc, target, basis_gates=None, seed=0):
    """Item 46: the Rust template over item 44's (and 48's) schedules; item 47's refusal; the Python for a ring the
    Rust does not port; PSF_RUST_TEMPLATE=check compares with the Python's."""
    schedules = _schedules(qc.num_qubits)
    ans = _rust_build(qc, target, schedules)
    status = ans.get("status")
    if status == "out_of_scope":
        A24_STATS["out_of_scope"] += 1
        raise OutOfScopeError("item 47 (formula only): " + ans.get("reason", ""))
    if status == "refused":
        A24_STATS["rust_refused"] += 1
        tpl = _ring_template_py(qc, target, basis_gates, seed, schedules)
    elif status == "built":
        tpl = _from_rust(ans["template"], qc)
        A24_STATS["rust_built"] += 1
    elif status == "none":
        tpl = None
        A24_STATS["rust_none"] += 1
    else:
        raise RuntimeError(f"item 46: unexpected answer from rust/psfbind: {status!r}")
    if status != "refused" and os.environ.get("PSF_RUST_TEMPLATE", "") == "check":
        ref = _ring_template_py(qc, target, basis_gates, seed, schedules)
        diffs = (["one is None"] if (tpl is None) != (ref is None) else
                 [] if tpl is None else compare_templates(tpl, ref))
        if diffs:
            raise RuntimeError("item 46: the Rust template differs from the Python's: " + "; ".join(diffs[:5]))
        A24_STATS["checked"] += 1
    if tpl is not None and getattr(tpl, "schedule", None) == "zip":
        A24_STATS["zip_chosen"] += 1
    return tpl


def compare_templates(a, b, tol=1e-12):
    """Item 46: the differences between two RingTemplates (empty: identical; fixed angles and factors within `tol`)."""
    diffs = []
    if getattr(a, "schedule", None) != getattr(b, "schedule", None):
        diffs.append(f"schedule {getattr(a, 'schedule', None)} / {getattr(b, 'schedule', None)}")
    if list(a.layout) != list(b.layout):
        diffs.append(f"layout {a.layout} / {b.layout}")
    if bool(a.use_rx) != bool(b.use_rx):
        diffs.append("use_rx")
    if [tuple(v) for v in a.var_kinds] != [tuple(v) for v in b.var_kinds]:
        diffs.append("var_kinds")
    ca, cb = a.circuit, b.circuit
    if ca.num_qubits != cb.num_qubits or ca.num_clbits != cb.num_clbits:
        diffs.append("circuit sizes")
    if len(ca.data) != len(cb.data):
        diffs.append(f"gate count {len(ca.data)} / {len(cb.data)}")
    worst = 0.0
    for k, (x, y) in enumerate(zip(ca.data, cb.data)):
        qx = [ca.find_bit(q).index for q in x.qubits]
        qy = [cb.find_bit(q).index for q in y.qubits]
        cx = [ca.find_bit(c).index for c in x.clbits]
        cy = [cb.find_bit(c).index for c in y.clbits]
        if x.operation.name != y.operation.name or qx != qy or cx != cy:
            diffs.append(f"gate {k}: {x.operation.name}{qx}{cx} / {y.operation.name}{qy}{cy}")
            continue
        for px, py in zip(x.operation.params, y.operation.params):
            fx = not hasattr(px, "parameters") or not px.parameters
            fy = not hasattr(py, "parameters") or not py.parameters
            if fx != fy:
                diffs.append(f"gate {k}: fixed / parameter")
            elif fx:
                worst = max(worst, abs(float(px) - float(py)))
            elif str(px) != str(py):
                diffs.append(f"gate {k}: {px} / {py}")
    if worst > tol:
        diffs.append(f"fixed angles differ by {worst:.3e}")
    if len(a.var_runs) != len(b.var_runs):
        diffs.append("number of runs")
    fw = 0.0
    for ra, rb in zip(a.var_runs, b.var_runs):
        if len(ra) != len(rb):
            diffs.append("run length")
            continue
        for fa, fb in zip(ra, rb):
            if (len(fa) == 3) != (len(fb) == 3):
                diffs.append("factor kind")
            elif len(fa) == 3:
                if tuple(fa[1:]) != tuple(fb[1:]):
                    diffs.append(f"factor {fa} / {fb}")
            else:
                fw = max(fw, max(abs(complex(u) - complex(v)) for u, v in zip(fa, fb)))
    if fw > tol:
        diffs.append(f"fixed factors differ by {fw:.3e}")
    return diffs


# ---------------------------------------------------------------------------------------------------------------------
# a26 (2026-10-26): item 50, the ring folded onto one of the device's own cycles, the default (PSF_RING_FOLD=0: a25).
# Ported from TOQB toqb/fold.py (fold:rule, 145f38b; results/flow-e-2026-10-26, flow-f-2026-10-26).
# ---------------------------------------------------------------------------------------------------------------------

from collections import defaultdict  # noqa: E402

A26_STATS = {"fold_pend": 0, "fold_holes": 0, "fold_zipper_fewer": 0, "fold_no_cycle": 0, "fold_not_ring": 0,
             "fold_off": 0, "fold_permuted": 0}


class FoldRefused(RuntimeError):
    pass


def _fold_on():
    """Item 50: the fold is the default; PSF_RING_FOLD=0 gives a25's output."""
    return os.environ.get("PSF_RING_FOLD", "1") != "0"


def _fold_works(target, name, qa, max_error):
    if name not in target.operation_names:
        return False
    p = target[name].get(qa)
    return p is not None and (p.error is None or p.error < max_error)


def _fold_edges(target, max_error):
    """Working CZ couplers as sorted pairs: every direction the Target lists must work (a direction listed and
    failed removes the pair)."""
    pairs = defaultdict(list)
    for qa in target["cz"]:
        if qa is not None:
            pairs[tuple(sorted(qa))].append(_fold_works(target, "cz", qa, max_error))
    return {e for e, oks in pairs.items() if all(oks)}


def _fold_qubit_ok(target, q, max_error):
    return all(_fold_works(target, g, (q,), max_error) for g in ("sx", "measure")) and (
        "x" not in target.operation_names or _fold_works(target, "x", (q,), max_error))


def fold_lattice(edges, nq):
    """Item 50: heavy-hex read off the coupling graph: bridges (two neighbours, indices not consecutive), rows (what
    is left without them), and per row pair its bridges (upper qubit, bridge, lower qubit), left to right."""
    nb = defaultdict(set)
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    bridges = {q for q in range(nq) if len(nb[q]) == 2 and all(abs(x - q) > 1 for x in nb[q])}
    row_of, nrows = {}, 0
    for q in range(nq):
        if q in bridges or q in row_of or not nb[q]:
            continue
        stack = [q]
        row_of[q] = nrows
        while stack:
            u = stack.pop()
            for v in nb[u]:
                if v not in bridges and v not in row_of:
                    row_of[v] = nrows
                    stack.append(v)
        nrows += 1
    pairs = defaultdict(list)
    for b in bridges:
        u, v = sorted(nb[b], key=lambda x: row_of.get(x, -1))
        if u in row_of and v in row_of and row_of[u] != row_of[v]:
            pairs[(row_of[u], row_of[v])].append((u, b, v))
    for k in pairs:
        pairs[k].sort()
    return pairs


def _fold_cell(rp, j, pairs):
    (u1, b1, v1), (u2, b2, v2) = pairs[rp][j], pairs[rp][j + 1]
    e = set()
    for a in range(min(u1, u2), max(u1, u2)):
        e.add((a, a + 1))
    for a in range(min(v1, v2), max(v1, v2)):
        e.add((a, a + 1))
    for x, y in ((u1, b1), (b1, v1), (u2, b2), (b2, v2)):
        e.add((min(x, y), max(x, y)))
    return e


def _fold_as_cycle(e):
    nb = defaultdict(list)
    for a, b in e:
        nb[a].append(b)
        nb[b].append(a)
    if not nb or any(len(v) != 2 for v in nb.values()):
        return None
    start = min(nb)
    cyc, prev, cur = [start], None, start
    while True:
        nxt = nb[cur][0] if nb[cur][0] != prev else nb[cur][1]
        if nxt == start:
            break
        cyc.append(nxt)
        prev, cur = cur, nxt
    return cyc if len(cyc) == len(nb) else None


def fold_shapes(pairs):
    """Item 50: outlines of k cells side by side in a row pair (8k + 4 qubits), and of k cells plus k2 side by side
    in the next row pair down when the outline is one cycle: [(length, name, qubits in cycle order)], in a fixed order
    (row pairs top to bottom, cells left to right)."""
    out, keys = [], sorted(pairs)
    for rp in keys:
        nc = len(pairs[rp]) - 1
        for k in range(1, nc + 1):
            for j in range(nc - k + 1):
                e = set()
                for c in range(j, j + k):
                    e ^= _fold_cell(rp, c, pairs)
                cyc = _fold_as_cycle(e)
                if cyc:
                    out.append((len(cyc), ("row", rp, j, k), cyc))
                for rb in (kk for kk in keys if kk[0] == rp[1]):
                    nb2 = len(pairs[rb]) - 1
                    for k2 in range(1, nb2 + 1):
                        for c in range(nb2 - k2 + 1):
                            e2 = set(e)
                            for cc in range(c, c + k2):
                                e2 ^= _fold_cell(rb, cc, pairs)
                            cyc2 = _fold_as_cycle(e2)
                            if cyc and cyc2:
                                out.append((len(cyc2), ("row+row", rp, j, k, rb, c, k2), cyc2))
    return out


def _fold_cands(target, max_error):
    edges = _fold_edges(target, max_error)
    good = []
    for length, name, cyc in fold_shapes(fold_lattice(edges, target.num_qubits)):
        if all(_fold_qubit_ok(target, q, max_error) for q in cyc) and all(
                tuple(sorted((cyc[i], cyc[(i + 1) % length]))) in edges for i in range(length)):
            good.append((length, name, cyc))
    return edges, good


def fold_place_holes(target, n, max_error=FAILED_MAX_ERROR):
    """Item 50: (m, name, cycle) -- the shortest working outline of m >= n qubits, the first in the fixed order."""
    _, good = _fold_cands(target, max_error)
    for m in sorted({g[0] for g in good}):
        if m >= n:
            return next(g for g in good if g[0] == m)
    return None


def fold_place_pend(target, n, max_error=FAILED_MAX_ERROR):
    """Item 50: (m, name, cycle, {cycle position: pendant qubit}) -- the longest working outline of m <= n qubits
    with n - m cycle nodes that have a working outside neighbour; pendants spread evenly over those nodes in cycle
    order, each node's smallest working outside neighbour not used before."""
    edges, good = _fold_cands(target, max_error)
    nb = defaultdict(set)
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    for m in sorted({g[0] for g in good}, reverse=True):
        if m > n:
            continue
        k = n - m
        if k >= m:
            continue
        for length, name, cyc in good:
            if length != m:
                continue
            inside, used, spots = set(cyc), set(), []
            for p, q in enumerate(cyc):
                outs = sorted(x for x in nb[q] if x not in inside and _fold_qubit_ok(target, x, max_error))
                if outs:
                    spots.append((p, outs))
            if len(spots) < k:
                continue
            pick = [spots[(j * len(spots)) // k] for j in range(k)] if k else []
            pend = {}
            for p, outs in pick:
                free = [x for x in outs if x not in used]
                if not free:
                    break
                pend[p] = free[0]
                used.add(free[0])
            if len(pend) == k:
                return m, name, cyc, pend
    return None


def _fold_read(qc):
    """Item 50: a ring classifier (layers of single-qubit gates, each followed by the n CZs of one cycle through all
    qubits, the same cycle every layer; then single-qubit gates and measurements) as (cycle order, layers, final
    gates, measurements) with qubits relabelled along the cycle; FoldRefused otherwise."""
    n = qc.num_qubits
    if n < 4:
        raise FoldRefused("fewer than 4 qubits")
    layers, cur, seen, meas, ring = [], [[] for _ in range(n)], set(), [], None
    for ins in qc.data:
        nm = ins.operation.name
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if nm == "cz":
            e = frozenset(qs)
            if e in seen or (ring is not None and e not in ring):
                raise FoldRefused("not one CZ cycle per layer")
            seen.add(e)
            if len(seen) == n:
                if ring is None:
                    ring = set(seen)
                layers.append(cur)
                cur, seen = [[] for _ in range(n)], set()
        elif nm == "measure":
            meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))
        elif nm == "barrier":
            continue
        elif len(qs) == 1 and not seen:
            if meas:
                raise FoldRefused("a gate after a measurement")
            cur[qs[0]].append(ins.operation)
        else:
            raise FoldRefused(f"{nm} on {qs} inside a ring layer")
    if seen or ring is None:
        raise FoldRefused("an incomplete ring layer")
    nb = defaultdict(list)
    for e in ring:
        a, b = sorted(e)
        nb[a].append(b)
        nb[b].append(a)
    if any(len(nb[q]) != 2 for q in range(n)):
        raise FoldRefused("the CZs do not form one cycle")
    for q in nb:
        nb[q].sort()  # the cycle is read from qubit 0 towards its smaller neighbour (0, 1, 2, ... for TOQB's ring)
    order, prev, c = [0], None, 0
    while True:
        nx = nb[c][0] if nb[c][0] != prev else nb[c][1]
        if nx == 0:
            break
        order.append(nx)
        prev, c = c, nx
    if len(order) != n:
        raise FoldRefused("the CZs do not form one cycle")
    pos = {q: i for i, q in enumerate(order)}
    layers = [[layer[q] for q in order] for layer in layers]
    final = [cur[q] for q in order]
    meas = [(pos[q], c) for q, c in meas]
    return order, layers, final, meas


def _fold_spread(n, m):
    h = m - n
    at = {round((k + 0.5) * m / h) % m for k in range(h)} if h else set()
    while len(at) < h:
        at.add(max(set(range(m)) - at))
    occ, q = [], 0
    for p in range(m):
        if p in at:
            occ.append(None)
        else:
            occ.append(q)
            q += 1
    return occ


def _fold_holes(n, layers, final, meas, cyc, nq, ncl):
    """Item 50, holes: the m - n spare qubits of the cycle start in |0>; per layer CZ on every ring edge already
    adjacent (rounds of disjoint pairs), then for each run of holes the ring node before it moves across (CX a->h,
    CX h->a), then CZ across: n + 2(m - n) two-qubit gates per layer."""
    m = len(cyc)
    occ = _fold_spread(n, m)
    start = {r: cyc[p] for p, r in enumerate(occ) if r is not None}
    out = QuantumCircuit(nq, ncl)
    for layer in layers:
        pos = {r: p for p, r in enumerate(occ) if r is not None}
        for r in range(n):
            for op in layer[r]:
                out.append(op, [cyc[pos[r]]])
        edges = [p for p in range(m) if occ[p] is not None and occ[(p + 1) % m] is not None
                 and ((occ[p] + 1) % n == occ[(p + 1) % m] or (occ[(p + 1) % m] + 1) % n == occ[p])]
        rounds = []
        for p in edges:
            for rd in rounds:
                if all({p, (p + 1) % m}.isdisjoint({q, (q + 1) % m}) for q in rd):
                    rd.append(p)
                    break
            else:
                rounds.append([p])
        for rd in rounds:
            for p in rd:
                out.cz(cyc[p], cyc[(p + 1) % m])
        for p in [p for p in range(m) if occ[p] is None and occ[(p - 1) % m] is not None]:
            cur, pb = (p - 1) % m, (p + 1) % m
            while occ[pb] is None:
                pb = (pb + 1) % m
            while occ[(cur + 1) % m] is None:
                nx = (cur + 1) % m
                out.cx(cyc[cur], cyc[nx])
                out.cx(cyc[nx], cyc[cur])
                occ[nx], occ[cur] = occ[cur], None
                cur = nx
            out.cz(cyc[cur], cyc[pb])
    pos = {r: p for p, r in enumerate(occ) if r is not None}
    for r in range(n):
        for op in final[r]:
            out.append(op, [cyc[pos[r]]])
    for q, c in meas:
        out.measure(cyc[pos[q]], c)
    return out, [start[r] for r in range(n)], [cyc[pos[r]] for r in range(n)]


def _fold_pend(n, layers, final, meas, cyc, pend, nq, ncl):
    """Item 50, pendants: the ring on the cycle, its n - m extra nodes hung off it; per layer each pendant node
    trades places with its cycle node by one CZ + SWAP block (two two-qubit gates): n + (n - m) per layer."""
    m = len(cyc)
    if len(pend) >= m:
        raise FoldRefused("every cycle node has a pendant")
    oc, op, r = [None] * m, {}, 0
    for p in range(m):
        oc[p] = r
        r += 1
        if p in pend:
            op[p] = r
            r += 1
    if r != n:
        raise FoldRefused("pendants do not match the ring size")

    def where():
        w = {v: cyc[p] for p, v in enumerate(oc)}
        w.update({v: pend[p] for p, v in op.items()})
        return w

    start = where()
    out = QuantumCircuit(nq, ncl)
    for layer in layers:
        w = where()
        for i in range(n):
            for o in layer[i]:
                out.append(o, [w[i]])
        need = {frozenset((i, (i + 1) % n)) for i in range(n)}
        while need:
            progress = False
            rounds = []
            for p in range(m):
                e = frozenset((oc[p], oc[(p + 1) % m]))
                if e in need:
                    for rd in rounds:
                        if all({p, (p + 1) % m}.isdisjoint({q, (q + 1) % m}) for q in rd):
                            rd.append(p)
                            break
                    else:
                        rounds.append([p])
                    need.discard(e)
            for rd in rounds:
                for p in rd:
                    out.cz(cyc[p], cyc[(p + 1) % m])
                progress = True
            for p in sorted(op):
                u, v = oc[p], op[p]
                e = frozenset((u, v))
                other = frozenset((u, (u - 1) % n)) if (u + 1) % n == v else frozenset((u, (u + 1) % n))
                if e in need and other not in need:
                    a, b = cyc[p], pend[p]
                    out.h(b)  # CZ then SWAP on (a, b) = H(b), CX(b, a), CX(a, b), H(a)
                    out.cx(b, a)
                    out.cx(a, b)
                    out.h(a)
                    oc[p], op[p] = v, u
                    need.discard(e)
                    progress = True
            if not progress:
                raise FoldRefused("no schedule step left")
    w = where()
    for i in range(n):
        for o in final[i]:
            out.append(o, [w[i]])
    for q, c in meas:
        out.measure(w[q], c)
    return out, [start[i] for i in range(n)], [w[i] for i in range(n)]


def fold_choice(target, n, max_error=FAILED_MAX_ERROR):
    """Item 50's rule: of the extra two-qubit gates per layer -- pendants n - m_below, holes 2 (m_above - n), the
    zipper (5n - 8)/2 - n or (5n - 7)/2 - n -- the fewest decides (ties: pendants, then holes). ("pend", placement),
    ("holes", placement) or ("zipper", None)."""
    below, above = fold_place_pend(target, n, max_error), fold_place_holes(target, n, max_error)
    ep = (n - below[0]) if below else None
    eh = 2 * (above[0] - n) if above else None
    ez = ((5 * n - 8) // 2 if n % 2 == 0 else (5 * n - 7) // 2) - n
    best = min(e for e in (ep, eh, ez) if e is not None)
    if ep is not None and ep == best:
        return "pend", below
    if eh is not None and eh == best:
        return "holes", above
    return "zipper", None


class FoldTemplate:
    """Item 50: a ring classifier folded onto a device cycle, compiled once into a parametrised circuit (Qiskit's
    level-1 preset on the fixed layout, no routing); `bind(qc)` checks the structure and assigns the input's angles.
    `layout`: the device qubit of each input qubit at the start; `final_layout`: at the end; `schedule`: "pend" or
    "holes"; `cycle`: the cycle's length."""

    def __init__(self, sig, circuit, order, layout, final_layout, schedule, cycle):
        self.sig, self.circuit, self.order = sig, circuit, order
        self.layout, self.final_layout, self.schedule, self.cycle = layout, final_layout, schedule, cycle

    def values_of(self, qc):
        return [float(p) for i in qc.data for p in i.operation.params]

    def bind_values(self, values):
        present = set(self.circuit.parameters)
        out = self.circuit.assign_parameters({p: v for p, v in zip(self.order, values) if p in present})
        TEMPLATE_STATS["bound"] += 1
        return out

    def bind(self, qc):
        if _structure_sig(qc) != self.sig:
            TEMPLATE_STATS["structure_mismatch"] += 1
            raise ValueError("item 50: the circuit's structure differs from the template's")
        return self.bind_values(self.values_of(qc))


def fold_template(qc, target, seed=0, max_error=FAILED_MAX_ERROR):
    """Item 50: the FoldTemplate of `qc` on `target`, or None (not a ring classifier, no CZ, the zipper has fewer
    extra gates, no cycle by the formula, or an unmeasured circuit whose qubits would end permuted)."""
    from qiskit.circuit import ParameterVector
    if target is None or "cz" not in target.operation_names:
        return None
    try:
        order, _, _, _ = _fold_read(qc)
    except FoldRefused:
        A26_STATS["fold_not_ring"] += 1
        return None
    n = qc.num_qubits
    kind, placed = fold_choice(target, n, max_error)
    if kind == "zipper":
        A26_STATS["fold_zipper_fewer"] += 1
        return None
    values = [float(p) for i in qc.data for p in i.operation.params]
    pv = ParameterVector("f", len(values))
    par = qc.copy_empty_like()
    k = 0
    for ins in qc.data:
        op = ins.operation
        if op.params:
            op = op.to_mutable()
            op.params = [pv[k + j] for j in range(len(op.params))]
            k += len(op.params)
        par.append(op, ins.qubits, ins.clbits)
    _, layers, final, meas = _fold_read(par)
    try:
        if kind == "pend":
            m, _, cyc, pend = placed
            hc, init, fin = _fold_pend(n, layers, final, meas, cyc, pend, target.num_qubits, qc.num_clbits)
        else:
            m, _, cyc = placed
            hc, init, fin = _fold_holes(n, layers, final, meas, cyc, target.num_qubits, qc.num_clbits)
    except FoldRefused:
        A26_STATS["fold_no_cycle"] += 1
        return None
    measured = {q for q, _ in meas}
    if init != fin and len(measured) < n:
        A26_STATS["fold_permuted"] += 1
        return None
    pm = generate_preset_pass_manager(1, target=target, initial_layout=list(range(target.num_qubits)),
                                      routing_method="none", seed_transpiler=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tq = pm.run(hc)
    inv = {q: i for i, q in enumerate(order)}
    layout = [init[inv[q]] for q in range(n)]
    final_layout = [fin[inv[q]] for q in range(n)]
    A26_STATS["fold_pend" if kind == "pend" else "fold_holes"] += 1
    return FoldTemplate(_structure_sig(qc), tq, [pv[j] for j in range(len(values))], layout, final_layout, kind, m)


_A25_MODEL_TEMPLATE = model_template
_A25_COMPILE = compile_for_model_circuit


def model_template(qc, target, basis_gates=None, seed=0, max_error=0.5):  # noqa: F811
    """Item 50 (a26): an embeddable ring and every other input as a25; a ring classifier that does not embed is
    folded onto a device cycle by item 50's rule (FoldTemplate) when the rule picks pendants or holes; otherwise, and
    with PSF_RING_FOLD=0, a25's template."""
    if _fold_on() and target is not None and not _ring_embeds(qc, target, max_error):
        tpl = fold_template(qc, target, seed, max_error)
        if tpl is not None:
            A22_STATS["templates_ring"] += 1
            return tpl
    elif not _fold_on():
        A26_STATS["fold_off"] += 1
    return _A25_MODEL_TEMPLATE(qc, target, basis_gates, seed, max_error)


def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,  # noqa: F811
                              commutation_start=True, do_polish=True, layout_search=True, return_info=False,
                              target=None, **kwargs):
    """Item 50 (a26): as model_template -- a ring classifier that does not embed and that item 50's rule folds is
    built by the FoldTemplate and bound; everything else, and PSF_RING_FOLD=0, as a25."""
    seed = kwargs.get("seed_transpiler")
    seed = seed if seed is not None else (seeds[0] if seeds else 0)
    max_error = kwargs.get("prune_max_error", 0.5)
    if _fold_on() and target is not None and basis_gates is not None and not _ring_embeds(qc, target, max_error):
        tpl = fold_template(qc, target, seed, max_error)
        if tpl is not None:
            out = tpl.bind(qc)
            A19_STATS["formula"] += 1
            info = {"path": "formula", "chosen": "fold_" + tpl.schedule, "candidates": ["fold_" + tpl.schedule],
                    "version": AI_COMPILE_VERSION}
            return (out, info) if return_info else out
    return _A25_COMPILE(qc, coupling_map, basis_gates, entangling_basis=entangling_basis, seeds=seeds,
                        commutation_start=commutation_start, do_polish=do_polish, layout_search=layout_search,
                        return_info=return_info, target=target, **kwargs)
