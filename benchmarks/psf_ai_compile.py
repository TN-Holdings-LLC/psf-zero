"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-14.a18
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
Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

import psf_compile as pc

_A15_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches",  # release: from benchmarks/
                         "psf_ai_compile_a15_2026-10-14", "psf_ai_compile.py")
if not os.path.exists(_A15_FILE):  # run from outside the repository's patches folder: the repository root as cwd
    _A15_FILE = os.path.join(os.getcwd(), "patches", "psf_ai_compile_a15_2026-10-14", "psf_ai_compile.py")


def _load_a15():
    name = "psf_ai_compile_a15_for_a16"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _A15_FILE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


a15 = _load_a15()
a14 = a15.a14

AI_COMPILE_VERSION = "2026-10-14.a18"  # adopted 2026-10-14 with psf_compile 2026-10-14.1 (Addendum 492); a12 is psf_ai_compile_a12.py; prototype: a17 (workplace: items 28-31) + items 32-33 (no search; hand-off outside the ring formula)
SMALL_MAX_QUBITS = a15.SMALL_MAX_QUBITS
DEFAULT_SEEDS = a15.DEFAULT_SEEDS
CONSTRUCT_BY_DURATION = True  # items 23-29; False gives a14
SEARCH_FALLBACK = False  # item 27; False since a18 (item 32)
HAND_OFF = True  # item 33
RING_FORMULA = True  # item 28
RX_ONE_PULSE = True  # item 29
EXACT_CHECK = True  # item 30
EXACT_TVD = 1e-9  # item 30: largest total variation distance accepted between the candidate's and the input's outcomes
EXACT_STATE = 1e-12  # item 30: largest state infidelity accepted for an unmeasured input (item 39's check)
EXACT_MAX_ACTIVE = 20  # item 30: largest number of active qubits simulated for the check
A17_STATS = {"checked": 0, "inexact_dropped": 0, "unchecked": 0, "largest_tvd_dropped": 0.0, "all_inexact": 0}
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
    cycle, segments, tail, meas = read
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

    def do_czs(czs):
        for a, b in czs:
            out.cz(wire_at[pos[a]], wire_at[pos[b]])
    for layer, seg in enumerate(segments):
        for op, q in seg:
            out.append(op, [here(q)])
        if layer % 2 == 0:
            for czs, swaps in rounds:
                do_czs(czs)
                do_swaps(swaps)
        else:
            for czs, swaps in reversed(rounds):
                do_swaps(list(reversed(swaps)))
                do_czs(czs)
    for op, q in tail:
        out.append(op, [here(q)])
    for q, cb in meas:
        out.measure(here(q), cb)
    return out, p_of_wire


def _ring_candidate(qc, target, basis_gates, seed, rounds_fn=None):
    """Item 28: N2's formula on a line of the device, or None."""
    if not RING_FORMULA:
        return None
    read = _ring_read(qc)
    if read is None:
        A16_STATS["ring_not_read"] += 1
        return None
    if not (hasattr(pc, "_chain_lattice") and hasattr(pc, "_chain_place")):
        A16_STATS["ring_no_item87"] += 1
        return None
    n, N = qc.num_qubits, target.num_qubits
    lat = pc._chain_lattice(target, a14.FAILED_MAX_ERROR)
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
    if not RX_ONE_PULSE or target is None or "rx" not in target.operation_names:
        return c
    new = PassManager([Optimize1qGatesDecomposition(basis=["rz", "rx"])]).run(c)
    new._layout = getattr(c, "_layout", None)
    A16_STATS["rx_merged"] += 1
    return new


def _outcomes(circ):
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
        if done & set(qs) or ins.operation.num_clbits or not hasattr(ins.operation, "to_matrix") and \
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
    """Item 33: `qc` handed on unchanged -- with a Target to psf_compile's default call, without one to Qiskit's level
    3 (coupling map and basis given, seed seeds[0]) -- and checked exact as item 30 does where it can be."""
    seed = seeds[0] if seeds else 0
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if target is not None:
            out = pc.compile_for_hardware(qc, target=target, basis_gates=basis_gates, entangling_basis=entangling_basis,
                                          layout_search=layout_search, seed_transpiler=seed, **kwargs)
            label = "compiler"
            A18_STATS["handed_compiler"] += 1
        else:
            from qiskit import transpile
            cmap = coupling_map if coupling_map is None or isinstance(coupling_map, CouplingMap) else CouplingMap(coupling_map)
            out = transpile(qc, coupling_map=cmap, basis_gates=basis_gates, optimization_level=3, seed_transpiler=seed)
            label = "QK3"
            A18_STATS["handed_qiskit"] += 1
    verdict = _exact(qc, out)
    if verdict is None:
        A18_STATS["handoff_unchecked"] += 1
    elif verdict:
        A18_STATS["handoff_exact"] += 1
    else:
        A18_STATS["handoff_inexact"] += 1
        warnings.warn(f"psf_ai_compile item 33: the handed-on output ({label}) is not exact within EXACT_TVD; returned "
                      "unchanged", RuntimeWarning, stacklevel=3)
    info = {"path": "hand-off", "chosen": label, "candidates": [label], "version": AI_COMPILE_VERSION,
            "exact": verdict}
    return (out, info) if return_info else out


def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,
                              commutation_start=True, do_polish=True, layout_search=True, return_info=False,
                              target=None, **kwargs):
    """a18: for a ring classifier (item 28's reader), a16's candidates -- a15's (items 23-26), N2's ring formula and
    the second ring schedule (items 28, 31), single-qubit runs in one pulse with rx (item 29), every candidate checked
    exact (item 30), no search (item 32) -- the shortest returned; every other input handed on unchanged (item 33)."""
    if HAND_OFF and (not CONSTRUCT_BY_DURATION or target is None or basis_gates is None
                     or qc.num_qubits > SMALL_MAX_QUBITS or not hasattr(pc, "_better") or _ring_read(qc) is None):
        return _hand_off(qc, coupling_map, basis_gates, target, seeds, entangling_basis, layout_search, return_info,
                         kwargs)
    if (not CONSTRUCT_BY_DURATION or target is None or basis_gates is None or qc.num_qubits > SMALL_MAX_QUBITS
            or not hasattr(pc, "_better")):
        return a14.compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis=entangling_basis,
                                             seeds=seeds, commutation_start=commutation_start, do_polish=do_polish,
                                             layout_search=layout_search, return_info=return_info, target=target,
                                             **kwargs)
    A16_STATS["calls"] += 1
    seed = seeds[0] if seeds else 0

    def device_call(circ):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pc.compile_for_hardware(circ, target=target, basis_gates=basis_gates,
                                           entangling_basis=entangling_basis, layout_search=layout_search,
                                           seed_transpiler=seed, **kwargs)

    starts = a15._starts(qc) if commutation_start else [("input", qc)]
    cands = []  # (label, circuit), in a fixed order: earlier wins a full tie
    for sname, circ in starts:
        try:
            out = device_call(circ)
        except pc.FailedElementsError:
            raise
        except Exception:  # noqa: BLE001
            A16_STATS["compile_failed"] += 1
            continue
        cands.append((sname, out))
        if do_polish:
            cands.append((sname + "+polish", a14.polish(out, basis_gates)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ring = _ring_candidate(qc, target, basis_gates, seed)
    if ring is not None:
        cands.append(("ring", ring))
        with warnings.catch_warnings():  # item 31
            warnings.simplefilter("ignore")
            ring2 = _ring_candidate(qc, target, basis_gates, seed, ring_rounds_q4)
        if ring2 is not None:
            cands.append(("ring2", ring2))
    if SEARCH_FALLBACK:  # item 27: last, so it loses every tie
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cands.append(("search", a14.compile_for_model_circuit(
                    qc, coupling_map, basis_gates, entangling_basis=entangling_basis, seeds=seeds,
                    commutation_start=commutation_start, do_polish=do_polish, layout_search=layout_search,
                    target=target, **kwargs)))
        except Exception:  # noqa: BLE001
            A16_STATS["search_failed"] += 1
    cands = [(lb, _executable(_rx_merged(c, target), target)) for lb, c in cands]  # item 29
    cands = [(lb, c) for lb, c in cands if c is not None]
    cands = _exact_filter(qc, cands)  # item 30
    A16_STATS["candidates"] += len(cands)
    edges, bad = pc._failed_elements(target, a14.FAILED_MAX_ERROR)
    ok = [(lb, c) for lb, c in cands if not pc._uses_failed(c, edges, bad)]  # item 26
    if len(ok) < len(cands):
        A16_STATS["failed_skipped"] += len(cands) - len(ok)
    if not ok:
        A16_STATS["release_fallback"] += 1
        best, label = _executable(_rx_merged(device_call(qc), target), target), "REL"
        if best is not None and _exact(qc, best) is False:  # item 30: nothing exact was found; counted
            A17_STATS["all_inexact"] += 1
    else:
        label, best = ok[0]
        for lb, c in ok[1:]:
            if pc._better(c, best, target):  # item 25
                label, best = lb, c
    A16_STATS["chosen_start"][label] = A16_STATS["chosen_start"].get(label, 0) + 1
    if label == "search":
        A16_STATS["chosen_search"] += 1
    if label in ("ring", "ring2"):
        A16_STATS["chosen_ring"] += 1

    def _release():
        return _rx_merged(device_call(qc), target)

    kept = a14._keep_direction(best, target, _release)  # a11 (item 16)
    info = {"path": "small-constructed", "chosen": label, "candidates": [lb for lb, _ in cands],
            "version": AI_COMPILE_VERSION, "exact_check": dict(A17_STATS), "direction_fix": kept is not best}
    return (kept, info) if return_info else kept
