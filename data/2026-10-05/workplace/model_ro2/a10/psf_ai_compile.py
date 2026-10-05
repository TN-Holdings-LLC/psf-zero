"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model (prototype).

The compiler itself is not changed: this module only calls `psf_compile.compile_for_hardware()` (and its
Rust core) and works around it. It is meant for what a model writes in the vLLM loop: small circuits
(a handful of qubits), many SWAPs and redundant gates, often given as PennyLane-style `unitary` gates,
compiled many times but each in milliseconds. Small circuits leave time for things a large-circuit path
cannot afford:

  1. a commutation-aware clean-up of the input (CommutativeCancellation) as a second starting point,
  2. several routing seeds per starting point, keeping the circuit with the fewest 2-qubit gates,
  3. a polish of the routed circuit: commutative cancellation plus re-synthesis (by the PSF-Zero core) of
     every 2-qubit block whose optimal CX count is below what it holds, repeated while it helps.

a1 (2026-10-01) adds, after the held-out test of a0 (Addendum TBD):
  4. when the input has gates on 3 or more qubits (ccz, ccx, cswap, ...), two more starting points with those
     gates decomposed (as written and commutatively cancelled), so the 2-qubit machinery sees them; a0's own
     starting points are kept first, so a1's candidate set contains a0's;
  5. a third starting point, only when the input has 2-qubit `unitary` gates: every 2-qubit gate re-expressed in
     CX form by `psf_compile.compile` and then commutatively cancelled, so `unitary` gates no longer hide
     commuting structure;
  6. one more candidate per starting point that routes from the initial layout Qiskit's level-3 search picks
     (only the layout is borrowed; synthesis and routing stay PSF-Zero's level-1 pipeline).

a2 (2026-10-01) adds, after a noisy-simulation test showed that once 2-qubit counts are close the physical qubits
chosen decide the fidelity (Addendum TBD), and only when a `target` with error rates is given:
  7. each of the best candidates (fewest 2-qubit gates, within REMAP_EXTRA_2Q of the best, at most REMAP_TOP) is
     re-placed on the device: every placement of its used qubits and couplers that preserves the couplings
     (subgraph isomorphism, up to MAX_MAPPINGS) is scored by the target's own gate errors, and the best is kept
     (like Qiskit's VF2PostLayout, applied to PSF-Zero's routed circuit);
  8. the final choice is by estimated fidelity (sum of -log(1 - error) over the gates), not by 2-qubit count;
  9. one more candidate per starting point routes from the initial layout Qiskit's level-3 search picks when it
     is given the target (error-aware); again only the layout is borrowed.
Without a target, a2 behaves exactly as a1.

a3 (2026-10-01) changes only the error used in 7-8. A replay of model-written circuits showed both error-aware
compilers (a2 and Qiskit level 3 with the target) behind error-blind Qiskit level 3 on FakeAuckland. The cause:
on some qubits the reported gate error is LOWER than what the same snapshot's T1/T2 and gate duration allow (qubit
24, T2 = 26 us: cx(24,25) reported 0.0055, decoherence limit 0.0089), so placing by reported error lures circuits
onto those qubits. a3 uses, per gate, max(reported error, decoherence limit), the limit being the average gate
infidelity of thermal relaxation (T1, T2 <= 2 T1) on each of the gate's qubits for the gate's duration.

a4 (2026-10-01) replaces the score itself. Exploration on development circuits showed that the sum of average gate
infidelities ranks the compiled candidates of one circuit no better than chance on FakeAuckland (63 of 117 pairs),
because relaxation and dephasing hardly affect a qubit that sits in a basis state while depolarizing noise affects
every state. a4 scores with a state-aware first-order estimate: every gate's error is split into Pauli components
(Pauli-twirled thermal relaxation per qubit from T1, T2 and the gate duration, plus the depolarizing remainder up to
the reported error), and each component P with probability p costs p * (1 - <P>^2), <P> taken on the ideal state of
the routed circuit right after the gate (statevector of the used qubits). The expectation values do not depend on
where the circuit is placed, so they are computed once per candidate and every placement is scored from
precomputed sums. (109 of 117 pairs ranked correctly on the same development circuits.) The model behind the
estimate makes the same physical assumptions as qiskit-aer's device noise model, so a noisy-simulation test is
favourable to it by construction; hardware is the real test.

a5 (2026-10-01) fixes a defect that only shows in long-running use: a3/a4 cached per-gate errors under the key
id(target). Python reuses the id of a freed object, so a new Target (for example after a calibration update)
could be served the old Target's numbers without any error (199 of 200 laps in a check that replaced the Target
every lap). a5 keys every cache entry by the values it was computed from (gate error, duration, T1 and T2 of the
gate's qubits), never by object identity, and bounds the caches' size. Results for one fixed Target are unchanged.

a7 (2026-10-02, home) adds one candidate, only when a `target` is given:
  12. Qiskit level 3's own output for the target (the transpile a2 already runs to borrow its layout) joins the
      candidates that are re-placed and scored by the state-aware estimate, whenever its two-qubit count is within
      REMAP_EXTRA_2Q of PSF-Zero's best (it does not count against REMAP_TOP). A diagnosis (Addendum 314) found a5 losing
      to that output on every F3 Heisenberg-chain circuit on FakeAuckland, on the same six qubits and with the same
      two-qubit count, while a5's own estimate ranked that output better in every case: a5's candidates never
      included it. The output is used as Qiskit returns it (not polished). No extra compile is made. Without a target,
      a7 behaves exactly as a5.

a8 (2026-10-04, home; adopted 2026-10-04, Addenda 336-338) changes only what happens above `SMALL_MAX_QUBITS` when a `target` is given:
  13. Until a7, such circuits went straight to `compile_for_hardware()` WITHOUT the target (the `target` argument
      was consumed by this function and never forwarded), so the compile saw neither error rates nor failed
      elements. WIDE (Addendum 335) measured the result at 9-10 logical qubits: 1.14-2.84 times the infidelity of
      the release's recommended call, and failed directions or qubits used in 302 of 1,188 circuits. a8 hands such
      circuits to the release's recommended call with the target (`target`, `placement_refine=True`,
      `final_resynthesis="select"`, `compare_level3=True`; caller kwargs override), which used no failed element in
      any test. Without a target, and at or below `SMALL_MAX_QUBITS`, a8 behaves exactly as a7.

a9 (2026-10-05, home; candidate) checks the one circuit it takes from Qiskit whole:
  14. Item 12 added Qiskit level 3's output for the target as a candidate without checking that it is equivalent to
      the input. On cx devices that output is exposed to Qiskit issue #17057 (Addendum 294): a workplace probe
      (Addendum 340) found level 3 wrong on 8 of 10 explicit near-boundary unitary circuits on FakeAuckland, the kind
      of `unitary` gate a model writes. a9 uses that output only if the release's `_implements` (psf_compile item
      39) confirms it; if the release has no such check, the output is not used. Everything else is a8's (its large-
      circuit path calls the release, which checks its own Qiskit-made circuits from item 39 on).

a10 (2026-10-05, workplace; candidate) counts readout in the state-aware estimate:
  15. The state-aware estimate (a4, `_state_weights` / `_score_weights`) skipped `measure`, so neither the choice among
      candidates nor the state-aware re-placement saw readout error; model-written circuits are sampled, so their
      measured qubits' readout enters every result. a10 records each measured circuit qubit once (weight 1) and charges
      the Target's measure error of the physical qubit it is mapped to. A circuit without measurements gets exactly
      a9's estimate (and output). The same blind spot in the release's item 38 is item 40 of candidate psf_compile
      2026-10-05.c13 (workplace READOUT test, 2026-10-05).

Circuits above `SMALL_MAX_QUBITS` go to `compile_for_hardware()`: without a target unchanged (a7), with a target by
the release's recommended call (a8, item 13).
The layout of the routed circuit (initial and final) is preserved by every step.
"""
import contextlib
import math

import numpy as np
import io
import warnings

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (BasisTranslator, Collect2qBlocks, CommutativeCancellation,
                                      ConsolidateBlocks, Optimize1qGatesDecomposition)

import psf_compile as pc

AI_COMPILE_VERSION = "2026-10-05.a10"  # candidate (workplace): a9 + readout of measured qubits in the state-aware estimate (item 15)
SMALL_MAX_QUBITS = 8
DEFAULT_SEEDS = (0, 1, 2, 3)
L3_LAYOUT_CANDIDATE = True
EXPANDED_START = True
REMAP_TOP = 6            # a2: candidates (fewest 2-qubit gates first) that are re-placed by error
REMAP_EXTRA_2Q = 2       # a2: ... within this many 2-qubit gates of the best
MAX_MAPPINGS = 5000      # a2: placements tried per candidate
L3T_LAYOUT_CANDIDATE = True
L3T_OUTPUT_CANDIDATE = True  # a7: level 3's own output (target given) is also a candidate
DECOHERENCE_FLOOR = True  # a3: a gate cannot be better than T1/T2 allow during its duration
STATE_AWARE = True  # a4: score placements and candidates by the state-aware first-order estimate  # a2: also route from Qiskit level 3's error-aware layout (target given)
POLISH_ROUNDS = 3
FAST_PATH_TARGET = True  # a8: above SMALL_MAX_QUBITS with a target, use the release's recommended call (item 13)
FAST_PATH_RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True)
L3T_CHECK_STATS = {"accepted": 0, "refused": 0, "unavailable": 0}  # a9 (item 14)


def _two_q(c):
    return sum(1 for inst in c.data if len(inst.qubits) == 2)


def _depth2q(c):
    return c.depth(lambda inst: len(inst.qubits) == 2)


def _resynthesize_blocks(circ, verify, tol, on_unsupported):
    """Consolidate every 2-qubit block that costs more CX than its optimum and re-synthesise it with the
    PSF-Zero core (CX basis). Returns a new circuit (layout not yet restored)."""
    pm = PassManager([
        Collect2qBlocks(filter_fn=lambda dag, block: pc._block_saves_cx(block)),
        ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True),
    ])
    blocked = pm.run(circ)
    idx = [i for i, inst in enumerate(blocked.data)
           if len(inst.qubits) == 2 and inst.operation.name == "unitary"]
    if not idx:
        return circ
    synth = pc.SU4GeodesicPSFSynthesizer(
        pc.GeodesicPSFHyper(tol=tol, on_unsupported=on_unsupported, entangling_basis="cx"), verify=verify)
    done = dict(zip(idx, synth.synthesize_many([blocked.data[i].operation.to_matrix() for i in idx])))
    out = blocked.copy_empty_like()
    out.global_phase = blocked.global_phase
    for i, inst in enumerate(blocked.data):
        if i in done:
            out.compose(done[i][0], inst.qubits, inplace=True)
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def polish(routed, basis_gates, rounds=POLISH_ROUNDS, verify=True, tol=1e-5, on_unsupported="keep"):
    """Commutative cancellation + block re-synthesis + translation, repeated while the 2-qubit count drops.
    Gates stay on the qubits they were routed to, so the result needs no re-routing."""
    layout = routed._layout
    tidy = PassManager([
        BasisTranslator(SessionEquivalenceLibrary, list(basis_gates)),
        Optimize1qGatesDecomposition(basis=list(basis_gates)),
    ])
    best = routed
    for _ in range(rounds):
        c = PassManager([CommutativeCancellation(basis_gates=list(basis_gates))]).run(best)
        c = _resynthesize_blocks(c, verify, tol, on_unsupported)
        c = tidy.run(c)
        c._layout = layout
        if _two_q(c) < _two_q(best):
            best = c
        else:
            break
    return best


def _split_multi_qubit(qc, max_rounds=6):
    """Decompose every gate acting on 3 or more qubits (repeatedly) into smaller gates."""
    for _ in range(max_rounds):
        names = sorted({inst.operation.name for inst in qc.data
                        if len(inst.qubits) >= 3 and inst.operation.name not in ("barrier", "measure")})
        if not names:
            break
        qc = qc.decompose(gates_to_decompose=names)
    return qc


def _decoherence_limit(target, qubits, duration):
    """Average gate infidelity of thermal relaxation on `qubits` for `duration` (closed form; the process
    fidelity of one qubit is (1 + 2 exp(-t/T2) + exp(-t/T1)) / 4, products over qubits, F_avg = (d F + 1)/(d + 1))."""
    if not duration:
        return 0.0
    qp = getattr(target, "qubit_properties", None)
    if not qp:
        return 0.0
    f = 1.0
    for q in qubits:
        p = qp[q] if q < len(qp) else None
        t1 = getattr(p, "t1", None) if p is not None else None
        t2 = getattr(p, "t2", None) if p is not None else None
        if not t1:
            return 0.0
        t2 = min(t2, 2 * t1) if t2 else 2 * t1
        f *= (1.0 + 2.0 * math.exp(-duration / t2) + math.exp(-duration / t1)) / 4.0
    d = 2 ** len(qubits)
    return 1.0 - (d * f + 1.0) / (d + 1.0)


_COST_CACHE = {}
_CACHE_LIMIT = 200_000  # a5: entries; a cache that grows past this is cleared (values only, never identities)


def _props_of(target, name, qubits):
    try:
        props = target[name].get(tuple(qubits))
        if props is None and len(qubits) == 2:
            props = target[name].get((qubits[1], qubits[0]))
    except KeyError:
        props = None
    return props


def _qubit_times(target, qubits):
    qp = getattr(target, "qubit_properties", None)
    out = []
    for q in qubits:
        p = qp[q] if qp and q < len(qp) else None
        out.append((getattr(p, "t1", None) if p is not None else None,
                    getattr(p, "t2", None) if p is not None else None))
    return tuple(out)


def _gate_cost(target, name, qubits):
    """-log(1 - error) of one gate from the target; 0 when the target has no error for it.
    a3: error = max(reported error, decoherence limit) when DECOHERENCE_FLOOR is on."""
    if name in ("rz", "barrier", "delay", "id", "measure", "reset"):
        return 0.0
    props = _props_of(target, name, qubits)
    key = (name, len(qubits), DECOHERENCE_FLOOR,
           getattr(props, "error", None) if props is not None else None,
           getattr(props, "duration", None) if props is not None else None,
           _qubit_times(target, qubits) if (DECOHERENCE_FLOOR and props is not None) else None)
    if key in _COST_CACHE:
        return _COST_CACHE[key]
    if len(_COST_CACHE) > _CACHE_LIMIT:
        _COST_CACHE.clear()
    err = getattr(props, "error", None) if props is not None else None
    err = err or 0.0
    if DECOHERENCE_FLOOR and props is not None:
        err = max(err, _decoherence_limit(target, list(qubits), getattr(props, "duration", None)))
    cost = -math.log(max(1e-12, 1.0 - err)) if err else 0.0
    _COST_CACHE[key] = cost
    return cost


def estimated_cost(circ, target):
    """Sum of -log(1 - error) over the circuit's gates (lower is better)."""
    total = 0.0
    for inst in circ.data:
        total += _gate_cost(target, inst.operation.name, [circ.find_bit(q).index for q in inst.qubits])
    return total


def _remap(out, perm):
    """Relabel the physical qubits of a routed circuit by the permutation `perm` (old -> new) and carry the
    initial and final layouts along."""
    from qiskit.transpiler import Layout, TranspileLayout
    new = out.copy_empty_like()
    new.global_phase = out.global_phase
    for inst in out.data:
        new.append(inst.operation, [new.qubits[perm[out.find_bit(q).index]] for q in inst.qubits], inst.clbits)
    L = out.layout
    init = Layout({vq: perm[p] for vq, p in L.initial_layout.get_virtual_bits().items()})
    fin = None
    if L.final_layout is not None:
        fin = Layout({perm[p]: new.qubits[perm[out.find_bit(q).index]]
                      for p, q in L.final_layout.get_physical_bits().items()})
    new._layout = TranspileLayout(init, L.input_qubit_mapping, fin, L._input_qubit_count, list(new.qubits))
    return new


def best_placement(out, target, coupling_map, max_mappings=MAX_MAPPINGS):
    """Return (circuit, cost): `out` re-placed on the device where the target's gate errors are lowest."""
    import rustworkx as rx
    used, edges = set(), {}
    one = {}
    for inst in out.data:
        qs = [out.find_bit(q).index for q in inst.qubits]
        used.update(qs)
        if len(qs) == 2:
            e = (qs[0], qs[1])
            edges[e] = edges.get(e, []) + [inst.operation.name]
        elif len(qs) == 1:
            one.setdefault(qs[0], []).append(inst.operation.name)
    if not edges:
        return out, estimated_cost(out, target)
    n = coupling_map.size()
    cg = rx.PyGraph()
    cg.add_nodes_from(range(n))
    seen = set()
    for a, b in coupling_map.get_edges():
        k = (min(a, b), max(a, b))
        if a != b and k not in seen:
            seen.add(k)
            cg.add_edge(a, b, None)
    U = sorted(used)
    idx = {q: i for i, q in enumerate(U)}
    pg = rx.PyGraph()
    pg.add_nodes_from(range(len(U)))
    pseen = set()
    for (a, b) in edges:
        k = (min(idx[a], idx[b]), max(idx[a], idx[b]))
        if k not in pseen:
            pseen.add(k)
            pg.add_edge(k[0], k[1], None)
    base = estimated_cost(out, target)
    best_cost, best_map = base, None
    count = 0
    for mapping in rx.vf2_mapping(cg, pg, subgraph=True, induced=False, call_limit=200 * max_mappings):
        inv = {pat: phys for phys, pat in mapping.items()}   # pattern node -> device qubit
        m = {U[i]: inv[i] for i in range(len(U)) if i in inv}
        if len(m) != len(U):
            continue
        c = 0.0
        for q, names in one.items():
            for nm in names:
                c += _gate_cost(target, nm, [m[q]])
        for (a, b), names in edges.items():
            for nm in names:
                c += _gate_cost(target, nm, [m[a], m[b]])
        if c < best_cost - 1e-15:
            best_cost, best_map = c, m
        count += 1
        if count >= max_mappings:
            break
    if best_map is None:
        return out, base
    perm = [None] * n
    for q, p in best_map.items():
        perm[q] = p
    free_new = [p for p in range(n) if p not in set(best_map.values())]
    it = iter(free_new)
    for q in range(n):
        if perm[q] is None:
            perm[q] = next(it)
    return _remap(out, perm), best_cost


_PAULI = {"X": np.array([[0, 1], [1, 0]], dtype=complex), "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex), "I": np.eye(2, dtype=complex)}


def _thermal_pauli(target, q, duration):
    """(p_x = p_y, p_z) of Pauli-twirled thermal relaxation of qubit q for `duration`."""
    qp = getattr(target, "qubit_properties", None)
    p = qp[q] if qp and q < len(qp) else None
    t1 = getattr(p, "t1", None) if p is not None else None
    if not t1 or not duration:
        return 0.0, 0.0
    t2 = getattr(p, "t2", None)
    t2 = min(t2, 2 * t1) if t2 else 2 * t1
    l1, l2 = math.exp(-duration / t1), math.exp(-duration / t2)
    pxy = (1.0 - l1) / 4.0
    return pxy, max(0.0, (1.0 - l2) / 2.0 - pxy)


_PARAM_CACHE = {}


def _gate_params(target, name, qubits):
    """(p_dep per non-identity Pauli, [(p_xy, p_z) per qubit]) for a gate on device qubits, or None."""
    props = _props_of(target, name, qubits)
    if props is None:
        return None
    key = (len(qubits), getattr(props, "error", None), getattr(props, "duration", None),
           _qubit_times(target, qubits))
    if key in _PARAM_CACHE:
        return _PARAM_CACHE[key]
    if len(_PARAM_CACHE) > _CACHE_LIMIT:
        _PARAM_CACHE.clear()
    dur = getattr(props, "duration", None) or 0.0
    rep = getattr(props, "error", None) or 0.0
    th = [_thermal_pauli(target, q, dur) for q in qubits]
    d = 2 ** len(qubits)
    fp = 1.0
    for pxy, pz in th:
        fp *= 1.0 - (2 * pxy + pz)
    e_th = 1.0 - (d * fp + 1.0) / (d + 1.0)
    e_dep = max(0.0, rep - e_th)
    out = (e_dep * (d + 1) / d / (d * d - 1), th)
    _PARAM_CACHE[key] = out
    return out


def _state_weights(circ):
    """Per (gate name, circuit qubits): summed weights of the state-aware estimate, from the ideal state after each
    gate. Returns {(name, qubits): [W_dep, [(w_xy, w_z) per qubit]]} (rz/id/barrier/delay/measure excluded)."""
    used = sorted({circ.find_bit(q).index for inst in circ.data for q in inst.qubits})
    idx = {q: i for i, q in enumerate(used)}
    m = len(used)
    psi = np.zeros(2 ** m, dtype=complex)
    psi[0] = 1.0
    psi = psi.reshape([2] * m)
    acc = {}
    for inst in circ.data:
        name = inst.operation.name
        if name == "measure":  # a10 (item 15): each measured qubit once, weight 1
            acc.setdefault(("measure", (circ.find_bit(inst.qubits[0]).index,)), [1.0, [[0.0, 0.0]]])
            continue
        if name in ("barrier", "delay", "reset"):
            continue
        qs = [circ.find_bit(q).index for q in inst.qubits]
        k = len(qs)
        U = inst.operation.to_matrix().reshape([2] * (2 * k))
        in_axes = [idx[qs[j]] for j in reversed(range(k))]
        psi = np.tensordot(U, psi, axes=(list(range(k, 2 * k)), in_axes))
        psi = np.moveaxis(psi, list(range(k)), in_axes)
        if name in ("rz", "id") or k > 2:
            continue
        v = np.moveaxis(psi, [idx[q] for q in qs], list(range(k))).reshape(2 ** k, -1)
        rho = v @ v.conj().T  # qs[0] is the most significant index here
        ent = acc.setdefault((name, tuple(qs)), [0.0, [[0.0, 0.0] for _ in qs]])
        if k == 1:
            ev = {pn: float(np.real(np.trace(rho @ _PAULI[pn]))) for pn in "XYZ"}
            ent[0] += sum(1 - ev[pn] ** 2 for pn in "XYZ")
            ent[1][0][0] += (1 - ev["X"] ** 2) + (1 - ev["Y"] ** 2)
            ent[1][0][1] += 1 - ev["Z"] ** 2
        else:
            for a in "IXYZ":
                for b in "IXYZ":
                    if a == b == "I":
                        continue
                    ev = float(np.real(np.trace(rho @ np.kron(_PAULI[a], _PAULI[b]))))
                    ent[0] += 1 - ev * ev
                    if b == "I" and a != "I":
                        ent[1][0][0 if a in "XY" else 1] += 1 - ev * ev
                    if a == "I" and b != "I":
                        ent[1][1][0 if b in "XY" else 1] += 1 - ev * ev
    return acc


def _score_weights(weights, target, mapping):
    total = 0.0
    for (name, qs), (wdep, wq) in weights.items():
        if name == "measure":  # a10 (item 15): the Target's measure error of the mapped qubit
            props = _props_of(target, "measure", [mapping[qs[0]]])
            if props is not None and getattr(props, "error", None) is not None:
                total += float(props.error) * wdep
            continue
        prm = _gate_params(target, name, [mapping[q] for q in qs])
        if prm is None:
            continue
        pdep, th = prm
        total += pdep * wdep
        for (pxy, pz), (wxy, wz) in zip(th, wq):
            total += pxy * wxy + pz * wz
    return total


def state_aware_cost(circ, target):
    """Estimated loss of fidelity of a routed circuit on the target (first order, state-aware)."""
    used = {circ.find_bit(q).index for inst in circ.data for q in inst.qubits}
    return _score_weights(_state_weights(circ), target, {q: q for q in used})


def best_placement_state_aware(out, target, coupling_map, max_mappings=MAX_MAPPINGS):
    """Like best_placement, scored by the state-aware estimate (weights computed once, placements from sums)."""
    import rustworkx as rx
    weights = _state_weights(out)
    used = sorted({out.find_bit(q).index for inst in out.data for q in inst.qubits})
    edges = {tuple(sorted(qs)) for (name, qs) in weights if len(qs) == 2}
    base = _score_weights(weights, target, {q: q for q in used})
    if not edges:
        return out, base
    n = coupling_map.size()
    cg = rx.PyGraph()
    cg.add_nodes_from(range(n))
    seen = set()
    for a, b in coupling_map.get_edges():
        k = (min(a, b), max(a, b))
        if a != b and k not in seen:
            seen.add(k)
            cg.add_edge(a, b, None)
    U = used
    pidx = {q: i for i, q in enumerate(U)}
    pg = rx.PyGraph()
    pg.add_nodes_from(range(len(U)))
    for a, b in edges:
        pg.add_edge(pidx[a], pidx[b], None)
    best_cost, best_map, count = base, None, 0
    for mapping in rx.vf2_mapping(cg, pg, subgraph=True, induced=False, call_limit=200 * max_mappings):
        inv = {pat: phys for phys, pat in mapping.items()}
        if len(inv) != len(U):
            continue
        mp = {U[i]: inv[i] for i in range(len(U))}
        c = _score_weights(weights, target, mp)
        if c < best_cost - 1e-15:
            best_cost, best_map = c, mp
        count += 1
        if count >= max_mappings:
            break
    if best_map is None:
        return out, base
    perm = [None] * n
    for q, p in best_map.items():
        perm[q] = p
    taken = set(best_map.values())
    it = iter([p for p in range(n) if p not in taken])
    for q in range(n):
        if perm[q] is None:
            perm[q] = next(it)
    return _remap(out, perm), best_cost


def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,
                              commutation_start=True, do_polish=True, layout_search=True, return_info=False,
                              target=None, **kwargs):
    """Best of several PSF-Zero compiles of a small circuit (fewest 2-qubit gates, then 2-qubit depth).

    kwargs are forwarded to `psf_compile.compile_for_hardware()`."""
    def cfh(circ, seed):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pc.compile_for_hardware(circ, coupling_map=coupling_map, basis_gates=basis_gates,
                                           entangling_basis=entangling_basis, layout_search=layout_search,
                                           seed_transpiler=seed, **kwargs)

    if qc.num_qubits > SMALL_MAX_QUBITS or basis_gates is None:
        if target is not None and basis_gates is not None and qc.num_qubits > SMALL_MAX_QUBITS and FAST_PATH_TARGET:
            # a8 (item 13): the release's recommended call with the target; caller kwargs override
            fk = dict(FAST_PATH_RECOMMENDED, target=target)
            fk.update(kwargs)
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = pc.compile_for_hardware(qc, coupling_map=coupling_map, basis_gates=basis_gates,
                                              entangling_basis=entangling_basis, layout_search=layout_search,
                                              seed_transpiler=seeds[0] if seeds else None, **fk)
            return (out, {"path": "fast-target", "version": AI_COMPILE_VERSION}) if return_info else out
        out = cfh(qc, seeds[0] if seeds else None)
        return (out, {"path": "fast"}) if return_info else out

    # a0's starting points first (unchanged), so a1 never loses a candidate a0 had
    starts = [("input", qc)]
    if commutation_start:
        cc = PassManager([CommutativeCancellation()]).run(qc)
        if _two_q(cc) < _two_q(qc) or [i.operation.name for i in cc.data] != [i.operation.name for i in qc.data]:
            starts.append(("commuted", cc))
    split = _split_multi_qubit(qc)
    if split is not qc:
        starts.append(("split", split))
        if commutation_start:
            starts.append(("split+commuted", PassManager([CommutativeCancellation()]).run(split)))
        qc = split
    if EXPANDED_START and any(len(i.qubits) == 2 and i.operation.name == "unitary" for i in qc.data):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex = pc.compile(qc, entangling_basis="cx")
        ex = PassManager([CommutativeCancellation()]).run(ex)
        starts.append(("expanded", ex))
    l3_layouts = []
    l3t_out = None  # a7
    if L3_LAYOUT_CANDIDATE:
        from qiskit import transpile
        try:
            t = transpile(qc, coupling_map=coupling_map, basis_gates=basis_gates, optimization_level=3,
                          seed_transpiler=0)
            l3_layouts.append(("L3layout", list(t.layout.initial_index_layout()[:qc.num_qubits])))
        except Exception:
            pass
        if target is not None and L3T_LAYOUT_CANDIDATE:
            # a2: the layout Qiskit's level-3 search picks when it can see the error rates
            try:
                t = transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
                l3t_out = t
                lay_t = list(t.layout.initial_index_layout()[:qc.num_qubits])
                if all(lay_t != l for _, l in l3_layouts):
                    l3_layouts.append(("L3Tlayout", lay_t))
            except Exception:
                pass
    tried = []
    cands = []
    best, best_key = None, None
    for sname, circ in starts:
        last = None
        # a0's starting points get every seed; the extra a1 starting points get the first seed only
        # (plus the level-3-layout candidate below), to bound the time
        for seed in (seeds if sname in ("input", "commuted") else seeds[:1]):
            out = cfh(circ, seed)
            if do_polish:
                out = polish(out, basis_gates)
            key = (_two_q(out), _depth2q(out))
            tried.append((sname, seed, key[0]))
            cands.append((key, out))
            if best_key is None or key < best_key:
                best, best_key = out, key
            # identical result from two seeds in a row: the layout search fixed the layout, stop early
            sig = (key, tuple(out.layout.initial_index_layout()[:qc.num_qubits]))
            if sig == last:
                break
            last = sig
        for lname, l3_layout in l3_layouts:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = pc.compile_for_hardware(circ, coupling_map=coupling_map, basis_gates=basis_gates,
                                              entangling_basis=entangling_basis, layout_search=False,
                                              initial_layout=l3_layout, seed_transpiler=seeds[0] if seeds else 0,
                                              **kwargs)
            if do_polish:
                out = polish(out, basis_gates)
            key = (_two_q(out), _depth2q(out))
            tried.append((sname, lname, key[0]))
            cands.append((key, out))
            if key < best_key:
                best, best_key = out, key
    if target is not None:
        top = sorted(cands, key=lambda kc: kc[0])
        top = [kc for kc in top if kc[0][0] <= top[0][0][0] + REMAP_EXTRA_2Q][:REMAP_TOP]
        if l3t_out is not None and L3T_OUTPUT_CANDIDATE:
            # a9 (item 14): only if the release confirms that level 3's output implements the input
            check = getattr(pc, "_implements", None)
            if check is None:
                L3T_CHECK_STATS["unavailable"] += 1
                l3t_out = None
            elif not check(qc, l3t_out):
                L3T_CHECK_STATS["refused"] += 1
                l3t_out = None
            else:
                L3T_CHECK_STATS["accepted"] += 1
        if l3t_out is not None and L3T_OUTPUT_CANDIDATE:
            # a7: always scored when within REMAP_EXTRA_2Q of PSF-Zero's best, not subject to the REMAP_TOP cut
            key = (_two_q(l3t_out), _depth2q(l3t_out))
            tried.append(("L3T", "output", key[0]))
            if key[0] <= top[0][0][0] + REMAP_EXTRA_2Q:
                top.append((key, l3t_out))
        scored = []
        for key, out in top:
            if STATE_AWARE:
                placed, cost = best_placement_state_aware(out, target, coupling_map)
            else:
                placed, cost = best_placement(out, target, coupling_map)
            scored.append((cost, key, placed, "L3T" if out is l3t_out else "PSF"))
        scored.sort(key=lambda t: (t[0], t[1]))
        best = scored[0][2]
        best_key = scored[0][1]
        chosen = scored[0][3]
        tried.append(("placement", "cost", round(scored[0][0], 6)))
    else:
        chosen = "PSF"
    info = {"path": "small", "tried": tried, "best": best_key, "version": AI_COMPILE_VERSION, "chosen": chosen}
    return (best, info) if return_info else best
