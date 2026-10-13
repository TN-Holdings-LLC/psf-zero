"""Build psf_ai_compile 2026-10-13.a14 from the adopted a12 (benchmarks/psf_ai_compile.py). Usage: make_a14.py SRC OUT"""
import sys

src, dst = sys.argv[1], sys.argv[2]
t = open(src, encoding="utf-8").read()


def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (old[:80], t.count(old))
    t = t.replace(old, new)


DOC = '''a14 (2026-10-13, workplace; candidate, not adopted) changes what happens at or below SMALL_MAX_QUBITS when a `target`
is given, after HW1 (Addendum 455) found that an estimate charging nothing for time mis-ranks compiled circuits on
the real device (design note: docs/findings/a14-design-2026-10-13.md, fixed before it was evaluated on any data):
  19. Every starting point also gets one compile by the release's call WITH the Target (first seed), so the release's
      device-aware steps (items 58-64) produce candidates too; they are always scored. a12's candidates are kept.
  20. The estimate adds an idle term: each qubit's first-order loss of state fidelity while it waits between
      instructions (as-soon-as-possible schedule from the Target's durations), from amplitude damping (T1) and pure
      dephasing (1/T_phi = 1/T2 - 1/(2 T1)), not Pauli-twirled: t/T1 [(1 - z)/2 - (x^2 + y^2)/4] + t/T_phi (1 - z^2)/2
      with (x, y, z) the qubit's reduced state after its previous instruction (|0> costs nothing, |1> t/T1, |+>
      t/(2 T2)). It is added in the re-placement (schedule of the candidate's own placement, T1/T2 of each
      placement) and in the final choice (each placed candidate re-scheduled). It assumes no dynamical decoupling.
  21. The DURATION_EXTRA candidates with the shortest scheduled duration are scored as well as a12's (fewest
      two-qubit gates within REMAP_EXTRA_2Q, at most REMAP_TOP).
  22. The returned circuit uses no element the Target reports failed (release `_failed_elements(target, 0.5)`): the
      best-scored candidate that avoids them is taken, else the release's recommended call (which raises
      FailedElementsError when nothing avoids them).
Without a target a14 is a12. Above SMALL_MAX_QUBITS nothing changes.

Circuits above `SMALL_MAX_QUBITS` go to'''
rep("\nCircuits above `SMALL_MAX_QUBITS` go to", "\n" + DOC)
import re
m = re.search(r'^AI_COMPILE_VERSION = .*$', t, re.M)
t = t[:m.start()] + ('AI_COMPILE_VERSION = "2026-10-13.a14"  # candidate (workplace, 2026-10-13): a12 + items 19-22 '
                     '(device-aware candidates, idle term, duration-diverse candidates, failed-element guarantee)') + t[m.end():]
rep('L3T_CHECK_STATS = {"accepted": 0, "refused": 0, "unavailable": 0}  # a9 (item 14)\n',
    'L3T_CHECK_STATS = {"accepted": 0, "refused": 0, "unavailable": 0}  # a9 (item 14)\n'
    'DEVICE_CANDIDATES = True  # a14 (item 19)\n'
    'IDLE_AWARE = True         # a14 (item 20)\n'
    'DURATION_EXTRA = 3        # a14 (item 21)\n'
    'FAILED_GUARD = True       # a14 (item 22)\n'
    'FAILED_MAX_ERROR = 0.5    # a14 (item 22): as the release\'s prune_max_error\n'
    'A14_STATS = {"device_candidates": 0, "device_failed": 0, "duration_extra": 0, "chosen_dev": 0,\n'
    '             "chosen_duration_extra": 0, "failed_skipped": 0, "release_fallback": 0}\n')

NEW = r'''
_IDLE_FREE = ("rz", "barrier", "id")


def _duration_of(target, name, qubits):
    """a14: an instruction's duration on device qubits from the Target, in seconds (0 when unknown or virtual)."""
    if name in _IDLE_FREE:
        return 0.0
    props = _props_of(target, name, list(qubits))
    d = getattr(props, "duration", None) if props is not None else None
    return float(d) if d else 0.0


def idle_loss_rate_terms(rho01_abs2, z):
    """a14 (item 20): (a, b) with loss = t/T1 * a + t/T_phi * b for a qubit with |rho_01|^2 and <Z> = z."""
    return (1.0 - z) / 2.0 - rho01_abs2, (1.0 - z * z) / 2.0


def idle_weights(circ, target):
    """a14 (item 20): ({circuit qubit: (A, B)}, makespan in seconds) of the as-soon-as-possible schedule of `circ` on
    its own qubits; A = sum of t * a, B = sum of t * b over the qubit's idle windows (see idle_loss_rate_terms)."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    used = sorted({pos[q] for inst in circ.data for q in inst.qubits})
    if not used:
        return {}, 0.0
    idx = {q: i for i, q in enumerate(used)}
    m = len(used)
    psi = np.zeros(2 ** m, dtype=complex)
    psi[0] = 1.0
    psi = psi.reshape([2] * m)
    avail = {q: 0.0 for q in used}
    state = {q: (0.0, 1.0) for q in used}  # (|rho01|^2, z): |0>
    measured = set()
    A = {q: 0.0 for q in used}
    B = {q: 0.0 for q in used}

    def charge(q, t):
        if t > 0.0 and q not in measured:
            a, b = idle_loss_rate_terms(*state[q])
            A[q] += t * a
            B[q] += t * b

    for inst in circ.data:
        name = inst.operation.name
        qs = [pos[q] for q in inst.qubits]
        if not qs:
            continue
        s = max(avail[q] for q in qs)
        for q in qs:
            charge(q, s - avail[q])
            avail[q] = s
        if name == "barrier":
            continue
        if name == "measure":
            d = _duration_of(target, name, qs)
            for q in qs:
                measured.add(q)
                avail[q] = s + d
            continue
        if name == "delay":
            d = float(getattr(inst.operation, "duration", 0) or 0)
            if getattr(inst.operation, "unit", "dt") == "dt":
                d *= float(getattr(target, "dt", None) or 0.0)
            for q in qs:
                charge(q, d)
                avail[q] = s + d
            continue
        d = _duration_of(target, name, qs)
        if name == "reset":
            for q in qs:
                avail[q] = s + d
            continue
        k = len(qs)
        U = inst.operation.to_matrix().reshape([2] * (2 * k))
        in_axes = [idx[qs[j]] for j in reversed(range(k))]
        psi = np.tensordot(U, psi, axes=(list(range(k, 2 * k)), in_axes))
        psi = np.moveaxis(psi, list(range(k)), in_axes)
        for q in qs:
            v = np.moveaxis(psi, idx[q], 0).reshape(2, -1)
            r00 = float(np.real(np.vdot(v[0], v[0])))
            r11 = float(np.real(np.vdot(v[1], v[1])))
            r01 = np.vdot(v[1], v[0])  # sum v0 * conj(v1)
            state[q] = (float(abs(r01) ** 2), r00 - r11)
            avail[q] = s + d
    T = max(avail.values())
    for q in used:
        charge(q, T - avail[q])
    return {q: (A[q], B[q]) for q in used}, T


def idle_cost(iw, target, mapping):
    """a14 (item 20): the idle term for placement `mapping` (circuit qubit -> device qubit)."""
    qp = getattr(target, "qubit_properties", None)
    if not qp:
        return 0.0
    total = 0.0
    for q, (a, b) in iw.items():
        p = mapping.get(q, q)
        pr = qp[p] if p < len(qp) else None
        t1 = getattr(pr, "t1", None) if pr is not None else None
        if not t1:
            continue
        t2 = getattr(pr, "t2", None)
        t2 = min(t2, 2.0 * t1) if t2 else 2.0 * t1
        total += a / t1 + b * max(0.0, 1.0 / t2 - 1.0 / (2.0 * t1))
    return total


def estimate_a14(circ, target):
    """a14: gate (a4) + readout (a10) + idle (item 20) estimate of a routed circuit where it stands."""
    pos = {q: i for i, q in enumerate(circ.qubits)}
    used = {pos[q] for inst in circ.data for q in inst.qubits}
    ident = {q: q for q in used}
    iw, _ = idle_weights(circ, target)
    return _score_weights(_state_weights(circ), target, ident) + idle_cost(iw, target, ident)


def best_placement_a14(out, target, coupling_map, max_mappings=MAX_MAPPINGS, cache=None, iw=None):
    """a14: best_placement_state_aware with the idle term added to every placement's score."""
    import rustworkx as rx
    if cache is None:
        cache = {}
    weights = _state_weights(out)
    if iw is None:
        iw, _ = idle_weights(out, target)
    used = sorted({out.find_bit(q).index for inst in out.data for q in inst.qubits})
    edges = {tuple(sorted(qs)) for (name, qs) in weights if len(qs) == 2}
    ident = {q: q for q in used}
    base = _score_weights(weights, target, ident) + idle_cost(iw, target, ident)
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
        c = _score_weights_cached(weights, target, mp, cache)
        if c != float("inf"):
            c += idle_cost(iw, target, mp)
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


def _uses_failed_target(circ, target):
    """a14 (item 22): True if `circ` uses an element the Target reports failed (error >= FAILED_MAX_ERROR)."""
    fe = getattr(pc, "_failed_elements", None)
    uf = getattr(pc, "_uses_failed", None)
    if fe is None or uf is None:
        return False
    edges, qubits = fe(target, FAILED_MAX_ERROR)
    if not edges and not qubits:
        return False
    return uf(circ, edges, qubits)


def compile_for_model_circuit('''
rep("\ndef compile_for_model_circuit(", NEW, 1)

# item 19: device-aware candidates, after each starting point's a12 candidates
rep('''            key = (_two_q(out), _depth2q(out))
            tried.append((sname, lname, key[0]))
            cands.append((key, out))
            if key < best_key:
                best, best_key = out, key
''', '''            key = (_two_q(out), _depth2q(out))
            tried.append((sname, lname, key[0]))
            cands.append((key, out))
            if key < best_key:
                best, best_key = out, key
        if target is not None and DEVICE_CANDIDATES:
            # a14 (item 19): the release's call with the Target (its device-aware steps), used as it returns it
            try:
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = pc.compile_for_hardware(circ, coupling_map=coupling_map, basis_gates=basis_gates,
                                                  entangling_basis=entangling_basis, layout_search=layout_search,
                                                  seed_transpiler=seeds[0] if seeds else None, target=target,
                                                  **kwargs)
                key = (_two_q(out), _depth2q(out))
                tried.append((sname, "device", key[0]))
                dev_cands.append((key, out))
                A14_STATS["device_candidates"] += 1
            except Exception:
                A14_STATS["device_failed"] += 1
''')
rep('''    tried = []
    cands = []
    best, best_key = None, None
''', '''    tried = []
    cands = []
    dev_cands = []  # a14 (item 19)
    best, best_key = None, None
''')

# items 19-21 in the scoring
rep('''        scored = []
        cache, done = {}, {}  # a12 (item 17): scoring terms of this call; re-placements of identical circuits
        for key, out in top:''', '''        labels = {id(out): "PSF" for _, out in top}
        if l3t_out is not None:
            labels[id(l3t_out)] = "L3T"
        iws = {}
        if IDLE_AWARE:
            # a14 (item 21): the shortest scheduled candidates are scored too
            for key, out in cands + dev_cands:
                iws[id(out)] = idle_weights(out, target)
            if DURATION_EXTRA:
                ids = {id(o) for _, o in top}
                extra = [kc for kc in sorted(cands, key=lambda kc: (iws[id(kc[1])][1], kc[0])) if id(kc[1]) not in ids]
                for kc in extra[:DURATION_EXTRA]:
                    top.append(kc)
                    labels[id(kc[1])] = "DUR"
                    A14_STATS["duration_extra"] += 1
        ids = {id(o) for _, o in top}
        for kc in dev_cands:  # a14 (item 19): always scored
            if id(kc[1]) not in ids:
                top.append(kc)
                labels[id(kc[1])] = "DEV"
        scored = []
        cache, done = {}, {}  # a12 (item 17): scoring terms of this call; re-placements of identical circuits
        for key, out in top:''')
rep('''            if sg is not None and sg in done:
                placed, cost = done[sg]
            elif STATE_AWARE:''', '''            if sg is not None and sg in done:
                placed, cost = done[sg]
            elif IDLE_AWARE:  # a14 (item 20)
                iw = iws.get(id(out))
                placed, cost = best_placement_a14(out, target, coupling_map, cache=cache,
                                                  iw=iw[0] if iw is not None else None)
            elif STATE_AWARE:''')
rep('''            scored.append((cost, key, placed, "L3T" if out is l3t_out else "PSF"))
        scored.sort(key=lambda t: (t[0], t[1]))
        best = scored[0][2]
        best_key = scored[0][1]
        chosen = scored[0][3]
        tried.append(("placement", "cost", round(scored[0][0], 6)))''', '''            if IDLE_AWARE and cost != float("inf"):
                cost = estimate_a14(placed, target)  # a14: final ranking on the placed circuit's own schedule
            scored.append((cost, key, placed, labels.get(id(out), "PSF")))
        scored.sort(key=lambda t: (t[0], t[1]))
        pick = 0
        if FAILED_GUARD:  # a14 (item 22)
            pick = next((i for i, s in enumerate(scored) if not _uses_failed_target(s[2], target)), None)
            if pick is None:
                A14_STATS["release_fallback"] += 1
                pick = -1
            elif pick > 0:
                A14_STATS["failed_skipped"] += 1
        if pick == -1:
            fk = dict(FAST_PATH_RECOMMENDED, target=target)
            fk.update(kwargs)
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best = pc.compile_for_hardware(qc, coupling_map=coupling_map, basis_gates=basis_gates,
                                               entangling_basis=entangling_basis, layout_search=layout_search,
                                               seed_transpiler=seeds[0] if seeds else None, **fk)
            best_key = (_two_q(best), _depth2q(best))
            chosen = "REL"
            tried.append(("placement", "cost", None))
        else:
            best = scored[pick][2]
            best_key = scored[pick][1]
            chosen = scored[pick][3]
            tried.append(("placement", "cost", round(scored[pick][0], 9)))
        if chosen == "DEV":
            A14_STATS["chosen_dev"] += 1
        if chosen == "DUR":
            A14_STATS["chosen_duration_extra"] += 1''')
open(dst, "w", encoding="utf-8").write(t)
print("ok", len(t))
