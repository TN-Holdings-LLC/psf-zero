

# ---------------------------------------------------------------------------------------------------------------------
# Workplace proposal on c54 (2026-10-14): item 94, runs of commuting ZZ terms re-layered by edge colouring before the
# compile (R1: 3-regular QAOA)
# ---------------------------------------------------------------------------------------------------------------------

ZZ_RELAYER = True  # item 94 (proposal)
ZZ94_STATS = {"tried": 0, "not_read": 0, "unchanged": 0, "kept_relayered": 0, "kept_input": 0, "inexact": 0,
              "error": 0}
_ZZ94_DEPTH = [0]


def _zz_relayered(qc):
    """Item 94: `qc` with every maximal run of pairwise commuting two-qubit ZZ terms (items 72-73's reader, in their
    order) regrouped into layers of disjoint pairs by a first-fit edge colouring in that order, each layer written
    as rzz gates; other terms as their own PauliEvolutionGate on their support; the reader's leading gates first,
    its final gates and measurements last. Exact: only commuting terms change order. None when the circuit is not
    read, permutes its qubits, or no run changes."""
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    ev = _st_parse(qc)
    ev = ev + ([], None) if ev is not None else _st_parse_gates(qc)
    if ev is None:
        return None
    prefix, meas, triples, phase, suffix, perm = ev
    n = qc.num_qubits
    if perm and list(perm) != list(range(n)):
        return None
    out = QuantumCircuit(*qc.qregs, *qc.cregs, global_phase=phase)
    for op, q in prefix:
        out.append(op, [q])
    changed = [False]
    run = []

    def zz(lb):
        sup = [n - 1 - i for i, ch in enumerate(lb) if ch != "I"]
        return len(sup) == 2 and all(lb[n - 1 - q] == "Z" for q in sup) and tuple(sorted(sup))

    def flush():
        layers = []
        for (a, b), ang in run:
            for lay in layers:
                if a not in lay[0] and b not in lay[0]:
                    lay[1].append((a, b, ang))
                    lay[0].update((a, b))
                    break
            else:
                layers.append(({a, b}, [(a, b, ang)]))
        flat = [g for lay in layers for g in lay[1]]
        if [(a, b) for a, b, _ in flat] != [e for e, _ in run]:
            changed[0] = True
        for a, b, ang in flat:
            out.rzz(2 * ang, a, b)
        run.clear()

    for lb, a, t in triples:
        ang = a * t
        e = zz(lb)
        if e:
            run.append((e, ang))
            continue
        flush()
        sup = [n - 1 - i for i, ch in enumerate(lb) if ch != "I"]
        out.append(PauliEvolutionGate(SparsePauliOp("".join(lb[n - 1 - q] for q in reversed(sup))), time=ang), sup)
    flush()
    for op, w in suffix:
        out.append(op, [w])
    for q, cb in meas:
        out.measure(q, cb)
    return out if changed[0] else None


def compile_for_hardware(qc, *args, **kwargs):
    """Item 94 (workplace proposal on c54): `_compile_for_hardware_c54` (c54, item 93's exactness check included) on the
    input, and on the outermost call also on its ZZ-relayered rewrite (`_zz_relayered`, exact by construction); the
    rewrite's output is returned when item 84's rule (`_better`) prefers it and item 93's check against the input does
    not refuse it.

    """
    out = _compile_for_hardware_c54(qc, *args, **kwargs)
    if _ZZ94_DEPTH[0] or _X93_DEPTH[0] or _FRAC_DEPTH[0] or _CHAIN_DEPTH[0] or not ZZ_RELAYER or kwargs.get("_inner"):
        return out
    target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
    if target is None:
        return out
    ZZ94_STATS["tried"] += 1
    try:
        hl = _zz_relayered(qc)
    except Exception:  # noqa: BLE001
        ZZ94_STATS["error"] += 1
        return out
    if hl is None:
        ZZ94_STATS["unchanged"] += 1
        return out
    _ZZ94_DEPTH[0] += 1
    try:
        alt = _compile_for_hardware_c54(hl, *args, **kwargs)
    except (TranspilerError, ValueError, FailedElementsError):
        ZZ94_STATS["error"] += 1
        return out
    finally:
        _ZZ94_DEPTH[0] -= 1
    if not _better(alt, out, target):
        ZZ94_STATS["kept_input"] += 1
        return out
    if _exact93(qc, alt) is False:
        ZZ94_STATS["inexact"] += 1
        return out
    ZZ94_STATS["kept_relayered"] += 1
    return alt


compile_for_hardware.__doc__ += _compile_for_hardware_c54.__doc__ or ""
