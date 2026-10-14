

# ---------------------------------------------------------------------------------------------------------------------
# Prototype c54: item 93, every output checked exact at a tight tolerance where it can be (home, after the workplace's
# Q3: Qiskit's two-qubit synthesis can snap a nearly degenerate block, 1.4e-5 in distribution, inside a level-3
# transpile)
# ---------------------------------------------------------------------------------------------------------------------

EXACT_CHECK = True  # item 93
EXACT_CHECK_TOL = 1e-6  # item 93: largest difference of an outcome probability (or state infidelity unmeasured)
EXACT93_STATS = {"checked": 0, "passed": 0, "not_checkable": 0, "failed": 0, "fallback_usual": 0, "fallback_level1": 0,
                 "raised": 0, "largest_passed": 0.0}
_X93_DEPTH = [0]
_X93_BASIS = ["cx", "cz", "rz", "sx", "x", "h", "rzz", "rx", "ry", "u", "measure"]  # standard names only (barriers kept)


def _dist93(circ, limit=RESYNTH_MAX_QUBITS):
    """Item 93: the exact outcome distribution over a circuit's classical bits ({bits: p}, clbit 0 rightmost) when every
    measurement is final and at most `limit` qubits are active; None otherwise."""
    from qiskit.quantum_info import Statevector
    meas, body, done = {}, [], set()
    for ins in circ.data:
        nm = ins.operation.name
        qs = [circ.find_bit(q).index for q in ins.qubits]
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = qs[0]
            done.add(qs[0])
            continue
        if nm in ("barrier", "delay"):
            continue
        if ins.clbits or any(q in done for q in qs) or getattr(ins.operation, "blocks", None) is not None:
            return None
        body.append((ins.operation, qs))
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    if not meas or len(active) > limit:
        return None
    pos = {q: i for i, q in enumerate(active)}
    small = QuantumCircuit(len(active))
    for op, qs in body:
        small.append(op, [pos[q] for q in qs])
    try:
        probs = Statevector(small).probabilities()
    except Exception:  # noqa: BLE001 - an instruction without a matrix: not checkable
        return None
    nclb = circ.num_clbits
    out = {}
    for idx in np.nonzero(probs > 1e-15)[0]:
        bits = ["0"] * nclb
        for cb, q in meas.items():
            if (int(idx) >> pos[q]) & 1:
                bits[nclb - 1 - cb] = "1"
        k = "".join(bits)
        out[k] = out.get(k, 0.0) + float(probs[idx])
    return out


def _exact93(qc, out):
    """Item 93: True if `out` gives `qc`'s outcome distribution within EXACT_CHECK_TOL (measured circuits), or implements
    it within EXACT_CHECK_TOL in state infidelity (unmeasured ones, item 39's check); False if it does not; None when
    the check cannot be made (more than RESYNTH_MAX_QUBITS active qubits, mid-circuit measurement, no matrix)."""
    measured = any(i.operation.name == "measure" for i in qc.data)
    if measured:
        try:
            ref_c = transpile(qc, basis_gates=_X93_BASIS, optimization_level=0)
        except Exception:  # noqa: BLE001
            return None
        a, b = _dist93(ref_c), _dist93(out)
        if a is None or b is None:
            return None
        diff = max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))
        if diff <= EXACT_CHECK_TOL:
            EXACT93_STATS["largest_passed"] = max(EXACT93_STATS["largest_passed"], diff)
            return True
        return False
    if qc.num_qubits > RESYNTH_MAX_QUBITS or len(_touched_qubits(out)) > RESYNTH_MAX_QUBITS:
        return None
    try:  # the input's own synthesis (a Pauli evolution's product formula, not its exact exponential)
        ref_u = transpile(qc, basis_gates=_X93_BASIS, optimization_level=0)
    except Exception:  # noqa: BLE001
        return None
    if _ops_of(ref_u) is None or _ops_of(out) is None:
        return None
    return bool(_implements(ref_u, out, tol=EXACT_CHECK_TOL))


def compile_for_hardware(qc, *args, **kwargs):
    """Item 93 (prototype c54): `_compile_for_hardware_c53g` (items 87-91 and everything before them, unchanged), and
    on the outermost call the output is checked exact at EXACT_CHECK_TOL where it can be (`_exact93`). If it is not
    exact, the usual path (`steiner=False, level3_candidate=False`) and then Qiskit's level-1 preset on the device are
    tried in that order, each checked the same way (and the second refused if it uses a failed element); the first
    exact one is returned. If none is, TranspilerError is raised: an inexact circuit is never returned silently.

    """
    if _X93_DEPTH[0] or not EXACT_CHECK:
        return _compile_for_hardware_c53g(qc, *args, **kwargs)
    _X93_DEPTH[0] += 1
    try:
        out = _compile_for_hardware_c53g(qc, *args, **kwargs)
    finally:
        _X93_DEPTH[0] -= 1
    verdict = _exact93(qc, out)
    if verdict is None:
        EXACT93_STATS["not_checkable"] += 1
        return out
    EXACT93_STATS["checked"] += 1
    if verdict:
        EXACT93_STATS["passed"] += 1
        return out
    EXACT93_STATS["failed"] += 1
    _X93_DEPTH[0] += 1
    try:
        try:
            alt = _compile_for_hardware_c53g(qc, *args, **dict(kwargs, steiner=False, level3_candidate=False))
        except (TranspilerError, ValueError):
            alt = None
        if alt is not None and _exact93(qc, alt):
            EXACT93_STATS["fallback_usual"] += 1
            return alt
        target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
        if target is not None:
            alt = generate_preset_pass_manager(1, target=target,
                                               seed_transpiler=kwargs.get("seed_transpiler")).run(qc)
            edges, bad = _failed_elements(target, kwargs.get("prune_max_error", 0.5))
            if not _uses_failed(alt, edges, bad) and _exact93(qc, alt):
                EXACT93_STATS["fallback_level1"] += 1
                return alt
    finally:
        _X93_DEPTH[0] -= 1
    EXACT93_STATS["raised"] += 1
    raise TranspilerError("no exact output at the tight tolerance (changelog item 93)")


compile_for_hardware.__doc__ += _compile_for_hardware_c53g.__doc__ or ""
