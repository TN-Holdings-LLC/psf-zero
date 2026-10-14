

# ---------------------------------------------------------------------------------------------------------------------
# Prototype c49: item 84 (choices by duration, not by the device's error table) and item 85 (failed qubits cut first)
# ---------------------------------------------------------------------------------------------------------------------

SELECT_BY = "duration"  # item 84: "duration" (c49) or "esp" (c48's summed -ln(1 - error))
SELECT_STATS = {"duration": 0, "cx_units": 0, "tie_2q": 0, "tie_kept": 0}
CUT_FAILED_QUBITS = True  # item 85: compile on the device without its failed qubits
CUT_STATS = {"cut": 0, "qubits_cut": 0, "too_few": 0, "no_placement": 0}


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
    if SELECT_BY == "esp":
        return ca < cb
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


def _working_target(target, cut):
    """Item 85: (a Target of the device without the qubits in `cut`, renumbered in order, and the list of the
    device's qubits it keeps). Every instruction, property (errors and durations as reported) and qubit property
    of the kept qubits is copied; operations offered on all qubits stay global."""
    from qiskit.transpiler import Target
    keep = [q for q in range(target.num_qubits) if q not in cut]
    idx = {q: i for i, q in enumerate(keep)}
    qprops = None
    if target.qubit_properties is not None:
        qprops = [target.qubit_properties[q] for q in keep]
    sub = Target(description=getattr(target, "description", None), num_qubits=len(keep), dt=target.dt,
                 granularity=target.granularity, min_length=target.min_length,
                 pulse_alignment=target.pulse_alignment, acquire_alignment=target.acquire_alignment,
                 qubit_properties=qprops)
    for name in target.operation_names:
        op = target.operation_from_name(name)
        if isinstance(op, type):  # variadic or control-flow operations: offered on every qubit
            sub.add_instruction(op, name=name)
            continue
        props = {}
        for qargs, p in target[name].items():
            if qargs is None:
                props[None] = p
            elif all(q in idx for q in qargs):
                props[tuple(idx[q] for q in qargs)] = p
        if not props:
            continue
        if None in props and len(props) == 1:
            sub.add_instruction(op, None, name=name)
        else:
            sub.add_instruction(op, props, name=name)
    return sub, keep


def _embed_working(o, keep, N):
    """Item 85: circuit `o`, compiled on `_working_target`'s renumbered device, written on the device's own N qubits
    (qubit i of `o` on keep[i]), with its layout carried over: the cut qubits hold fresh ancillas that no operation
    touches and that do not move."""
    from qiskit.circuit import AncillaRegister, QuantumRegister
    from qiskit.transpiler import Layout, TranspileLayout
    full = QuantumCircuit(QuantumRegister(N, "q"), *o.cregs, name=o.name, global_phase=o.global_phase,
                          metadata=dict(o.metadata or {}))
    loose = [c for c in o.clbits if not o.find_bit(c).registers]
    if loose:
        full.add_bits(loose)
    qmap = {o.qubits[i]: full.qubits[keep[i]] for i in range(len(keep))}
    for ins in o.data:
        full.append(ins.operation, [qmap[q] for q in ins.qubits], ins.clbits, copy=False)
    lay = getattr(o, "layout", None)
    if lay is None:
        return full
    cut = [q for q in range(N) if q not in set(keep)]
    anc = AncillaRegister(len(cut), "cut_failed") if cut else None
    v2p = {v: keep[p] for v, p in lay.initial_layout.get_virtual_bits().items()}
    if anc is not None:
        v2p.update({anc[k]: cut[k] for k in range(len(cut))})
    initial = Layout(v2p)
    for reg in lay.initial_layout.get_registers():
        initial.add_register(reg)
    if anc is not None:
        initial.add_register(anc)
    mapping = dict(lay.input_qubit_mapping)
    base = len(mapping)
    if anc is not None:
        mapping.update({anc[k]: base + k for k in range(len(cut))})
    final = None
    if lay.final_layout is not None:
        opos = {q: i for i, q in enumerate(o.qubits)}
        fl = {full.qubits[keep[opos[q]]]: keep[p] for q, p in lay.final_layout.get_virtual_bits().items()}
        fl.update({full.qubits[c]: c for c in cut})
        final = Layout(fl)
    full._layout = TranspileLayout(initial_layout=initial, input_qubit_mapping=mapping, final_layout=final,
                                   _input_qubit_count=lay._input_qubit_count, _output_qubit_list=list(full.qubits))
    return full


def _cut_failed_qubits(kw):
    """Item 85: the output of compile_for_hardware(**kw) compiled on the device without its failed qubits and written
    back on the device's own indices, or None when the device has no failed qubit (or too few working qubits and
    on_failed_elements="keep": the call then goes on as in c48). `kw` is the call's arguments, the device's Target
    resolved."""
    target, qc = kw["target"], kw["qc"]
    cut = sorted(_failed_elements(target, kw["prune_max_error"])[1])
    if not cut:
        return None
    if target.num_qubits - len(cut) < qc.num_qubits:
        CUT_STATS["too_few"] += 1
        if kw["on_failed_elements"] == "raise":
            raise FailedElementsError(
                f"the device has {target.num_qubits - len(cut)} working qubits (failed: {cut}) and the circuit "
                f"{qc.num_qubits} (changelog item 85)")
        return None
    sub, keep = _working_target(target, set(cut))
    CUT_STATS["cut"] += 1
    CUT_STATS["qubits_cut"] += len(cut)
    try:
        out = compile_for_hardware(**dict(kw, target=sub, backend=None))
    except TranspilerError as exc:  # FailedElementsError included: no placement on the working qubits
        if isinstance(exc, FailedElementsError):
            raise
        CUT_STATS["no_placement"] += 1
        if kw["on_failed_elements"] == "raise":
            raise FailedElementsError(f"no placement on the device's working qubits (failed: {cut}; changelog "
                                      f"item 85): {exc}") from exc
        return None
    return _embed_working(out, keep, target.num_qubits)
