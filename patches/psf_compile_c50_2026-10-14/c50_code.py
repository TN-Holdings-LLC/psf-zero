

# ---------------------------------------------------------------------------------------------------------------------
# Prototype c50: item 86 (the device's native fractional gates: rzz and rx)
# ---------------------------------------------------------------------------------------------------------------------

FRACTIONAL = "auto"  # item 86: "auto" re-synthesises the output with rzz / rx when the Target offers rzz; False: never
FRAC_STATS = {"tried": 0, "kept": 0, "longer": 0, "not_isa": 0, "not_exact": 0, "unverified": 0, "error": 0}
_FRAC_DEPTH = [0]


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


def _isa_ok(circ, target):
    pos = {q: i for i, q in enumerate(circ.qubits)}
    names = set(target.operation_names)
    for ins in circ.data:
        nm = ins.operation.name
        if nm == "barrier":
            continue
        if nm not in names:
            return False
        qargs = tuple(pos[q] for q in ins.qubits)
        if qargs not in target[nm] and None not in target[nm]:
            return False
    return True


def _fractional(out, target):
    """Item 86: `out` re-synthesised on the same physical qubits with the device's native fractional gates, or None.

    Two-qubit blocks are consolidated and re-synthesised (Qiskit's ConsolidateBlocks and UnitarySynthesis on an
    error-free copy of the Target, so no reported error chooses), single-qubit runs are merged (rx is one pulse), rzz
    angles are folded into IBM's executable range 0 < theta <= pi/2 (qiskit-ibm-runtime's FoldRzzAngle). The placement
    and the layout are those of `out`. None when qiskit-ibm-runtime's pass is unavailable or the result is not native
    to the Target."""
    try:
        from qiskit.circuit.library import GlobalPhaseGate  # noqa: F401
        from qiskit.transpiler.passes import (Collect2qBlocks, ConsolidateBlocks, Optimize1qGatesDecomposition,
                                              UnitarySynthesis)
        from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
    except Exception:  # noqa: BLE001 - the item needs qiskit-ibm-runtime's fold; without it nothing changes
        FRAC_STATS["error"] += 1
        return None
    nt = _errorless(target)
    pm = PassManager([Collect2qBlocks(), ConsolidateBlocks(target=nt), UnitarySynthesis(target=nt),
                      Optimize1qGatesDecomposition(target=nt), FoldRzzAngle(),
                      Optimize1qGatesDecomposition(target=nt)])
    try:
        new = pm.run(out)
    except Exception:  # noqa: BLE001
        FRAC_STATS["error"] += 1
        return None
    phase = 0.0
    kept = []
    for ins in new.data:  # FoldRzzAngle may leave a GlobalPhaseGate: carried into the circuit's phase
        if ins.operation.name == "global_phase":
            phase += float(ins.operation.params[0])
        else:
            kept.append(ins)
    if phase:
        new.data = kept
        new.global_phase += phase
    new._layout = getattr(out, "_layout", None)
    if not _isa_ok(new, target):
        FRAC_STATS["not_isa"] += 1
        return None
    return new


def _fractional_choice(out, target):
    """Item 86: the re-synthesised circuit when it is shorter (item 84's `_better`) and, where item 39's check can be
    made, acts as `out` does; otherwise `out`."""
    if FRACTIONAL is False or target is None or "rzz" not in target.operation_names:
        return out
    FRAC_STATS["tried"] += 1
    new = _fractional(out, target)
    if new is None:
        return out
    if not _better(new, out, target):
        FRAC_STATS["longer"] += 1
        return out
    a = out.remove_final_measurements(inplace=False)
    b = new.remove_final_measurements(inplace=False)
    if len(_touched_qubits(a) | _touched_qubits(b)) <= RESYNTH_MAX_QUBITS:
        if not _same_action(a, b):
            FRAC_STATS["not_exact"] += 1
            return out
    else:
        FRAC_STATS["unverified"] += 1
    FRAC_STATS["kept"] += 1
    return new


def compile_for_hardware(qc, *args, **kwargs):
    """Item 86 (prototype c50): `_compile_for_hardware_c49` (the compiler as in c49, unchanged), then, on the outermost
    call only, the device's native fractional gates where they shorten the circuit (`_fractional_choice`).

    """
    _FRAC_DEPTH[0] += 1
    try:
        out = _compile_for_hardware_c49(qc, *args, **kwargs)
    finally:
        _FRAC_DEPTH[0] -= 1
    if _FRAC_DEPTH[0] == 0 and not kwargs.get("_inner", False):
        target = kwargs.get("target")
        if target is None and kwargs.get("backend") is not None:
            target = kwargs["backend"].target
        out = _fractional_choice(out, target)
    return out


compile_for_hardware.__doc__ += _compile_for_hardware_c49.__doc__ or ""
