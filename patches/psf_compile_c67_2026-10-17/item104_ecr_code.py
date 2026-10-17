
# ======================================================================================================================
# Item 104 (prototype c67, home, 2026-10-17): devices whose two-qubit gate is ECR (IBM Eagle), by formula.
#
# Every formula path (items 87, 91, 99) builds circuits from CZ. CZ and ECR are the same two-qubit class: CZ(a, b) is
# ECR(a, b) with fixed single-qubit gates around it (below; checked at import). So on an ECR device the call is made on
# a CZ view of the device (`_e104_view`: CZ on every coupler that has ECR in either direction, with that ECR's error
# and duration; every other instruction as the device gives it), and every CZ of the output is written as the ECR the
# coupler has, in its direction, with its fixed gates; the single-qubit runs are then merged into rz, sx and x by
# Qiskit's one-qubit synthesis (a closed form per run, no search). Placement, scope and refusals are the CZ path's, on
# the view. Templates (compile_template) on ECR devices are not covered: OutOfScopeError as before.
# ----------------------------------------------------------------------------------------------------------------------
ECR104_STATS = {"calls": 0, "views": 0, "cz_written": 0}
_E104_VIEWS = {}


def _e104_cz_as_ecr():
    """CZ(x, y) as [(gate, qubit index in (x, y))...] around ECR(x, y), and the global phase: verified at import."""
    from qiskit.circuit.library import ECRGate, RZGate, SXGate
    pi = math.pi
    before = [(RZGate(3 * pi / 4), 1), (SXGate(), 1), (RZGate(pi / 2), 1)]
    after = [(SXGate(), 0), (SXGate(), 0), (RZGate(-pi / 2), 0), (RZGate(pi / 2), 1), (SXGate(), 1),
             (RZGate(-pi / 4), 1)]
    phase = pi / 4
    from qiskit.quantum_info import Operator
    c = QuantumCircuit(2, global_phase=phase)
    for g, q in before:
        c.append(g, [q])
    c.append(ECRGate(), [0, 1])
    for g, q in after:
        c.append(g, [q])
    cz = np.diag([1, 1, 1, -1]).astype(complex)
    if not np.allclose(Operator(c).data, cz, atol=1e-12):
        raise RuntimeError("item 104: the CZ-as-ECR form does not multiply out to CZ")
    return before, after, phase


_E104_FORM = _e104_cz_as_ecr()


def _e104_view(target):
    """The CZ view of an ECR device (see above), cached per Target object."""
    key = id(target)
    hit = _E104_VIEWS.get(key)
    if hit is not None and hit[0] is target:
        return hit[1]
    from qiskit.circuit.library import CZGate
    from qiskit.transpiler import InstructionProperties, Target
    view = Target(num_qubits=target.num_qubits, dt=target.dt, qubit_properties=target.qubit_properties,
                  description="item 104: CZ view of an ECR device")
    for name in target.operation_names:
        if name in ("ecr", "switch_case", "for_loop", "if_else", "while_loop"):
            continue
        view.add_instruction(target.operation_from_name(name), dict(target[name]))
    cz = {}
    for (a, b), p in target["ecr"].items():
        pr = None if p is None else InstructionProperties(error=p.error, duration=p.duration)
        for qa in ((a, b), (b, a)):
            if qa not in cz:
                cz[qa] = pr
    view.add_instruction(CZGate(), cz)
    _E104_VIEWS[key] = (target, view)
    ECR104_STATS["views"] += 1
    return view


def _e104_write(out, target):
    """`out` (built on the CZ view) with every cz written as the device's ECR, single-qubit runs merged."""
    from qiskit.circuit.library import ECRGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition
    before, after, phase = _E104_FORM
    ecr = target["ecr"]
    new = out.copy_empty_like()
    for ins in out.data:
        if ins.operation.name != "cz":
            new.append(ins.operation, ins.qubits, ins.clbits)
            continue
        qa, qb = ins.qubits
        a, b = out.find_bit(qa).index, out.find_bit(qb).index
        if (a, b) in ecr:
            x, y = qa, qb
        elif (b, a) in ecr:
            x, y = qb, qa
        else:
            raise TranspilerError(f"item 104: no ECR on the coupler ({a}, {b})")
        pair = (x, y)
        for g, k in before:
            new.append(g, [pair[k]])
        new.append(ECRGate(), [x, y])
        for g, k in after:
            new.append(g, [pair[k]])
        new.global_phase += phase
        ECR104_STATS["cz_written"] += 1
    merged = PassManager([Optimize1qGatesDecomposition(basis=["rz", "sx", "x"])]).run(new)
    merged._layout = getattr(out, "_layout", None)
    return merged


_e104_compile_cz = compile_for_hardware


def compile_for_hardware(qc, *args, **kwargs):  # noqa: F811  (item 104 wraps the CZ path)
    """Item 104 on items 97-99: on a device whose two-qubit gate is ECR (and not CZ), the CZ path on its CZ view, each
    CZ then written as the coupler's ECR (see the item's comment). Every other call is the CZ path unchanged."""
    target = kwargs.get("target") or getattr(kwargs.get("backend"), "target", None)
    if target is None or args or "cz" in target.operation_names or "ecr" not in target.operation_names:
        return _e104_compile_cz(qc, *args, **kwargs)
    ECR104_STATS["calls"] += 1
    kw = dict(kwargs)
    kw.pop("backend", None)
    kw["target"] = _e104_view(target)
    return _e104_write(_e104_compile_cz(qc, **kw), target)
