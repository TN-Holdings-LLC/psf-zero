"""Builds candidate 2026-10-10.c31 from release 2026-10-10.2 (psf_compile.py) by exact substitutions."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1, after=None):
    global s
    start = s.index(after) if after else 0
    head, tail = s[:start], s[start:]
    n = tail.count(old)
    assert n >= count and (count != 1 or n == 1 or after), (old[:80], n)
    s = head + tail.replace(old, new, count)


sub("VERSION: 2026-10-10.2 -- release, adopted on 2026-10-10 from candidate 2026-10-10.c30 of Addendum 426 (previous release: 2026-10-10.1)",
    "VERSION: 2026-10-10.c31 -- candidate (from release 2026-10-10.2): items 58 and 59a")
sub('VERSION = "2026-10-10.2"  # release (from candidate 2026-10-10.c30 of Addendum 426): 2026-10-10.1 + item 57b (the estimates\' and checks\' loops in psf_zero_core57, when installed)',
    'VERSION = "2026-10-10.c31"  # candidate: release 2026-10-10.2 + item 58 (failed elements avoided whenever the device is known, never returned silently) + item 59a (exactness margins recorded)')

sub("""    six qubits), the Python code runs. CORE57_STATS counts the calls made each way.
\"\"\"""", """    six qubits), the Python code runs. CORE57_STATS counts the calls made each way.
58. **SAFETY: a circuit that uses an element the device reports as failed is never returned silently (candidate
    2026-10-10.c31).** An output with a gate on a failed coupler or qubit gives no usable result on the device, and
    nothing in the output says so. Item 31 already avoided failed elements, but only when the caller passed `target`
    (the recommended call), and item 43 then returned the failing circuit with a warning when no placement avoided
    them. Now:
    - `backend=` (new) supplies the device: its Target, and from it the coupling map and the basis, unless given.
      `coupling_map` may be left out when `target` or `backend` is given. With a device, the default call avoids
      failed elements exactly as item 31 does; with the same arguments as before, every output is unchanged.
    - Without a Target, the first call in a process warns once that failed elements cannot be avoided
      (`WARN_WITHOUT_TARGET` switches this off).
    - A qubit whose measurement error is >= `prune_max_error` is a failed qubit too (`_failed_elements`,
      `prune_coupling_map`): a result read from it is noise.
    - After item 31's recompile on the pruned map the output is checked again. If no placement avoids the failed
      elements, or the output still uses one, `on_failed_elements="raise"` (the default) raises `FailedElementsError`
      instead of returning it; `"keep"` returns it with item 43's warning, as before.
59. **EXACTNESS (part a: measurement only; candidate 2026-10-10.c31).** Item 39 accepts a circuit when its state
    infidelity against the reference is <= `EXACT_TOL` (1e-6). Dropping a Z rotation of angle theta, as Qiskit
    level 3's CommutativeCancellation does below |theta| = 1.2566e-4 (Addendum 247), changes the infidelity by at
    most theta^2 / 4 (4e-9), far inside that tolerance. Before the tolerance is tightened, the infidelities that
    exact paths actually produce are recorded: `EXACT_SEEN` keeps, per check, the largest infidelity found and
    where (`_same_action`, `_implements`), up to `EXACT_SEEN_MAX` entries. No decision changes.
\"\"\"""")

# item 58: exception, switches, readout errors
sub("""PRUNE_STATS = {"calls": 0,""", """class FailedElementsError(TranspilerError):
    \"\"\"Item 58: no placement of the circuit avoids the couplers and qubits the device reports as failed.\"\"\"


WARN_WITHOUT_TARGET = True  # item 58: warn once per process when compile_for_hardware has no device Target
_WARNED = {"no_target": False}
_BACKEND_BASIS = ("cx", "cz", "ecr", "rz", "sx", "x", "id")  # item 58: the basis taken from a backend's Target


def readout_errors_from_target(target) -> dict:
    \"\"\"Item 58: `{q: error}` of `measure`, for the qubits that have an error value.\"\"\"
    out = {}
    if "measure" not in target.operation_names:
        return out
    for qargs, props in target["measure"].items():
        if qargs is None or props is None or props.error is None:
            continue
        out[qargs[0]] = props.error
    return out


PRUNE_STATS = {"calls": 0,""")
sub(""""unavoidable": 0}""", """"unavoidable": 0, "still_failed": 0, "raised": 0}""")
sub('''def _recompile_pruned(args, first):
    """Item 43: item 31's recompile on the pruned map, or `first` (with a warning) when no placement exists."""
    try:
        return compile_for_hardware(**args)
    except TranspilerError as exc:
        PRUNE_STATS["unavoidable"] += 1
        warnings.warn(f"no placement on the coupling map without the failed elements ({exc}); keeping the output "
                      f"that uses them (changelog item 43)", RuntimeWarning, stacklevel=3)
        return first''', '''def _recompile_pruned(args, first, edges=None, qubits=None, mode="keep"):
    """Item 43: item 31's recompile on the pruned map, or `first` (with a warning) when no placement exists.
    Item 58: with `edges` and `qubits`, the recompiled circuit is checked again; with `mode="raise"`, a circuit that
    cannot avoid the failed elements raises FailedElementsError instead of being returned."""
    try:
        out = compile_for_hardware(**args)
    except QiskitError as exc:  # item 58: TranspilerError, and CouplingError on a map left without edges
        PRUNE_STATS["unavoidable"] += 1
        if mode == "raise":
            PRUNE_STATS["raised"] += 1
            raise FailedElementsError(f"no placement avoids the failed couplers and qubits ({exc}) (changelog "
                                      f"item 58)") from exc
        warnings.warn(f"no placement on the coupling map without the failed elements ({exc}); keeping the output "
                      f"that uses them (changelog item 43)", RuntimeWarning, stacklevel=3)
        return first
    if edges is not None and _uses_failed(out, edges, qubits):
        PRUNE_STATS["still_failed"] += 1
        if mode == "raise":
            PRUNE_STATS["raised"] += 1
            raise FailedElementsError("the circuit compiled on the map without the failed couplers still uses a "
                                      "failed qubit (changelog item 58)")
        warnings.warn("the output still uses a failed qubit; keeping it (changelog item 58)", RuntimeWarning,
                      stacklevel=3)
    return out''')
# readout errors count as failed qubits
sub("""    qubits = {q for q, e in qubit_errors_from_target(target, qubit_gate).items() if e >= max_error}
    return edges, qubits""", """    qubits = {q for q, e in qubit_errors_from_target(target, qubit_gate).items() if e >= max_error}
    qubits |= {q for q, e in readout_errors_from_target(target).items() if e >= max_error}  # item 58
    return edges, qubits""")
sub("""    bad_q = {q for q, e in qubit_errors_from_target(target, qubit_gate).items() if e >= max_error}
    out = CouplingMap()""", """    bad_q = {q for q, e in qubit_errors_from_target(target, qubit_gate).items() if e >= max_error}
    bad_q |= {q for q, e in readout_errors_from_target(target).items() if e >= max_error}  # item 58
    out = CouplingMap()""")

# signature
sub("""def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,""", """def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: CouplingMap | None = None,""")
sub("""    _refine_target=None,
    _cancel_done: bool = False,
) -> QuantumCircuit:""", """    _refine_target=None,
    _cancel_done: bool = False,
    backend=None,
    on_failed_elements: str = "raise",
    _inner: bool = False,
) -> QuantumCircuit:""")
sub("""    call_args = dict(locals())  # item 50: this call's arguments, for the two runs of the pipeline
""", """    if backend is not None:  # item 58: the device, from which the Target, map and basis follow unless given
        if target is None:
            target = backend.target
        if basis_gates is None:
            basis_gates = [g for g in target.operation_names if g in _BACKEND_BASIS]
    if coupling_map is None:
        if target is None:
            raise ValueError("compile_for_hardware needs coupling_map, target or backend (changelog item 58)")
        coupling_map = target.build_coupling_map()
    if on_failed_elements not in ("raise", "keep"):
        raise ValueError('on_failed_elements must be "raise" or "keep" (changelog item 58)')
    if target is None and not _inner and WARN_WITHOUT_TARGET and not _WARNED["no_target"]:
        _WARNED["no_target"] = True
        warnings.warn("compile_for_hardware was called without the device's Target (target= or backend=): couplers "
                      "and qubits the device reports as failed cannot be avoided, and a result that uses one is "
                      "noise (changelog item 58). This warning is given once per process.", UserWarning, stacklevel=2)
    call_args = dict(locals())  # item 50: this call's arguments, for the two runs of the pipeline
""")
sub("""                    placement_call_limit=placement_call_limit, placement_max_trials=placement_max_trials,
                    _refine_target=target if placement_refine else None)""", """                    placement_call_limit=placement_call_limit, placement_max_trials=placement_max_trials,
                    _refine_target=target if placement_refine else None, _inner=True)""")
sub("""            args["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
            out = _recompile_pruned(args, out)""", """            args["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
            out = _recompile_pruned(args, out, edges, qubits, on_failed_elements)  # item 58""")
sub("""            kw = dict(call_args, qc=qc, _cancel_done=True)""", """            kw = dict(call_args, qc=qc, _cancel_done=True, _inner=True)""")

# item 59a: record the infidelities
sub("""EXACT_STATS = {"checked": 0, "refused_resynthesis": 0, "refused_floor": 0, "refused_level3": 0, "not_checkable": 0}
""", """EXACT_STATS = {"checked": 0, "refused_resynthesis": 0, "refused_floor": 0, "refused_level3": 0, "not_checkable": 0}
EXACT_SEEN = []  # item 59a: (where, largest state infidelity over the seeds) per check that was made
EXACT_SEEN_MAX = 10_000
""")
sub("""        x, y = _apply_ops(psi, a, pos), _apply_ops(psi, b, pos)
        if 1.0 - abs(np.vdot(x.ravel(), y.ravel())) ** 2 > tol:
            return False
    return True""", """        x, y = _apply_ops(psi, a, pos), _apply_ops(psi, b, pos)
        inf = 1.0 - abs(np.vdot(x.ravel(), y.ravel())) ** 2
        worst = max(worst, inf)
        if inf > tol:
            _seen("same_action", worst)
            return False
    _seen("same_action", worst)
    return True""")
sub("""    pos = {p: j for j, p in enumerate(touched)}
    for seed in EXACT_SEEDS:
        rng = np.random.default_rng(seed)
        psi = np.array(1.0 + 0j)""", """    pos = {p: j for j, p in enumerate(touched)}
    worst = 0.0
    for seed in EXACT_SEEDS:
        rng = np.random.default_rng(seed)
        psi = np.array(1.0 + 0j)""")
sub("""        red = np.moveaxis(red, [kept.index(fin[v]) for v in range(n)], list(range(n)))
        if 1.0 - abs(np.vdot(ideal.ravel(), red.ravel())) ** 2 > tol:
            return False
    return True""", """        red = np.moveaxis(red, [kept.index(fin[v]) for v in range(n)], list(range(n)))
        inf = 1.0 - abs(np.vdot(ideal.ravel(), red.ravel())) ** 2
        worst = max(worst, inf)
        if inf > tol:
            _seen("implements", worst)
            return False
    _seen("implements", worst)
    return True""")
sub("""    rest = [pos[p] for p in touched if p not in set(fin)]
    for seed in EXACT_SEEDS:""", """    rest = [pos[p] for p in touched if p not in set(fin)]
    worst = 0.0
    for seed in EXACT_SEEDS:""")
sub("""def _same_action(ref, new, tol=EXACT_TOL):""", """def _seen(where, value):
    \"\"\"Item 59a: records a check's largest infidelity.\"\"\"
    if len(EXACT_SEEN) < EXACT_SEEN_MAX:
        EXACT_SEEN.append((where, float(value)))


def _same_action(ref, new, tol=EXACT_TOL):""")
sub('''    changelog for why this exists.
    """
    if backend is not None:''', '''    changelog for why this exists.
    `backend`, `on_failed_elements` (candidate 2026-10-10.c31, item 58): `backend` supplies the device's Target, and
    from it `coupling_map` and `basis_gates` when they are not given; `coupling_map` may then be left out. With a
    Target, failed couplers and qubits (error >= `prune_max_error`, measurement included) are avoided as in item 31;
    when they cannot be, "raise" (default) raises `FailedElementsError` and "keep" returns the circuit with a warning.
    Without a Target, the first call in a process warns that they cannot be avoided.
    """
    if backend is not None:''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
