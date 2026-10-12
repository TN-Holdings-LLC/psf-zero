"""Builds prototype 2026-10-12.c45 from prototype 2026-10-11.c44 by exact substitutions:
  - item 78: item 75's re-placement at Qiskit's routing stage (`strict_direction=False`, before translation) no longer
    gives up when the circuit has gates the device does not run natively (swap, cx on a cz device, h, ...). Such gates
    are scored by what they will cost once translated: a two-qubit gate as k native two-qubit gates on that coupler
    (swap 3, gates locally equivalent to cx 1, others 2), a single-qubit gate as 0, 1 or 2 sx gates on that qubit.
    Diagnosis (gen_random_n48 on FakeBoston): Qiskit level 3's ESP is 0.029 with its VF2PostLayout and 5.5e-8 without
    it; c44 skipped this call on every circuit, so its routed circuits stayed where an error-blind layout put them.
    python make_c45.py <c44 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"expected {count} occurrence(s), found {n}: {old[:80]!r}")
    s = s.replace(old, new)


OLD_W = '''        weights = {}
        for name in {n for n, _ in one} | {n for n, _ in two}:
            if name not in t.operation_names:
                DET_STATS["skipped"] += 1
                return
            table = {}
            for qargs, props in t[name].items():
                e = props.error if props is not None and props.error is not None else 0.0
                table[None if qargs is None else tuple(qargs)] = math.inf if e >= 1.0 else -math.log1p(-max(e, 0.0))
            weights[name] = table

        def w(name, qargs):
            table = weights[name]
            v = table.get(qargs)
            if v is None and not self.strict and len(qargs) == 2:
                v = table.get(qargs[::-1])
            if v is None:
                v = 0.0 if None in table else math.inf
            return v
'''

NEW_W = '''        weights = {}
        native2 = next((g for g in ("cz", "ecr", "cx") if g in t.operation_names), None)
        proxy = {}  # item 78: names the device does not run, scored as what they translate to (non-strict only)
        for name in {n for n, _ in one} | {n for n, _ in two} | ({n for n in (native2, "sx") if n in t.operation_names} if self.proxy else set()):
            if name not in t.operation_names or self.proxy and name not in (native2, "sx", "measure"):
                arity = 2 if any(n == name for n, _ in two) else 1
                if not self.proxy or (arity == 2 and native2 is None) or (arity == 1 and "sx" not in t.operation_names):
                    DET_STATS["skipped"] += 1
                    return
                proxy[name] = (native2, _NATIVE2_COUNT.get(name, 2)) if arity == 2 else ("sx", _SX_COUNT.get(name, 2))
                continue
            table = {}
            for qargs, props in t[name].items():
                e = props.error if props is not None and props.error is not None else 0.0
                failed = e >= 1.0 or self.proxy and e >= DET_FAILED_ERROR  # item 78
                table[None if qargs is None else tuple(qargs)] = math.inf if failed else -math.log1p(-max(e, 0.0))
            weights[name] = table

        def w(name, qargs):
            k = 1
            if name in proxy:
                name, k = proxy[name]
                if k == 0:
                    return 0.0
            table = weights[name]
            v = table.get(qargs)
            if v is None and not self.strict and len(qargs) == 2:
                v = table.get(qargs[::-1])
            if v is None:
                v = 0.0 if None in table else math.inf
            return k * v
'''

OLD_E = '''            dev_edges = None
            for name in names2:
                ok = {tuple(q) for q in t[name] if q is not None}
                if not self.strict:
                    ok |= {(b, a) for a, b in ok}
                dev_edges = ok if dev_edges is None else dev_edges & ok
'''

NEW_E = '''            dev_edges = None
            for name in names2:
                name = proxy.get(name, (name, 1))[0]  # item 78
                ok = {tuple(q) for q in t[name] if q is not None}
                if not self.strict:
                    ok |= {(b, a) for a, b in ok}
                dev_edges = ok if dev_edges is None else dev_edges & ok
'''

OLD_STATS = '''DET_STATS = {"calls": 0, "applied": 0, "trials": 0, "skipped": 0}  # item 75
'''

NEW_STATS = '''DET_STATS = {"calls": 0, "applied": 0, "trials": 0, "skipped": 0}  # item 75

# Item 78: at Qiskit's routing stage the circuit is not yet in the device's gates. A two-qubit gate is scored as this
# many native two-qubit gates on its coupler (2 if not listed: a general two-qubit gate needs 3, most controlled
# rotations 2); a single-qubit gate as this many sx gates on its qubit (2 if not listed; diagonal gates are free, as rz
# is virtual on these devices).
_NATIVE2_COUNT = {"swap": 3, "cx": 1, "cz": 1, "ecr": 1, "cy": 1, "ch": 1, "dcx": 2, "iswap": 2}
_SX_COUNT = {"rz": 0, "p": 0, "u1": 0, "z": 0, "s": 0, "sdg": 0, "t": 0, "tdg": 0, "id": 0,
             "x": 1, "sx": 1, "sxdg": 1, "h": 1, "y": 1, "u2": 1}
'''

sub(OLD_W, NEW_W)
sub(OLD_E, NEW_E)
sub(OLD_STATS, NEW_STATS)


# item 79: basis gates from a Target given alone
sub('''    if backend is not None:  # item 58: the device, from which the Target, map and basis follow unless given
        if target is None:
            target = backend.target
        if basis_gates is None:
            basis_gates = [g for g in target.operation_names if g in _BACKEND_BASIS]
''', '''    if backend is not None:  # item 58: the device, from which the Target, map and basis follow unless given
        if target is None:
            target = backend.target
    if basis_gates is None and target is not None:  # item 79: also when the Target is given alone
        basis_gates = [g for g in target.operation_names if g in _BACKEND_BASIS]
''')

# item 80: Qiskit level 3, deterministic, as a candidate by ESP
sub('''    if steiner == "auto":  # item 68: decided before the device fills in the coupling map
        steiner = (backend is not None or target is not None) and coupling_map is None and initial_layout is None
''', '''    if steiner == "auto":  # item 68: decided before the device fills in the coupling map
        steiner = (backend is not None or target is not None) and coupling_map is None and initial_layout is None
    if level3_candidate == "auto":  # item 80: as item 68's "auto"
        level3_candidate = (backend is not None or target is not None) and coupling_map is None and \\
            initial_layout is None
''')
sub('''    near_failed_error: float | None = 0.1,
    _inner: bool = False,
) -> QuantumCircuit:''', '''    near_failed_error: float | None = 0.1,
    level3_candidate: Union[bool, str] = "auto",
    _inner: bool = False,
) -> QuantumCircuit:''')
sub('''        if not _inner:  # item 74
            out = _near_alternative(out, call_args, target, near_failed_error, prune_max_error)
''', '''        if not _inner:  # item 74
            out = _near_alternative(out, call_args, target, near_failed_error, prune_max_error)
        if not _inner and level3_candidate:  # item 80
            out = _level3_candidate(qc, out, target, prune_max_error, seed_transpiler)
''')
sub('''NEAR_STATS = {"checked": 0, "tried": 0, "kept_alternative": 0, "kept_first": 0, "no_alternative": 0}  # item 74
''', '''NEAR_STATS = {"checked": 0, "tried": 0, "kept_alternative": 0, "kept_first": 0, "no_alternative": 0}  # item 74

L3C_STATS = {"tried": 0, "kept_psf": 0, "kept_level3": 0, "failed_elements": 0, "not_exact": 0, "unverified": 0,
             "error": 0}  # item 80


def _level3_candidate(qc, out, target, max_error, seed_transpiler):
    """Item 80: `out`, or Qiskit's level-3 preset on the device's Target when its summed -ln(1 - error) on the device
    (item 69's measure) is lower. Qiskit's circuit is built under item 75's `_DeterministicVF2(target)`, so it is the
    same on every call; it is refused if it uses a failed element, or, where item 39's check can be made, if it does
    not implement the input. A tie keeps `out`."""
    L3C_STATS["tried"] += 1
    try:
        with _DeterministicVF2(target):
            pm = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=seed_transpiler)
        l3 = pm.run(qc)
    except (TranspilerError, QiskitError, ValueError):
        L3C_STATS["error"] += 1
        return out
    edges, qubits = _failed_elements(target, max_error)
    if _uses_failed(l3, edges, qubits):
        L3C_STATS["failed_elements"] += 1
        return out
    if not _st_cost(l3, target) < _st_cost(out, target):
        L3C_STATS["kept_psf"] += 1
        return out
    if _checkable_logical(qc) and not any(i.operation.name == "measure" for i in qc.data):
        if not _implements(qc, l3):
            L3C_STATS["not_exact"] += 1
            return out
    else:
        L3C_STATS["unverified"] += 1
    L3C_STATS["kept_level3"] += 1
    return l3
''')

sub("""        alt = compile_for_hardware(**dict(call_args, backend=None, target=marked, prune_max_error=max_error,
                                          near_failed_error=None, steiner=False, on_failed_elements="raise"))""",
    """        alt = compile_for_hardware(**dict(call_args, backend=None, target=marked, prune_max_error=max_error,
                                          near_failed_error=None, steiner=False, on_failed_elements="raise",
                                          level3_candidate=False))  # item 80 runs once, on the real Target""")

sub("""            return _DeterministicPostLayout(target, call_limit or 300_000, max_trials or 2_500, strict_direction)""",
    """            if not routing_stage:
                return _DeterministicPostLayout(target, call_limit or 300_000, max_trials or 2_500, strict_direction)
            p = _DeterministicPostLayout(target, min(call_limit or 300_000, DET_CALL_CAP),  # item 78
                                         min(max_trials or 2_500, DET_TRIALS_CAP), strict_direction)
            p.proxy = not strict_direction
            return p""")
sub("""    def __init__(self, target):
        self.target = target

    def __enter__(self):""", """    def __init__(self, target, routing_stage=False):
        self.target = target
        self.routing_stage = routing_stage  # item 78: score the routing stage's gates by their translated cost

    def __enter__(self):""")
sub("""        self.saved = [m.VF2PostLayout for m in self.mods]
        target = self.target
""", """        self.saved = [m.VF2PostLayout for m in self.mods]
        target = self.target
        routing_stage = self.routing_stage
""")
sub("""        self.max_trials = max_trials
        self.strict = strict_direction
""", """        self.max_trials = max_trials
        self.strict = strict_direction
        self.proxy = False  # item 78: set by _DeterministicVF2(target, routing_stage=True) for the non-strict call
""")
sub("""        with _DeterministicVF2(target):
            pm = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=seed_transpiler)""",
    """        with _DeterministicVF2(target, routing_stage=True):  # item 78
            pm = generate_preset_pass_manager(optimization_level=3, target=target, seed_transpiler=seed_transpiler)""")
sub("""DET_STATS = {"calls": 0, "applied": 0, "trials": 0, "skipped": 0}  # item 75
""", """DET_STATS = {"calls": 0, "applied": 0, "trials": 0, "skipped": 0}  # item 75
DET_TRIALS_CAP, DET_CALL_CAP = 25_000, 3_000_000  # item 78: per search order, whatever Qiskit's preset asks for
DET_FAILED_ERROR = 0.5  # item 78: at the routing stage, an element at or above this error is never placed on
""")

sub("""        cost1 = {}
        for (name, q), c in one.items():
            row = cost1.setdefault(q, [0.0] * N)
            for p in range(N):
                row[p] += c * w(name, (p,))
        zero = [0.0] * N
""", """        cost1 = {}
        for (name, q), c in one.items():
            row = cost1.setdefault(q, [0.0] * N)
            for p in range(N):
                row[p] += c * w(name, (p,))
        if self.proxy:  # item 78: no qubit with any operation goes to a failed qubit (gates come with translation)
            bad_p = set()
            for name in [n for n in ("sx", "x", "measure") if n in t.operation_names]:
                for qargs, props in t[name].items():
                    if qargs is not None and props is not None and props.error is not None and \\
                            props.error >= DET_FAILED_ERROR:
                        bad_p.add(qargs[0])
            for q in active:
                row = cost1.setdefault(q, [0.0] * N)
                for p in bad_p:
                    row[p] = math.inf
        zero = [0.0] * N
""")

sub('VERSION: 2026-10-11.c44 -- prototype (from c43 = release 2026-10-11.4): items 75 revised, 77',
    'VERSION: 2026-10-12.c45 -- prototype (from c44; c43 = release 2026-10-11.4): items 75 revised, 77-80')
sub('VERSION = "2026-10-11.c44"  # prototype: release 2026-10-11.4 + item 75 revised (search order) + item 77 (no time '
    'budget in the layout search)',
    'VERSION = "2026-10-12.c45"  # prototype: release 2026-10-11.4 + item 75 revised (search order) + item 77 (no time '
    'budget in the layout search) + items 78-80 (re-placement at the routing stage, basis from the Target, Qiskit level 3 as a candidate)')

OLD_TAIL = '''    path interactions (psf_smart_layout) keeps its own short time limit; it did not bind in those 154 compiles.
"""'''
NEW_TAIL = '''    path interactions (psf_smart_layout) keeps its own short time limit; it did not bind in those 154 compiles.
78. **ACCURACY: item 75's re-placement at Qiskit's routing stage, for item 80 (prototype c45).** Qiskit's preset pass
    managers re-place the routed circuit right after routing, before it is translated to the device's gates
    (`strict_direction=False`). Item 75 gave up there whenever the circuit had a gate the device does not run natively
    (swap, cx on a cz device, h, ...), that is, on almost every circuit. In item 80's level-3 compile
    (`_DeterministicVF2(target, routing_stage=True)`) such gates are now scored by what they translate to: every
    two-qubit gate as 1-3 native two-qubit gates on its coupler (swap 3, gates locally equivalent to cx 1, others 2:
    `_NATIVE2_COUNT`), every single-qubit gate but measure as 0-2 sx gates on its qubit (`_SX_COUNT`); an element of
    error >= `DET_FAILED_ERROR` (0.5) is never placed on, nor is any qubit with an operation placed on a qubit whose
    sx, x or readout error is (a qubit with only rz and measure there gets sx gates later; found on FakeBoston, qubit
    146). Qiskit's preset asks for up to 250,000 trials (30 million VF2 calls); scored in Python that took 38 s on a
    100-qubit Bernstein-Vazirani circuit, so each search order is capped at `DET_TRIALS_CAP` = 25,000 trials and
    `DET_CALL_CAP` = 3,000,000 calls (on five circuits: the same ESP as 250,000 trials, 4.8 s against 38 s on that
    circuit; 2,500 trials lost 22% there).
    Diagnosis: on gen_random_n48 and FakeBoston, Qiskit level 3's ESP is 0.029 with its own pass, 5.5e-8 without it,
    and 0.029 with this one. **PSF-Zero's own pipeline keeps c44's behaviour** (the routing-stage call is skipped as
    before): tried there, it did not help (ESP 0.996 times c44's on 118 runnable tests of C44-VAL, development data)
    and lost 45% on one (Bernstein-Vazirani n = 100, FakeTorino): the proxy's best placement before translation
    narrowed item 33's later search.
79. **FIX: the basis gates from a Target given alone (prototype c45).** `basis_gates` was taken from the Target only
    when `backend=` was given; with `target=` alone the output was routed but not translated (swap, h, cp, ... left in
    it: not runnable on the device). It is now taken from the Target in both cases.
80. **ACCURACY: Qiskit level 3 as a candidate, by ESP (prototype c45).** With the device (`backend=` or `target=`, and
    no `coupling_map` or `initial_layout` of the caller's: `level3_candidate="auto"`), the input is also compiled by
    Qiskit's level-3 preset on the device's Target, under item 75's deterministic re-placement, and that circuit is
    returned when its summed -ln(1 - error) on the device is lower than PSF-Zero's (a tie keeps PSF-Zero's). It is
    refused if it uses a failed element, or, where item 39's check can be made (at most `RESYNTH_MAX_QUBITS` qubits, no
    measurement), if it does not implement the input; above that size Qiskit's circuit is not checked (counted as
    "unverified" in `L3C_STATS`). `level3_candidate=False` turns it off. This is a stopgap, not a method: where it is
    chosen, PSF-Zero's own pipeline lost to Qiskit's on this device, mostly by an error-blind layout (item 78's
    diagnosis). Item 36's opt-in `compare_level3` (by `excitation_cost`, small circuits) is unchanged.
"""'''
sub(OLD_TAIL, NEW_TAIL)

open(dst, "w", encoding="utf-8").write(s)
print("wrote", dst)
