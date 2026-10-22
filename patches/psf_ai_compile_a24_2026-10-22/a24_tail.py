# ---------------------------------------------------------------------------------------------------------------------
# a24 (home, 2026-10-22): item 46, the ring template built in Rust; item 47, formula-only placement (E5); item 48, the
# zipper schedule as an option.
# ---------------------------------------------------------------------------------------------------------------------

import ctypes  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

A24_STATS = {"rust_built": 0, "rust_none": 0, "rust_refused": 0, "checked": 0, "out_of_scope": 0, "zip_chosen": 0}
PSFBIND_RING_MIN = 2  # item 46: "psfbind ring r2" or later (r2 has the zipper)


def _zip_on(n):
    """Item 48: the zipper is a candidate only with PSF_RING_ZIP=1, for an even ring of n >= 4."""
    return os.environ.get("PSF_RING_ZIP", "") == "1" and n >= 4 and n % 2 == 0


def _snake_line(n, K, fadj, kadj, snake):
    """Item 47: the ring's line, the longest piece of the snake on the couplers of the most common gate length, else
    on every working coupler (psf_compile item 87's first rule, both by formula), or OutOfScopeError where only VF2's
    search would find one."""
    for adj in (fadj, kadj):
        line, how = pc._chain_place(n, K, adj, snake)
        if line is not None and how == "snake":
            return line
    A24_STATS["out_of_scope"] += 1
    raise OutOfScopeError("item 47 (formula only): no piece of the device's snake holds the "
                          f"{n}-qubit ring; a placement would need VF2's search, which is outside the scope")


def ring_rounds_zip(n):
    """Item 48: the zipper schedule (the workplace's prototype P1, 2026-10-22; TOQB's ring_zip.py, unchanged):
    (start order: ring index at each line position, rounds [(CZ pairs of ring indices, SWAPs (p, p + 1))])."""
    if n < 4 or n % 2:
        raise RuntimeError("zipper: n must be even and >= 4")
    top = list(range(1, n))
    bot = list(range(n - 1, 0, -1))
    order, used, t, b, from_top = [0], {0}, 0, 0, True
    while len(order) < n:
        src = top if from_top else bot
        k = 0
        while k < 2 and len(order) < n:
            i = t if from_top else b
            while src[i] in used:
                i += 1
            order.append(src[i])
            used.add(src[i])
            if from_top:
                t = i + 1
            else:
                b = i + 1
            k += 1
        from_top = not from_top
    occ = list(order)
    edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}

    def take(p):
        a, b_ = occ[p], occ[p + 1]
        e = (min(a, b_), max(a, b_))
        if e not in edges:
            raise RuntimeError(f"zipper: positions ({p}, {p + 1}) hold {a}, {b_}, not a ring edge")
        edges.discard(e)
        return (a, b_)

    rounds = []
    odd = list(range(1, n - 2, 2))
    rounds.append(([take(p) for p in odd], []))
    ends = [0, n - 2]
    czb = [take(p) for p in ends]
    rounds.append((czb, [(p, p + 1) for p in ends]))
    for p in ends:
        occ[p], occ[p + 1] = occ[p + 1], occ[p]
    mid = [(p, p + 1) for p in range(2, n - 3, 2)]
    if mid:
        rounds.append(([], mid))
        for p, p1 in mid:
            occ[p], occ[p1] = occ[p1], occ[p]
    rounds.append(([take(p) for p in odd], []))
    if edges:
        raise RuntimeError(f"zipper: edges left {sorted(edges)}")
    return order, rounds


_SCHED_FN = {"n2": None, "q4": ring_rounds_q4, "zip": ring_rounds_zip}


def _schedules(n):
    """Items 44 and 48: the schedules to build, in order (item 44's rule's, then the zipper when asked)."""
    first = "n2" if _ring_rounds_for(n)[0] is None else "q4"
    return [first, "zip"] if _zip_on(n) else [first]


# ----------------------------------------------------------------------------------------- item 46: the Rust bridge

def _f(x):
    return None if x is None else float(x)


def _device_json(target):
    names = sorted(target.operation_names)
    entries = []
    for name in names:
        for qargs, props in target[name].items():
            err = getattr(props, "error", None) if props is not None else None
            dur = getattr(props, "duration", None) if props is not None else None
            entries.append([name, None if qargs is None else [int(q) for q in qargs], _f(err), _f(dur)])
    return {"num_qubits": int(target.num_qubits), "names": names, "entries": entries}


def _input_json(qc):
    ops = []
    for ins in qc.data:
        op = ins.operation
        qs = [qc.find_bit(q).index for q in ins.qubits]
        cs = [qc.find_bit(c).index for c in ins.clbits]
        angle = mat = None
        if len(qs) == 1 and op.name not in ("measure", "barrier"):
            parameterised = getattr(op, "is_parameterized", lambda: False)()
            if op.name in ("rz", "ry", "rx"):
                if not parameterised:
                    angle = float(op.params[0])
            elif not parameterised:
                m = op.to_matrix()
                mat = [float(v) for z in (m[0][0], m[0][1], m[1][0], m[1][1]) for v in (complex(z).real,
                                                                                      complex(z).imag)]
        ops.append([op.name, qs, cs, angle, mat])
    return {"num_qubits": qc.num_qubits, "ops": ops}


_BLOCKS_JSON = []


def _blocks_json(max_len=4):
    """Every sequence of cz / swap up to `max_len`, decomposed by this module's `_block_ops` (Qiskit's
    TwoQubitBasisDecomposer, checked there): constants handed to the builder, not chosen by it."""
    if not _BLOCKS_JSON:
        out = []
        for m in range(1, max_len + 1):
            for kinds in itertools.product(("cz", "swap"), repeat=m):
                try:
                    seq = _block_ops(tuple(kinds))
                except RuntimeError:
                    continue
                ops = []
                for op in seq:
                    if op[0] == "cz":
                        ops.append(["cz"])
                    else:
                        M = op[2]
                        ops.append(["1q", int(op[1]), [float(v) for z in (M[0, 0], M[0, 1], M[1, 0], M[1, 1])
                                                       for v in (complex(z).real, complex(z).imag)]])
                out.append([list(kinds), ops])
        _BLOCKS_JSON.append(out)
    return _BLOCKS_JSON[0]


_PSFBIND = []


def _psfbind_path():
    env = os.environ.get("PSF_BIND_LIB")
    if env:
        return env
    roots = (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), os.getcwd())
    for root in roots:
        for name in ("libpsfbind.so", "libpsfbind.dylib", "psfbind.dll"):
            p = os.path.join(root, "rust", "psfbind", "target", "release", name)
            if os.path.exists(p):
                return p
    return None


def _psfbind():
    """Item 46: the library, checked to be "psfbind ring r2" or later; RuntimeError otherwise (no fallback)."""
    if _PSFBIND:
        return _PSFBIND[0]
    path = _psfbind_path()
    if path is None or not os.path.exists(path) or not path.endswith((".so", ".dylib", ".dll")):
        raise RuntimeError("item 46: the ring template is built by PSF-Zero's rust/psfbind, which was not found; "
                           "build it with `cargo build --release --manifest-path rust/psfbind/Cargo.toml` or set "
                           "PSF_BIND_LIB to the library")
    lib = ctypes.CDLL(path)
    try:
        lib.psfbind_version.restype = ctypes.c_char_p
        lib.psfbind_version.argtypes = []
        version = lib.psfbind_version().decode()
    except AttributeError:
        version = ""
    m = re.match(r"psfbind ring r(\d+)", version)
    if m is None or int(m.group(1)) < PSFBIND_RING_MIN:
        raise RuntimeError(f"item 46: {path} is an older psfbind ({version or 'no version'}); rebuild it with "
                           "`cargo build --release --manifest-path rust/psfbind/Cargo.toml`")
    lib.psfbind_ring_template.restype = ctypes.c_void_p
    lib.psfbind_ring_template.argtypes = [ctypes.c_char_p]
    lib.psfbind_free.argtypes = [ctypes.c_void_p]
    lib.psfbind_free.restype = None
    lib.version = version
    _PSFBIND.append(lib)
    return lib


def _rust_build(qc, target, schedules, max_error=FAILED_MAX_ERROR):
    req = {"device": _device_json(target), "input": _input_json(qc), "blocks": _blocks_json(),
           "max_error": float(max_error), "schedules": list(schedules)}
    lib = _psfbind()
    p = lib.psfbind_ring_template(json.dumps(req).encode())
    try:
        return json.loads(ctypes.string_at(p).decode())
    finally:
        lib.psfbind_free(p)


def _from_rust(t, qc):
    from qiskit.circuit import ParameterVector
    N = int(t["num_qubits"])
    pv = ParameterVector("psf_t", 3 * int(t["nvar"]))
    out = QuantumCircuit(N, name=qc.name)
    for cr in qc.cregs:
        out.add_register(cr)
    for name, qs, cs, p in t["ops"]:
        qs = [int(q) for q in qs]
        if name == "cz":
            out.cz(qs[0], qs[1])
        elif name == "measure":
            out.measure(qs[0], int(cs[0]))
        elif name == "sx":
            out.sx(qs[0])
        elif name in ("rz", "rx"):
            getattr(out, name)(pv[int(p["p"])] if isinstance(p, dict) else float(p), qs[0])
        else:
            raise RuntimeError(f"item 46: unknown operation {name!r} in the Rust template")
    var_runs = []
    for run in t["var_runs"]:
        factors = []
        for f in run:
            if len(f) == 8:
                factors.append(tuple(complex(f[2 * i], f[2 * i + 1]) for i in range(4)))
            else:
                factors.append(("v", int(f[0]), f[1]))
        var_runs.append(factors)
    var_kinds = [(int(k), nm) for k, nm in t["var_kinds"]]
    tpl = RingTemplate(_structure_sig(qc), out, var_runs, list(pv), bool(t["use_rx"]),
                       [int(q) for q in t["layout"]], var_kinds)
    tpl.schedule = t["schedule"]
    return tpl


def _ring_template_py(qc, target, basis_gates, seed, schedules):
    """a23's Python builder (item 47's placement) over `schedules`, item 84's rule choosing (a full tie keeps the
    earlier): the reference of PSF_RUST_TEMPLATE=check, and the builder of a ring the Rust does not port."""
    best, bc = None, None
    for name in schedules:
        try:
            t = _ring_template_one(qc, target, basis_gates, seed, _SCHED_FN[name])
        except RuntimeError:
            t = None
        if t is None:
            continue
        t.schedule = name
        c = t.bind(qc)
        if best is None or pc._better(c, bc, target):
            best, bc = t, c
    return best


def _ring_template_rust(qc, target, basis_gates=None, seed=0):
    """Item 46: the Rust template over item 44's (and 48's) schedules; item 47's refusal; the Python for a ring the
    Rust does not port; PSF_RUST_TEMPLATE=check compares with the Python's."""
    schedules = _schedules(qc.num_qubits)
    ans = _rust_build(qc, target, schedules)
    status = ans.get("status")
    if status == "out_of_scope":
        A24_STATS["out_of_scope"] += 1
        raise OutOfScopeError("item 47 (formula only): " + ans.get("reason", ""))
    if status == "refused":
        A24_STATS["rust_refused"] += 1
        tpl = _ring_template_py(qc, target, basis_gates, seed, schedules)
    elif status == "built":
        tpl = _from_rust(ans["template"], qc)
        A24_STATS["rust_built"] += 1
    elif status == "none":
        tpl = None
        A24_STATS["rust_none"] += 1
    else:
        raise RuntimeError(f"item 46: unexpected answer from rust/psfbind: {status!r}")
    if status != "refused" and os.environ.get("PSF_RUST_TEMPLATE", "") == "check":
        ref = _ring_template_py(qc, target, basis_gates, seed, schedules)
        diffs = (["one is None"] if (tpl is None) != (ref is None) else
                 [] if tpl is None else compare_templates(tpl, ref))
        if diffs:
            raise RuntimeError("item 46: the Rust template differs from the Python's: " + "; ".join(diffs[:5]))
        A24_STATS["checked"] += 1
    if tpl is not None and getattr(tpl, "schedule", None) == "zip":
        A24_STATS["zip_chosen"] += 1
    return tpl


def compare_templates(a, b, tol=1e-12):
    """Item 46: the differences between two RingTemplates (empty: identical; fixed angles and factors within `tol`)."""
    diffs = []
    if getattr(a, "schedule", None) != getattr(b, "schedule", None):
        diffs.append(f"schedule {getattr(a, 'schedule', None)} / {getattr(b, 'schedule', None)}")
    if list(a.layout) != list(b.layout):
        diffs.append(f"layout {a.layout} / {b.layout}")
    if bool(a.use_rx) != bool(b.use_rx):
        diffs.append("use_rx")
    if [tuple(v) for v in a.var_kinds] != [tuple(v) for v in b.var_kinds]:
        diffs.append("var_kinds")
    ca, cb = a.circuit, b.circuit
    if ca.num_qubits != cb.num_qubits or ca.num_clbits != cb.num_clbits:
        diffs.append("circuit sizes")
    if len(ca.data) != len(cb.data):
        diffs.append(f"gate count {len(ca.data)} / {len(cb.data)}")
    worst = 0.0
    for k, (x, y) in enumerate(zip(ca.data, cb.data)):
        qx = [ca.find_bit(q).index for q in x.qubits]
        qy = [cb.find_bit(q).index for q in y.qubits]
        cx = [ca.find_bit(c).index for c in x.clbits]
        cy = [cb.find_bit(c).index for c in y.clbits]
        if x.operation.name != y.operation.name or qx != qy or cx != cy:
            diffs.append(f"gate {k}: {x.operation.name}{qx}{cx} / {y.operation.name}{qy}{cy}")
            continue
        for px, py in zip(x.operation.params, y.operation.params):
            fx = not hasattr(px, "parameters") or not px.parameters
            fy = not hasattr(py, "parameters") or not py.parameters
            if fx != fy:
                diffs.append(f"gate {k}: fixed / parameter")
            elif fx:
                worst = max(worst, abs(float(px) - float(py)))
            elif str(px) != str(py):
                diffs.append(f"gate {k}: {px} / {py}")
    if worst > tol:
        diffs.append(f"fixed angles differ by {worst:.3e}")
    if len(a.var_runs) != len(b.var_runs):
        diffs.append("number of runs")
    fw = 0.0
    for ra, rb in zip(a.var_runs, b.var_runs):
        if len(ra) != len(rb):
            diffs.append("run length")
            continue
        for fa, fb in zip(ra, rb):
            if (len(fa) == 3) != (len(fb) == 3):
                diffs.append("factor kind")
            elif len(fa) == 3:
                if tuple(fa[1:]) != tuple(fb[1:]):
                    diffs.append(f"factor {fa} / {fb}")
            else:
                fw = max(fw, max(abs(complex(u) - complex(v)) for u, v in zip(fa, fb)))
    if fw > tol:
        diffs.append(f"fixed factors differ by {fw:.3e}")
    return diffs
