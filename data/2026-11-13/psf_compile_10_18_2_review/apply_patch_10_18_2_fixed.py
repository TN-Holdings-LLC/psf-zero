
# apply_patch_10_18_2.py として保存して python3 apply_patch_10_18_2.py で実行
from pathlib import Path

p = Path("psf_compile.py")
src = p.read_text(encoding="utf-8").replace("\u00a0", " ")

replacements = [
    # 0. Version bump
    (
        'VERSION = "2026-10-18.1"',
        'VERSION = "2026-10-18.2"',
    ),
    # 1. _st_flatten: O(1) bit lookup instead of repeated find_bit()
    (
        "    for ins in qc.data:\n"
        "        op = ins.operation\n"
        "        qs = [qc.find_bit(q).index for q in ins.qubits]\n"
        "        cs = [qc.find_bit(c).index for c in ins.clbits]",
        "    qmap = {q: i for i, q in enumerate(qc.qubits)}\n"
        "    cmap = {c: i for i, c in enumerate(qc.clbits)}\n"
        "    for ins in qc.data:\n"
        "        op = ins.operation\n"
        "        qs = [qmap[q] for q in ins.qubits]\n"
        "        cs = [cmap[c] for c in ins.clbits]",
    ),
    # 2. _st_parse_gates: fix 0.0-angle fallback bug on p/u1 & O(1) bit lookup
    (
        '            try:\n'
        '                lam = {"t": math.pi / 4, "tdg": -math.pi / 4}.get(name) or float(op.params[0])\n'
        '            except (TypeError, ValueError):\n'
        '                return False',
        '            try:\n'
        '                lam = (math.pi / 4 if name == "t" else -math.pi / 4) if name in ("t", "tdg") else float(op.params[0])\n'
        '            except (TypeError, ValueError, IndexError):\n'
        '                return False',
    ),
    (
        "    for ins in qc.data:\n"
        "        op = ins.operation\n"
        "        name = op.name\n"
        '        if name == "barrier":\n'
        "            continue\n"
        "        qs = [qc.find_bit(q).index for q in ins.qubits]\n"
        '        if name == "measure":\n'
        "            meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))",
        "    qmap = {q: i for i, q in enumerate(qc.qubits)}\n"
        "    cmap = {c: i for i, c in enumerate(qc.clbits)}\n"
        "    for ins in qc.data:\n"
        "        op = ins.operation\n"
        "        name = op.name\n"
        '        if name == "barrier":\n'
        "            continue\n"
        "        qs = [qmap[q] for q in ins.qubits]\n"
        '        if name == "measure":\n'
        "            meas.append((qs[0], cmap[ins.clbits[0]]))",
    ),
    # 3. _chain_lattice: cache per (id(target), max_error) with FIFO eviction
    (
        "_CHAIN_VF2_CALLS = 5_000_000\n\n\n"
        "def _chain_support(lb):",
        "_CHAIN_VF2_CALLS = 5_000_000\n"
        "_LATTICE_CACHE = {}\n"
        "_LATTICE_CACHE_MAX = 32\n\n\n"
        "def _chain_support(lb):",
    ),
    (
        "def _chain_lattice(target, max_error):\n"
        '    """Item 87: (K, its adjacency, the adjacency restricted to couplers of the device\'s most common two-qubit gate\n'
        "    length, the snake path) -- W5's cut: qubits with sx or readout error >= max_error and couplers with error >=\n"
        '    max_error removed, the largest connected component kept (ties: the smallest qubit)."""\n'
        "    N = target.num_qubits",
        "def _chain_lattice(target, max_error):\n"
        '    """Item 87: (K, its adjacency, the adjacency restricted to couplers of the device\'s most common two-qubit gate\n'
        '    length, the snake path) -- cached per (id(target), max_error)."""\n'
        "    _key = (id(target), max_error)\n"
        "    _hit = _LATTICE_CACHE.get(_key)\n"
        "    if _hit is not None and _hit[0] is target:\n"
        "        return _hit[1]\n"
        "    N = target.num_qubits",
    ),
    (
        "    if not comps:\n"
        "        return None\n"
        "    K = max(comps, key=lambda c: (len(c), -min(c)))\n"
        "    kadj = {q: {v for v in gadj[q] if v in K} for q in K}\n"
        "    lens = {}\n"
        "    for a in K:\n"
        "        for b in kadj[a]:\n"
        "            if a < b:\n"
        "                k = round(dur[(a, b)] * 1e12)\n"
        "                lens[k] = lens.get(k, 0) + 1\n"
        "    modal = max(lens, key=lambda x: (lens[x], -x)) if lens else None\n"
        "    fadj = {q: {v for v in kadj[q] if round(dur[(min(q, v), max(q, v))] * 1e12) == modal} for q in K}\n"
        "    return K, kadj, fadj, _chain_snake(N, adj)",
        "    if not comps:\n"
        "        _res = None\n"
        "    else:\n"
        "        K = max(comps, key=lambda c: (len(c), -min(c)))\n"
        "        kadj = {q: {v for v in gadj[q] if v in K} for q in K}\n"
        "        lens = {}\n"
        "        for a in K:\n"
        "            for b in kadj[a]:\n"
        "                if a < b:\n"
        "                    k = round(dur[(a, b)] * 1e12)\n"
        "                    lens[k] = lens.get(k, 0) + 1\n"
        "        modal = max(lens, key=lambda x: (lens[x], -x)) if lens else None\n"
        "        fadj = {q: {v for v in kadj[q] if round(dur[(min(q, v), max(q, v))] * 1e12) == modal} for q in K}\n"
        "        _res = (K, kadj, fadj, _chain_snake(N, adj))\n"
        "    if len(_LATTICE_CACHE) >= _LATTICE_CACHE_MAX:\n"
        "        _LATTICE_CACHE.pop(next(iter(_LATTICE_CACHE)))\n"
        "    _LATTICE_CACHE[_key] = (target, _res)\n"
        "    return _res",
    ),
    # 4. _chain_frame & _scope96_edges & _n99_read: O(1) bit lookup + FIFO cache eviction
    (
        "    f = [0] * n\n"
        "    for ins in log.data:\n"
        "        nm = ins.operation.name\n"
        "        qs = [log.find_bit(q).index for q in ins.qubits]\n"
        '        if nm == "barrier":\n'
        "            out.append(ins.operation, qs)\n"
        "            continue\n"
        '        if nm == "measure":\n'
        "            q = qs[0]\n"
        "            if f[q]:\n"
        "                out.x(q)\n"
        "                f[q] = 0\n"
        "            out.measure(q, log.find_bit(ins.clbits[0]).index)",
        "    f = [0] * n\n"
        "    qmap = {q: i for i, q in enumerate(log.qubits)}\n"
        "    cmap = {c: i for i, c in enumerate(log.clbits)}\n"
        "    for ins in log.data:\n"
        "        nm = ins.operation.name\n"
        "        qs = [qmap[q] for q in ins.qubits]\n"
        '        if nm == "barrier":\n'
        "            out.append(ins.operation, qs)\n"
        "            continue\n"
        '        if nm == "measure":\n'
        "            q = qs[0]\n"
        "            if f[q]:\n"
        "                out.x(q)\n"
        "                f[q] = 0\n"
        "            out.measure(q, cmap[ins.clbits[0]])",
    ),
    (
        "    edges = set()\n"
        "    for ins in qc.data:\n"
        '        if ins.operation.name in ("barrier", "measure", "delay"):\n'
        "            continue\n"
        "        qs = [qc.find_bit(q).index for q in ins.qubits]",
        "    edges = set()\n"
        "    qmap = {q: i for i, q in enumerate(qc.qubits)}\n"
        "    for ins in qc.data:\n"
        '        if ins.operation.name in ("barrier", "measure", "delay"):\n'
        "            continue\n"
        "        qs = [qmap[q] for q in ins.qubits]",
    ),
    (
        "        if len(_S96_GRAPH_CACHE) >= _S96_GRAPH_CACHE_MAX:\n"
        "            _S96_GRAPH_CACHE.clear()",
        "        if len(_S96_GRAPH_CACHE) >= _S96_GRAPH_CACHE_MAX:\n"
        "            _S96_GRAPH_CACHE.pop(next(iter(_S96_GRAPH_CACHE)))",
    ),
    (
        "    ops, meas, measured = [], [], set()\n"
        "    for ins in qc.data:\n"
        "        op = ins.operation\n"
        "        nm = op.name\n"
        '        if nm == "barrier":\n'
        "            continue\n"
        '        if getattr(op, "condition", None) is not None or getattr(op, "blocks", None) is not None:\n'
        "            return None\n"
        "        qs = tuple(qc.find_bit(q).index for q in ins.qubits)\n"
        '        if nm == "measure":\n'
        "            if qs[0] in measured:\n"
        "                return None\n"
        "            measured.add(qs[0])\n"
        "            meas.append((qs[0], qc.find_bit(ins.clbits[0]).index))",
        "    ops, meas, measured = [], [], set()\n"
        "    qmap = {q: i for i, q in enumerate(qc.qubits)}\n"
        "    cmap = {c: i for i, c in enumerate(qc.clbits)}\n"
        "    for ins in qc.data:\n"
        "        op = ins.operation\n"
        "        nm = op.name\n"
        '        if nm == "barrier":\n'
        "            continue\n"
        '        if getattr(op, "condition", None) is not None or getattr(op, "blocks", None) is not None:\n'
        "            return None\n"
        "        qs = tuple(qmap[q] for q in ins.qubits)\n"
        '        if nm == "measure":\n'
        "            if qs[0] in measured:\n"
        "                return None\n"
        "            measured.add(qs[0])\n"
        "            meas.append((qs[0], cmap[ins.clbits[0]]))",
    ),
    # 5. Item 104: Closed-form 1Q synthesis in _e104_write (eliminates Qiskit PassManager on ECR)
    (
        "_E104_FORM = _e104_cz_as_ecr()",
        "_E104_FORM = _e104_cz_as_ecr()\n"
        "_E104_BEFORE_T = [(_n99_t(g), k) for g, k in _E104_FORM[0]]\n"
        "_E104_AFTER_T = [(_n99_t(g), k) for g, k in _E104_FORM[1]]",
    ),
    (
        "def _e104_write(out, target):\n"
        '    """`out` (built on the CZ view) with every cz written as the device\'s ECR, single-qubit runs merged."""\n'
        "    from qiskit.circuit.library import ECRGate\n"
        "    from qiskit.transpiler import PassManager\n"
        "    from qiskit.transpiler.passes import Optimize1qGatesDecomposition\n"
        "    before, after, phase = _E104_FORM\n"
        '    ecr = target["ecr"]\n'
        "    new = out.copy_empty_like()\n"
        "    for ins in out.data:\n"
        '        if ins.operation.name != "cz":\n'
        "            new.append(ins.operation, ins.qubits, ins.clbits)\n"
        "            continue\n"
        "        qa, qb = ins.qubits\n"
        "        a, b = out.find_bit(qa).index, out.find_bit(qb).index\n"
        "        if (a, b) in ecr:\n"
        "            x, y = qa, qb\n"
        "        elif (b, a) in ecr:\n"
        "            x, y = qb, qa\n"
        "        else:\n"
        '            raise TranspilerError(f"item 104: no ECR on the coupler ({a}, {b})")\n'
        "        pair = (x, y)\n"
        "        for g, k in before:\n"
        "            new.append(g, [pair[k]])\n"
        "        new.append(ECRGate(), [x, y])\n"
        "        for g, k in after:\n"
        "            new.append(g, [pair[k]])\n"
        "        new.global_phase += phase\n"
        '        ECR104_STATS["cz_written"] += 1\n'
        '    merged = PassManager([Optimize1qGatesDecomposition(basis=["rz", "sx", "x"])]).run(new)\n'
        '    merged._layout = getattr(out, "_layout", None)\n'
        "    return merged",
        "def _e104_write(out, target):\n"
        '    """`out` (built on the CZ view) with every cz written as the device\'s ECR, single-qubit runs merged in closed form."""\n'
        "    from qiskit.circuit.library import ECRGate\n"
        "    _, _, phase = _E104_FORM\n"
        '    ecr = target["ecr"]\n'
        '    use_x = "x" in target.operation_names\n'
        "    new = out.copy_empty_like()\n"
        "    qmap = {q: i for i, q in enumerate(out.qubits)}\n"
        "    pending = {}\n"
        "    ecr_gate = ECRGate()\n\n"
        "    def flush(q_idx):\n"
        "        U = pending.pop(q_idx, None)\n"
        "        if U is None:\n"
        "            return\n"
        "        cls, ang = _n99_run_class(U, False)\n"
        '        if use_x and cls == "full" and abs(ang[1] - 2 * math.pi) < _N99_TOL:\n'
        "            new.rz(ang[0] - math.pi / 2, q_idx)\n"
        "            new.x(q_idx)\n"
        "            new.rz(ang[2] - math.pi / 2, q_idx)\n"
        "            return\n"
        "        _n99_emit_run(new, q_idx, cls, ang, False)\n\n"
        "    for ins in out.data:\n"
        "        nm = ins.operation.name\n"
        "        qs = [qmap[q] for q in ins.qubits]\n"
        '        if nm == "cz":\n'
        "            a, b = qs\n"
        "            if (a, b) in ecr:\n"
        "                x, y = a, b\n"
        "            elif (b, a) in ecr:\n"
        "                x, y = b, a\n"
        "            else:\n"
        '                raise TranspilerError(f"item 104: no ECR on the coupler ({a}, {b})")\n'
        "            pair = (x, y)\n"
        "            for mat, k in _E104_BEFORE_T:\n"
        "                pk = pair[k]\n"
        "                pending[pk] = _n99_mul(mat, pending.get(pk, _T100_ONE))\n"
        "            flush(x)\n"
        "            flush(y)\n"
        "            new.append(ecr_gate, [new.qubits[x], new.qubits[y]])\n"
        "            for mat, k in _E104_AFTER_T:\n"
        "                pk = pair[k]\n"
        "                pending[pk] = _n99_mul(mat, pending.get(pk, _T100_ONE))\n"
        "            new.global_phase += phase\n"
        '            ECR104_STATS["cz_written"] += 1\n'
        '        elif len(qs) == 1 and not ins.clbits and nm not in ("barrier", "delay", "measure"):\n'
        "            q_idx = qs[0]\n"
        "            pending[q_idx] = _n99_mul(_n99_t(ins.operation), pending.get(q_idx, _T100_ONE))\n"
        "        else:\n"
        "            for q_idx in qs:\n"
        "                flush(q_idx)\n"
        "            new.append(ins.operation, [new.qubits[i] for i in qs], ins.clbits)\n"
        "    for q_idx in sorted(pending):\n"
        "        flush(q_idx)\n"
        '    new._layout = getattr(out, "_layout", None)\n'
        "    return new",
    ),
    # 6. Item 105: Fast-path QASM3 dump (skip transpile when already in basis) & binary path cache
    (
        '_PSFZ105_BASIS = ["rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "sx", "p", "cx", "cz", "rzz", "rxx", "ryy"]\n'
        '_PSFZ105_KEYS = {"target", "backend", "seed_transpiler", "prune_max_error"}\n'
        "_PSFZ105_DEVICES = {}  # id(target) -> (target, device file text)\n\n\n"
        "def _psfz105_program():\n"
        '    """Item 105: psfz\'s executable, or None."""\n'
        '    p = _os105.environ.get("PSFZ_BIN")\n'
        "    if p:\n"
        "        return p if _os105.path.isfile(p) else None\n"
        "    here = _os105.path.dirname(_os105.path.abspath(__file__))\n"
        "    for up in (here, _os105.path.dirname(here), _os105.path.dirname(_os105.path.dirname(here))):\n"
        '        for name in ("psfz", "psfz.exe"):\n'
        '            c = _os105.path.join(up, "rust", "psfz", "target", "release", name)\n'
        "            if _os105.path.isfile(c):\n"
        "                return c\n"
        "    return None",
        '_PSFZ105_BASIS = ["rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "sx", "p", "cx", "cz", "rzz", "rxx", "ryy"]\n'
        '_PSFZ105_DIRECT_OPS = frozenset(_PSFZ105_BASIS) | frozenset(("measure", "barrier"))\n'
        '_PSFZ105_KEYS = {"target", "backend", "seed_transpiler", "prune_max_error"}\n'
        "_PSFZ105_DEVICES = {}  # id(target) -> (target, device file text)\n"
        "_PSFZ105_BIN_CACHE = [None]\n\n\n"
        "def _psfz105_program():\n"
        '    """Item 105: psfz\'s executable, or None (cached unless $PSFZ_BIN is set)."""\n'
        '    p = _os105.environ.get("PSFZ_BIN")\n'
        "    if p:\n"
        "        return p if _os105.path.isfile(p) else None\n"
        "    cached = _PSFZ105_BIN_CACHE[0]\n"
        "    if cached is not None and _os105.path.isfile(cached):\n"
        "        return cached\n"
        "    here = _os105.path.dirname(_os105.path.abspath(__file__))\n"
        "    for up in (here, _os105.path.dirname(here), _os105.path.dirname(_os105.path.dirname(here))):\n"
        '        for name in ("psfz", "psfz.exe"):\n'
        '            c = _os105.path.join(up, "rust", "psfz", "target", "release", name)\n'
        "            if _os105.path.isfile(c):\n"
        "                _PSFZ105_BIN_CACHE[0] = c\n"
        "                return c\n"
        "    return None",
    ),
    (
        "    if len(_PSFZ105_DEVICES) >= 16:\n"
        "        _PSFZ105_DEVICES.clear()",
        "    if len(_PSFZ105_DEVICES) >= 16:\n"
        "        _PSFZ105_DEVICES.pop(next(iter(_PSFZ105_DEVICES)))",
    ),
    (
        "    try:\n"
        "        from qiskit import qasm3\n"
        "        from qiskit import transpile as _transpile105\n"
        "        src = qasm3.dumps(_transpile105(qc, basis_gates=_PSFZ105_BASIS, optimization_level=0))",
        "    try:\n"
        "        from qiskit import qasm3\n"
        "        if all(ins.operation.name in _PSFZ105_DIRECT_OPS for ins in qc.data):\n"
        "            src = qasm3.dumps(qc)\n"
        "        else:\n"
        "            from qiskit import transpile as _transpile105\n"
        "            src = qasm3.dumps(_transpile105(qc, basis_gates=_PSFZ105_BASIS, optimization_level=0))",
    ),
]

for idx, (old, new) in enumerate(replacements, 1):
    assert old in src, f"Hunk {idx} target not found"
    src = src.replace(old, new, 1)

p.write_text(src, encoding="utf-8")
print("Successfully patched psf_compile.py to VERSION 2026-10-18.2")

