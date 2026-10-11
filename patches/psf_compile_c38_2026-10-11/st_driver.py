

_ST_FAILED = 0.5  # set from prune_max_error by _st_compile
_ST_VF2_CHECK, _ST_VF2_MAPS, _ST_KEEP = 20_000, 200, 6
_ST_EMBED = 2  # STEINER5's VF2 embeddings added per distance mode as initial placements
_ST_BIG_TERMS = 2000  # above this many terms: the error mode only, one greedy start and one embedding
STEINER_STATS = {"not_applicable": 0, "not_commuting": 0, "used": 0, "fallback": 0}  # changelog item 68


def _st_evolution(qc):
    """Item 68: (PauliEvolutionGate instruction, [(qubit, clbit)] of the final measurements) when `qc` is one
    PauliEvolutionGate on all its qubits in order (barriers and final measurements aside), with Qiskit's default
    first-order product formula (LieTrotter, one repetition), a bound time and real coefficients; else None."""
    evo, meas = None, []
    for ins in qc.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        if name == "PauliEvolution" and evo is None and not meas:
            evo = ins
        elif name == "measure" and evo is not None:
            meas.append((qc.find_bit(ins.qubits[0]).index, qc.find_bit(ins.clbits[0]).index))
        else:
            return None
    if evo is None:
        return None
    gate = evo.operation
    if isinstance(gate.operator, list):
        return None
    syn = getattr(gate, "synthesis", None)
    if syn is not None and (type(syn).__name__ != "LieTrotter" or getattr(syn, "reps", 1) != 1):
        return None
    if [qc.find_bit(q).index for q in evo.qubits] != list(range(qc.num_qubits)):
        return None
    try:
        t = float(gate.time)
    except (TypeError, ValueError):
        return None
    coeffs = np.asarray(gate.operator.coeffs)
    if coeffs.dtype == object or np.any(np.abs(np.imag(coeffs)) > 1e-12):
        return None
    triples = [(lb, float(np.real(c)), t) for lb, c in gate.operator.to_list()]
    return evo, meas, triples


def _st_all_commute(triples):
    """Item 68: True if every two terms commute (checked on the Pauli bit matrices, in blocks)."""
    labels = [lb for lb, _, _ in triples]
    n = len(labels[0]) if labels else 0
    X = np.array([[ch in "XY" for ch in reversed(lb)] for lb in labels], dtype=np.int32).reshape(len(labels), n)
    Z = np.array([[ch in "ZY" for ch in reversed(lb)] for lb in labels], dtype=np.int32).reshape(len(labels), n)
    for i0 in range(0, len(labels), 512):
        a = (X[i0:i0 + 512] @ Z.T + Z[i0:i0 + 512] @ X.T) & 1
        if a.any():
            return False
    return True


def _st_construct(n, triples, target, meas):
    """Item 68: STEINER6's construction (Addendum 444, run v6): candidates from the greedy placements (hop and error
    distances, two starts each) and STEINER5's embeddings, each run by _StMover2; the lowest estimated -ln ESP kept.
    (mover, initial placement) or None."""
    adj, _ = _st_working_graph(target)
    if len(adj) < n:
        return None
    dist, par = _st_bfs_all(adj)
    derr, perr, wedge = _st_error_distances(adj, target)
    if not wedge:
        return None
    readout = {}
    if "measure" in target.operation_names:
        for qa, pr in target["measure"].items():
            if qa is not None and pr is not None and pr.error is not None:
                readout[qa[0]] = -math.log(1.0 - min(pr.error, 0.999))
    terms = [(lb, c) for lb, c, _ in triples]
    w_med = sorted(wedge.values())[len(wedge) // 2]
    starts_h = sorted(adj, key=lambda v: (-len(adj[v]), v))[:2]
    nb = {v: sum(wedge.get((min(v, u), max(v, u)), w_med) for u in adj[v]) / max(1, len(adj[v])) for v in adj}
    starts_e = sorted((v for v in adj if len(adj[v]) >= 2), key=lambda v: (nb[v], v))[:2]
    inits = {"H": [("greedy", _st_placement(n, terms, adj, dist, s0)) for s0 in starts_h],
             "E": [("greedy", _st_placement(n, terms, adj, derr, s0)) for s0 in starts_e]}
    for mode, (dd, pp, ew) in (("H", (dist, par, None)), ("E", (derr, perr, wedge))):
        try:
            emb, _, _ = _st_embeddings(n, terms, adj, dd)
        except Exception:  # noqa: BLE001 - no embedding within the call limit: the greedy placements remain
            emb = []
        scored = sorted(((_st_placement_cost(e, terms, dd, pp, ew), k) for k, e in enumerate(emb)))[:_ST_EMBED]
        inits[mode] += [("vf2", emb[k]) for _, k in scored]
    modes = ("H", "E")
    if len(triples) > _ST_BIG_TERMS:
        modes = ("E",)
        inits["E"] = inits["E"][:1] + [x for x in inits["E"] if x[0] == "vf2"][:1]
    cands = []
    for mode in modes:
        for label, pi0 in inits[mode]:
            if any(p not in adj for p in pi0):
                continue
            m = _StMover2(n, triples, adj, dist, par, pi0, err=None if mode == "H" else (derr, perr, wedge)).build()
            est = 0.0
            for name, q, _ in m.ops:
                if len(q) == 2:
                    est += wedge.get((min(q), max(q)), w_med) * (1 if name == "cx" else 3)
            est -= 2 * w_med * m.stats["merged_swaps"]
            est += sum(readout.get(m.pi[q], 0.0) for q, _ in meas)
            cands.append((est, mode, label, m, pi0))
    if not cands:
        return None
    best = min(cands, key=lambda c: (c[0], c[1], c[2]))
    return best[3], list(best[4])


def _st_compile(qc, target, max_error, seed, call_limit, max_trials):
    """Item 68: the circuit compiled by STEINER6 with its layout, or None when the path does not apply (not one
    Pauli evolution, or its terms do not all commute) or its output would use a failed element (then the call goes on
    by the usual path)."""
    global _ST_FAILED
    from qiskit.circuit import QuantumRegister
    from qiskit.transpiler import Layout, TranspileLayout
    ev = _st_evolution(qc)
    if ev is None:
        STEINER_STATS["not_applicable"] += 1
        return None
    evo, meas, triples = ev
    if not triples or not _st_all_commute(triples):
        STEINER_STATS["not_commuting"] += 1
        return None
    _ST_FAILED = max_error
    n, N = qc.num_qubits, target.num_qubits
    got = _st_construct(n, triples, target, meas)
    if got is None:
        STEINER_STATS["fallback"] += 1
        return None
    m, pi0 = got
    phys = QuantumCircuit(N, qc.num_clbits)
    at = list(range(N))  # at[p]: the physical qubit whose state is at p (SWAPs move idle qubits' states too)
    for name, q, prm in m.ops:
        if name == "cx":
            phys.cx(*q)
        elif name == "swap":
            phys.swap(*q)
            at[q[0]], at[q[1]] = at[q[1]], at[q[0]]
        elif name == "rz":
            phys.rz(prm[0], q[0])
        else:
            getattr(phys, name)(q[0])
    phys.global_phase = qc.global_phase + m.gphase
    for q, cb in meas:
        phys.measure(m.pi[q], cb)
    out = transpile(phys, target=target, initial_layout=list(range(N)), routing_method="none", optimization_level=3,
                    seed_transpiler=seed)
    # Item 33's exact re-placement, as a renaming of the physical qubits
    dag = circuit_to_dag(out)
    vf2 = VF2PostLayout(target=target, seed=-1, call_limit=call_limit, max_trials=max_trials, strict_direction=True)
    vf2.run(dag)
    rel = list(range(N))
    if vf2.property_set["VF2PostLayout_stop_reason"] == VF2PostLayoutStopReason.SOLUTION_FOUND:
        post = vf2.property_set["post_layout"]
        rel = [post[dag.qubits[i]] for i in range(N)]
    final = QuantumCircuit(N, qc.num_clbits, global_phase=out.global_phase)
    for ins in out.data:
        final.append(ins.operation, [final.qubits[rel[out.find_bit(b).index]] for b in ins.qubits],
                     [final.clbits[out.find_bit(c).index] for c in ins.clbits])
    edges, bad = _failed_elements(target, max_error)
    if _uses_failed(final, edges, bad):
        STEINER_STATS["fallback"] += 1
        return None
    pos = [0] * N  # pos[p]: where the state that started at physical p ends
    for p_now, p_start in enumerate(at):
        pos[p_start] = p_now
    anc = QuantumRegister(N - n, "ancilla") if N > n else None
    virt = list(qc.qubits) + (list(anc) if anc is not None else [])
    used = set(pi0)
    init_phys = list(pi0) + [p for p in range(N) if p not in used]
    initial_layout = Layout({virt[k]: rel[init_phys[k]] for k in range(N)})
    for reg in qc.qregs:
        initial_layout.add_register(reg)
    if anc is not None:
        initial_layout.add_register(anc)
    final_layout = Layout({final.qubits[rel[p]]: rel[pos[p]] for p in range(N)})
    final._layout = TranspileLayout(initial_layout=initial_layout,
                                    input_qubit_mapping={virt[k]: k for k in range(N)},
                                    final_layout=final_layout, _input_qubit_count=qc.num_qubits,
                                    _output_qubit_list=list(final.qubits))
    STEINER_STATS["used"] += 1
    return final


