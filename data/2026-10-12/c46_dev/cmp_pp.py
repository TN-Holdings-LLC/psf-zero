# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).
import os, sys, os, math, warnings, contextlib, io, time
warnings.simplefilter("ignore")
sys.path[:0] = ["benchmarks", ".", "data/2026-10-12/c46_dev"]
from c41_exact_sweep import load
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import fake_provider
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
M = load("patches/psf_compile_c46_2026-10-12/psf_compile.py", "c46")
import pp_synth as PP

def nl(e): return 30.0 if e is None or e >= 0.5 else -math.log1p(-e)
def regions(t, n, K=3):
    adj, _ = M._st_working_graph(t)
    qc_ = {p: 0.0 for p in adj}
    for g in ("sx", "measure"):
        for qa, pr in t[g].items():
            if qa[0] in qc_: qc_[qa[0]] += nl(pr.error if pr else None)
    g2 = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)
    ec = {}
    for qa, pr in t[g2].items():
        k = (min(qa), max(qa)); ec[k] = max(ec.get(k, 0), nl(pr.error if pr else None))
    out = []
    for s in sorted(adj):
        R = [s]; inR = {s}; tot = qc_[s]
        while len(R) < n:
            best = None
            for p in R:
                for q in adj[p]:
                    if q in inR: continue
                    v = qc_[q] + 4 * ec[(min(p, q), max(p, q))]
                    if best is None or (v, q) < best: best = (v, q)
            if best is None: break
            R.append(best[1]); inR.add(best[1]); tot += best[0]
        if len(R) == n: out.append((tot, R))
    out.sort()
    sel = []
    for sc, R in out:
        if all(len(set(R) & set(o)) < 0.7 * n for _, o in sel): sel.append((sc, R))
        if len(sel) >= K: break
    return [R for _, R in sel]

def map_logical(qc, R, t):
    # logical BFS by interaction weight onto region order (R is in growth order)
    n = qc.num_qubits
    w = {}
    for ins in qc.data:
        if len(ins.qubits) == 2:
            a, b = sorted(qc.find_bit(q).index for q in ins.qubits); w[(a, b)] = w.get((a, b), 0) + 1
    deg = [0] * n
    for (a, b), c in w.items(): deg[a] += c; deg[b] += c
    order = [max(range(n), key=lambda q: (deg[q], -q))]
    while len(order) < n:
        cand = [q for q in range(n) if q not in order]
        order.append(max(cand, key=lambda q: (sum(w.get((min(q, o), max(q, o)), 0) for o in order), deg[q], -q)))
    phys = [None] * n
    for i, q in enumerate(order): phys[q] = R[i]
    return phys

def run_pp(qc, b, qk=None):
    t = b.target
    best = None
    cands = [map_logical(qc, R, t) for R in regions(t, qc.num_qubits)]
    if qk is not None:
        cands.append(list(qk.layout.initial_index_layout(filter_ancillas=True)[:qc.num_qubits]))
    for phys in cands:
        try:
            got = PP.compile_pp(M, qc, t, phys)
        except RuntimeError:
            continue
        if got is None: continue
        out = got[0]
        with M._DeterministicVF2(t):
            o = transpile(out, target=t, initial_layout=list(range(t.num_qubits)), routing_method="none", optimization_level=3, seed_transpiler=0)
        c = M._st_cost(o, t)
        if best is None or c < best[0]: best = (c, o)
    return best

root = os.environ.get("BENCHPRESS_QASM", "benchpress/qasm/")
n2 = lambda c: sum(1 for i in c.data if len(i.qubits) == 2)
for rel in sys.argv[1].split(","):
    qc = QuantumCircuit.from_qasm_file(root + rel)
    if not any(i.operation.name == "measure" for i in qc.data): qc.measure_all()
    for d in sys.argv[2].split(","):
        b = getattr(fake_provider, d)(); t = b.target
        if qc.num_qubits > t.num_qubits: continue
        q = generate_preset_pass_manager(3, b, seed_transpiler=0).run(qc)
        with contextlib.redirect_stdout(io.StringIO()):
            u = M.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0, level3_candidate=False)
        t0 = time.time()
        try:
            got = run_pp(qc, b, q)
        except Exception as e:
            got = None; err = repr(e)[:80]
        tp = time.time() - t0
        row = [rel.split("/")[-1][:24], d[4:], f"QK3 {math.exp(-M._st_cost(q,t)):.3g} ({n2(q)})", f"C46 {math.exp(-M._st_cost(u,t)):.3g} ({n2(u)})"]
        row.append(f"PP {math.exp(-got[0]):.3g} ({n2(got[1])}) {tp:.1f}s" if got else f"PP none {err if 'err' in dir() else ''}")
        print(*row, flush=True)
