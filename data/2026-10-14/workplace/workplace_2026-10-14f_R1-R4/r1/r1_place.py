"""R1 feasibility: deterministic placements for 3-regular QAOA, Qiskit level 3 (seed 0 only) routes.
 HP: the first Hamiltonian path of the interaction graph (DFS from qubit 0, neighbours in index order) put on item
     87's line (snake piece / first VF2 path, common gate length first); the hand-layered circuit routed by level 3.
 MST: (reference) VF2 of nothing -- just item 87's line in logical index order."""
import sys, os, contextlib, io, tempfile, importlib.util
sys.argv = ["family_map.py", tempfile.mkdtemp() + "/fm"]
spec = importlib.util.spec_from_file_location("fm", "benchmarks/family_map.py"); FM = importlib.util.module_from_spec(spec)
sys.modules["fm"] = FM; spec.loader.exec_module(FM)
P = FM.H.load_module("patches/psf_compile_c54_2026-10-14/psf_compile.py", "c54_r1"); P.WARN_WITHOUT_TARGET = False
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
from qiskit_ibm_runtime.fake_provider import FakeFez
b = FakeFez(); TG = {"T": b.target, "F": FM.fractional(b.target)}


def graph(qc):
    ev = FM.terms_of(qc); n = qc.num_qubits
    E = set()
    for lb, a, t in ev[2]:
        sup = [n - 1 - i for i, ch in enumerate(lb) if ch != "I"]
        if len(sup) == 2: E.add(tuple(sorted(sup)))
    return n, sorted(E)


def ham_path(n, E):
    adj = {q: sorted(v for e in E for v in e if q in e and v != q) for q in range(n)}
    best = None
    def dfs(path, seen):
        nonlocal best
        if len(path) == n: best = list(path); return True
        for v in adj[path[-1]]:
            if v not in seen:
                seen.add(v); path.append(v)
                if dfs(path, seen): return True
                path.pop(); seen.discard(v)
        return False
    for s in range(n):
        if dfs([s], {s}): return best
    return None


def run(qc, t, layout):
    pm = generate_preset_pass_manager(3, target=t, initial_layout=layout, seed_transpiler=0)
    c = pm.run(qc)
    if "rzz" in t.operation_names: c = PassManager([FoldRzzAngle()]).run(c)
    return c


for name in ("qaoa_n12_p2", "qaoa_n20_p2", "qaoa3reg_n20_r3"):
    qc = FM.build(name); hl = FM.hand_layered(qc)
    n, E = graph(qc)
    hp = ham_path(n, E)
    for tk, t in TG.items():
        K, kadj, fadj, snake = P._chain_lattice(t, 0.5)
        line, _ = P._chain_place(n, K, fadj, snake)
        if line is None: line, _ = P._chain_place(n, K, kadj, snake)
        lay_hp = [None] * n
        for i, q in enumerate(hp): lay_hp[q] = line[i]
        c_hp = run(hl, t, lay_hp)
        c_id = run(hl, t, line[:n])
        with contextlib.redirect_stdout(io.StringIO()):
            psf = P.compile_for_hardware(qc, target=t, seed_transpiler=0)
        print(f"{name:16s} {tk} | HP line + HL + L3s0 {FM.dur(c_hp, t):6.2f} ({FM.n2(c_hp)}) | index line + HL {FM.dur(c_id, t):6.2f} | c54 {FM.dur(psf, t):6.2f} | exact {FM.exact(qc, c_hp)}", flush=True)
    print("   path", hp)

# --- more fixed placements: reverse Cuthill-McKee order on item 87's line, and on a compact BFS patch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
print("--- RCM")
for name in ("qaoa_n12_p2", "qaoa_n20_p2", "qaoa3reg_n20_r3"):
    qc = FM.build(name); hl = FM.hand_layered(qc)
    n, E = graph(qc)
    A = np.zeros((n, n))
    for a, c in E: A[a, c] = A[c, a] = 1
    order = list(reverse_cuthill_mckee(csr_matrix(A), symmetric_mode=True))
    bw = max(abs(order.index(a) - order.index(c)) for a, c in E)
    for tk, t in TG.items():
        K, kadj, fadj, snake = P._chain_lattice(t, 0.5)
        line, _ = P._chain_place(n, K, fadj, snake)
        if line is None: line, _ = P._chain_place(n, K, kadj, snake)
        lay = [None] * n
        for i, q in enumerate(order): lay[q] = line[i]
        c1 = run(hl, t, lay)
        # compact patch: BFS from the line's middle qubit over K, in BFS order
        start = line[n // 2]; seen = [start]; fr = [start]
        while fr and len(seen) < n:
            u = fr.pop(0)
            for v in sorted(kadj[u]):
                if v not in seen and len(seen) < n: seen.append(v); fr.append(v)
        lay2 = [None] * n
        for i, q in enumerate(order): lay2[q] = seen[i]
        c2 = run(hl, t, lay2)
        print(f"{name:16s} {tk} bandwidth {bw} | RCM line {FM.dur(c1, t):6.2f} ({FM.n2(c1)}) | RCM patch {FM.dur(c2, t):6.2f} ({FM.n2(c2)})", flush=True)
