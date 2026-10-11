"""steiner6_chain_probe.py -- chain probe (chemistry); BENCHPRESS=<clone> python benchmarks/steiner6_chain_probe.py
<device> <test substrings...>: logical qubits 0..n-1 on a simple path of the device's working graph, in order; each term's
CNOT tree on the path only (a ladder toward its lowest logical qubit, bridges through the path), so that the ladders
of consecutive terms coincide and cancel as in Qiskit's own synthesis. No SWAPs. Then level 3 on the fixed layout."""
import os, sys, json, math, warnings, time
warnings.simplefilter("ignore")
REPO = os.getcwd()  # the psf-zero repository root
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
import steiner3 as S3, steiner5 as S5, bp_mock as B
import steiner6m as S6
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import fake_provider
BP = os.environ["BENCHPRESS"]  # a Benchpress clone


def long_paths(adj, wedge, n, limit=200000, keep=8):
    """Simple paths of n working qubits (DFS, a call budget), the ones of least summed coupler weight first."""
    out = []
    calls = [0]
    starts = sorted(adj, key=lambda v: (len(adj[v]), v))
    def dfs(path, seen, w):
        calls[0] += 1
        if calls[0] > limit:
            return
        if len(path) == n:
            out.append((w, list(path)))
            return
        u = path[-1]
        for v in sorted(adj[u], key=lambda v: (wedge.get((min(u, v), max(u, v)), 1), v)):
            if v not in seen:
                seen.add(v); path.append(v)
                dfs(path, seen, w + wedge.get((min(u, v), max(u, v)), 1))
                path.pop(); seen.discard(v)
                if calls[0] > limit:
                    return
    for s in starts[:40]:
        dfs([s], {s}, 0.0)
        if calls[0] > limit:
            break
    out.sort(key=lambda x: x[0])
    return [p for _, p in out[:keep]]


for name in sys.argv[2:]:
    dev = sys.argv[1]
    backend = getattr(fake_provider, dev)(); target = backend.target
    t = [t for t in S3.tests(BP) if name in t[1]][0]
    qc, _ = B.build(BP, t[2], t[3])
    evo = [i for i in qc.data if i.operation.name == "PauliEvolution"][0]
    n = evo.operation.num_qubits
    adj, _ = S3.working_graph(target)
    derr, perr, wedge = S5.error_distances(adj, target)
    paths = long_paths(adj, wedge, n)
    triples = S3._variants(evo.operation)["given"]
    best = None
    t0 = time.perf_counter()
    for path in paths[:3]:
        padj = {q: set() for q in path}
        for a, b in zip(path, path[1:]):
            padj[a].add(b); padj[b].add(a)
        pd, pp = S3.bfs_all(padj)
        out = QuantumCircuit(target.num_qubits)
        gp = 0.0
        for lb, c, tt in triples:
            for nm, q, prm in S3.gadget_ops(lb, c, tt, path, pd, pp):
                if nm == "gphase": gp += prm[0]
                elif nm == "cx": out.cx(*q)
                elif nm == "rz": out.rz(prm[0], q[0])
                else: getattr(out, nm)(q[0])
        out.global_phase = gp
        pre = sum(1 for i in out.data if i.operation.name == "cx")
        o3 = transpile(out, target=target, initial_layout=list(range(target.num_qubits)), routing_method="none",
                       optimization_level=3, seed_transpiler=0)
        esp, f = S3.esp_of(o3, target)
        if best is None or esp > best[0]:
            best = (esp, S6.q2(o3), pre, path, out)
    q3 = generate_preset_pass_manager(3, backend, seed_transpiler=0).run(qc)
    e3, f3 = S3.esp_of(q3, target)
    ex = S6.check(evo, best[3], best[3], best[4])
    print(f"{dev} {name}: chain q2 {best[1]} (pre {best[2]}) esp {best[0]} | QK3 q2 {S6.q2(q3)} esp {e3} failed {f3} | "
          f"ESP ratio {10 ** (best[0] - e3):.3f} exact {ex} paths {len(paths)} t {time.perf_counter() - t0:.1f}")
