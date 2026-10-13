"""Where STEINER6's CNOTs go: per test, the mover's gadget CNOTs, SWAPs, the lower bound sum 2(w-1), and after Qiskit."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import os, sys, os, math, warnings, contextlib, io, time, random
warnings.simplefilter("ignore")
sys.path[:0] = ["benchmarks", "."]
from c41_exact_sweep import load
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import fake_provider
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
M = load(os.environ.get("PSFC", "patches/psf_compile_c45_2026-10-12/psf_compile.py"), "psfm")
root = os.environ.get("BENCHPRESS_QASM", "benchpress/qasm/")
n2 = lambda c: sum(1 for i in c.data if len(i.qubits) == 2)

def gen(name):
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter, SuzukiTrotter
    kind, n, reps = name.split("_"); n = int(n[1:]); reps = int(reps[1:])
    rng = random.Random("C47-DEV|" + name)
    def lab(d):
        l = ["I"] * n
        for q, p in d.items(): l[n - 1 - q] = p
        return "".join(l)
    if kind == "tfim":
        terms = [(lab({q: "Z", (q + 1) % n: "Z"}), rng.uniform(-1, 1)) for q in range(n)] + [(lab({q: "X"}), rng.uniform(-1, 1)) for q in range(n)]
    elif kind == "heis":
        terms = [(lab({q: p, (q + 1) % n: p}), rng.uniform(-1, 1)) for q in range(n) for p in "XYZ"]
    elif kind == "heis2d":
        L = int(round(n ** 0.5)); terms = []
        for x in range(L):
            for y in range(L):
                q = x * L + y
                for nb in (((x + 1) % L) * L + y, x * L + (y + 1) % L):
                    for p in "XYZ": terms.append((lab({q: p, nb: p}), rng.uniform(-1, 1)))
    elif kind == "chem":  # random weight-2..6 strings, like a fermionic Hamiltonian under Jordan-Wigner
        d = {}
        for _ in range(6 * n):
            k = rng.choice((2, 2, 4, 4, 6)); qs = sorted(rng.sample(range(n), k))
            d[lab({q: ("Z" if q not in (qs[0], qs[-1]) else rng.choice("XY")) for q in range(qs[0], qs[-1] + 1)})] = rng.uniform(-1, 1)
        terms = sorted(d.items())
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=0.5, synthesis=SuzukiTrotter(order=2, reps=reps) if kind != "chem" else LieTrotter(reps=reps)), range(n))
    qc.measure_all()
    return qc

def qaoa_reg(name):
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    _, n, reps = name.split("_"); n = int(n[1:]); reps = int(reps[1:])
    rng = random.Random("C47-DEV|" + name)
    while True:
        stubs = [v for v in range(n) for _ in range(3)]; rng.shuffle(stubs)
        E = set(); ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or (min(a, b), max(a, b)) in E: ok = False; break
            E.add((min(a, b), max(a, b)))
        if ok: break
    def lab(a, b):
        l = ["I"] * n; l[n - 1 - a] = "Z"; l[n - 1 - b] = "Z"; return "".join(l)
    qa = QAOAAnsatz(SparsePauliOp.from_list([(lab(a, b), 1.0) for a, b in sorted(E)]), reps=reps)
    qc = qa.assign_parameters([rng.uniform(0, 3) for _ in range(qa.num_parameters)])
    qc.measure_all()
    return qc

def build(name):
    if name.startswith("qaoa3reg"):
        return qaoa_reg(name)
    if name.startswith("bp:"):
        qc = QuantumCircuit.from_qasm_file(root + name[3:])
        if not any(i.operation.name == "measure" for i in qc.data): qc.measure_all()
        return qc
    return gen(name)

for name in sys.argv[1].split(","):
    qc = build(name)
    for d in sys.argv[2].split(","):
        b = getattr(fake_provider, d)(); t = b.target
        ev = M._st_parse(qc)
        ev = ev + ([], None) if ev is not None else M._st_parse_gates(qc)
        if ev is None: print(name, d, "not read"); continue
        triples = M._st_merge(ev[2])
        M._ST_FAILED = 0.5
        t0 = time.time()
        got = M._st_construct(qc.num_qubits, triples, t, ev[1])
        mv = got[0]
        cx = sum(1 for nm, q, _ in mv.ops if nm == "cx"); sw = sum(1 for nm, q, _ in mv.ops if nm == "swap")
        lb = sum(2 * (sum(c != "I" for c in lb_) - 1) for lb_, _, _ in triples if sum(c != "I" for c in lb_) > 1)
        with contextlib.redirect_stdout(io.StringIO()):
            st = M._st_compile(qc, t, 0.5, 0, 300_000, 2_500)
            us = M.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0, steiner=False, level3_candidate=False)
        q = generate_preset_pass_manager(3, b, seed_transpiler=0).run(qc)
        e = lambda c: math.exp(-M._st_cost(c, t))
        print(f"{name:28s} {d[4:]:9s} terms {len(triples):5d} | gadget-cx {cx:5d} (LB {lb:5d}, +{100*(cx-lb)/max(lb,1):.0f}%) swaps {sw:4d} (merged {mv.stats['merged_swaps']}) bridged {mv.stats['bridged_terms']} | ST out 2q {n2(st) if st is not None else '-':>5} ESP {e(st) if st is not None else 0:.3g} | usual {n2(us)} {e(us):.3g} | QK3 {n2(q)} {e(q):.3g} | {time.time()-t0:.1f}s", flush=True)
