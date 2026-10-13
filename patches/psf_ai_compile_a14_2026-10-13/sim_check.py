"""Exploratory (not pre-registered): a12, a14 and Qiskit level 3 (target) on small circuits, Aer noise of the fake device
with relaxation on idle windows (circuits scheduled ASAP with delays). Favourable to a14 by construction: Aer's
relaxation is the physics a14's idle term assumes."""
import contextlib, io, os, sys, time, warnings
warnings.simplefilter("ignore")
REPO, A14 = sys.argv[1], sys.argv[2]
devs = sys.argv[3].split(",")
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
import numpy as np
import core_fix_c2_eval as H
H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
a12 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
a14 = H.load_module(A14, "psf_ai_compile_a14")
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, random_unitary, hellinger_fidelity
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ASAPScheduleAnalysis, PadDelay
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider

def gen(seed):
    rng = np.random.default_rng(seed)
    out = []
    for n in (4, 5, 6):
        qc = QuantumCircuit(n, name=f"qaoa_ring{n}")
        qc.h(range(n))
        for p in range(2):
            g, b = rng.uniform(0.2, 1.2, 2)
            for i in range(n):
                qc.rzz(g, i, (i + 1) % n)
            qc.rx(b, range(n))
        out.append(qc)
        qc = QuantumCircuit(n, name=f"tfim{n}")
        for r in range(3):
            for i in range(0, n - 1, 1):
                qc.rzz(0.4, i, i + 1)
            qc.rx(0.3, range(n))
        out.append(qc)
        qc = QuantumCircuit(n, name=f"model{n}")  # model-like: unitary blocks, swaps, redundant cx pairs
        qc.h(0)
        for k in range(3 * n):
            a, b = rng.choice(n, 2, replace=False)
            r = rng.integers(4)
            if r == 0: qc.cx(a, b); qc.cx(a, b); qc.cx(a, b)
            elif r == 1: qc.swap(a, b)
            elif r == 2: qc.append(UnitaryGate(random_unitary(4, seed=int(rng.integers(1e9)))), [a, b])
            else: qc.ry(float(rng.uniform(-1, 1)), a); qc.cz(a, b)
        out.append(qc)
    return out

def ideal_probs(qc):
    p = Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict()
    return {k: v for k, v in p.items() if v > 1e-12}

rows = []
for dev in devs:
    be = getattr(fake_provider, dev)()
    t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    cm = t.build_coupling_map()
    sim = AerSimulator.from_backend(be)
    for qc in gen(20261013):
        ideal = ideal_probs(qc)
        m = qc.copy(); m.measure_all()
        outs = {}
        with contextlib.redirect_stdout(io.StringIO()):
            outs["a12"] = a12.compile_for_model_circuit(m, cm, basis, target=t)
            outs["a14"] = a14.compile_for_model_circuit(m, cm, basis, target=t)
        outs["L3T"] = transpile(m, target=t, optimization_level=3, seed_transpiler=0)
        res = {}
        for arm, out in outs.items():
            assert rel._implements(m, out) or arm == "L3T"
            sch = PassManager([ASAPScheduleAnalysis(target=t), PadDelay(target=t)]).run(out)
            cnt = sim.run(sch, shots=8000, seed_simulator=7).result().get_counts()
            # measured clbits are in logical order (measure_all on the input)
            f = hellinger_fidelity(ideal, {k: v / 8000 for k, v in cnt.items()})
            _, T = a14.idle_weights(out, t)
            res[arm] = (f, a14._two_q(out), T * 1e6, a14.estimate_a14(out, t))
        rows.append((dev, qc.name, res))
        print(dev, f"{qc.name:12s}", "  ".join(f"{a}: F={v[0]:.3f} 2q={v[1]:3d} dur={v[2]:5.2f}us est={v[3]:.3f}" for a, v in res.items()), flush=True)

def gm(x): return float(np.exp(np.mean(np.log(x))))
for a, b in (("a14", "a12"), ("a14", "L3T"), ("a12", "L3T")):
    r = [x[2][a][0] / x[2][b][0] for x in rows]
    w = sum(1 for v in r if v > 1.0 + 1e-9); l = sum(1 for v in r if v < 1.0 - 1e-9)
    print(f"{a}/{b}: geometric mean infidelity-free F ratio {gm(r):.4f}; better {w}, worse {l}, of {len(r)}")
    ri = [(1 - x[2][a][0]) / max(1e-9, 1 - x[2][b][0]) for x in rows]
    print(f"   infidelity ratio (1-F) geometric mean {gm(ri):.3f}")
