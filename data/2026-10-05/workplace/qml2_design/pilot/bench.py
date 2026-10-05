import os, time, sys, warnings, contextlib, io
warnings.simplefilter("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # psf_compile.py next to this file
import psf_compile as P
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider as fp
def model(n, L, rng):
    qc = QuantumCircuit(n)
    x = rng.uniform(-1, 1, n)
    for l in range(L):
        for q in range(n): qc.ry(np.pi * x[q], q)
        for q in range(n): qc.ry(rng.normal(), q); qc.rz(rng.normal(), q)
        for q in range(n): qc.cz(q, (q + 1) % n)
    return qc
for dname in sys.argv[1:]:
    be = getattr(fp, dname)(); t = be.target
    sim = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(be))
    for n in (6, 8, 10, 12):
        for L in (2, 4):
            qc = model(n, L, np.random.default_rng(1))
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                out = P.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=[g for g in t.operation_names if g in ("cx","cz","ecr","rz","sx","x","id")],
                    entangling_basis="cx", layout_search=True, seed_transpiler=0, target=t, placement_refine=True,
                    final_resynthesis="select", compare_level3=True, compare_floor=True, candidate_score="hybrid")
            tc = time.perf_counter() - t0
            active = sorted({out.find_bit(b).index for i in out.data for b in i.qubits})
            idx = {p: i for i, p in enumerate(active)}
            red = QuantumCircuit(len(active))
            for i in out.data: red.append(i.operation, [idx[out.find_bit(b).index] for b in i.qubits])
            # map noise to physical qubits: simulate full device restricted via initial_layout trick
            tr = transpile(red, sim, optimization_level=0)  # keeps gates; noise model applies by qubit index of red (approx timing only)
            tr.save_density_matrix()
            t0 = time.perf_counter(); sim.run(tr).result(); ts = time.perf_counter() - t0
            n2q = sum(1 for i in out.data if len(i.qubits) == 2)
            print(f"{dname} n={n} L={L} active={len(active)} 2q={n2q} compile {tc:.2f}s  DMsim {ts:.2f}s", flush=True)
