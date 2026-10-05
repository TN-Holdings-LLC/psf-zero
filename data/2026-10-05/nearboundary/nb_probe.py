"""Exploratory probe (not pre-registered, not a test): are release 2026-10-04.1's Qiskit-resynthesis paths
(item 35 final_resynthesis, item 36 compare_level3) exposed to Qiskit #17057 on near-boundary workloads?
Workloads from benchmarks/b17_practice_eval.py (W1 Trotter cells where PSFNG failed; W2 near-boundary unitaries).
Exactness is checked as the noiseless output-state infidelity of the compiled circuit (random product input,
prepended to the circuit), on the touched qubits only."""
import contextlib, io, os, sys, warnings, json, time
import numpy as np
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psf_compile as P
import b17_practice_eval as B
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity, random_unitary
from qiskit.circuit.library import UnitaryGate
from qiskit_ibm_runtime import fake_provider as fp

assert P.VERSION == "2026-10-04.1", P.VERSION


def with_prep(qc, seed):
    rng = np.random.default_rng(seed)
    out = QuantumCircuit(qc.num_qubits)
    for q in range(qc.num_qubits):
        out.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
    out.compose(qc, inplace=True)
    return out


def state_infid(qc, out):
    ideal = Statevector(qc)
    active = sorted({out.find_bit(b).index for ins in out.data for b in ins.qubits})
    fin = out.layout.final_index_layout(filter_ancillas=True) if out.layout is not None else list(range(qc.num_qubits))
    active = sorted(set(active) | set(fin))
    idx = {p: i for i, p in enumerate(active)}
    red = QuantumCircuit(len(active))
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        red.append(ins.operation, [idx[out.find_bit(b).index] for b in ins.qubits])
    sv = Statevector(red)
    keep = [idx[p] for p in fin]
    trace_out = [i for i in range(len(active)) if i not in keep]
    rho = partial_trace(sv, trace_out) if trace_out else sv
    # reorder: partial_trace keeps remaining qubits in ascending order; build a permutation to virtual order
    order = sorted(keep)
    perm = [order.index(k) for k in keep]       # virtual v sits at position perm[v] of rho
    from qiskit.quantum_info import DensityMatrix
    rho = DensityMatrix(rho)
    # permute rho's subsystems so that subsystem v is virtual qubit v
    n = len(keep)
    t = rho.data.reshape([2] * (2 * n))
    # qiskit little-endian: axis (n-1-j) is qubit j for rows
    row_axes = [n - 1 - perm[v] for v in range(n)][::-1]
    col_axes = [2 * n - 1 - perm[v] for v in range(n)][::-1]
    t = np.transpose(t, row_axes + col_axes).reshape(2**n, 2**n)
    return float(1 - state_fidelity(DensityMatrix(t), ideal))


def arms(backend):
    t = backend.target
    cm = t.build_coupling_map()
    basis = list(t.operation_names)
    ent = "cx"  # the release accepts canonical or cx; its README uses cx on every device
    base = dict(coupling_map=cm, basis_gates=basis, entangling_basis=ent, layout_search=True, seed_transpiler=0,
                target=t, placement_refine=True)
    return {
        "R_102_2": lambda qc: P.compile_for_hardware(qc, **base),
        "R_resynth_always": lambda qc: P.compile_for_hardware(qc, **base, final_resynthesis=True),
        "R_recommended": lambda qc: P.compile_for_hardware(qc, **base, final_resynthesis="select",
                                                           compare_level3=True, compare_floor=True,
                                                           candidate_score="hybrid"),
        "L3T": lambda qc: transpile(qc, target=t, optimization_level=3, seed_transpiler=0, approximation_degree=1.0),
    }


def main():
    devs = sys.argv[1].split(",")
    nseed = int(sys.argv[2])
    rows = []
    for dname in devs:
        be = getattr(fp, dname)()
        A = arms(be)
        work = []
        for n in (4, 6):
            for p, qc in B.circuits("W1", n, smoke=False):
                if (p["dt"], p["r"]) in ((1e-3, 1e-4), (1e-2, 1e-5), (0.1, 1.0)) and p["seed"] < nseed:
                    work.append((f"W1 n{n} dt{p['dt']} r{p['r']}", with_prep(qc, 7000 + p["seed"])))
            k = 0
            for p, qc in B.circuits("W2", n, smoke=False):
                if k >= nseed:
                    break
                work.append((f"W2 n{n}", with_prep(qc, 8000 + k)))
                k += 1
        for cell, qc in work:
            for a, f in A.items():
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    out = f(qc)
                inf = state_infid(qc, out)
                rows.append(dict(dev=dname, cell=cell, arm=a, infid=inf, s=round(time.perf_counter() - t0, 3)))
        print(dname, "done", flush=True)
    json.dump(rows, open(sys.argv[3], "w"))
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["dev"], r["cell"], r["arm"])].append(r["infid"])
    for k in sorted(agg):
        v = agg[k]
        bad = sum(x > 1e-6 for x in v)
        print(f"{k[0]:13s} {k[1]:24s} {k[2]:17s} wrong {bad}/{len(v)}  max {max(v):.2e}")


if __name__ == "__main__":
    main()
