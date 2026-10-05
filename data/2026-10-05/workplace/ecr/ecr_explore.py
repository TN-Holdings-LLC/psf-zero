"""ecr_explore.py -- workplace EXPLORATION (not pre-registered, not a test), 2026-10-05: do the current candidates
(psf_compile c13 with the recommended call; AI front end a10) work on ecr (Eagle) devices, where Addendum 293 found
reported gate errors below the T1/T2 floor most often, and how do they compare with Qiskit level 3?

Circuits (fresh seeds, 72,000,000 +): the MODEL-RO2 generator's families (ai10_eval2._family/_model_style), 3 per
(family, n) cell = 48, plus 8 measured classifier circuits (depth_eval.circuit, n 4 and 6, L 4, random parameters
and inputs). All with final measurements.
Arms: RPSF (psf_compile c13, target + placement_refine), C13 (recommended call), A11 (a11 with c13 underneath;
the first pass used a10, kept in out_a10/),
L3TM (Qiskit level 3 with the Target, approximation_degree 1.0).
Per circuit: state infidelity (exactness), off-target instructions, two-qubit gates in a direction reported failed
(error >= 0.5) or on a failed qubit (sx error >= 0.5), summed measure error, classical infidelity of the sampled
distribution under the device noise (restricted noise model + readout, as MODEL-RO2), two-qubit count, compile time.
"""
import contextlib, io, json, os, sys, time, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import psf_compile as pc  # c13
import depth_eval as DE
import readout_eval as RE
import ai10_eval2 as E2
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, random_unitary
from qiskit_ibm_runtime import fake_provider as fp


def circuits():
    out = []
    k = 0
    for name, ns in E2.FAMILIES:
        for n in ns:
            for _ in range(3):
                rng = np.random.default_rng(72_000_000 + k); k += 1
                qc = E2._model_style(E2._family(name, n, rng), rng)
                for q in range(n):
                    qc.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
                out.append((f"{name}{n}", qc))
    for j, n in enumerate((4, 4, 4, 4, 6, 6, 6, 6)):
        rng = np.random.default_rng(72_500_000 + j)
        out.append((f"CLS{n}", DE.circuit(rng.uniform(-1, 1, n), rng.normal(0, 1, DE.n_params(n, 4)), n, 4)))
    return out


def main(dev):
    A10 = E2.load("psf_ai_compile.py", "x_a11")  # a11: the psf_ai_compile.py next to this script
    be = getattr(fp, dev)(); t = be.target; cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    base = dict(coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True, seed_transpiler=0,
                target=t, placement_refine=True)
    full = dict(base, final_resynthesis="select", compare_level3=True, compare_floor=True, candidate_score="hybrid")
    arms = {"RPSF": lambda qc: pc.compile_for_hardware(qc, **base),
            "C13": lambda qc: pc.compile_for_hardware(qc, **full),
            "A11": lambda qc: A10.compile_for_model_circuit(qc, cm, basis, target=t),
            "L3TM": lambda qc: transpile(qc, target=t, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)}
    edges, fq = pc._failed_elements(t, 0.5)
    sim = DE.Noisy(be)
    rows = []
    for name, qc0 in circuits():
        qc = qc0.copy(); qc.measure_all()
        ideal = Statevector(qc0).probabilities()
        row = dict(name=name, n=qc0.num_qubits)
        for a, f in arms.items():
            t0 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    out = f(qc)
            except Exception as e:
                row[a] = dict(error=f"{type(e).__name__}: {e}"[:300]); continue
            cs = time.perf_counter() - t0
            off = failed = 0
            for ins in out.data:
                nm = ins.operation.name
                if nm in ("barrier", "measure", "delay"):
                    continue
                q = tuple(out.find_bit(b).index for b in ins.qubits)
                if nm not in t.operation_names or q not in t[nm]:
                    off += 1
                if any(i in fq for i in q) or (len(q) == 2 and q in edges):
                    failed += 1
            p0, mq, cl = E2.reduced_probs(out, sim, False)
            pn, _, _ = E2.reduced_probs(out, sim, True)
            qd = E2.apply_readout(pn, mq, sim)
            row[a] = dict(state_infid=RE.state_infid(qc0, RE.strip_measure(out)), off_target=off, failed_uses=failed,
                          meas_err=float(sum(t["measure"][(m,)].error for m in mq)),
                          infid=float(1 - np.sum(np.sqrt(np.clip(ideal, 0, None) * np.clip(qd, 0, None))) ** 2),
                          n2q=sum(1 for g in out.data if len(g.qubits) == 2), compile_s=round(cs, 4))
        rows.append(row)
    json.dump(dict(device=dev, rows=rows), open(os.path.join(HERE, "out", f"ecr_{dev}.json"), "w"))
    print("done", dev, len(rows), flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    main(sys.argv[1])
