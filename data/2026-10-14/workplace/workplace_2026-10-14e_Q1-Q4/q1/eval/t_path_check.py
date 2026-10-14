"""T-path (CZ) correctness at n > 24: the Clifford-angle twin compiled for the device's own Target, read by Aer MPS
(<Z_i>, <Z_a Z_b> on interaction edges), against the logical twin's stabilizer state."""
import os, sys, os, contextlib, io, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cliff_check import *  # noqa
from cliff_check import _snap, StabilizerState, Clifford, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider
NEW = compiler(os.environ.get("PSF_NEW", REPO + "/patches/psf_compile_c53_91_2026-10-14/psf_compile.py"), "n91")
out = []
dev, fam, n_arg, s_arg = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
for dev in (dev,):
    T = getattr(fake_provider, dev)().target
    K, kadj, fadj, _ = NEW._chain_lattice(T, 0.5)
    for fam, n, s in ((fam, len(K) if n_arg == "all" else int(n_arg), s_arg),):
        e = hh_patch(T, n, min(K), adj=kadj) if fam == "hh_ising" else bfs_tree(T, n, min(K), adj=kadj)
        cq = clifford_ising(n, e, s)
        with contextlib.redirect_stdout(io.StringIO()):
            co = NEW.compile_for_hardware(cq, target=T, seed_transpiler=0)
        from qiskit import transpile
        lg = transpile(cq, basis_gates=["rz", "rx", "rzz", "h", "s", "sdg", "x", "cx", "cz", "sx"], optimization_level=0)
        c0, m0 = _snap(lg)
        s0 = StabilizerState(Clifford(c0))
        # MPS on the compiled circuit's active qubits
        meas, body = {}, []
        for ins in co.data:
            nm = ins.operation.name
            if nm == "measure": meas[co.find_bit(ins.clbits[0]).index] = co.find_bit(ins.qubits[0]).index
            elif nm not in ("barrier", "delay"): body.append((ins.operation, [co.find_bit(q).index for q in ins.qubits]))
        active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
        pos = {q: i for i, q in enumerate(active)}
        red = QuantumCircuit(len(active))
        for op, qs in body: red.append(op, [pos[q] for q in qs])
        obs = [[i] for i in range(n)] + [[a, b] for a, b in e]
        for k, o in enumerate(obs):
            red.save_expectation_value(Pauli("Z" * len(o)), [pos[meas[q]] for q in o], label=f"o{k}")
        t0 = time.perf_counter()
        d = AerSimulator(method="matrix_product_state").run(red).result().data()
        def ev0(o):
            lab_ = ["I"] * c0.num_qubits
            for q in o: lab_[c0.num_qubits - 1 - m0[q]] = "Z"
            return s0.expectation_value(Pauli("".join(lab_)))
        worst = max(abs(d[f"o{k}"] - ev0(o)) for k, o in enumerate(obs))
        r = dict(device=dev, family=fam, n=n, steps=s, target="T", max_diff=float(worst), observables=len(obs), mps_s=round(time.perf_counter() - t0, 1))
        out.append(r); print(r, flush=True)
open("t_path_check.jsonl", "a").write(json.dumps(out[0]) + "\n")
